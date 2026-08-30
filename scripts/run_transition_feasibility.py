#!/usr/bin/env python3
"""Collect and test a strictly bounded Autophase-transition feasibility subset.

The only LLVM interaction is the prescribed reset / frozen-action / Autophase
sequence collection. Candidate values and policy45 prefix data are loaded after
collection from existing frozen artifacts; this script never observes ObjectText.
"""
from __future__ import annotations

import argparse
import ast
import collections
import gzip
import json
import math
import os
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from mamba_ssm import Mamba
from torch import nn


K, FEATURE_DIM, PAD_TOKEN, PAD_LENGTH = 50, 56, 124, 20
_WORKER_ENV: Any | None = None


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_jsonl_gzip(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def load_candidates(path: Path) -> list[list[int]]:
    rows = [list(ast.literal_eval(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != K or any(not 4 <= len(row) <= PAD_LENGTH for row in rows):
        raise ValueError("frozen candidate count or candidate length mismatch")
    if any(action < 0 or action >= PAD_TOKEN for row in rows for action in row):
        raise ValueError("frozen candidate action outside vocabulary")
    return rows


def validate_config(cfg: Mapping[str, Any]) -> None:
    frozen, population, collection, probes = cfg["frozen_inputs"], cfg["population"], cfg["collection"], cfg["controlled_probes"]
    if (frozen["candidate_count"], frozen["autophase_dimension"], frozen["target_formula"]) != (K, FEATURE_DIM, "softmax(raw_candidate_value / 0.05)"):
        raise ValueError("frozen K50/Autophase/target contract mismatch")
    if (population["train_complete_k50"], population["validation_complete_k50"], population["max_train_programs_per_source"], population["max_validation_programs_per_source"]) != (28159, 4488, 8, 4):
        raise ValueError("frozen subset contract mismatch")
    if collection["objecttext_observations"] != 0 or collection["retry_count"] != 0 or collection["workers"] < 1:
        raise ValueError("collection safety contract mismatch")
    if probes["seeds"] != [1, 2, 3] or probes["d_model"] != 64 or probes["total_steps"] != 1500 or probes["checkpoint_selection"] != "none; evaluate exactly at the frozen final step":
        raise ValueError("controlled probe contract mismatch")
    predictor = cfg["predictor_if_signal_supported"]
    if predictor["seeds"] != [1, 2, 3] or predictor["d_model"] != 64 or predictor["total_steps"] != 1500:
        raise ValueError("predictor contract mismatch")
    if population["final_or_ood_access"] is not False:
        raise ValueError("final/OOD access is forbidden")


def complete_targets(path: Path, split: str, expected: int, candidates: Sequence[Sequence[int]]) -> list[dict[str, Any]]:
    rows = read_jsonl_gzip(path)
    if len(rows) != expected:
        raise ValueError(f"{split} target population mismatch: {len(rows)}")
    programs = [str(row["program_id"]) for row in rows]
    if len(programs) != len(set(programs)):
        raise ValueError(f"{split} target program IDs are not unique")
    for row in rows:
        if row.get("split") != split or row.get("training_target_validity") != "valid_complete_K50":
            raise ValueError(f"{split} target is not a valid complete-K50 row")
        if row.get("candidate_ids") != list(range(K)) or row.get("candidate_sequences") != list(candidates):
            raise ValueError(f"{split} candidate ordering differs from frozen candidate configuration")
        if len(row.get("normalized_target", [])) != K or len(row.get("raw_candidate_value", [])) != K:
            raise ValueError(f"{split} candidate-value target is not K=50")
    return rows


def select_by_source(rows: Sequence[Mapping[str, Any]], maximum: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        grouped[str(row["dataset_id"])].append(row)
    selected: list[dict[str, Any]] = []
    for source in sorted(grouped):
        selected.extend(dict(row) for row in sorted(grouped[source], key=lambda item: str(item["program_id"]))[:maximum])
    if not selected or len({row["program_id"] for row in selected}) != len(selected):
        raise ValueError("invalid deterministic source-stratified selection")
    return selected


def normalize_autophase(raw: Any, program_id: str) -> list[float]:
    value = np.asarray(raw, dtype=np.float32).reshape(-1)
    if value.size != FEATURE_DIM or not np.isfinite(value).all() or value[51] <= 0:
        raise ValueError(f"invalid Autophase state for {program_id}")
    normalized = value / value[51]
    if not np.isfinite(normalized).all():
        raise ValueError(f"non-finite normalized Autophase state for {program_id}")
    return normalized.tolist()


def _init_worker() -> None:
    global _WORKER_ENV
    import compiler_gym

    _WORKER_ENV = compiler_gym.make("llvm-v0", reward_space=None)


def _collect_one(task: tuple[int, Mapping[str, Any], Sequence[Sequence[int]], str]) -> tuple[int, list[dict[str, Any]]]:
    index, row, candidates, split = task
    if _WORKER_ENV is None:
        raise RuntimeError("transition collector worker was not initialized")
    program_id, source = str(row["program_id"]), str(row["dataset_id"])
    output: list[dict[str, Any]] = []
    for candidate_id, candidate in enumerate(candidates):
        _WORKER_ENV.reset(benchmark=program_id)
        states = [normalize_autophase(_WORKER_ENV.observation["Autophase"], program_id)]
        for action in candidate:
            _, _, done, info = _WORKER_ENV.step(int(action))
            if done:
                raise RuntimeError(f"episode ended before frozen candidate completion for {program_id}: {info}")
            states.append(normalize_autophase(_WORKER_ENV.observation["Autophase"], program_id))
        output.append({"program_id": program_id, "source_id": source, "split": split, "candidate_id": candidate_id, "pass_ids": list(candidate), "states": states})
    return index, output


def validate_trajectory(row: Mapping[str, Any], candidates: Sequence[Sequence[int]], split: str) -> None:
    candidate_id, states, passes = int(row["candidate_id"]), row["states"], row["pass_ids"]
    if row.get("split") != split or candidate_id not in range(K) or list(passes) != list(candidates[candidate_id]):
        raise ValueError("trajectory identity or frozen pass order mismatch")
    if len(states) != len(passes) + 1 or any(len(state) != FEATURE_DIM for state in states):
        raise ValueError("trajectory state schema mismatch")
    if not np.isfinite(np.asarray(states, dtype=np.float32)).all():
        raise ValueError("trajectory has non-finite state")


def write_trajectories(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    temporary.replace(path)


def collect_split(selected: Sequence[Mapping[str, Any]], candidates: Sequence[Sequence[int]], split: str, workers: int, output: Path) -> tuple[int, int]:
    tasks = [(index, row, candidates, split) for index, row in enumerate(selected)]
    by_index: dict[int, list[dict[str, Any]]] = {}
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as pool:
        for index, trajectories in pool.map(_collect_one, tasks, chunksize=1):
            if len(trajectories) != K:
                raise ValueError("collector did not return a complete K=50 program")
            for row in trajectories:
                validate_trajectory(row, candidates, split)
            states0 = np.asarray([row["states"][0] for row in trajectories], dtype=np.float32)
            if not np.array_equal(states0, states0[:1].repeat(K, axis=0)):
                raise ValueError("state_0 differs across independent resets of one program")
            by_index[index] = trajectories
    ordered = [row for index in range(len(selected)) for row in by_index[index]]
    write_trajectories(output, ordered)
    return len(ordered), sum(len(row["pass_ids"]) for row in ordered)


def load_trajectories(path: Path, split: str, candidates: Sequence[Sequence[int]], selected: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in read_jsonl_gzip(path):
        validate_trajectory(row, candidates, split)
        grouped[str(row["program_id"])].append(row)
    selected_ids = {str(row["program_id"]) for row in selected}
    if set(grouped) != selected_ids:
        raise ValueError(f"{split} trajectory program IDs do not exactly match selected source subset")
    for program, rows in grouped.items():
        rows.sort(key=lambda row: int(row["candidate_id"]))
        if [row["candidate_id"] for row in rows] != list(range(K)):
            raise ValueError(f"{split} {program} trajectory candidate cohort mismatch")
    return dict(grouped)


def transition_statistics(grouped: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    deltas, pass_ids, sources, programs, positions = [], [], [], [], []
    for program, rows in grouped.items():
        for row in rows:
            states = np.asarray(row["states"], dtype=np.float64)
            current = states[1:] - states[:-1]
            deltas.append(current); pass_ids.extend(row["pass_ids"]); sources.extend([row["source_id"]] * len(current)); programs.extend([program] * len(current)); positions.extend(range(len(current)))
    matrix = np.concatenate(deltas, axis=0)
    norms = np.sqrt(np.mean(np.square(matrix), axis=1))
    changed = np.any(matrix != 0.0, axis=1)
    pass_array, source_array, program_array, position_array = np.asarray(pass_ids), np.asarray(sources), np.asarray(programs), np.asarray(positions)
    per_feature = {str(index): float(np.mean(matrix[:, index] != 0.0)) for index in range(FEATURE_DIM)}
    pass_variance, pass_program_variance, pass_position_variance = {}, {}, {}
    for action in sorted(set(pass_ids)):
        mask = pass_array == action
        action_rows = matrix[mask]
        pass_variance[str(action)] = float(np.mean(np.var(action_rows, axis=0)))
        per_program = [action_rows[program_array[mask] == program].mean(axis=0) for program in sorted(set(program_array[mask]))]
        per_position = [action_rows[position_array[mask] == position].mean(axis=0) for position in sorted(set(position_array[mask]))]
        pass_program_variance[str(action)] = float(np.mean(np.var(np.asarray(per_program), axis=0))) if len(per_program) > 1 else 0.0
        pass_position_variance[str(action)] = float(np.mean(np.var(np.asarray(per_position), axis=0))) if len(per_position) > 1 else 0.0
    source_stats = {}
    for source in sorted(set(sources)):
        current = matrix[source_array == source]
        source_stats[str(source)] = {"transition_count": int(len(current)), "nonzero_change_rate": float(np.mean(np.any(current != 0.0, axis=1))), "delta_rms_mean": float(np.mean(np.sqrt(np.mean(np.square(current), axis=1)))), "delta_rms_median": float(np.median(np.sqrt(np.mean(np.square(current), axis=1))))}
    return {"transition_count": int(len(matrix)), "nonzero_change_rate": float(np.mean(changed)), "delta_rms_mean": float(np.mean(norms)), "delta_rms_median": float(np.median(norms)), "delta_rms_p95": float(np.quantile(norms, 0.95)), "per_feature_change_frequency": per_feature, "same_pass_delta_variance_mean": float(np.mean(list(pass_variance.values()))), "same_pass_program_mean_delta_variance_mean": float(np.mean(list(pass_program_variance.values()))), "same_pass_position_mean_delta_variance_mean": float(np.mean(list(pass_position_variance.values()))), "per_pass_delta_variance": pass_variance, "per_pass_program_mean_delta_variance": pass_program_variance, "per_pass_position_mean_delta_variance": pass_position_variance, "source_statistics": source_stats}


def transition_summary(states: np.ndarray) -> np.ndarray:
    delta = states[1:] - states[:-1]
    return np.concatenate((states[-1] - states[0], delta.mean(axis=0), np.abs(delta).max(axis=0), np.abs(delta).sum(axis=0)), axis=0).astype(np.float32)


def build_probe_arrays(grouped: Mapping[str, Sequence[Mapping[str, Any]]], selected: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    target_by_program = {str(row["program_id"]): row for row in selected}
    initials, summaries, targets, records = [], [], [], []
    for program in sorted(grouped):
        trajectory_rows, target = grouped[program], target_by_program[program]
        states0 = np.asarray([row["states"][0] for row in trajectory_rows], dtype=np.float32)
        if not np.array_equal(states0, states0[:1].repeat(K, axis=0)):
            raise ValueError("state_0 does not correspond consistently to reset benchmark")
        initials.append(states0[0]); summaries.append(np.stack([transition_summary(np.asarray(row["states"], dtype=np.float32)) for row in trajectory_rows])); targets.append(np.asarray(target["normalized_target"], dtype=np.float32)); records.append(target)
    return np.stack(initials), np.stack(summaries), np.stack(targets), records


def read_selected_label_matrix(shards: Path, selected: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    wanted = {str(row["program_id"]) for row in selected}
    result: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(shards.glob("*.json.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        summary = payload["program_summary"]
        program = str(summary["program_id"])
        if program not in wanted:
            continue
        if summary.get("program_training_target_validity") != "valid_complete_K50":
            raise ValueError("selected validation program is not complete K50 in frozen labels")
        records = sorted(payload["records"], key=lambda row: row["candidate_id"])
        if len(records) != K or [row["candidate_id"] for row in records] != list(range(K)):
            raise ValueError("selected validation label candidate order mismatch")
        result[program] = records
    if set(result) != wanted:
        raise ValueError("selected validation targets do not exactly match frozen label matrix")
    return result


def soft_ce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return -(targets * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()


class CandidateValueProbe(nn.Module):
    def __init__(self, candidates: Sequence[Sequence[int]], with_transition: bool, d_model: int = 64) -> None:
        super().__init__()
        tokens = torch.full((K, PAD_LENGTH), PAD_TOKEN, dtype=torch.long)
        lengths = torch.empty(K, dtype=torch.long)
        for index, row in enumerate(candidates):
            tokens[index, :len(row)] = torch.tensor(row, dtype=torch.long); lengths[index] = len(row)
        self.register_buffer("tokens", tokens); self.register_buffer("lengths", lengths)
        self.with_transition = with_transition
        self.program = nn.Linear(FEATURE_DIM, d_model)
        self.embedding = nn.Embedding(PAD_TOKEN + 1, d_model, padding_idx=PAD_TOKEN)
        self.position = nn.Embedding(PAD_LENGTH, d_model)
        self.sequence = nn.GRU(d_model, d_model, batch_first=True)
        self.transition = nn.Linear(4 * FEATURE_DIM, d_model) if with_transition else None
        width = 2 * d_model + (d_model if with_transition else 0)
        self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, d_model), nn.GELU(), nn.Linear(d_model, 1))

    def forward(self, initial: torch.Tensor, summaries: torch.Tensor | None = None) -> torch.Tensor:
        batch = len(initial); device = initial.device
        tokens = self.tokens.unsqueeze(0).expand(batch, -1, -1).reshape(batch * K, PAD_LENGTH)
        positions = torch.arange(PAD_LENGTH, device=device)
        conditioned = self.embedding(tokens) + self.position(positions)[None] + self.program(initial)[:, None, :].expand(-1, K, -1).reshape(batch * K, 1, -1)
        encoded, _ = self.sequence(conditioned)
        last = encoded[torch.arange(batch * K, device=device), self.lengths.repeat(batch).to(device) - 1].reshape(batch, K, -1)
        parts = [last, self.program(initial)[:, None, :].expand(-1, K, -1)]
        if self.with_transition:
            if summaries is None or self.transition is None:
                raise ValueError("transition oracle requires real transition summary")
            parts.append(self.transition(summaries))
        return self.head(torch.cat(parts, dim=-1)).squeeze(-1)


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sample_source_balanced(dataset_ids: Sequence[str], batch_size: int, generator: torch.Generator, device: torch.device) -> torch.Tensor:
    pools: dict[str, list[int]] = collections.defaultdict(list)
    for index, source in enumerate(dataset_ids):
        pools[source].append(index)
    sources = sorted(pools)
    selected = torch.randint(len(sources), (batch_size,), generator=generator)
    result = torch.empty(batch_size, dtype=torch.long)
    for source_index in selected.unique().tolist():
        mask, pool = selected == source_index, pools[sources[source_index]]
        result[mask] = torch.tensor(pool, dtype=torch.long)[torch.randint(len(pool), (int(mask.sum()),), generator=generator)]
    return result.to(device)


def policy_metrics(logits: np.ndarray, records: Sequence[Mapping[str, Any]], matrix: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    by_source: dict[str, list[float]] = collections.defaultdict(list)
    regrets, top1 = [], 0
    for scores, row in zip(logits, records):
        budget, observed = 45, []
        for candidate in sorted(range(K), key=lambda index: (-float(scores[index]), index)):
            prefix = matrix[str(row["program_id"])][candidate]["prefix_object_text_size_bytes"]
            take = min(budget, len(prefix)); observed.extend(prefix[:take]); budget -= take
            if budget == 0:
                break
        if budget != 0:
            raise ValueError("frozen policy45 did not consume exactly 45 pass observations")
        policy = min(observed); oracle = min(row["best_object_text_size"]); oz = float(row["S_Oz"])
        by_source[str(row["dataset_id"])].append((oz - policy) / oz)
        regrets.append(policy - oracle)
        chosen = int(np.argmax(scores)); top1 += int(row["best_object_text_size"][chosen] == oracle)
    per_source = {source: float(np.mean(values)) for source, values in sorted(by_source.items())}
    return {"candidate_top1_oracle_tie_accuracy": top1 / len(records), "policy45_mean_over_oz": float(np.mean(list(per_source.values()))), "policy45_regret_mean_bytes": float(np.mean(regrets)), "per_source_policy45_mean_over_oz": per_source}


def evaluate_probe(model: CandidateValueProbe, initial: torch.Tensor, summary: torch.Tensor, targets: torch.Tensor, records: Sequence[Mapping[str, Any]], matrix: Mapping[str, Sequence[Mapping[str, Any]],], batch_size: int) -> dict[str, Any]:
    rows, loss_sum = [], 0.0
    model.eval()
    with torch.no_grad():
        for start in range(0, len(initial), batch_size):
            current_summary = summary[start:start + batch_size] if model.with_transition else None
            scores = model(initial[start:start + batch_size], current_summary)
            rows.append(scores.cpu()); loss_sum += float(soft_ce(scores, targets[start:start + len(scores)]).cpu()) * len(scores)
    result = policy_metrics(torch.cat(rows).numpy(), records, matrix)
    result["candidate_value_cross_entropy"] = loss_sum / len(initial)
    return result


def run_probe(kind: str, candidates: Sequence[Sequence[int]], cfg: Mapping[str, Any], train_initial: np.ndarray, train_summary: np.ndarray, train_target: np.ndarray, train_records: Sequence[Mapping[str, Any]], val_initial: np.ndarray, val_summary: np.ndarray, val_target: np.ndarray, val_records: Sequence[Mapping[str, Any]], matrix: Mapping[str, Sequence[Mapping[str, Any]]], device: torch.device) -> dict[str, Any]:
    with_transition = kind == "REAL_TRANSITION_ORACLE"
    reports = []
    train_x, train_s, train_y = (torch.tensor(train_initial, device=device), torch.tensor(train_summary, device=device), torch.tensor(train_target, device=device))
    val_x, val_s, val_y = (torch.tensor(val_initial, device=device), torch.tensor(val_summary, device=device), torch.tensor(val_target, device=device))
    for seed in cfg["seeds"]:
        seed_all(int(seed)); model = CandidateValueProbe(candidates, with_transition, int(cfg["d_model"])).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"]))
        generator = torch.Generator().manual_seed(int(seed)); curve = []
        for step in range(1, int(cfg["total_steps"]) + 1):
            index = sample_source_balanced([str(row["dataset_id"]) for row in train_records], int(cfg["batch_size"]), generator, device)
            model.train(); scores = model(train_x[index], train_s[index] if with_transition else None); loss = soft_ce(scores, train_y[index])
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            if step % 250 == 0 or step == int(cfg["total_steps"]):
                curve.append({"step": step, "train_candidate_value_cross_entropy": float(loss.detach().cpu())})
        metric = evaluate_probe(model, val_x, val_s, val_y, val_records, matrix, int(cfg["batch_size"]))
        reports.append({"seed": int(seed), "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad), "training_curve": curve, "validation": metric})
    keys = ["candidate_value_cross_entropy", "policy45_mean_over_oz", "candidate_top1_oracle_tie_accuracy", "policy45_regret_mean_bytes"]
    aggregate = {key: float(np.mean([report["validation"][key] for report in reports])) for key in keys}
    per_source = {source: float(np.mean([report["validation"]["per_source_policy45_mean_over_oz"][source] for report in reports])) for source in reports[0]["validation"]["per_source_policy45_mean_over_oz"]}
    return {"name": kind, "input": "initial Autophase + ordered pass sequence" + (" + real transition summary" if with_transition else ""), "seeds": reports, "three_seed_mean": aggregate, "three_seed_per_source_policy45_mean_over_oz": per_source}


class TransitionPredictor(nn.Module):
    def __init__(self, d_model: int, d_state: int, d_conv: int, expand: int) -> None:
        super().__init__()
        self.program = nn.Linear(FEATURE_DIM, d_model)
        self.embedding = nn.Embedding(PAD_TOKEN + 1, d_model, padding_idx=PAD_TOKEN)
        self.position = nn.Embedding(PAD_LENGTH, d_model)
        self.norms = nn.ModuleList([nn.LayerNorm(d_model), nn.LayerNorm(d_model)])
        self.blocks = nn.ModuleList([Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand, layer_idx=index) for index in range(2)])
        self.output = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, FEATURE_DIM))

    def forward(self, initial: torch.Tensor, histories: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(PAD_LENGTH, device=initial.device)
        hidden = self.embedding(histories) + self.position(positions)[None] + self.program(initial)[:, None, :]
        for norm, block in zip(self.norms, self.blocks):
            hidden = hidden + block(norm(hidden))
        return self.output(hidden[torch.arange(len(hidden), device=initial.device), lengths - 1])


def transition_examples(grouped: Mapping[str, Sequence[Mapping[str, Any]]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    initial, histories, lengths, target, sources = [], [], [], [], []
    for rows in grouped.values():
        for row in rows:
            states, passes = np.asarray(row["states"], dtype=np.float32), list(row["pass_ids"])
            for position in range(len(passes)):
                history = passes[:position + 1] + [PAD_TOKEN] * (PAD_LENGTH - position - 1)
                initial.append(states[0]); histories.append(history); lengths.append(position + 1); target.append(states[position + 1] - states[position]); sources.append(str(row["source_id"]))
    return np.asarray(initial), np.asarray(histories), np.asarray(lengths), np.asarray(target), sources


def error_metrics(prediction: np.ndarray, target: np.ndarray, scale: np.ndarray, sources: Sequence[str]) -> dict[str, Any]:
    normalized = (prediction - target) / scale
    squared, absolute = np.mean(np.square(normalized), axis=1), np.mean(np.abs(normalized), axis=1)
    source_array = np.asarray(sources)
    return {"normalized_mse": float(np.mean(squared)), "normalized_mae": float(np.mean(absolute)), "per_feature_normalized_mse": np.mean(np.square(normalized), axis=0).tolist(), "per_source_normalized_mse": {source: float(np.mean(squared[source_array == source])) for source in sorted(set(sources))}}


def run_predictor(cfg: Mapping[str, Any], train_grouped: Mapping[str, Sequence[Mapping[str, Any]]], val_grouped: Mapping[str, Sequence[Mapping[str, Any]]], device: torch.device) -> dict[str, Any]:
    train_x, train_h, train_l, train_y, train_sources = transition_examples(train_grouped)
    val_x, val_h, val_l, val_y, val_sources = transition_examples(val_grouped)
    scale = np.sqrt(np.mean(np.square(train_y), axis=0)).clip(1e-6)
    global_mean = train_y.mean(axis=0)
    pass_sum, pass_count = np.zeros((PAD_TOKEN, FEATURE_DIM), dtype=np.float64), np.zeros(PAD_TOKEN, dtype=np.int64)
    for action, delta in zip(train_h[:, 0], train_y): pass_sum[action] += delta; pass_count[action] += 1
    per_pass = np.zeros_like(val_y)
    for index, action in enumerate(val_h[:, 0]): per_pass[index] = pass_sum[action] / pass_count[action] if pass_count[action] else global_mean
    baselines = {"zero_delta": error_metrics(np.zeros_like(val_y), val_y, scale, val_sources), "global_train_mean_delta": error_metrics(np.repeat(global_mean[None], len(val_y), axis=0), val_y, scale, val_sources), "per_pass_train_mean_delta": error_metrics(per_pass, val_y, scale, val_sources)}
    tx, th, tl, ty = (torch.tensor(train_x, device=device), torch.tensor(train_h, device=device), torch.tensor(train_l, device=device), torch.tensor(train_y, device=device))
    vx, vh, vl = (torch.tensor(val_x, device=device), torch.tensor(val_h, device=device), torch.tensor(val_l, device=device))
    reports = []
    for seed in cfg["seeds"]:
        seed_all(int(seed)); model = TransitionPredictor(int(cfg["d_model"]), int(cfg["mamba_d_state"]), int(cfg["mamba_d_conv"]), int(cfg["mamba_expand"])).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"])); generator = torch.Generator().manual_seed(int(seed))
        for _ in range(int(cfg["total_steps"])):
            index = torch.randint(len(tx), (int(cfg["batch_size"]),), generator=generator, device=device)
            prediction = model(tx[index], th[index], tl[index]); loss = torch.mean(torch.square((prediction - ty[index]) / torch.tensor(scale, device=device)))
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad(): prediction = model(vx, vh, vl).cpu().numpy()
        reports.append({"seed": int(seed), "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad), "validation": error_metrics(prediction, val_y, scale, val_sources)})
    learned = {"normalized_mse": float(np.mean([row["validation"]["normalized_mse"] for row in reports])), "normalized_mae": float(np.mean([row["validation"]["normalized_mae"] for row in reports])), "per_feature_normalized_mse": np.mean([row["validation"]["per_feature_normalized_mse"] for row in reports], axis=0).tolist(), "per_source_normalized_mse": {source: float(np.mean([row["validation"]["per_source_normalized_mse"][source] for row in reports])) for source in sorted(set(val_sources))}}
    reference = baselines["per_pass_train_mean_delta"]["per_source_normalized_mse"]
    deltas = {source: reference[source] - learned["per_source_normalized_mse"][source] for source in reference}
    supported = learned["normalized_mse"] < min(row["normalized_mse"] for row in baselines.values()) and learned["normalized_mae"] < baselines["per_pass_train_mean_delta"]["normalized_mae"] and sum(value > 0 for value in deltas.values()) > sum(value < 0 for value in deltas.values())
    return {"executed": True, "train_transition_count": int(len(train_y)), "validation_transition_count": int(len(val_y)), "normalization_scale": "per-feature train delta RMS clamped at 1e-6", "baselines": baselines, "learned": learned, "per_source_mse_improvement_vs_per_pass_mean": deltas, "positive_source_count": sum(value > 0 for value in deltas.values()), "negative_source_count": sum(value < 0 for value in deltas.values()), "seed_reports": reports, "decision": "TRANSITION_PREDICTABILITY_SUPPORTED" if supported else "TRANSITION_PREDICTABILITY_NOT_SUPPORTED"}


def signal_decision(base: Mapping[str, Any], oracle: Mapping[str, Any]) -> dict[str, Any]:
    base_mean, oracle_mean = base["three_seed_mean"], oracle["three_seed_mean"]
    per_source_delta = {source: oracle["three_seed_per_source_policy45_mean_over_oz"][source] - base["three_seed_per_source_policy45_mean_over_oz"][source] for source in base["three_seed_per_source_policy45_mean_over_oz"]}
    seed_policy_delta = [oracle["seeds"][index]["validation"]["policy45_mean_over_oz"] - base["seeds"][index]["validation"]["policy45_mean_over_oz"] for index in range(3)]
    conditions = {"lower_three_seed_candidate_cross_entropy": oracle_mean["candidate_value_cross_entropy"] < base_mean["candidate_value_cross_entropy"], "higher_policy45_in_at_least_two_paired_seeds": sum(value > 0 for value in seed_policy_delta) >= 2, "positive_median_source_policy_delta": float(np.median(list(per_source_delta.values()))) > 0, "positive_sources_outnumber_negative_sources": sum(value > 0 for value in per_source_delta.values()) > sum(value < 0 for value in per_source_delta.values())}
    return {"base_to_oracle_three_seed_delta": {key: oracle_mean[key] - base_mean[key] for key in oracle_mean}, "per_source_policy45_delta": per_source_delta, "median_source_policy45_delta": float(np.median(list(per_source_delta.values()))), "positive_source_count": sum(value > 0 for value in per_source_delta.values()), "negative_source_count": sum(value < 0 for value in per_source_delta.values()), "paired_seed_policy45_deltas": seed_policy_delta, "gate_conditions": conditions, "decision": "TRANSITION_SIGNAL_SUPPORTED" if all(conditions.values()) else "TRANSITION_SIGNAL_NOT_SUPPORTED"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/transition_feasibility_v1.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/transition_feasibility_v1"))
    args = parser.parse_args()
    cfg = read_json(args.config); validate_config(cfg)
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing experiment directory: {args.output_dir}")
    candidates = load_candidates(Path(cfg["frozen_inputs"]["candidate_sequences"]))
    train = complete_targets(Path(cfg["frozen_inputs"]["train_targets"]), "train", 28159, candidates)
    validation = complete_targets(Path(cfg["frozen_inputs"]["validation_targets"]), "validation", 4488, candidates)
    selected_train = select_by_source(train, int(cfg["population"]["max_train_programs_per_source"]))
    selected_validation = select_by_source(validation, int(cfg["population"]["max_validation_programs_per_source"]))
    args.output_dir.mkdir(parents=True)
    frozen_cfg = dict(cfg)
    frozen_cfg["audit"] = {"existing_intermediate_autophase_states": False, "audited_artifacts": [cfg["frozen_inputs"]["train_targets"], cfg["frozen_inputs"]["validation_targets"], "outputs/autophase_feature_cache_v6", "outputs/rlcompopt_route_a_objecttext_v6_parallel12/shards", "scripts/generate_rlcompopt_objecttext_labels.py", "scripts/export_autophase_feature_cache.py", "project_notes/progress.md"], "finding": "existing target and ObjectText shards do not contain state_0..state_L Autophase; feature cache contains initial Autophase only"}
    frozen_cfg["selected_population"] = {"train": [{"program_id": row["program_id"], "dataset_id": row["dataset_id"]} for row in selected_train], "validation": [{"program_id": row["program_id"], "dataset_id": row["dataset_id"]} for row in selected_validation]}
    write_json(args.output_dir / "config.json", frozen_cfg)
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[variable] = "1"
    train_count, train_transitions = collect_split(selected_train, candidates, "train", int(cfg["collection"]["workers"]), args.output_dir / "train_transitions.jsonl.gz")
    val_count, val_transitions = collect_split(selected_validation, candidates, "validation", int(cfg["collection"]["workers"]), args.output_dir / "validation_transitions.jsonl.gz")
    grouped_train = load_trajectories(args.output_dir / "train_transitions.jsonl.gz", "train", candidates, selected_train)
    grouped_val = load_trajectories(args.output_dir / "validation_transitions.jsonl.gz", "validation", candidates, selected_validation)
    train_initial, train_summary, train_target, train_records = build_probe_arrays(grouped_train, selected_train)
    val_initial, val_summary, val_target, val_records = build_probe_arrays(grouped_val, selected_validation)
    matrix = read_selected_label_matrix(Path(cfg["frozen_inputs"]["validation_label_shards"]), val_records)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = run_probe("BASE", candidates, cfg["controlled_probes"], train_initial, train_summary, train_target, train_records, val_initial, val_summary, val_target, val_records, matrix, device)
    oracle = run_probe("REAL_TRANSITION_ORACLE", candidates, cfg["controlled_probes"], train_initial, train_summary, train_target, train_records, val_initial, val_summary, val_target, val_records, matrix, device)
    signal = signal_decision(base, oracle)
    predictor = run_predictor(cfg["predictor_if_signal_supported"], grouped_train, grouped_val, device) if signal["decision"] == "TRANSITION_SIGNAL_SUPPORTED" else {"executed": False, "decision": "NOT_EXECUTED_TRANSITION_SIGNAL_NOT_SUPPORTED"}
    report = {"step_execution": "COMPLETE", "audit": frozen_cfg["audit"], "collection": {"train_program_count": len(selected_train), "validation_program_count": len(selected_validation), "train_candidate_trajectory_count": train_count, "validation_candidate_trajectory_count": val_count, "train_pass_transition_count": train_transitions, "validation_pass_transition_count": val_transitions, "train_sources": sorted({row["dataset_id"] for row in selected_train}), "validation_sources": sorted({row["dataset_id"] for row in selected_validation}), "compiler_actions": train_transitions + val_transitions, "objecttext_observations": 0, "final_or_ood_accessed": False}, "transition_statistics": {"train": transition_statistics(grouped_train), "validation": transition_statistics(grouped_val)}, "controlled_probes": {"BASE": base, "REAL_TRANSITION_ORACLE": oracle}, "transition_signal": signal, "transition_predictor": predictor, "final_decision": {"transition_signal": signal["decision"], "transition_predictability": predictor["decision"]}}
    write_json(args.output_dir / "experiment_report.json", report)
    print(json.dumps(report["final_decision"], sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
