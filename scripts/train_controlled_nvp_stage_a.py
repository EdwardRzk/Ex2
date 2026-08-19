#!/usr/bin/env python3
"""Run frozen Stage-A controlled ObjectText NVP architecture selection."""
from __future__ import annotations

import argparse
import ast
import gzip
import json
import math
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from mamba_ssm import Mamba
from torch import nn

K, AUTOPHASE_DIM = 50, 56
_ENV: Any = None


def _init_feature_worker() -> None:
    global _ENV
    import compiler_gym
    _ENV = compiler_gym.make("llvm-v0", reward_space=None)


def _feature(program_id: str) -> tuple[str, list[float]]:
    _ENV.reset(benchmark=program_id)
    raw = np.asarray(_ENV.observation["Autophase"], dtype=np.float32).reshape(-1)
    if raw.size != AUTOPHASE_DIM or raw[51] <= 0:
        raise ValueError(f"invalid Autophase feature for {program_id}")
    return program_id, (raw / raw[51]).tolist()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def read_label_matrix(shards: Path) -> dict[str, list[dict[str, Any]]]:
    matrix: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(shards.glob("*.json.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        summary, records = payload["program_summary"], payload["records"]
        if summary["program_training_target_validity"] == "valid_complete_K50":
            matrix[summary["program_id"]] = sorted(records, key=lambda row: row["candidate_id"])
    return matrix


def extract_features(records: list[dict[str, Any]], workers: int) -> dict[str, list[float]]:
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_feature_worker) as pool:
        return dict(pool.map(_feature, (row["program_id"] for row in records), chunksize=32))


def load_candidates(path: Path, *, pad_token_id: int, padded_length: int) -> tuple[torch.Tensor, torch.Tensor]:
    sequences = [list(ast.literal_eval(line)) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if len(sequences) != K or max(map(len, sequences)) != padded_length:
        raise ValueError("candidate count or frozen padded length mismatch")
    if min(map(len, sequences)) < 1 or any(action < 0 or action >= pad_token_id for row in sequences for action in row):
        raise ValueError("candidate action is outside frozen vocabulary")
    tokens = torch.full((K, padded_length), pad_token_id, dtype=torch.long)
    lengths = torch.tensor([len(row) for row in sequences], dtype=torch.long)
    for index, row in enumerate(sequences):
        tokens[index, : len(row)] = torch.tensor(row, dtype=torch.long)
    return tokens, lengths


class _CommonCandidateInterface(nn.Module):
    def __init__(self, cfg: Mapping[str, Any], tokens: torch.Tensor, lengths: torch.Tensor) -> None:
        super().__init__()
        d_model, max_length = int(cfg["d_model"]), int(cfg["padded_length"])
        self.d_model = d_model
        self.register_buffer("candidate_tokens", tokens)
        self.register_buffer("candidate_lengths", lengths)
        self.program_projection = nn.Linear(AUTOPHASE_DIM, d_model)
        self.token_embedding = nn.Embedding(int(cfg["vocabulary_size"]), d_model, padding_idx=int(cfg["pad_token_id"]))
        self.position_embedding = nn.Embedding(max_length, d_model)
        self.input_norm = nn.LayerNorm(d_model)

    def candidate_inputs(self, program: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, length = program.shape[0], self.candidate_tokens.shape[1]
        tokens = self.candidate_tokens.unsqueeze(0).expand(batch, -1, -1)
        positions = torch.arange(length, device=program.device)
        condition = self.program_projection(program)[:, None, None, :]
        hidden = self.input_norm(self.token_embedding(tokens) + self.position_embedding(positions)[None, None, :, :] + condition)
        lengths = self.candidate_lengths.unsqueeze(0).expand(batch, -1).reshape(-1)
        return hidden.reshape(batch * K, length, self.d_model), lengths, tokens.reshape(batch * K, length)


class ControlledCandidateModel(_CommonCandidateInterface):
    def __init__(self, name: str, cfg: Mapping[str, Any], tokens: torch.Tensor, lengths: torch.Tensor) -> None:
        super().__init__(cfg, tokens, lengths)
        self.name = name
        d_model = self.d_model
        if name == "MLP":
            self.encoder = nn.Sequential(nn.Linear(int(cfg["padded_length"]) * d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model), nn.ReLU())
        elif name == "LSTM":
            self.encoder = nn.LSTM(d_model, d_model, num_layers=int(cfg["layers"]), dropout=0.0, batch_first=True)
        elif name == "Transformer":
            layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=int(cfg["attention_heads"]), dim_feedforward=int(cfg["feedforward_dimension"]), dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(layer, num_layers=int(cfg["layers"]), norm=nn.LayerNorm(d_model))
        elif name == "Mamba":
            self.block_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(int(cfg["layers"]))])
            self.blocks = nn.ModuleList([Mamba(d_model=d_model, d_state=int(cfg["d_state"]), d_conv=int(cfg["d_conv"]), expand=int(cfg["expand"]), use_fast_path=bool(cfg["use_fast_path"]), layer_idx=index) for index in range(int(cfg["layers"]))])
        else:
            raise ValueError(f"unsupported controlled architecture: {name}")
        self.output_norm = nn.LayerNorm(d_model)
        self.value_head = nn.Linear(d_model, 1)

    def forward(self, program: torch.Tensor) -> torch.Tensor:
        hidden, lengths, tokens = self.candidate_inputs(program)
        if self.name == "MLP":
            mask = (tokens != self.token_embedding.padding_idx).unsqueeze(-1)
            encoded = self.encoder((hidden * mask).reshape(hidden.shape[0], -1))
        elif self.name == "LSTM":
            encoded, _ = self.encoder(hidden)
            encoded = encoded[torch.arange(len(encoded), device=hidden.device), lengths - 1]
        elif self.name == "Transformer":
            positions = torch.arange(hidden.shape[1], device=hidden.device)
            encoded = self.encoder(hidden, src_key_padding_mask=positions[None, :] >= lengths[:, None])
            encoded = encoded[torch.arange(len(encoded), device=hidden.device), lengths - 1]
        else:
            encoded = hidden
            for norm, block in zip(self.block_norms, self.blocks):
                encoded = encoded + block(norm(encoded))
            encoded = encoded[torch.arange(len(encoded), device=hidden.device), lengths - 1]
        return self.value_head(self.output_norm(encoded)).reshape(program.shape[0], K)


def soft_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return -(targets * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()


def policy_metrics(logits: torch.Tensor, records: list[dict[str, Any]], matrix: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    by_dataset: dict[str, list[float]] = {}
    regrets: list[int] = []
    positives = 0
    for score, target in zip(logits.tolist(), records):
        budget, observed = 45, []
        for candidate_id in sorted(range(K), key=lambda index: (-score[index], index)):
            prefix = matrix[target["program_id"]][candidate_id]["prefix_object_text_size_bytes"]
            take = min(budget, len(prefix))
            observed.extend(prefix[:take]); budget -= take
            if budget == 0:
                break
        if not observed:
            raise ValueError(f"empty policy rollout: {target['program_id']}")
        policy, oracle, oz = min(observed), min(target["best_object_text_size"]), target["S_Oz"]
        reduction = (oz - policy) / oz
        by_dataset.setdefault(target["dataset_id"], []).append(reduction)
        regrets.append(policy - oracle)
        positives += int(reduction > 0)
    per_dataset = {name: sum(values) / len(values) for name, values in sorted(by_dataset.items())}
    return {"per_dataset": per_dataset, "ValidationFinalMeanOverOz": sum(per_dataset.values()) / len(per_dataset), "policy45_regret_mean_bytes": float(np.mean(regrets)), "policy45_regret_median_bytes": float(np.median(regrets)), "positive_program_count_vs_Oz": positives, "N_total": len(records), "N_primary_valid": len(records), "N_failed_or_invalid": 0}


def evaluate(model: nn.Module, features: torch.Tensor, targets: torch.Tensor, records: list[dict[str, Any]], matrix: dict[str, list[dict[str, Any]]], batch_size: int) -> dict[str, Any]:
    model.eval()
    logits_parts, total_ce = [], 0.0
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            current = model(features[start:start + batch_size])
            count = len(current)
            logits_parts.append(current.cpu())
            total_ce += float(soft_cross_entropy(current, targets[start:start + count]).cpu()) * count
    metric = policy_metrics(torch.cat(logits_parts), records, matrix)
    metric["validation_ce"] = total_ce / len(features)
    return metric


def learning_rate(cfg: Mapping[str, Any], step: int) -> float:
    base, warmup, total = float(cfg["learning_rate"]), int(cfg["warmup_steps"]), int(cfg["total_steps"])
    return base * (step / warmup if step <= warmup else (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * (step - warmup) / (total - warmup)))))


def train_one(name: str, model_cfg: Mapping[str, Any], common: Mapping[str, Any], tokens: torch.Tensor, lengths: torch.Tensor, train_x: torch.Tensor, train_y: torch.Tensor, val_x: torch.Tensor, val_y: torch.Tensor, val_records: list[dict[str, Any]], matrix: dict[str, list[dict[str, Any]]], output_dir: Path) -> dict[str, Any]:
    torch.manual_seed(int(common["training"]["seed"]))
    model = ControlledCandidateModel(name, {**common["candidate_representation"], **model_cfg}, tokens, lengths).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(common["training"]["learning_rate"]), weight_decay=float(common["training"]["weight_decay"]))
    rng = torch.Generator().manual_seed(int(common["training"]["seed"]))
    out = output_dir / name.lower(); out.mkdir()
    curve, best, step, epoch_offset = [], None, 0, 0
    batch, total = int(common["training"]["batch_size"]), int(common["training"]["total_steps"])
    while step < total:
        order = torch.randperm(len(train_x), generator=rng)
        for begin in range(0, len(order), batch):
            step += 1
            index = order[begin:begin + batch].cuda(non_blocking=True)
            lr = learning_rate(common["training"], step)
            for group in optimizer.param_groups:
                group["lr"] = lr
            model.train(); loss = soft_cross_entropy(model(train_x[index]), train_y[index]); optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            if step % int(common["training"]["checkpoint_evaluation_cadence_steps"]) == 0 or step == total:
                metric = evaluate(model, val_x, val_y, val_records, matrix, int(common["training"]["evaluation_batch_size"]))
                metric.update({"step": step, "train_loss": float(loss.detach().cpu()), "lr": lr})
                curve.append(metric)
                if best is None or metric["ValidationFinalMeanOverOz"] > best["ValidationFinalMeanOverOz"]:
                    best = metric
                    torch.save({"architecture": name, "model_config": dict(model_cfg), "state_dict": model.state_dict(), "step": step, "metrics": metric}, out / "model.pt")
            if step == total:
                break
        epoch_offset += 1
    (out / "learning_curve.json").write_text(json.dumps(curve, indent=2, sort_keys=True) + "\n")
    (out / "experiment_report.json").write_text(json.dumps({"architecture": name, "step_execution": "COMPLETE", "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad), "selection_metric": "ValidationFinalMeanOverOz policy-45 dataset macro mean", "selected": best}, indent=2, sort_keys=True) + "\n")
    return {"architecture": name, "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad), "selected": best}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-shards", type=Path, required=True)
    parser.add_argument("--validation-shards", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    random.seed(cfg["training"]["seed"]); np.random.seed(cfg["training"]["seed"]); torch.manual_seed(cfg["training"]["seed"])
    if not torch.cuda.is_available():
        raise RuntimeError("Stage A requires CUDA")
    train, validation = read_jsonl(Path(cfg["target_files"]["train"])), read_jsonl(Path(cfg["target_files"]["validation"]))
    if len(train) != 28159 or len(validation) != 4488:
        raise ValueError("unexpected frozen target population")
    tokens, lengths = load_candidates(Path(cfg["candidate_representation"]["candidate_sequences"]), pad_token_id=int(cfg["candidate_representation"]["pad_token_id"]), padded_length=int(cfg["candidate_representation"]["padded_length"]))
    matrix = read_label_matrix(args.validation_shards)
    if set(matrix) != {row["program_id"] for row in validation}:
        raise ValueError("validation target/label cohort mismatch")
    train_features, validation_features = extract_features(train, args.workers), extract_features(validation, args.workers)
    train_x = torch.tensor([train_features[row["program_id"]] for row in train], device="cuda")
    train_y = torch.tensor([row["normalized_target"] for row in train], device="cuda")
    val_x = torch.tensor([validation_features[row["program_id"]] for row in validation], device="cuda")
    val_y = torch.tensor([row["normalized_target"] for row in validation], device="cuda")
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "shared_interface_config.json").write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n")
    results = [train_one(name, model_cfg, cfg, tokens.cuda(), lengths.cuda(), train_x, train_y, val_x, val_y, validation, matrix, args.output_dir) for name, model_cfg in cfg["models"].items()]
    report = {"step_execution": "COMPLETE", "selection_metric": "ValidationFinalMeanOverOz policy-45 dataset macro mean", "sampling": False, "offline_label_matrix_only": True, "train_programs": len(train), "validation_programs": len(validation), "validation_cohort": {"N_total": len(validation), "N_primary_valid": len(validation), "N_failed_or_invalid": 0}, "frozen_autophase_nvp_reference": 0.0629247173, "fixed_route_a_oracle": 0.07743661591867755, "models": results}
    (args.output_dir / "comparison_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
