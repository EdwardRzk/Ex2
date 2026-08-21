#!/usr/bin/env python3
"""Train the validation-only SetConditionedMambaRanker from frozen offline artifacts."""
from __future__ import annotations

import argparse
import ast
import copy
import gzip
import json
import math
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from mamba_ssm import Mamba
from torch import nn


K, AUTOPHASE_DIM = 50, 56
METHOD = "SetConditionedMambaRanker"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def load_feature_cache(path: Path, split: str, expected_program_ids: Sequence[str]) -> dict[str, list[float]]:
    features: dict[str, list[float]] = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["split"] != split or len(row["raw_autophase"]) != AUTOPHASE_DIM or len(row["normalized_autophase"]) != AUTOPHASE_DIM:
                raise ValueError(f"invalid cached Autophase row in {path}")
            program_id = str(row["program_id"])
            if program_id in features:
                raise ValueError(f"duplicate cached feature: {program_id}")
            features[program_id] = list(row["normalized_autophase"])
    if set(features) != set(expected_program_ids):
        raise ValueError(f"cached feature population mismatch: {path}")
    return features


def read_label_matrix(shards: Path) -> dict[str, list[dict[str, Any]]]:
    matrix: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(shards.glob("*.json.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        summary, records = payload["program_summary"], payload["records"]
        if summary["program_training_target_validity"] == "valid_complete_K50":
            ordered = sorted(records, key=lambda row: row["candidate_id"])
            if len(ordered) != K or [row["candidate_id"] for row in ordered] != list(range(K)):
                raise ValueError(f"invalid frozen K50 labels: {path}")
            matrix[str(summary["program_id"])] = ordered
    return matrix


def load_candidates(path: Path, *, pad_token_id: int, padded_length: int) -> tuple[torch.Tensor, torch.Tensor]:
    sequences = [list(ast.literal_eval(line)) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if len(sequences) != K or max(map(len, sequences)) != padded_length or min(map(len, sequences)) < 1:
        raise ValueError("frozen candidate count or padded length mismatch")
    if any(action < 0 or action >= pad_token_id for sequence in sequences for action in sequence):
        raise ValueError("candidate action is outside frozen vocabulary")
    tokens = torch.full((K, padded_length), pad_token_id, dtype=torch.long)
    lengths = torch.tensor([len(sequence) for sequence in sequences], dtype=torch.long)
    for candidate_id, sequence in enumerate(sequences):
        tokens[candidate_id, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
    return tokens, lengths


class CandidateInterface(nn.Module):
    def __init__(self, cfg: Mapping[str, Any], tokens: torch.Tensor, lengths: torch.Tensor) -> None:
        super().__init__()
        self.d_model, self.padded_length = int(cfg["d_model"]), int(cfg["padded_length"])
        self.register_buffer("candidate_tokens", tokens)
        self.register_buffer("candidate_lengths", lengths)
        self.program_projection = nn.Linear(AUTOPHASE_DIM, self.d_model)
        self.token_embedding = nn.Embedding(int(cfg["vocabulary_size"]), self.d_model, padding_idx=int(cfg["pad_token_id"]))
        self.position_embedding = nn.Embedding(self.padded_length, self.d_model)
        self.input_norm = nn.LayerNorm(self.d_model)

    def candidate_inputs(self, program: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = program.shape[0]
        tokens = self.candidate_tokens.unsqueeze(0).expand(batch, -1, -1)
        positions = torch.arange(self.padded_length, device=program.device)
        condition = self.program_projection(program)[:, None, None, :]
        hidden = self.input_norm(self.token_embedding(tokens) + self.position_embedding(positions)[None, None, :, :] + condition)
        lengths = self.candidate_lengths.unsqueeze(0).expand(batch, -1).reshape(-1)
        return hidden.reshape(batch * K, self.padded_length, self.d_model), lengths


class SetConditionedMambaRanker(CandidateInterface):
    def __init__(self, cfg: Mapping[str, Any], tokens: torch.Tensor, lengths: torch.Tensor) -> None:
        super().__init__(cfg, tokens, lengths)
        d_model = self.d_model
        self.block_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(int(cfg["layers"]))])
        self.blocks = nn.ModuleList([
            Mamba(d_model=d_model, d_state=int(cfg["d_state"]), d_conv=int(cfg["d_conv"]), expand=int(cfg["expand"]), use_fast_path=bool(cfg["use_fast_path"]), layer_idx=index)
            for index in range(int(cfg["layers"]))
        ])
        interaction = cfg["candidate_interaction"]
        self.attention_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(int(interaction["layers"]))])
        self.attention = nn.ModuleList([
            nn.MultiheadAttention(d_model, int(interaction["num_heads"]), dropout=float(interaction["dropout"]), batch_first=True)
            for _ in range(int(interaction["layers"]))
        ])
        self.output_norm = nn.LayerNorm(d_model)
        self.ranking_head = nn.Linear(d_model, 1)

    def forward(self, program: torch.Tensor) -> torch.Tensor:
        hidden, lengths = self.candidate_inputs(program)
        for norm, block in zip(self.block_norms, self.blocks):
            hidden = hidden + block(norm(hidden))
        rows = torch.arange(len(hidden), device=hidden.device)
        candidates = hidden[rows, lengths - 1].reshape(program.shape[0], K, -1)
        for norm, attention in zip(self.attention_norms, self.attention):
            update, _ = attention(norm(candidates), norm(candidates), norm(candidates), need_weights=False)
            candidates = candidates + update
        return self.ranking_head(self.output_norm(candidates)).squeeze(-1)

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)


def ranking_permutation(values: torch.Tensor) -> torch.Tensor:
    """Higher reward first; stable sort preserves frozen candidate-ID order on ties."""
    return torch.argsort(values, dim=1, descending=True, stable=True)


def listmle_loss(scores: torch.Tensor, permutation: torch.Tensor) -> torch.Tensor:
    ordered = scores.gather(1, permutation)
    return (torch.logcumsumexp(ordered.flip(1), dim=1).flip(1) - ordered)[:, :-1].mean()


def sampled_pairwise_loss(scores: torch.Tensor, values: torch.Tensor, pairs_per_program: int) -> torch.Tensor:
    batch = scores.shape[0]
    first = torch.randint(K, (batch, pairs_per_program), device=scores.device)
    second = torch.randint(K, (batch, pairs_per_program), device=scores.device)
    first_value, second_value = values.gather(1, first), values.gather(1, second)
    preferred_first = first_value > second_value
    preferred_second = second_value > first_value
    valid = preferred_first | preferred_second
    if not bool(valid.any()):
        return scores.new_zeros(())
    first_score, second_score = scores.gather(1, first), scores.gather(1, second)
    margin = torch.where(preferred_first, first_score - second_score, second_score - first_score)
    return torch.nn.functional.softplus(-margin)[valid].mean()


def learning_rate(cfg: Mapping[str, Any], step: int) -> float:
    base, warmup, total = float(cfg["learning_rate"]), int(cfg["warmup_steps"]), int(cfg["total_steps"])
    return base * (step / warmup if step <= warmup else (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * (step - warmup) / (total - warmup)))))


def policy_metrics(logits: torch.Tensor, records: Sequence[Mapping[str, Any]], matrix: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    by_dataset: dict[str, list[float]] = {}
    regrets: list[int] = []
    positives = 0
    for score, target in zip(logits.tolist(), records):
        budget, observed = 45, []
        for candidate_id in sorted(range(K), key=lambda index: (-score[index], index)):
            prefix = matrix[str(target["program_id"])][candidate_id]["prefix_object_text_size_bytes"]
            take = min(budget, len(prefix))
            observed.extend(prefix[:take])
            budget -= take
            if budget == 0:
                break
        if budget != 0 or not observed:
            raise ValueError(f"policy45 did not consume exactly 45 frozen prefix values: {target['program_id']}")
        policy, oracle, oz = min(observed), min(target["best_object_text_size"]), int(target["S_Oz"])
        reduction = (oz - policy) / oz
        by_dataset.setdefault(str(target["dataset_id"]), []).append(reduction)
        regrets.append(policy - oracle)
        positives += int(reduction > 0)
    per_dataset = {name: sum(values) / len(values) for name, values in sorted(by_dataset.items())}
    return {
        "per_dataset": per_dataset,
        "ValidationFinalMeanOverOz": sum(per_dataset.values()) / len(per_dataset),
        "policy45_regret_mean_bytes": float(np.mean(regrets)),
        "policy45_regret_median_bytes": float(np.median(regrets)),
        "positive_program_count_vs_Oz": positives,
        "N_total": len(records),
        "N_primary_valid": len(records),
        "N_failed_or_invalid": 0,
    }


def average_ranks(values: np.ndarray, *, descending: bool) -> np.ndarray:
    order = np.argsort(-values if descending else values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def ranking_diagnostics(logits: torch.Tensor, values: torch.Tensor) -> dict[str, Any]:
    top1_correct, correlations, undefined = 0, [], 0
    for score, reward in zip(logits.cpu().numpy(), values.cpu().numpy()):
        prediction = min(range(K), key=lambda index: (-float(score[index]), index))
        top1_correct += int(reward[prediction] == reward.max())
        target_ranks, predicted_ranks = average_ranks(reward, descending=True), average_ranks(score, descending=True)
        if np.std(target_ranks) == 0.0 or np.std(predicted_ranks) == 0.0:
            undefined += 1
        else:
            correlations.append(float(np.corrcoef(target_ranks, predicted_ranks)[0, 1]))
    return {
        "top1_accuracy": top1_correct / len(logits),
        "ranking_correlation": {
            "name": "tie-aware Spearman rank correlation of predicted scores against frozen raw_candidate_value",
            "mean": float(np.mean(correlations)) if correlations else None,
            "N_defined": len(correlations),
            "N_undefined_all_tied_or_constant": undefined,
        },
    }


def evaluate(model: nn.Module, features: torch.Tensor, values: torch.Tensor, records: Sequence[Mapping[str, Any]], matrix: Mapping[str, Sequence[Mapping[str, Any]]], batch_size: int) -> dict[str, Any]:
    model.eval()
    logits_parts, listmle_total = [], 0.0
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            current_values = values[start:start + batch_size]
            current = model(features[start:start + batch_size])
            count = len(current)
            logits_parts.append(current.cpu())
            listmle_total += float(listmle_loss(current, ranking_permutation(current_values)).cpu()) * count
    logits = torch.cat(logits_parts)
    metrics = policy_metrics(logits, records, matrix)
    metrics["validation_listmle_loss"] = listmle_total / len(features)
    metrics.update(ranking_diagnostics(logits, values))
    return metrics


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def validate_config(cfg: Mapping[str, Any], controlled: Mapping[str, Any]) -> None:
    if cfg["final_seed_set"] != [1, 2, 3] or cfg["frozen_data_population"] != {"train_complete_k50": 28159, "validation_complete_k50": 4488}:
        raise ValueError("frozen cohort or seed set mismatch")
    if controlled["candidate_representation"]["K"] != K or controlled["candidate_representation"]["padded_length"] != 20 or controlled["candidate_representation"]["pad_token_id"] != 124:
        raise ValueError("frozen candidate representation mismatch")
    if cfg["architecture"]["candidate_interaction"] != {"layers": 2, "num_heads": 4, "dropout": 0.0}:
        raise ValueError("candidate interaction must remain frozen")
    if cfg["objective"]["pairwise_weight"] != 0.1 or cfg["objective"]["pairwise_pairs_per_program"] != 32:
        raise ValueError("listwise objective must remain frozen")
    if cfg["training"]["total_steps"] != 10000 or cfg["training"]["early_stopping"] or cfg["training"]["checkpoint_evaluation_cadence_steps"] != 100:
        raise ValueError("training budget or checkpoint policy mismatch")
    if cfg["validation"]["sampling"] or cfg["validation"]["scored_pass_budget"] != 45:
        raise ValueError("frozen validation inference mismatch")


def frozen_references(cfg: Mapping[str, Any]) -> dict[str, Any]:
    stage_b = load_json(Path(cfg["frozen_reference_reports"]["stage_b"]))
    models = {item["architecture"]: item for item in stage_b["models"]}
    mamba_nvp = load_json(Path(cfg["frozen_reference_reports"]["mamba_nvp_v1"]))["comparison"]
    cross = load_json(Path(cfg["frozen_reference_reports"]["cross_candidate_mambanvp"]))["cross_candidate_mambanvp"]
    return {
        "NVP": {"MeanOverOz_3seed": models["NVP"]["ValidationFinalMeanOverOz_3seed"], "oracle_recovery": models["NVP"]["oracle_opportunity_recovered"], "policy45_regret_mean_bytes": models["NVP"]["policy45_regret_mean_bytes_3seed"], "trainable_parameters": models["NVP"]["seed_results"][0]["trainable_parameters"]},
        "Mamba": {"MeanOverOz_3seed": models["Mamba"]["ValidationFinalMeanOverOz_3seed"], "oracle_recovery": models["Mamba"]["oracle_opportunity_recovered"], "policy45_regret_mean_bytes": models["Mamba"]["policy45_regret_mean_bytes_3seed"], "trainable_parameters": models["Mamba"]["seed_results"][0]["trainable_parameters"]},
        "MambaNVP_v1": {"MeanOverOz_3seed": mamba_nvp["ValidationFinalMeanOverOz_3seed"], "oracle_recovery": mamba_nvp["ValidationFinalMeanOverOz_3seed"] / stage_b["fixed_route_a_oracle"], "policy45_regret_mean_bytes": mamba_nvp["policy45_regret_mean_bytes_3seed"], "trainable_parameters": mamba_nvp["seed_results"][0]["residual_trainable_parameters"]},
        "CrossCandidateMambaNVP": {"MeanOverOz_3seed": cross["ValidationFinalMeanOverOz_3seed"], "oracle_recovery": cross["oracle_recovery_3seed"], "policy45_regret_mean_bytes": cross["policy45_regret_mean_bytes_3seed"], "trainable_parameters": cross["seed_results"][0]["trainable_parameters"]},
    }


def train_seed(cfg: Mapping[str, Any], controlled: Mapping[str, Any], seed: int, tokens: torch.Tensor, lengths: torch.Tensor, train_x: torch.Tensor, train_values: torch.Tensor, val_x: torch.Tensor, val_values: torch.Tensor, validation: Sequence[Mapping[str, Any]], matrix: Mapping[str, Sequence[Mapping[str, Any]]], checkpoint_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    seed_everything(seed)
    model_cfg = {**controlled["candidate_representation"], **controlled["models"]["Mamba"], "candidate_interaction": cfg["architecture"]["candidate_interaction"]}
    model = SetConditionedMambaRanker(model_cfg, tokens, lengths).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["training"]["learning_rate"]), weight_decay=float(cfg["training"]["weight_decay"]))
    order_rng = torch.Generator().manual_seed(seed)
    curve: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    step = 0
    checkpoint_dir.mkdir(parents=True)
    while step < int(cfg["training"]["total_steps"]):
        order = torch.randperm(len(train_x), generator=order_rng)
        for begin in range(0, len(train_x), int(cfg["training"]["batch_size"])):
            step += 1
            indices = order[begin : begin + int(cfg["training"]["batch_size"])].cuda(non_blocking=True)
            lr = learning_rate(cfg["training"], step)
            for group in optimizer.param_groups:
                group["lr"] = lr
            model.train()
            scores = model(train_x[indices])
            values = train_values[indices]
            ranking = listmle_loss(scores, ranking_permutation(values))
            pairwise = sampled_pairwise_loss(scores, values, int(cfg["objective"]["pairwise_pairs_per_program"]))
            loss = ranking + float(cfg["objective"]["pairwise_weight"]) * pairwise
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if step % int(cfg["training"]["checkpoint_evaluation_cadence_steps"]) == 0 or step == int(cfg["training"]["total_steps"]):
                metric = evaluate(model, val_x, val_values, validation, matrix, int(cfg["training"]["evaluation_batch_size"]))
                metric.update({"step": step, "train_total_loss": float(loss.detach().cpu()), "train_listmle_loss": float(ranking.detach().cpu()), "train_pairwise_loss": float(pairwise.detach().cpu()), "lr": lr})
                curve.append(metric)
                print(json.dumps({"architecture": METHOD, "seed": seed, **metric}, sort_keys=True), flush=True)
                if best is None or metric["ValidationFinalMeanOverOz"] > best["ValidationFinalMeanOverOz"]:
                    best = metric
                    torch.save({"stage": "Route-A Set-Conditioned Listwise Mamba Ranker v1", "architecture": METHOD, "seed": seed, "step": step, "metrics": metric, "state_dict": model.state_dict(), "model_config": model_cfg, "ranking_target": copy.deepcopy(dict(cfg["ranking_target"])), "objective": copy.deepcopy(dict(cfg["objective"]))}, checkpoint_dir / "model.pt")
            if step == int(cfg["training"]["total_steps"]):
                break
    if best is None:
        raise RuntimeError("no checkpoint evaluated")
    report = {"architecture": METHOD, "seed": seed, "step_execution": "COMPLETE", "trainable_parameters": model.trainable_parameter_count(), "selection_metric": "ValidationFinalMeanOverOz policy-45 dataset macro mean", "selected": best}
    return report, curve


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {args.output_dir}")
    if not torch.cuda.is_available():
        raise RuntimeError("formal training requires CUDA")
    cfg = load_json(args.config)
    controlled = load_json(Path(cfg["candidate_representation_source"]))
    validate_config(cfg, controlled)
    train, validation = read_jsonl(Path(cfg["target_files"]["train"])), read_jsonl(Path(cfg["target_files"]["validation"]))
    if len(train) != 28159 or len(validation) != 4488:
        raise ValueError("frozen target population mismatch")
    if any(row["training_target_validity"] != "valid_complete_K50" or len(row["raw_candidate_value"]) != K for row in train + validation):
        raise ValueError("incomplete frozen K50 ranking target")
    tokens, lengths = load_candidates(Path(controlled["candidate_representation"]["candidate_sequences"]), pad_token_id=124, padded_length=20)
    matrix = read_label_matrix(Path(cfg["validation_label_shards"]))
    if set(matrix) != {str(row["program_id"]) for row in validation}:
        raise ValueError("validation target/label cohort mismatch")
    train_features = load_feature_cache(Path(cfg["autophase_feature_cache"]["train"]), "train", [str(row["program_id"]) for row in train])
    validation_features = load_feature_cache(Path(cfg["autophase_feature_cache"]["validation"]), "validation", [str(row["program_id"]) for row in validation])
    train_x = torch.tensor([train_features[str(row["program_id"])] for row in train], dtype=torch.float32, device="cuda")
    train_values = torch.tensor([row["raw_candidate_value"] for row in train], dtype=torch.float32, device="cuda")
    val_x = torch.tensor([validation_features[str(row["program_id"])] for row in validation], dtype=torch.float32, device="cuda")
    val_values = torch.tensor([row["raw_candidate_value"] for row in validation], dtype=torch.float32, device="cuda")
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "config.json").write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    reports, curves = [], {}
    for seed in cfg["final_seed_set"]:
        report, curve = train_seed(cfg, controlled, seed, tokens.cuda(), lengths.cuda(), train_x, train_values, val_x, val_values, validation, matrix, args.output_dir / "checkpoints" / f"seed{seed}")
        reports.append(report)
        curves[str(seed)] = curve
    (args.output_dir / "training_curve.json").write_text(json.dumps(curves, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    mean = sum(report["selected"]["ValidationFinalMeanOverOz"] for report in reports) / len(reports)
    oracle = load_json(Path(cfg["frozen_reference_reports"]["stage_b"]))["fixed_route_a_oracle"]
    ranker = {
        "architecture": METHOD,
        "ValidationFinalMeanOverOz_3seed": mean,
        "oracle_recovery_3seed": mean / oracle,
        "policy45_regret_mean_bytes_3seed": sum(report["selected"]["policy45_regret_mean_bytes"] for report in reports) / len(reports),
        "top1_accuracy_3seed": sum(report["selected"]["top1_accuracy"] for report in reports) / len(reports),
        "ranking_correlation_3seed": sum(report["selected"]["ranking_correlation"]["mean"] for report in reports) / len(reports),
        "ranking_correlation_defined_programs_per_seed": [report["selected"]["ranking_correlation"]["N_defined"] for report in reports],
        "ranking_correlation_undefined_programs_per_seed": [report["selected"]["ranking_correlation"]["N_undefined_all_tied_or_constant"] for report in reports],
        "trainable_parameters": reports[0]["trainable_parameters"],
        "seed_results": reports,
    }
    references = frozen_references(cfg)
    report = {
        "step_execution": "COMPLETE",
        "training_only": True,
        "compiler_gym_initialized": False,
        "llvm_execution": False,
        "candidate_rollouts": 0,
        "objecttext_measurements": 0,
        "label_regeneration": False,
        "final_test_accessed": False,
        "ood_accessed": False,
        "runtime_accessed": False,
        "validation_cohort": {"N_total": 4488, "N_primary_valid": 4488, "N_failed_or_invalid": 0},
        "set_conditioned_mamba_ranker": ranker,
        "frozen_references": references,
        "differences": {name: mean - reference["MeanOverOz_3seed"] for name, reference in references.items()},
    }
    (args.output_dir / "comparison_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
