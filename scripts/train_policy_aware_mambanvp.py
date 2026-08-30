#!/usr/bin/env python3
"""Train and freeze Counterfactual Policy-Aware MambaNVP from offline artifacts."""
from __future__ import annotations

import argparse
import collections
import copy
import gzip
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

if __package__:
    from scripts.evaluate_mamba_nvp_final_objecttext import load_final_features, policy45, read_final_artifacts
    from scripts.train_adaptive_mamba_nvp_router import SourceBalancedSampler
    from scripts.train_controlled_nvp_stage_a import ControlledCandidateModel, evaluate, learning_rate, load_candidates, read_jsonl, read_label_matrix, soft_cross_entropy
    from scripts.train_mamba_nvp_objecttext import K, load_feature_cache, load_frozen_nvp, load_json, seed_everything
else:
    from evaluate_mamba_nvp_final_objecttext import load_final_features, policy45, read_final_artifacts
    from train_adaptive_mamba_nvp_router import SourceBalancedSampler
    from train_controlled_nvp_stage_a import ControlledCandidateModel, evaluate, learning_rate, load_candidates, read_jsonl, read_label_matrix, soft_cross_entropy
    from train_mamba_nvp_objecttext import K, load_feature_cache, load_frozen_nvp, load_json, seed_everything


METHOD = "PA-MambaNVP"
EPSILON = 1e-12


@dataclass(frozen=True)
class PairSet:
    preferred: np.ndarray
    other: np.ndarray
    weight: np.ndarray
    categories: tuple[tuple[str, ...], ...]


def policy45_utility(order: Sequence[int], target: Mapping[str, Any], records: Sequence[Mapping[str, Any]]) -> float:
    if sorted(order) != list(range(K)):
        raise ValueError("policy rank order must be a K=50 permutation")
    scores = [0.0] * K
    for position, candidate in enumerate(order):
        scores[candidate] = float(K - position)
    return (int(target["S_Oz"]) - policy45(scores, records)) / int(target["S_Oz"])


def policy45_admission(order: Sequence[int], records: Sequence[Mapping[str, Any]]) -> list[int]:
    budget, admitted = 45, []
    for candidate in order:
        take = min(budget, len(records[candidate]["prefix_object_text_size_bytes"]))
        if take:
            admitted.append(candidate)
        budget -= take
        if budget == 0:
            return admitted
    raise ValueError("frozen candidates did not consume the exact policy45 budget")


def swap_order(order: Sequence[int], first: int, second: int) -> list[int]:
    changed = list(order)
    changed[first], changed[second] = changed[second], changed[first]
    return changed


def build_policy_pairs(order: Sequence[int], target: Mapping[str, Any], records: Sequence[Mapping[str, Any]], tolerance: float = EPSILON) -> tuple[PairSet | None, dict[str, Any]]:
    """Build only policy-sensitive counterfactual pairs for one frozen NVP ranking."""
    base = policy45_utility(order, target, records)
    tagged: dict[tuple[int, int], set[str]] = collections.defaultdict(set)
    for first in range(10):
        for second in range(first + 1, 10):
            tagged[(first, second)].add("top1_10_internal")
    for first in range(10):
        for second in range(10, 20):
            tagged[(first, second)].add("top10_vs_11_20_crossing")
    admission = policy45_admission(order, records)
    frontier = len(admission)
    for first in range(max(0, frontier - 3), min(20, frontier)):
        for second in range(max(frontier, 0), min(20, frontier + 3)):
            if first < second:
                tagged[(first, second)].add("policy45_admission_frontier")
    preferred: list[int] = []
    other: list[int] = []
    raw: list[float] = []
    categories: list[tuple[str, ...]] = []
    deltas: list[float] = []
    for (first, second), names in sorted(tagged.items()):
        delta = policy45_utility(swap_order(order, first, second), target, records) - base
        if abs(delta) <= tolerance:
            continue
        # delta>0 means moving original second candidate ahead of original first helps.
        winner, loser = (order[second], order[first]) if delta > 0 else (order[first], order[second])
        preferred.append(winner); other.append(loser); raw.append(abs(delta)); categories.append(tuple(sorted(names))); deltas.append(delta)
    if not raw:
        return None, {"base_utility": base, "pair_count": 0, "mean_abs_delta_u": 0.0, "category_counts": {}}
    denominator = float(sum(raw)) + EPSILON
    pair_set = PairSet(np.asarray(preferred, dtype=np.int64), np.asarray(other, dtype=np.int64), np.asarray(raw, dtype=np.float32) / denominator, tuple(categories))
    category_counts = collections.Counter(name for names in categories for name in names)
    return pair_set, {"base_utility": base, "pair_count": len(raw), "mean_abs_delta_u": float(np.mean(np.abs(deltas))), "category_counts": dict(sorted(category_counts.items()))}


def pairwise_policy_loss(scores: torch.Tensor, pair_sets: Sequence[PairSet | None], index: torch.Tensor) -> torch.Tensor:
    """Per-program normalized weighted LambdaRank loss; zero-pair rows have zero effect."""
    total = scores.new_zeros(())
    for local, global_index in enumerate(index.detach().cpu().tolist()):
        pairs = pair_sets[global_index]
        if pairs is None:
            continue
        preferred = torch.as_tensor(pairs.preferred, device=scores.device)
        other = torch.as_tensor(pairs.other, device=scores.device)
        weights = torch.as_tensor(pairs.weight, device=scores.device)
        difference = scores[local, preferred] - scores[local, other]
        total = total + (weights * torch.nn.functional.softplus(-difference)).sum()
    return total / len(index)


def validate_candidate_alignment(records: Sequence[Mapping[str, Any]], tokens: torch.Tensor, lengths: torch.Tensor) -> None:
    if len(records) != K or [int(row["candidate_id"]) for row in records] != list(range(K)):
        raise ValueError("candidate IDs do not retain frozen K=50 ordering")
    for row, length in zip(records, lengths.tolist()):
        prefix = row["prefix_object_text_size_bytes"]
        if not prefix or len(prefix) > int(length):
            raise ValueError("candidate length/prefix label misalignment")


class PolicyAwareMambaNVP(nn.Module):
    def __init__(self, nvp: nn.Module, residual_cfg: Mapping[str, Any], tokens: torch.Tensor, lengths: torch.Tensor) -> None:
        super().__init__()
        self.nvp = nvp
        self.residual = ControlledCandidateModel("Mamba", residual_cfg, tokens, lengths)
        nn.init.zeros_(self.residual.value_head.weight); nn.init.zeros_(self.residual.value_head.bias)
        for parameter in self.nvp.parameters():
            parameter.requires_grad_(False)
        self.nvp.eval()

    def train(self, mode: bool = True) -> "PolicyAwareMambaNVP":
        super().train(mode); self.nvp.eval(); return self

    def components(self, program: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            anchor = self.nvp(program)
        raw = self.residual(program)
        centered = raw - raw.mean(dim=1, keepdim=True)
        return anchor, centered, anchor + centered

    def forward(self, program: torch.Tensor) -> torch.Tensor:
        return self.components(program)[2]

    def residual_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.residual.parameters() if parameter.requires_grad)


def policy_pair_statistics(pair_sets: Sequence[PairSet | None], details: Sequence[Mapping[str, Any]], datasets: Sequence[str], seed: int) -> dict[str, Any]:
    counts = np.asarray([0 if item is None else len(item.weight) for item in pair_sets], dtype=np.int64)
    category_counts = collections.Counter()
    mean_deltas = []
    by_source: dict[str, list[int]] = collections.defaultdict(list)
    for detail, dataset in zip(details, datasets):
        category_counts.update(detail["category_counts"]); mean_deltas.append(detail["mean_abs_delta_u"]); by_source[str(dataset)].append(int(detail["pair_count"]))
    return {"seed": seed, "programs": int(len(counts)), "mean_policy_sensitive_pairs_per_program": float(counts.mean()), "median_policy_sensitive_pairs_per_program": float(np.median(counts)), "program_fraction_with_at_least_one_pair": float(np.mean(counts > 0)), "mean_abs_delta_U_over_programs": float(np.mean(mean_deltas)), "pair_category_occurrences": dict(sorted(category_counts.items())), "per_source_pair_count_mean": {name: float(np.mean(values)) for name, values in sorted(by_source.items())}}


def nvp_orders(nvp: nn.Module, features: torch.Tensor, batch_size: int) -> list[list[int]]:
    orders: list[list[int]] = []
    nvp.eval()
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            scores = nvp(features[start:start + batch_size]).cpu().tolist()
            orders.extend([sorted(range(K), key=lambda candidate: (-row[candidate], candidate)) for row in scores])
    return orders


def extra_metrics(logits: torch.Tensor, records: Sequence[Mapping[str, Any]], matrix: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    top1 = top5 = 0; admission_count = []; admission_changed = 0
    for score, target in zip(logits.tolist(), records):
        order = sorted(range(K), key=lambda candidate: (-score[candidate], candidate))
        values = list(target["raw_candidate_value"]); maximum = max(values)
        top1 += int(values[order[0]] == maximum); top5 += int(any(values[candidate] == maximum for candidate in order[:5]))
        admission_count.append(len(policy45_admission(order, matrix[target["program_id"]])))
    return {"top1_oracle_tie_accuracy": top1 / len(records), "top5_oracle_coverage": top5 / len(records), "admitted_candidate_count_mean": float(np.mean(admission_count)), "admitted_candidate_count_median": float(np.median(admission_count))}


def evaluate_pa(model: PolicyAwareMambaNVP, features: torch.Tensor, targets: torch.Tensor, records: list[dict[str, Any]], matrix: dict[str, list[dict[str, Any]]], batch_size: int) -> dict[str, Any]:
    model.eval(); parts = []; ce_sum = residual_sq = 0.0
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            anchor, centered, scores = model.components(features[start:start + batch_size]); count = len(scores)
            parts.append(scores.cpu()); ce_sum += float(soft_cross_entropy(scores, targets[start:start + count]).cpu()) * count; residual_sq += float(centered.square().mean().cpu()) * count
    logits = torch.cat(parts)
    values = evaluate.__globals__["policy_metrics"](logits, records, matrix)
    values.update(extra_metrics(logits, records, matrix))
    values["validation_ce"] = ce_sum / len(features); values["centered_residual_mse"] = residual_sq / len(features)
    return values


def checkpoint_payload(model: PolicyAwareMambaNVP, cfg: Mapping[str, Any], seed: int, step: int, metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {"stage": "Counterfactual Policy-Aware MambaNVP v1", "architecture": METHOD, "seed": seed, "step": step, "metrics": dict(metrics), "residual_state_dict": model.residual.state_dict(), "residual_trainable_parameters": model.residual_parameter_count(), "nvp_checkpoint": str(Path(cfg["nvp_checkpoint_root"]) / f"seed{seed}" / "model.pt"), "nvp_frozen": True, "fusion": cfg["architecture"]["fusion"], "config": copy.deepcopy(dict(cfg))}


def train_seed(cfg: Mapping[str, Any], controlled: Mapping[str, Any], seed: int, tokens: torch.Tensor, lengths: torch.Tensor, train_x: torch.Tensor, train_y: torch.Tensor, val_x: torch.Tensor, val_y: torch.Tensor, train: list[dict[str, Any]], validation: list[dict[str, Any]], train_matrix: Mapping[str, Sequence[Mapping[str, Any]]], val_matrix: Mapping[str, Sequence[Mapping[str, Any]]], output: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    seed_everything(seed)
    nvp = load_frozen_nvp(Path(cfg["nvp_checkpoint_root"]) / f"seed{seed}" / "model.pt", seed).cuda()
    orders = nvp_orders(nvp, train_x, int(cfg["training"]["evaluation_batch_size"]))
    pairs_and_details = [build_policy_pairs(order, target, train_matrix[target["program_id"]], float(cfg["target_and_objective"]["delta_tolerance"])) for order, target in zip(orders, train)]
    pair_sets, details = [item[0] for item in pairs_and_details], [item[1] for item in pairs_and_details]
    stats = policy_pair_statistics(pair_sets, details, [row["dataset_id"] for row in train], seed)
    residual_cfg = {**controlled["candidate_representation"], **controlled["models"]["Mamba"]}
    model = PolicyAwareMambaNVP(nvp, residual_cfg, tokens, lengths).cuda()
    if any(parameter.requires_grad for parameter in model.nvp.parameters()) or model.nvp.training: raise RuntimeError("NVP must be frozen eval")
    optimizer = torch.optim.Adam(model.residual.parameters(), lr=float(cfg["training"]["learning_rate"]), weight_decay=float(cfg["training"]["weight_decay"]))
    sampler = SourceBalancedSampler([row["dataset_id"] for row in train], seed)
    output.mkdir(parents=True); curve: list[dict[str, Any]] = []; best: dict[str, Any] | None = None
    for step in range(1, int(cfg["training"]["total_steps"]) + 1):
        index = sampler.sample(int(cfg["training"]["batch_size"]), torch.device("cuda"))
        lr = learning_rate(cfg["training"], step)
        for group in optimizer.param_groups: group["lr"] = lr
        model.train(); anchor, centered, scores = model.components(train_x[index]); policy_loss = pairwise_policy_loss(scores, pair_sets, index); ce_loss = soft_cross_entropy(scores, train_y[index]); residual_loss = centered.square().mean(); loss = policy_loss + .25 * ce_loss + .001 * residual_loss
        optimizer.zero_grad(set_to_none=True); loss.backward()
        if any(parameter.grad is not None for parameter in model.nvp.parameters()): raise RuntimeError("frozen NVP received gradients")
        optimizer.step()
        if step % int(cfg["training"]["checkpoint_evaluation_cadence_steps"]) == 0 or step == int(cfg["training"]["total_steps"]):
            metric = evaluate_pa(model, val_x, val_y, validation, dict(val_matrix), int(cfg["training"]["evaluation_batch_size"]))
            metric.update({"step": step, "train_total_loss": float(loss.detach().cpu()), "train_policy_loss": float(policy_loss.detach().cpu()), "train_ce_loss": float(ce_loss.detach().cpu()), "train_centered_residual_loss": float(residual_loss.detach().cpu()), "lr": lr}); curve.append(metric)
            print(json.dumps({"architecture": METHOD, "seed": seed, **metric}, sort_keys=True), flush=True)
            if best is None or metric["ValidationFinalMeanOverOz"] > best["ValidationFinalMeanOverOz"]:
                best = metric; torch.save(checkpoint_payload(model, cfg, seed, step, metric), output / "model.pt")
    if best is None: raise RuntimeError("no validation checkpoint")
    return {"architecture": METHOD, "seed": seed, "step_execution": "COMPLETE", "nvp_frozen": True, "residual_trainable_parameters": model.residual_parameter_count(), "selection_metric": "ValidationFinalMeanOverOz policy-45 dataset macro mean", "selected": best}, curve, stats


def aggregate_validation(reports: Sequence[Mapping[str, Any]], references: Mapping[str, float]) -> dict[str, Any]:
    selected = [row["selected"] for row in reports]; datasets = sorted(selected[0]["per_dataset"])
    macro = float(np.mean([row["ValidationFinalMeanOverOz"] for row in selected])); per_dataset = {name: float(np.mean([row["per_dataset"][name] for row in selected])) for name in datasets}
    nvp_validation = float(references["NVP_validation"])
    deltas = {name: value - float(references["NVP_validation_per_dataset"][name]) for name, value in per_dataset.items()}
    return {"three_seed_mean_over_oz": macro, "per_seed_mean_over_oz": {str(row["seed"]): row["selected"]["ValidationFinalMeanOverOz"] for row in reports}, "per_dataset": per_dataset, "delta_vs_nvp_per_dataset": deltas, "positive_dataset_count_vs_nvp": sum(value > 0 for value in deltas.values()), "negative_dataset_count_vs_nvp": sum(value < 0 for value in deltas.values()), "median_dataset_delta_vs_nvp": float(np.median(list(deltas.values()))), "delta_vs_nvp": macro - nvp_validation, "delta_vs_mamba": macro - float(references["Mamba_validation"]), "policy45_regret_mean_bytes": float(np.mean([row["policy45_regret_mean_bytes"] for row in selected])), "top1_oracle_tie_accuracy": float(np.mean([row["top1_oracle_tie_accuracy"] for row in selected])), "top5_oracle_coverage": float(np.mean([row["top5_oracle_coverage"] for row in selected])), "admitted_candidate_count_mean": float(np.mean([row["admitted_candidate_count_mean"] for row in selected])), "validation_ce": float(np.mean([row["validation_ce"] for row in selected]))}


def final_seed(model: PolicyAwareMambaNVP, seed: int, programs: Sequence[str], matrix: Mapping[str, Sequence[Mapping[str, Any]]], summaries: Mapping[str, Mapping[str, Any]], features: Mapping[str, Sequence[float]], output: Path) -> list[dict[str, Any]]:
    eligible = [program for program in programs if program in matrix and summaries[program]["ratio_metric_validity"] == "valid_for_ObjectText_ratio_metric"]
    rows: list[dict[str, Any]] = []; model.eval()
    with torch.no_grad():
        for start in range(0, len(eligible), 128):
            current = eligible[start:start + 128]; scores = model(torch.tensor([features[p] for p in current], dtype=torch.float32, device="cuda")).cpu().tolist()
            for program, score in zip(current, scores):
                records, summary = matrix[program], summaries[program]; order = sorted(range(K), key=lambda candidate: (-score[candidate], candidate)); policy = policy45(score, records); oracle = min(int(row["best_object_text_size_bytes"]) for row in records); oz = int(summary["oz_object_text_size_bytes"]); values = [int(row["best_object_text_size_bytes"]) for row in records]; best = min(values)
                rows.append({"program_id": program, "dataset_id": summary["dataset_id"], "seed": seed, "valid": True, "policy45_object_text_size_bytes": policy, "oracle_object_text_size_bytes": oracle, "oz_object_text_size_bytes": oz, "mean_over_oz": (oz-policy)/oz, "policy45_regret_bytes": policy-oracle, "top1_oracle_tie": values[order[0]] == best, "top5_oracle_coverage": any(values[i] == best for i in order[:5]), "admitted_candidate_count": len(policy45_admission(order, records)), "selected_candidate_id": order[0]})
    output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output, "wt", encoding="utf-8") as handle:
        for row in rows: handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    return rows


def aggregate_final(rows_by_seed: Mapping[int, Sequence[Mapping[str, Any]]], nvp_per_dataset: Mapping[str, Any], oracle_macro: float) -> dict[str, Any]:
    datasets = sorted({row["dataset_id"] for rows in rows_by_seed.values() for row in rows})
    per_dataset = {}; seed_macro = {}
    for dataset in datasets:
        per_seed = {str(seed): float(np.mean([row["mean_over_oz"] for row in rows if row["dataset_id"] == dataset])) for seed, rows in rows_by_seed.items()}
        value = float(np.mean(list(per_seed.values()))); nvp = float(nvp_per_dataset[dataset]["NVP"]["three_seed_mean"]); per_dataset[dataset] = {"three_seed_mean": value, "per_seed": per_seed, "nvp": nvp, "delta_vs_nvp": value-nvp}
    for seed, rows in rows_by_seed.items(): seed_macro[str(seed)] = float(np.mean([float(np.mean([row["mean_over_oz"] for row in rows if row["dataset_id"] == dataset])) for dataset in datasets]))
    macro = float(np.mean(list(seed_macro.values()))); deltas = [row["delta_vs_nvp"] for row in per_dataset.values()]; no_llvm = [row["delta_vs_nvp"] for name, row in per_dataset.items() if name != "llvm-stress-v0"]
    all_rows = [row for rows in rows_by_seed.values() for row in rows]
    return {"three_seed_mean_over_oz": macro, "per_seed_mean_over_oz": seed_macro, "delta_vs_nvp_per_dataset": {name: row["delta_vs_nvp"] for name, row in per_dataset.items()}, "per_dataset": per_dataset, "median_dataset_delta_vs_nvp": float(np.median(deltas)), "positive_dataset_count_vs_nvp": sum(value > 0 for value in deltas), "negative_dataset_count_vs_nvp": sum(value < 0 for value in deltas), "leave_llvm_stress_out_13dataset_delta_vs_nvp": float(np.mean(no_llvm)), "policy45_regret_mean_bytes": float(np.mean([row["policy45_regret_bytes"] for row in all_rows])), "oracle_recovery": macro / oracle_macro, "top1_oracle_tie_accuracy": float(np.mean([row["top1_oracle_tie"] for row in all_rows])), "top5_oracle_coverage": float(np.mean([row["top5_oracle_coverage"] for row in all_rows])), "admitted_candidate_count_mean": float(np.mean([row["admitted_candidate_count"] for row in all_rows]))}


def load_pa_checkpoint(path: Path, seed: int, controlled: Mapping[str, Any]) -> PolicyAwareMambaNVP:
    payload = torch.load(path, map_location="cpu")
    if payload.get("stage") != "Counterfactual Policy-Aware MambaNVP v1" or payload.get("architecture") != METHOD or payload.get("seed") != seed or payload.get("nvp_frozen") is not True: raise ValueError(f"invalid frozen PA checkpoint: {path}")
    tokens, lengths = load_candidates(Path(controlled["candidate_representation"]["candidate_sequences"]), pad_token_id=124, padded_length=20)
    nvp = load_frozen_nvp(Path(payload["nvp_checkpoint"]), seed)
    model = PolicyAwareMambaNVP(nvp, {**controlled["candidate_representation"], **controlled["models"]["Mamba"]}, tokens, lengths)
    model.residual.load_state_dict(payload["residual_state_dict"], strict=True)
    return model.cuda().eval()


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", type=Path, required=True); parser.add_argument("--output-dir", type=Path, required=True); args = parser.parse_args()
    if args.output_dir.exists(): raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    if not torch.cuda.is_available(): raise RuntimeError("formal PA-MambaNVP requires CUDA")
    cfg, controlled = load_json(args.config), load_json(Path(load_json(args.config)["candidate_representation_source"]))
    if cfg["final_seed_set"] != [1, 2, 3] or cfg["training"]["total_steps"] != 10000 or cfg["target_and_objective"]["target_temperature"] != .05: raise ValueError("frozen protocol configuration mismatch")
    train, validation = read_jsonl(Path(cfg["target_files"]["train"])), read_jsonl(Path(cfg["target_files"]["validation"]))
    if len(train) != 28159 or len(validation) != 4488 or set(row["program_id"] for row in train) & set(row["program_id"] for row in validation): raise ValueError("frozen train/validation population mismatch")
    if any(len(row["normalized_target"]) != K or len(row["raw_candidate_value"]) != K for row in train + validation): raise ValueError("frozen K50 targets missing")
    tokens, lengths = load_candidates(Path(controlled["candidate_representation"]["candidate_sequences"]), pad_token_id=124, padded_length=20)
    train_matrix, val_matrix = read_label_matrix(Path(cfg["train_label_shards"])), read_label_matrix(Path(cfg["validation_label_shards"]))
    if set(train_matrix) != {row["program_id"] for row in train} or set(val_matrix) != {row["program_id"] for row in validation}: raise ValueError("target and label cohort mismatch")
    for record_set in list(train_matrix.values())[:1] + list(val_matrix.values())[:1]: validate_candidate_alignment(record_set, tokens, lengths)
    train_features = load_feature_cache(Path(cfg["autophase_feature_cache"]["train"]), "train", [row["program_id"] for row in train]); val_features = load_feature_cache(Path(cfg["autophase_feature_cache"]["validation"]), "validation", [row["program_id"] for row in validation])
    train_x = torch.tensor([train_features[row["program_id"]] for row in train], dtype=torch.float32, device="cuda"); train_y = torch.tensor([row["normalized_target"] for row in train], dtype=torch.float32, device="cuda")
    val_x = torch.tensor([val_features[row["program_id"]] for row in validation], dtype=torch.float32, device="cuda"); val_y = torch.tensor([row["normalized_target"] for row in validation], dtype=torch.float32, device="cuda")
    args.output_dir.mkdir(parents=True); (args.output_dir / "config.json").write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    references_source = load_json(Path(cfg["frozen_reference_reports"]["stage_b"])); stage_models = {row["architecture"]: row for row in references_source["models"]}; references = {"NVP_validation": stage_models["NVP"]["ValidationFinalMeanOverOz_3seed"], "Mamba_validation": stage_models["Mamba"]["ValidationFinalMeanOverOz_3seed"], "NVP_validation_per_dataset": stage_models["NVP"]["per_dataset_3seed"]}
    reports=[]; curves={}; pair_statistics=[]
    for seed in cfg["final_seed_set"]:
        report, curve, stats = train_seed(cfg, controlled, seed, tokens.cuda(), lengths.cuda(), train_x, train_y, val_x, val_y, train, validation, train_matrix, val_matrix, args.output_dir / "checkpoints" / f"seed{seed}")
        reports.append(report); curves[str(seed)] = curve; pair_statistics.append(stats)
    validation_summary = aggregate_validation(reports, references)
    (args.output_dir / "learning_curve.json").write_text(json.dumps(curves, indent=2, sort_keys=True) + "\n", encoding="utf-8"); (args.output_dir / "policy_pair_statistics.json").write_text(json.dumps({"per_seed": pair_statistics}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    # Final artifacts are intentionally inaccessible until all validation checkpoints above have frozen.
    programs, final_matrix, summaries = read_final_artifacts(Path(cfg["final_label_shards"])); eligible = [program for program in programs if program in final_matrix and summaries[program]["ratio_metric_validity"] == "valid_for_ObjectText_ratio_metric"]
    if len(programs) != 4683 or len(eligible) != 4679: raise ValueError("frozen final cohort mismatch")
    final_features = load_final_features(Path(cfg["autophase_feature_cache"]["final"]), eligible); final_rows={}
    for seed in cfg["final_seed_set"]:
        model = load_pa_checkpoint(args.output_dir / "checkpoints" / f"seed{seed}" / "model.pt", seed, controlled)
        final_rows[seed] = final_seed(model, seed, programs, final_matrix, summaries, final_features, args.output_dir / "final_results" / f"seed{seed}.jsonl.gz")
    baseline_final = load_json(Path(cfg["frozen_reference_reports"]["final_baseline"])); nvp_final = next(family for family in baseline_final["comparison_families"] if family["family"] == "H2a")
    oracle = baseline_final["offline_k50_oracle"]["dataset_macro"]
    final_summary = aggregate_final(final_rows, nvp_final["per_dataset"], oracle)
    direct = load_json(Path(cfg["frozen_reference_reports"]["direct_final"]))["combined_comparison"]["dataset_macro"]["MambaNVP"]["three_seed_mean"]
    anchored = load_json(Path(cfg["frozen_reference_reports"]["anchored_final"]))["combined_comparison"]["dataset_macro"]["GatedCalibratedMambaNVP"]["three_seed_mean"]
    final_summary["delta_vs_nvp"] = final_summary["three_seed_mean_over_oz"] - .08715469; final_summary["delta_vs_mamba"] = final_summary["three_seed_mean_over_oz"] - .08462666; final_summary["delta_vs_direct"] = final_summary["three_seed_mean_over_oz"] - direct; final_summary["delta_vs_anchored"] = final_summary["three_seed_mean_over_oz"] - anchored
    high_bar = {"validation_exceeds_mamba": validation_summary["three_seed_mean_over_oz"] > .06355292, "final_exceeds_anchored": final_summary["three_seed_mean_over_oz"] > .08778865, "leave_llvm_positive": final_summary["leave_llvm_stress_out_13dataset_delta_vs_nvp"] > 0, "median_delta_positive": final_summary["median_dataset_delta_vs_nvp"] > 0, "majority_datasets_positive": final_summary["positive_dataset_count_vs_nvp"] > final_summary["negative_dataset_count_vs_nvp"]}
    report = {"step_execution": "COMPLETE", "protocol": "frozen offline labels/features/checkpoints only", "compiler_gym_initialized": False, "llvm_execution": False, "candidate_rollouts": 0, "objecttext_measurements": 0, "label_regeneration": False, "runtime_accessed": False, "final_population": {"N_total": 4683, "N_complete_K50_valid": 4679, "N_invalid": 4}, "validation": validation_summary, "final": final_summary, "final_checkpoints_frozen_before_final": True, "high_bar": high_bar, "decision": "PASS" if all(high_bar.values()) else "FAIL"}
    (args.output_dir / "comparison_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"); (args.output_dir / "experiment_report.json").write_text(json.dumps({"seed_reports": reports, "final_checkpoint_freeze": True, "final_inference_only": True}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
