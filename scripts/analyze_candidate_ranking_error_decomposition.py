#!/usr/bin/env python3
"""Frozen, offline K=50 ranking-error decomposition using existing evaluator semantics."""
from __future__ import annotations

import argparse
import collections
import csv
import gzip
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from scripts.evaluate_gated_calibrated_mambanvp_final import load_gated, selected_steps_from_report
from scripts.evaluate_mamba_nvp_final_objecttext import K, load_final_features, load_mamba_nvp, policy45, read_final_artifacts
from scripts.train_adaptive_mamba_nvp_router import load_frozen_mamba
from scripts.train_controlled_nvp_stage_a import load_candidates, read_label_matrix
from scripts.train_mamba_nvp_objecttext import load_feature_cache, load_frozen_nvp, load_json


FEATURE_DIM, PAD_TOKEN, PAD_LENGTH = 56, 124, 20
METHODS = ("NVP", "Mamba", "DirectMambaNVP", "AnchoredMambaNVP")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def validate_config(cfg: Mapping[str, Any]) -> None:
    if cfg["models"] != list(METHODS) or cfg["seeds"] != [1, 2, 3]:
        raise ValueError("core method or seed set differs from frozen diagnostic contract")
    if cfg["candidate_space"] != {"K": 50, "candidate_length_range": [4, 20], "tie_break": "descending score, candidate ID ascending", "policy45_pass_budget": 45}:
        raise ValueError("frozen candidate/policy45 semantics mismatch")
    if cfg["splits"]["validation"]["complete_k50"] != 4488 or cfg["splits"]["final"]["complete_k50"] != 4679:
        raise ValueError("frozen valid cohort mismatch")
    if cfg["analysis"]["final_used_for_selection_or_tuning"]:
        raise ValueError("final/OOD selection or tuning is forbidden")


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp = np.exp(shifted)
    return exp / exp.sum()


def average_rank_desc(values: np.ndarray) -> np.ndarray:
    order = np.argsort(-values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return pearson(average_rank_desc(x), average_rank_desc(y))


def kendall_tau_b(x: np.ndarray, y: np.ndarray) -> float:
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    upper = np.triu(np.ones(dx.shape, dtype=bool), 1)
    sx, sy = np.sign(dx[upper]), np.sign(dy[upper])
    concordant, discordant = int(np.sum(sx * sy > 0)), int(np.sum(sx * sy < 0))
    tied_x, tied_y = int(np.sum((sx == 0) & (sy != 0))), int(np.sum((sx != 0) & (sy == 0)))
    denom = math.sqrt((concordant + discordant + tied_x) * (concordant + discordant + tied_y))
    return 0.0 if denom == 0 else (concordant - discordant) / denom


def true_top_inversions(predicted_order: Sequence[int], true_values: np.ndarray, limit: int) -> int:
    true_order = sorted(range(K), key=lambda index: (-float(true_values[index]), index))[:limit]
    predicted_rank = {candidate: rank for rank, candidate in enumerate(predicted_order)}
    return sum(predicted_rank[left] > predicted_rank[right] for pos, left in enumerate(true_order) for right in true_order[pos + 1 :] if true_values[left] != true_values[right])


def policy_details(scores: np.ndarray, records: Sequence[Mapping[str, Any]], true_values: np.ndarray) -> dict[str, Any]:
    ordered = sorted(range(K), key=lambda index: (-float(scores[index]), index))
    budget, admitted, observed = 45, [], []
    for rank, candidate in enumerate(ordered, start=1):
        prefix = records[candidate]["prefix_object_text_size_bytes"]
        take = min(budget, len(prefix))
        admitted.append({"candidate_id": candidate, "predicted_rank": rank, "candidate_length": len(prefix), "observed_prefix_count": take})
        observed.extend(prefix[:take]); budget -= take
        if budget == 0:
            break
    measured = policy45(scores.tolist(), records)
    if budget != 0 or not observed or min(observed) != measured:
        raise ValueError("diagnostic policy decomposition disagrees with existing policy45 evaluator")
    oracle_value = float(np.max(true_values))
    oracle_ids = [index for index, value in enumerate(true_values) if value == oracle_value]
    admitted_ids = [row["candidate_id"] for row in admitted]
    oracle_admitted = any(candidate in admitted_ids for candidate in oracle_ids)
    oracle_rank = min(ordered.index(candidate) + 1 for candidate in oracle_ids)
    best_admitted = float(max(true_values[candidate] for candidate in admitted_ids))
    return {"ordered": ordered, "admitted": admitted, "admitted_ids": admitted_ids, "policy45_object_text_size_bytes": measured, "oracle_admitted": oracle_admitted, "oracle_rank": oracle_rank, "best_admitted_true_value": best_admitted, "best_admitted_true_rank": min(sorted(range(K), key=lambda index: (-float(true_values[index]), index)).index(candidate) + 1 for candidate in admitted_ids if true_values[candidate] == best_admitted)}


def candidate_target_from_records(program: str, dataset: str, records: Sequence[Mapping[str, Any]], oz: int) -> dict[str, Any]:
    sizes = [int(row["best_object_text_size_bytes"]) for row in records]
    values = np.asarray([(oz - size) / oz for size in sizes], dtype=np.float64)
    target = softmax(values / 0.05)
    return {"program_id": program, "dataset_id": dataset, "S_Oz": oz, "best_object_text": sizes, "raw_candidate_value": values.tolist(), "normalized_target": target.tolist()}


def load_validation(cfg: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]], dict[str, list[float]], int]:
    settings = cfg["splits"]["validation"]
    targets = read_jsonl(Path(settings["targets"]))
    if len(targets) != settings["complete_k50"] or any(row.get("training_target_validity") != "valid_complete_K50" for row in targets):
        raise ValueError("frozen validation target cohort mismatch")
    for row in targets:
        row["best_object_text"] = list(row["best_object_text_size"])
    programs = [str(row["program_id"]) for row in targets]
    matrix = read_label_matrix(Path(settings["label_shards"]))
    if set(matrix) != set(programs):
        raise ValueError("validation target/label program IDs do not exactly match")
    features = load_feature_cache(Path(settings["features"]), "validation", programs)
    return targets, matrix, features, int(settings["total_programs"])


def load_final(cfg: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]], dict[str, list[float]], int]:
    settings = cfg["splits"]["final"]
    programs, matrix, summaries = read_final_artifacts(Path(settings["label_shards"]))
    eligible = [program for program in programs if program in matrix and summaries[program]["ratio_metric_validity"] == "valid_for_ObjectText_ratio_metric"]
    if len(programs) != settings["total_programs"] or len(eligible) != settings["complete_k50"]:
        raise ValueError("frozen final cohort mismatch")
    targets = [candidate_target_from_records(program, summaries[program]["dataset_id"], matrix[program], int(summaries[program]["oz_object_text_size_bytes"])) for program in eligible]
    features = load_final_features(Path(settings["features"]), eligible)
    return targets, matrix, features, int(settings["total_programs"])


def load_models(seed: int, cfg: Mapping[str, Any], device: torch.device) -> dict[str, torch.nn.Module]:
    source = cfg["score_sources"]
    controlled = load_json(Path(source["controlled_config"]))
    tokens, lengths = load_candidates(Path(controlled["candidate_representation"]["candidate_sequences"]), pad_token_id=PAD_TOKEN, padded_length=PAD_LENGTH)
    direct_final = load_json(Path(source["direct_final_config"])); direct_training = load_json(Path(source["direct_training_config"]))
    anchored_final = load_json(Path(source["anchored_final_config"])); anchored_training = load_json(Path(source["anchored_training_config"]))
    selected_steps = selected_steps_from_report(Path(anchored_final["gated_validation_report"]))
    return {
        "NVP": load_frozen_nvp(Path(source["nvp_checkpoint_root"]) / f"seed{seed}" / "model.pt", seed).to(device).eval(),
        "Mamba": load_frozen_mamba(Path(source["mamba_checkpoint_root"]) / f"seed{seed}" / "model.pt", seed, controlled, tokens, lengths).to(device).eval(),
        "DirectMambaNVP": load_mamba_nvp(seed, direct_final, direct_training, controlled, device),
        "AnchoredMambaNVP": load_gated(seed, anchored_final, anchored_training, selected_steps, device),
    }


def infer_scores(models: Mapping[str, torch.nn.Module], targets: Sequence[Mapping[str, Any]], features: Mapping[str, Sequence[float]], device: torch.device) -> dict[str, dict[str, np.ndarray]]:
    results = {method: {} for method in METHODS}
    with torch.no_grad():
        for start in range(0, len(targets), 128):
            rows = targets[start : start + 128]
            x = torch.tensor([features[str(row["program_id"])] for row in rows], dtype=torch.float32, device=device)
            values = {"NVP": models["NVP"](x), "Mamba": models["Mamba"](x), "DirectMambaNVP": models["DirectMambaNVP"](x), "AnchoredMambaNVP": models["AnchoredMambaNVP"].components(x)[-1]}
            for method, score in values.items():
                for row, current in zip(rows, score.cpu().numpy()):
                    results[method][str(row["program_id"])] = np.asarray(current, dtype=np.float64)
    return results


def program_metric(split: str, method: str, seed: int, target: Mapping[str, Any], records: Sequence[Mapping[str, Any]], features: Sequence[float], scores: np.ndarray, margin_thresholds: tuple[float, float]) -> dict[str, Any]:
    values = np.asarray(target["raw_candidate_value"], dtype=np.float64)
    soft_target = np.asarray(target["normalized_target"], dtype=np.float64)
    ordered = sorted(range(K), key=lambda index: (-float(scores[index]), index))
    predicted_rank = {candidate: rank + 1 for rank, candidate in enumerate(ordered)}
    oracle_value, oracle_ids = float(values.max()), [index for index, value in enumerate(values) if value == values.max()]
    selected = ordered[0]
    details = policy_details(scores, records, values)
    policy = details["policy45_object_text_size_bytes"]
    oracle_size, oz = min(target["best_object_text"]), int(target["S_Oz"])
    margin = oracle_value - float(np.sort(values)[-2])
    q1, q2 = margin_thresholds
    margin_bin = "small" if margin <= q1 else "medium" if margin <= q2 else "large"
    if details["oracle_admitted"]:
        case = "oracle_admitted_zero_regret" if policy == oracle_size else "oracle_admitted_nonzero_regret"
    else:
        case = "oracle_not_admitted_top10" if details["oracle_rank"] <= 10 else "oracle_not_admitted_beyond_top10"
    probabilities = softmax(scores)
    admitted = details["admitted"]
    result = {
        "split": split, "method": method, "seed": seed, "program_id": target["program_id"], "dataset_id": target["dataset_id"],
        "candidate_value_cross_entropy": float(-np.sum(soft_target * np.log(probabilities + np.finfo(np.float64).tiny))),
        "score_entropy": float(-np.sum(probabilities * np.log(probabilities + np.finfo(np.float64).tiny))),
        "oracle_rank": details["oracle_rank"], "oracle_top1_hit": int(selected in oracle_ids), "oracle_top3_hit": int(any(candidate in oracle_ids for candidate in ordered[:3])), "oracle_top5_hit": int(any(candidate in oracle_ids for candidate in ordered[:5])), "oracle_top10_hit": int(any(candidate in oracle_ids for candidate in ordered[:10])),
        "spearman": spearman(scores, values), "kendall_tau_b": kendall_tau_b(scores, values), "top5_pairwise_inversions": true_top_inversions(ordered, values, 5), "top10_pairwise_inversions": true_top_inversions(ordered, values, 10),
        "selected_candidate_id": selected, "selected_candidate_length": int(records[selected]["candidate_length"]), "selected_true_value": float(values[selected]), "selected_value_gap": oracle_value - float(values[selected]), "oracle_value": oracle_value, "oracle_margin": margin, "top5_value_spread": oracle_value - float(sorted(values, reverse=True)[4]), "margin_bin": margin_bin,
        "policy45_object_text_size_bytes": policy, "oracle_object_text_size_bytes": oracle_size, "policy45_regret_bytes": policy - oracle_size, "mean_over_oz": (oz - policy) / oz,
        "admitted_candidate_count": len(admitted), "mean_admitted_candidate_length": float(np.mean([row["candidate_length"] for row in admitted])), "mean_admitted_observed_prefix_length": float(np.mean([row["observed_prefix_count"] for row in admitted])), "admitted_candidate_ids": json.dumps(details["admitted_ids"]), "oracle_admitted": int(details["oracle_admitted"]), "best_admitted_true_value": details["best_admitted_true_value"], "best_admitted_true_rank": details["best_admitted_true_rank"], "policy45_case": case,
        "score_length_pearson": pearson(scores, np.asarray([record["candidate_length"] for record in records], dtype=np.float64)), "true_value_length_pearson": pearson(values, np.asarray([record["candidate_length"] for record in records], dtype=np.float64)), "initial_autophase_l2": float(np.linalg.norm(np.asarray(features, dtype=np.float64))),
    }
    for name, members in {"rank1": ordered[:1], "rank2_3": ordered[1:3], "rank4_5": ordered[3:5], "rank6_10": ordered[5:10], "rank11_50": ordered[10:]}.items():
        result[f"mean_length_{name}"] = float(np.mean([records[candidate]["candidate_length"] for candidate in members]))
    return result


def grouped(rows: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> dict[tuple[Any, ...], list[Mapping[str, Any]]]:
    output: dict[tuple[Any, ...], list[Mapping[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        output[tuple(row[key] for key in keys)].append(row)
    return output


SUMMARY_COLUMNS = ("mean_over_oz", "oracle_top1_hit", "oracle_top5_hit", "oracle_rank", "policy45_regret_bytes", "spearman", "kendall_tau_b", "admitted_candidate_count", "mean_admitted_candidate_length", "candidate_value_cross_entropy", "score_length_pearson", "top5_pairwise_inversions", "top10_pairwise_inversions")


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result = {"N_valid": len(rows)}
    for key in SUMMARY_COLUMNS:
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        result[f"mean_{key}"] = float(np.mean(values)); result[f"median_{key}"] = float(np.median(values))
    ce = np.asarray([row["candidate_value_cross_entropy"] for row in rows]); regret = np.asarray([row["policy45_regret_bytes"] for row in rows]); rank = np.asarray([row["oracle_rank"] for row in rows]); utility = np.asarray([row["mean_over_oz"] for row in rows])
    result["spearman_ce_vs_policy45_regret"] = spearman(ce, regret); result["spearman_ce_vs_oracle_rank"] = spearman(ce, rank); result["spearman_ce_vs_mean_over_oz"] = spearman(ce, utility)
    result["policy_cases"] = dict(collections.Counter(str(row["policy45_case"]) for row in rows))
    result["margin_bin_regret"] = {name: float(np.mean([row["policy45_regret_bytes"] for row in rows if row["margin_bin"] == name])) for name in ("small", "medium", "large") if any(row["margin_bin"] == name for row in rows)}
    result["selected_length_regret"] = {str(length): float(np.mean([row["policy45_regret_bytes"] for row in rows if row["selected_candidate_length"] == length])) for length in sorted({row["selected_candidate_length"] for row in rows})}
    return result


def correction_summary(rows: Sequence[Mapping[str, Any]], baseline: str, corrected: str) -> dict[str, Any]:
    index = {(row["split"], row["seed"], row["program_id"], row["method"]): row for row in rows}
    output: list[dict[str, Any]] = []
    for key, base in index.items():
        split, seed, program, method = key
        if method != baseline:
            continue
        changed = index[(split, seed, program, corrected)]
        if base["selected_candidate_id"] == changed["selected_candidate_id"]:
            outcome = "unchanged"
        elif changed["selected_true_value"] > base["selected_true_value"]:
            outcome = "beneficial"
        elif changed["selected_true_value"] < base["selected_true_value"]:
            outcome = "harmful"
        else:
            outcome = "neutral_tie"
        output.append({"split": split, "dataset_id": base["dataset_id"], "outcome": outcome, "top1_changed": int(outcome != "unchanged"), "oracle_rank_delta": changed["oracle_rank"] - base["oracle_rank"], "policy45_regret_delta": changed["policy45_regret_bytes"] - base["policy45_regret_bytes"], "admitted_set_changed": int(changed["admitted_candidate_ids"] != base["admitted_candidate_ids"])})
    return {"overall": correction_groups(output), "by_split": {key[0]: correction_groups(group) for key, group in grouped(output, ("split",)).items()}, "per_split_dataset": {f"{key[0]}:{key[1]}": correction_groups(group) for key, group in grouped(output, ("split", "dataset_id")).items()}}


def correction_groups(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts = collections.Counter(row["outcome"] for row in rows); total = len(rows)
    return {"N_program_seed": total, "decision_outcome_counts": dict(counts), "unchanged_rate": counts["unchanged"] / total, "beneficial_rate": counts["beneficial"] / total, "harmful_rate": counts["harmful"] / total, "neutral_tie_rate": counts["neutral_tie"] / total, "mean_oracle_rank_delta": float(np.mean([row["oracle_rank_delta"] for row in rows])), "mean_policy45_regret_delta": float(np.mean([row["policy45_regret_delta"] for row in rows])), "admitted_set_changed_rate": float(np.mean([row["admitted_set_changed"] for row in rows]))}


def disagreement_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    index = {(row["split"], row["seed"], row["program_id"], row["method"]): row for row in rows}; output = []
    for split, seed, program in sorted({key[:3] for key in index}):
        key = (split, seed, program)
        if key + ("NVP",) not in index or key + ("Mamba",) not in index:
            continue
        nvp, mamba = index[key + ("NVP",)], index[key + ("Mamba",)]
        if nvp["selected_true_value"] > mamba["selected_true_value"]:
            category = "nvp_better"
        elif mamba["selected_true_value"] > nvp["selected_true_value"]:
            category = "mamba_better"
        elif nvp["oracle_top1_hit"]:
            category = "both_good"
        else:
            category = "both_poor"
        output.append({"split": split, "dataset_id": nvp["dataset_id"], "category": category, "top1_agree": int(nvp["selected_candidate_id"] == mamba["selected_candidate_id"]), "oracle_margin": nvp["oracle_margin"], "initial_autophase_l2": nvp["initial_autophase_l2"], "nvp_entropy": nvp["score_entropy"], "mamba_entropy": mamba["score_entropy"], "nvp_selected_length": nvp["selected_candidate_length"], "mamba_selected_length": mamba["selected_candidate_length"], "nvp_admitted_count": nvp["admitted_candidate_count"], "mamba_admitted_count": mamba["admitted_candidate_count"], "nvp_regret": nvp["policy45_regret_bytes"], "mamba_regret": mamba["policy45_regret_bytes"]})
    def collapse(group: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        categories = collections.Counter(row["category"] for row in group)
        return {"N_program_seed": len(group), "top1_agreement_rate": float(np.mean([row["top1_agree"] for row in group])), "categories": dict(categories), "mean_oracle_margin": float(np.mean([row["oracle_margin"] for row in group])), "mean_nvp_minus_mamba_regret": float(np.mean([row["nvp_regret"] - row["mamba_regret"] for row in group])), "mean_nvp_minus_mamba_selected_length": float(np.mean([row["nvp_selected_length"] - row["mamba_selected_length"] for row in group])), "mean_nvp_minus_mamba_admitted_count": float(np.mean([row["nvp_admitted_count"] - row["mamba_admitted_count"] for row in group]))}
    return {"overall": collapse(output), "by_split": {key[0]: collapse(group) for key, group in grouped(output, ("split",)).items()}, "per_split_dataset": {f"{key[0]}:{key[1]}": collapse(group) for key, group in grouped(output, ("split", "dataset_id")).items()}}


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore"); writer.writeheader(); writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", type=Path, default=Path("configs/candidate_ranking_error_decomposition_v1.json")); parser.add_argument("--output-dir", type=Path, default=Path("outputs/candidate_ranking_error_decomposition_v1")); args = parser.parse_args()
    cfg = load_json(args.config); validate_config(cfg)
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite diagnostic output: {args.output_dir}")
    validation, validation_matrix, validation_features, validation_total = load_validation(cfg)
    final, final_matrix, final_features, final_total = load_final(cfg)
    margins = np.asarray([float(max(row["raw_candidate_value"]) - sorted(row["raw_candidate_value"])[-2]) for row in validation])
    margin_thresholds = tuple(float(value) for value in np.quantile(margins, [1 / 3, 2 / 3]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows: list[dict[str, Any]] = []
    for seed in cfg["seeds"]:
        models = load_models(int(seed), cfg, device)
        for split, targets, matrix, features in (("validation", validation, validation_matrix, validation_features), ("final", final, final_matrix, final_features)):
            recovered = infer_scores(models, targets, features, device)
            for method in METHODS:
                for target in targets:
                    program = str(target["program_id"])
                    rows.append(program_metric(split, method, int(seed), target, matrix[program], features[program], recovered[method][program], margin_thresholds))
        del models
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    expected_rows = (len(validation) + len(final)) * len(METHODS) * len(cfg["seeds"])
    if len(rows) != expected_rows:
        raise RuntimeError("incomplete score recovery")
    datasets = []
    for (split, dataset, method), members in sorted(grouped(rows, ("split", "dataset_id", "method")).items()):
        summary = summarize(members); summary.update({"split": split, "dataset_id": dataset, "method": method}); datasets.append(summary)
    by_dataset = {(row["split"], row["dataset_id"], row["method"]): row for row in datasets}
    for row in datasets:
        nvp = by_dataset[(row["split"], row["dataset_id"], "NVP")]
        row["delta_mean_over_oz_vs_nvp"] = row["mean_mean_over_oz"] - nvp["mean_mean_over_oz"]
        row["delta_mean_policy45_regret_vs_nvp"] = row["mean_policy45_regret_bytes"] - nvp["mean_policy45_regret_bytes"]
    overall = {f"{split}:{method}": summarize(members) for (split, method), members in sorted(grouped(rows, ("split", "method")).items())}
    dataset_macro = {f"{split}:{method}": {"dataset_macro_mean_over_oz": float(np.mean([row["mean_mean_over_oz"] for row in datasets if row["split"] == split and row["method"] == method])), "dataset_macro_policy45_regret": float(np.mean([row["mean_policy45_regret_bytes"] for row in datasets if row["split"] == split and row["method"] == method]))} for split in ("validation", "final") for method in METHODS}
    corrections = {"DirectMambaNVP_vs_NVP": correction_summary(rows, "NVP", "DirectMambaNVP"), "AnchoredMambaNVP_vs_NVP": correction_summary(rows, "NVP", "AnchoredMambaNVP")}
    disagreement = disagreement_summary(rows)
    report = {"step_execution": "COMPLETE", "offline_diagnostic_only": True, "compiler_gym_initialized": False, "llvm_execution": False, "candidate_rollouts": 0, "objecttext_observations": 0, "label_regeneration": False, "model_training": False, "checkpoint_selection": False, "device": str(device), "population": {"validation": {"total": validation_total, "valid": len(validation), "invalid": validation_total - len(validation)}, "final": {"total": final_total, "valid": len(final), "invalid": final_total - len(final)}}, "policy45_semantics": "Imported scripts.evaluate_mamba_nvp_final_objecttext.policy45; descending score, candidate ID ascending tie-break, sequential prefix consumption to exactly 45 passes, minimum observed prefix size.", "margin_thresholds_from_validation": {"q33": margin_thresholds[0], "q67": margin_thresholds[1]}, "overall": overall, "dataset_macro": dataset_macro, "per_dataset": datasets, "nvp_mamba_disagreement": disagreement, "correction_analysis": corrections, "diagnostic_interpretation": "Descriptive only. No final/OOD result was used to select a model or future hyperparameter."}
    args.output_dir.mkdir(parents=True); write_json(args.output_dir / "config.json", cfg); write_json(args.output_dir / "comparison_report.json", report); write_csv(args.output_dir / "per_dataset.csv", datasets); write_csv(args.output_dir / "per_program.csv", rows)
    final_nvp, final_mamba = overall["final:NVP"], overall["final:Mamba"]
    final_nvp_macro, final_mamba_macro = dataset_macro["final:NVP"], dataset_macro["final:Mamba"]
    analysis = "# Candidate-ranking error decomposition\n\n" + f"Frozen offline score recovery completed for {len(rows):,} program-method-seed rows. Validation/final valid cohorts are {len(validation):,}/{len(final):,}; no CompilerGym, LLVM, ObjectText observation, rollout, training, or checkpoint selection occurred.\n\n" + f"Final Mamba minus NVP: dataset-macro MeanOverOz {final_mamba_macro['dataset_macro_mean_over_oz'] - final_nvp_macro['dataset_macro_mean_over_oz']:+.8f}; program-micro MeanOverOz {final_mamba['mean_mean_over_oz'] - final_nvp['mean_mean_over_oz']:+.8f}; mean policy45 regret {final_mamba['mean_policy45_regret_bytes'] - final_nvp['mean_policy45_regret_bytes']:+.4f} bytes; top-1 oracle hit {final_mamba['mean_oracle_top1_hit'] - final_nvp['mean_oracle_top1_hit']:+.4f}; top-5 hit {final_mamba['mean_oracle_top5_hit'] - final_nvp['mean_oracle_top5_hit']:+.4f}.\n\nDetailed per-source, per-method, seed-matched correction, CE-alignment, rank, inversion, candidate-length, and exact policy45-admission statistics are in `comparison_report.json`, `per_dataset.csv`, and `per_program.csv`.\n"
    (args.output_dir / "analysis.md").write_text(analysis, encoding="utf-8")
    print(json.dumps({"rows": len(rows), "output": str(args.output_dir), "population": report["population"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
