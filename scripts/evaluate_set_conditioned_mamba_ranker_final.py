#!/usr/bin/env python3
"""Evaluate frozen SetConditionedMambaRanker checkpoints on final/OOD artifacts only."""
from __future__ import annotations

import argparse
import collections
import gzip
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

if __package__:
    from scripts.evaluate_cross_candidate_mambanvp_final import invalid_summary, read_results
    from scripts.evaluate_mamba_nvp_final_objecttext import K, aggregate, load_final_features, policy45, read_final_artifacts, regret_summary
    from scripts.train_set_conditioned_mamba_ranker import SetConditionedMambaRanker, load_candidates, load_json
else:
    from evaluate_cross_candidate_mambanvp_final import invalid_summary, read_results
    from evaluate_mamba_nvp_final_objecttext import K, aggregate, load_final_features, policy45, read_final_artifacts, regret_summary
    from train_set_conditioned_mamba_ranker import SetConditionedMambaRanker, load_candidates, load_json


METHOD = "SetConditionedMambaRanker"


def validate_config(cfg: Mapping[str, Any]) -> None:
    if cfg["final_seed_set"] != [1, 2, 3] or cfg["final_population"] != {"total": 4683, "complete_k50": 4679, "invalid": 4}:
        raise ValueError("frozen final population or seed set mismatch")
    if cfg["candidate_representation"] != {"K": 50, "padded_length": 20, "pad_token_id": 124}:
        raise ValueError("frozen candidate representation mismatch")
    if cfg["inference"] != {"sampling": False, "ranking": "descending scores; candidate ID ascending tie break", "scored_pass_budget": 45}:
        raise ValueError("frozen policy45 inference mismatch")
    if cfg["comparison_family"] != ["NVP", "Mamba", "MambaNVP", "CrossCandidateMambaNVP", METHOD]:
        raise ValueError("final comparison family mismatch")
    if cfg["checkpoint_selection"]["selected_steps"] != {"1": 7200, "2": 6200, "3": 6700}:
        raise ValueError("validation-selected Listwise checkpoints mismatch")


def selected_steps_from_report(path: Path) -> dict[int, int]:
    report = load_json(path)
    selected = {int(item["seed"]): int(item["selected"]["step"]) for item in report["set_conditioned_mamba_ranker"]["seed_results"]}
    if selected != {1: 7200, 2: 6200, 3: 6700}:
        raise ValueError("frozen Listwise validation selections mismatch")
    return selected


def load_ranker(seed: int, cfg: Mapping[str, Any], training_cfg: Mapping[str, Any], selected_steps: Mapping[int, int], device: torch.device) -> SetConditionedMambaRanker:
    path = Path(cfg["ranker_checkpoint_root"]) / f"seed{seed}" / "model.pt"
    payload = torch.load(path, map_location="cpu")
    if payload.get("stage") != "Route-A Set-Conditioned Listwise Mamba Ranker v1" or payload.get("architecture") != METHOD or payload.get("seed") != seed or payload.get("step") != selected_steps[seed]:
        raise ValueError(f"not the validation-selected frozen Listwise checkpoint: {path}")
    model_cfg = payload.get("model_config")
    if not isinstance(model_cfg, dict) or model_cfg.get("candidate_interaction") != training_cfg["architecture"]["candidate_interaction"]:
        raise ValueError("checkpoint model configuration mismatch")
    tokens, lengths = load_candidates(Path(training_cfg["candidate_representation_source"]).parent / "rlcompopt_action_seq_50.txt", pad_token_id=124, padded_length=20)
    model = SetConditionedMambaRanker(model_cfg, tokens, lengths)
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.to(device).eval()


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


def ranking_diagnostics(scores: Mapping[str, Sequence[float]], matrix: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    correct, correlations, undefined = 0, [], 0
    for program, score in scores.items():
        sizes = np.asarray([row["best_object_text_size_bytes"] for row in matrix[program]], dtype=np.float64)
        prediction = min(range(K), key=lambda index: (-float(score[index]), index))
        correct += int(sizes[prediction] == sizes.min())
        target_ranks, predicted_ranks = average_ranks(sizes, descending=False), average_ranks(np.asarray(score), descending=True)
        if np.std(target_ranks) == 0.0 or np.std(predicted_ranks) == 0.0:
            undefined += 1
        else:
            correlations.append(float(np.corrcoef(target_ranks, predicted_ranks)[0, 1]))
    return {"top1_accuracy": correct / len(scores), "ranking_correlation": {"name": "tie-aware Spearman rank correlation of predicted scores against final K=50 best ObjectText size", "mean": float(np.mean(correlations)) if correlations else None, "N_defined": len(correlations), "N_undefined_all_tied_or_constant": undefined}}


def infer_ranker(model: SetConditionedMambaRanker, seed: int, programs: Sequence[str], matrix: Mapping[str, Sequence[Mapping[str, Any]]], summaries: Mapping[str, Mapping[str, Any]], features: Mapping[str, Sequence[float]], device: torch.device) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    eligible = [program for program in programs if program in matrix and summaries[program]["ratio_metric_validity"] == "valid_for_ObjectText_ratio_metric"]
    if len(eligible) != 4679:
        raise ValueError("frozen final eligible population mismatch")
    scores: dict[str, list[float]] = {}
    with torch.no_grad():
        for start in range(0, len(eligible), 128):
            current = eligible[start : start + 128]
            scores.update(zip(current, model(torch.tensor([features[program] for program in current], dtype=torch.float32, device=device)).cpu().tolist()))
    rows: dict[str, dict[str, Any]] = {}
    failures: collections.Counter[str] = collections.Counter()
    for program in programs:
        summary = summaries[program]
        row: dict[str, Any] = {"program_id": program, "dataset_id": summary["dataset_id"], "model": METHOD, "seed": seed, "valid": False}
        if summary["oracle_K50_validity"] != "valid_complete_K50":
            row["failure_reason"] = "incomplete_K50"; failures["incomplete_K50"] += 1
        elif summary["ratio_metric_validity"] != "valid_for_ObjectText_ratio_metric":
            row["failure_reason"] = "invalid_ratio_denominator"; failures["invalid_ratio_denominator"] += 1
        else:
            policy = policy45(scores[program], matrix[program])
            oracle = min(record["best_object_text_size_bytes"] for record in matrix[program])
            oz = int(summary["oz_object_text_size_bytes"])
            row.update({"valid": True, "policy45_object_text_size_bytes": policy, "oracle_object_text_size_bytes": oracle, "oz_object_text_size_bytes": oz, "mean_over_oz": (oz - policy) / oz, "policy45_regret_bytes": policy - oracle})
        rows[program] = row
    diagnostics = ranking_diagnostics(scores, matrix)
    return rows, {"model": METHOD, "seed": seed, "N_total": 4683, "N_primary_valid": 4679, "failure_count_by_reason": dict(failures), **diagnostics}


def read_cross_candidate_rows(path: Path, seed: int) -> dict[str, dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if (item := json.loads(line)) and item["model"] == "CrossCandidateMambaNVP" and item["seed"] == seed]
    if len(rows) != 4683 or len({row["program_id"] for row in rows}) != 4683:
        raise ValueError("frozen Cross-Candidate result population mismatch")
    return {row["program_id"]: row for row in rows}


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", type=Path, required=True); parser.add_argument("--output-dir", type=Path, required=True); args = parser.parse_args()
    if args.output_dir.exists(): raise FileExistsError(f"refusing to overwrite existing output directory: {args.output_dir}")
    cfg = load_json(args.config); validate_config(cfg)
    training_cfg = load_json(Path(cfg["ranker_training_config"])); selected_steps = selected_steps_from_report(Path(cfg["checkpoint_selection"]["source"]))
    programs, matrix, summaries = read_final_artifacts(Path(cfg["final_label_shards"]))
    eligible = [program for program in programs if program in matrix and summaries[program]["ratio_metric_validity"] == "valid_for_ObjectText_ratio_metric"]
    features = load_final_features(Path(cfg["final_feature_cache"]), eligible)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result_maps: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    for method in ("NVP", "Mamba", "MambaNVP"):
        for seed in cfg["final_seed_set"]:
            result_maps[(method, seed)] = read_results(Path(cfg["existing_model_results"][method]) / f"seed{seed}.jsonl.gz", method, seed)
    for seed in cfg["final_seed_set"]:
        result_maps[("CrossCandidateMambaNVP", seed)] = read_cross_candidate_rows(Path(cfg["existing_model_results"]["CrossCandidateMambaNVP"]), seed)
    ranker_reports = []
    for seed in cfg["final_seed_set"]:
        model = load_ranker(seed, cfg, training_cfg, selected_steps, device)
        rows, seed_report = infer_ranker(model, seed, programs, matrix, summaries, features, device)
        result_maps[(METHOD, seed)] = rows; ranker_reports.append(seed_report)
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "config.json").write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    combined = aggregate(cfg["comparison_family"], programs, summaries, result_maps)
    oracle = load_json(Path(cfg["existing_final_report"]))["offline_k50_oracle"]
    ranker_mean = combined["dataset_macro"][METHOD]["three_seed_mean"]
    ranker_summary = {"three_seed_mean_over_oz": ranker_mean, "oracle_recovery": ranker_mean / oracle["dataset_macro"], "policy45_regret": regret_summary(METHOD, result_maps), "top1_accuracy_3seed": sum(row["top1_accuracy"] for row in ranker_reports) / 3, "ranking_correlation_3seed": sum(row["ranking_correlation"]["mean"] for row in ranker_reports) / 3, "seed_results": ranker_reports}
    (args.output_dir / "per_seed_results.json").write_text(json.dumps(ranker_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "per_dataset_results.json").write_text(json.dumps(combined["per_dataset"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = {"step_execution": "COMPLETE", "offline_only": True, "compiler_gym_initialized": False, "llvm_execution": False, "candidate_rollouts": 0, "objecttext_measurements": 0, "label_regeneration": False, "checkpoint_reselection": False, "model_training": False, "final_population": {"N_total": 4683, "N_complete_K50_valid": 4679, "N_invalid": 4}, "device": str(device), "common_cohort_invalid_statistics": invalid_summary(cfg["comparison_family"], result_maps), "combined_comparison": combined, "frozen_offline_k50_oracle": oracle, "set_conditioned_mamba_ranker": ranker_summary, "oracle_recovery": {method: combined["dataset_macro"][method]["three_seed_mean"] / oracle["dataset_macro"] for method in cfg["comparison_family"]}, "policy45_regret": {method: regret_summary(method, result_maps) for method in cfg["comparison_family"]}, "set_conditioned_minus": {method: ranker_mean - combined["dataset_macro"][method]["three_seed_mean"] for method in ("NVP", "Mamba", "MambaNVP", "CrossCandidateMambaNVP")}}
    (args.output_dir / "comparison_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
