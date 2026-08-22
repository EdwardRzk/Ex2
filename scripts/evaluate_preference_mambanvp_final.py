#!/usr/bin/env python3
"""Evaluate frozen PreferenceAwareMambaNVP checkpoints on final/OOD artifacts only."""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

if __package__:
    from scripts.evaluate_cross_candidate_mambanvp_final import invalid_summary, read_results
    from scripts.evaluate_mamba_nvp_final_objecttext import K, aggregate, load_final_features, policy45, read_final_artifacts, regret_summary
    from scripts.train_preference_mambanvp import METHOD, PreferenceAwareMambaNVP
    from scripts.train_set_conditioned_mamba_ranker import load_candidates, load_json
else:
    from evaluate_cross_candidate_mambanvp_final import invalid_summary, read_results
    from evaluate_mamba_nvp_final_objecttext import K, aggregate, load_final_features, policy45, read_final_artifacts, regret_summary
    from train_preference_mambanvp import METHOD, PreferenceAwareMambaNVP
    from train_set_conditioned_mamba_ranker import load_candidates, load_json


def validate_config(cfg: Mapping[str, Any]) -> None:
    if cfg["final_seed_set"] != [1, 2, 3] or cfg["final_population"] != {"total": 4683, "complete_k50": 4679, "invalid": 4}:
        raise ValueError("frozen final population or seed set mismatch")
    if cfg["candidate_representation"] != {"K": 50, "padded_length": 20, "pad_token_id": 124}:
        raise ValueError("frozen candidate representation mismatch")
    if cfg["inference"] != {"sampling": False, "ranking": "descending value logits; candidate ID ascending tie break", "scored_pass_budget": 45}:
        raise ValueError("frozen policy45 inference mismatch")
    if cfg["comparison_family"] != ["NVP", "Mamba", "MambaNVP", "CrossCandidateMambaNVP", METHOD]:
        raise ValueError("final comparison family mismatch")
    if cfg["checkpoint_selection"]["selected_steps"] != {"1": 5800, "2": 7500, "3": 6800}:
        raise ValueError("validation-selected PreferenceAware checkpoints mismatch")
    if cfg["preference_diagnostic"] != {"enabled": True, "definition": "strict final K=50 pair accuracy of the frozen preference head; a smaller best ObjectText size is preferred", "affects_ranking": False}:
        raise ValueError("frozen preference diagnostic mismatch")


def selected_checkpoints_from_report(cfg: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    report = load_json(Path(cfg["checkpoint_selection_source"]))
    entries = report["preference_aware_mambanvp"]["seed_results"]
    selected = {int(item["seed"]): item for item in entries}
    expected = {int(seed): step for seed, step in cfg["checkpoint_selection"]["selected_steps"].items()}
    if set(selected) != set(expected) or any(int(selected[seed]["selected_step"]) != step for seed, step in expected.items()):
        raise ValueError("frozen PreferenceAware validation selections mismatch")
    for seed, item in selected.items():
        if item["checkpoint_path"] != cfg["preference_checkpoint_paths"][str(seed)]:
            raise ValueError("checkpoint path differs from frozen validation record")
    return selected


def load_preference_model(seed: int, cfg: Mapping[str, Any], training_cfg: Mapping[str, Any], selected: Mapping[int, Mapping[str, Any]], device: torch.device) -> PreferenceAwareMambaNVP:
    path = Path(cfg["preference_checkpoint_paths"][str(seed)])
    payload = torch.load(path, map_location="cpu", weights_only=False)
    expected_hash = selected[seed]["checkpoint_sha256"]
    actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual_hash != expected_hash:
        raise ValueError(f"checkpoint hash mismatch: {path}")
    if (payload.get("stage") != "Route-A Preference-aware MambaNVP v1" or payload.get("architecture") != METHOD or payload.get("seed") != seed or payload.get("step") != selected[seed]["selected_step"] or payload.get("lambda_preference") != training_cfg["target_and_objective"]["lambda_preference"]):
        raise ValueError(f"not the validation-selected frozen PreferenceAware checkpoint: {path}")
    controlled = load_json(Path(training_cfg["candidate_representation_source"]))
    tokens, lengths = load_candidates(Path(controlled["candidate_representation"]["candidate_sequences"]), pad_token_id=124, padded_length=20)
    model = PreferenceAwareMambaNVP(payload["model_config"], tokens, lengths)
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.to(device).eval()


def final_diagnostics(model: PreferenceAwareMambaNVP, embeddings: torch.Tensor, sizes: torch.Tensor) -> tuple[int, int, int, int]:
    """Return strict preference/value correct counts and strict pair total from frozen final labels."""
    preference_correct = value_correct = pair_total = 0
    rows = torch.arange(len(embeddings), device=embeddings.device)
    scores = model.value_head(embeddings).squeeze(-1)
    for first in range(K):
        for second in range(first + 1, K):
            strict = sizes[:, first] != sizes[:, second]
            if bool(strict.any()):
                winner = torch.where(sizes[:, first] < sizes[:, second], torch.full_like(sizes[:, first], first, dtype=torch.long), torch.full_like(sizes[:, first], second, dtype=torch.long))
                loser = torch.where(winner == first, torch.full_like(winner, second), torch.full_like(winner, first))
                preference = model.preference_head(embeddings[rows, winner] - embeddings[rows, loser]).squeeze(-1)
                preference_correct += int((preference[strict] > 0).sum())
                value_correct += int((scores[rows, winner][strict] > scores[rows, loser][strict]).sum())
                pair_total += int(strict.sum())
    return preference_correct, value_correct, pair_total, len(embeddings)


def infer_preference(model: PreferenceAwareMambaNVP, seed: int, programs: Sequence[str], matrix: Mapping[str, Sequence[Mapping[str, Any]]], summaries: Mapping[str, Mapping[str, Any]], features: Mapping[str, Sequence[float]], device: torch.device) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    eligible = [program for program in programs if program in matrix and summaries[program]["ratio_metric_validity"] == "valid_for_ObjectText_ratio_metric"]
    if len(eligible) != 4679:
        raise ValueError("frozen final eligible population mismatch")
    logits: dict[str, list[float]] = {}
    preference_correct = value_correct = pair_total = 0
    with torch.no_grad():
        for start in range(0, len(eligible), 128):
            current = eligible[start : start + 128]
            program_features = torch.tensor([features[program] for program in current], dtype=torch.float32, device=device)
            embeddings = model.embeddings(program_features)
            logits.update(zip(current, model.value_head(embeddings).squeeze(-1).cpu().tolist()))
            sizes = torch.tensor([[record["best_object_text_size_bytes"] for record in matrix[program]] for program in current], dtype=torch.float32, device=device)
            pref_ok, value_ok, pairs, _ = final_diagnostics(model, embeddings, sizes)
            preference_correct += pref_ok; value_correct += value_ok; pair_total += pairs
    rows: dict[str, dict[str, Any]] = {}
    failures: collections.Counter[str] = collections.Counter()
    top1_correct = 0
    for program in programs:
        summary = summaries[program]
        row: dict[str, Any] = {"program_id": program, "dataset_id": summary["dataset_id"], "model": METHOD, "seed": seed, "valid": False}
        if summary["oracle_K50_validity"] != "valid_complete_K50":
            row["failure_reason"] = "incomplete_K50"; failures["incomplete_K50"] += 1
        elif summary["ratio_metric_validity"] != "valid_for_ObjectText_ratio_metric":
            row["failure_reason"] = "invalid_ratio_denominator"; failures["invalid_ratio_denominator"] += 1
        else:
            scores, records = logits[program], matrix[program]
            candidate_id = min(range(K), key=lambda index: (-scores[index], index))
            policy = policy45(scores, records)
            oracle = min(record["best_object_text_size_bytes"] for record in records)
            oz = int(summary["oz_object_text_size_bytes"])
            top1_correct += int(records[candidate_id]["best_object_text_size_bytes"] == oracle)
            row.update({"valid": True, "selected_candidate_id": candidate_id, "policy45_object_text_size_bytes": policy, "oracle_object_text_size_bytes": oracle, "oz_object_text_size_bytes": oz, "mean_over_oz": (oz - policy) / oz, "policy45_regret_bytes": policy - oracle})
        rows[program] = row
    return rows, {"model": METHOD, "seed": seed, "N_total": 4683, "N_primary_valid": 4679, "failure_count_by_reason": dict(failures), "top1_accuracy": top1_correct / len(eligible), "preference_accuracy": preference_correct / pair_total if pair_total else None, "value_pairwise_accuracy": value_correct / pair_total if pair_total else None, "final_strict_pair_count": pair_total}


def read_cross_candidate_rows(path: Path, seed: int) -> dict[str, dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        rows = [row for line in handle if (row := json.loads(line))["model"] == "CrossCandidateMambaNVP" and row["seed"] == seed]
    if len(rows) != 4683 or len({row["program_id"] for row in rows}) != 4683:
        raise ValueError("frozen Cross-Candidate result population mismatch")
    return {row["program_id"]: row for row in rows}


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", type=Path, required=True); parser.add_argument("--output-dir", type=Path, required=True); args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {args.output_dir}")
    cfg = load_json(args.config); validate_config(cfg)
    training_cfg = load_json(Path(cfg["preference_training_config"])); selected = selected_checkpoints_from_report(cfg)
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
    preference_reports = []
    for seed in cfg["final_seed_set"]:
        rows, seed_report = infer_preference(load_preference_model(seed, cfg, training_cfg, selected, device), seed, programs, matrix, summaries, features, device)
        result_maps[(METHOD, seed)] = rows; preference_reports.append(seed_report)
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "config.json").write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    combined = aggregate(cfg["comparison_family"], programs, summaries, result_maps)
    oracle = load_json(Path(cfg["existing_final_report"]))["offline_k50_oracle"]
    mean = combined["dataset_macro"][METHOD]["three_seed_mean"]
    summary = {"three_seed_mean_over_oz": mean, "oracle_recovery": mean / oracle["dataset_macro"], "policy45_regret": regret_summary(METHOD, result_maps), "top1_accuracy_3seed": sum(row["top1_accuracy"] for row in preference_reports) / 3, "preference_accuracy_3seed": sum(row["preference_accuracy"] for row in preference_reports) / 3, "value_pairwise_accuracy_3seed": sum(row["value_pairwise_accuracy"] for row in preference_reports) / 3, "final_strict_pair_count_per_seed": preference_reports[0]["final_strict_pair_count"], "seed_results": preference_reports}
    (args.output_dir / "per_seed_results.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "per_dataset_results.json").write_text(json.dumps(combined["per_dataset"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = {"step_execution": "COMPLETE", "offline_only": True, "compiler_gym_initialized": False, "llvm_execution": False, "candidate_rollouts": 0, "objecttext_measurements": 0, "label_regeneration": False, "invalid_program_retry": False, "checkpoint_reselection": False, "model_training": False, "final_population": {"N_total": 4683, "N_complete_K50_valid": 4679, "N_invalid": 4}, "device": str(device), "common_cohort_invalid_statistics": invalid_summary(cfg["comparison_family"], result_maps), "combined_comparison": combined, "frozen_offline_k50_oracle": oracle, "preference_aware_mambanvp": summary, "oracle_recovery": {method: combined["dataset_macro"][method]["three_seed_mean"] / oracle["dataset_macro"] for method in cfg["comparison_family"]}, "policy45_regret": {method: regret_summary(method, result_maps) for method in cfg["comparison_family"]}, "preference_aware_minus": {method: mean - combined["dataset_macro"][method]["three_seed_mean"] for method in ("NVP", "Mamba", "MambaNVP", "CrossCandidateMambaNVP")}}
    (args.output_dir / "comparison_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "experiment_report.json").write_text(json.dumps({"step_execution": "COMPLETE", "offline_only": True, "selected_checkpoints": [{"seed": seed, "path": cfg["preference_checkpoint_paths"][str(seed)], "step": selected[seed]["selected_step"], "sha256": selected[seed]["checkpoint_sha256"]} for seed in cfg["final_seed_set"]]}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
