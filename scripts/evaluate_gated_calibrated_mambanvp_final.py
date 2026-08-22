#!/usr/bin/env python3
"""Evaluate frozen GatedCalibratedMambaNVP checkpoints on final/OOD artifacts only."""
from __future__ import annotations

import argparse
import collections
import gzip
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

if __package__:
    from scripts.evaluate_cross_candidate_mambanvp_final import invalid_summary, read_results
    from scripts.evaluate_mamba_nvp_final_objecttext import K, aggregate, load_final_features, policy45, read_final_artifacts, regret_summary
    from scripts.train_gated_calibrated_mambanvp import METHOD, GatedCalibratedMambaNVP, kl_final_to_nvp, kl_nvp_to_final
    from scripts.train_mamba_nvp_objecttext import load_frozen_nvp
    from scripts.train_set_conditioned_mamba_ranker import load_candidates, load_json
else:
    from evaluate_cross_candidate_mambanvp_final import invalid_summary, read_results
    from evaluate_mamba_nvp_final_objecttext import K, aggregate, load_final_features, policy45, read_final_artifacts, regret_summary
    from train_gated_calibrated_mambanvp import METHOD, GatedCalibratedMambaNVP, kl_final_to_nvp, kl_nvp_to_final
    from train_mamba_nvp_objecttext import load_frozen_nvp
    from train_set_conditioned_mamba_ranker import load_candidates, load_json


def validate_config(cfg: Mapping[str, Any]) -> None:
    if cfg["final_seed_set"] != [1, 2, 3] or cfg["final_population"] != {"total": 4683, "complete_k50": 4679, "invalid": 4}:
        raise ValueError("frozen final population or seed set mismatch")
    if cfg["candidate_representation"] != {"K": 50, "padded_length": 20, "pad_token_id": 124}:
        raise ValueError("frozen candidate representation mismatch")
    if cfg["inference"] != {"sampling": False, "ranking": "descending final logits; candidate ID ascending tie break", "scored_pass_budget": 45}:
        raise ValueError("frozen policy45 inference mismatch")
    if cfg["comparison_family"] != ["NVP", "Mamba", "MambaNVP", "CrossCandidateMambaNVP", METHOD]:
        raise ValueError("final comparison family mismatch")
    if cfg["checkpoint_selection"]["selected_steps"] != {"1": 3400, "2": 500, "3": 3500}:
        raise ValueError("validation-selected Gated-Calibrated checkpoints mismatch")


def selected_steps_from_report(path: Path) -> dict[int, int]:
    report = load_json(path)
    selected = {int(item["seed"]): int(item["selected"]["step"]) for item in report["gated_calibrated_mambanvp"]["seed_results"]}
    if selected != {1: 3400, 2: 500, 3: 3500}:
        raise ValueError("frozen Gated-Calibrated validation selections mismatch")
    return selected


def load_gated(seed: int, cfg: Mapping[str, Any], training_cfg: Mapping[str, Any], selected_steps: Mapping[int, int], device: torch.device) -> GatedCalibratedMambaNVP:
    path = Path(cfg["gated_checkpoint_root"]) / f"seed{seed}" / "model.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (payload.get("stage") != "Route-A Gated-Calibrated MambaNVP v2" or payload.get("architecture") != METHOD or payload.get("seed") != seed or payload.get("step") != selected_steps[seed] or payload.get("nvp_frozen") is not True or payload.get("lambda_kl") != training_cfg["target_and_objective"]["lambda_kl"] or payload.get("fusion") != training_cfg["architecture"]["fusion"]):
        raise ValueError(f"not the validation-selected frozen Gated-Calibrated checkpoint: {path}")
    controlled = load_json(Path(training_cfg["candidate_representation_source"]))
    tokens, lengths = load_candidates(Path(controlled["candidate_representation"]["candidate_sequences"]), pad_token_id=124, padded_length=20)
    model = GatedCalibratedMambaNVP(load_frozen_nvp(Path(payload["nvp_checkpoint"]), seed), payload["model_config"], tokens, lengths)
    model.load_state_dict(payload["state_dict"], strict=True)
    if any(parameter.requires_grad for parameter in model.nvp.parameters()) or model.nvp.training:
        raise RuntimeError("frozen NVP invariant failed")
    return model.to(device).eval()


def infer_gated(model: GatedCalibratedMambaNVP, seed: int, programs: Sequence[str], matrix: Mapping[str, Sequence[Mapping[str, Any]]], summaries: Mapping[str, Mapping[str, Any]], features: Mapping[str, Sequence[float]], device: torch.device) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    eligible = [program for program in programs if program in matrix and summaries[program]["ratio_metric_validity"] == "valid_for_ObjectText_ratio_metric"]
    if len(eligible) != 4679:
        raise ValueError("frozen final eligible population mismatch")
    logits: dict[str, list[float]] = {}
    alpha_total = kl_forward_total = kl_reverse_total = 0.0
    with torch.no_grad():
        for start in range(0, len(eligible), 128):
            current = eligible[start:start + 128]
            program_features = torch.tensor([features[program] for program in current], dtype=torch.float32, device=device)
            nvp_logits, _, alpha, final_logits = model.components(program_features)
            logits.update(zip(current, final_logits.cpu().tolist()))
            alpha_total += float(alpha.sum().cpu())
            kl_forward_total += float(kl_final_to_nvp(final_logits, nvp_logits).cpu()) * len(current)
            kl_reverse_total += float(kl_nvp_to_final(final_logits, nvp_logits).cpu()) * len(current)
    rows: dict[str, dict[str, Any]] = {}
    failures: collections.Counter[str] = collections.Counter(); top1_correct = 0
    for program in programs:
        summary = summaries[program]
        row: dict[str, Any] = {"program_id": program, "dataset_id": summary["dataset_id"], "model": METHOD, "seed": seed, "valid": False}
        if summary["oracle_K50_validity"] != "valid_complete_K50":
            row["failure_reason"] = "incomplete_K50"; failures["incomplete_K50"] += 1
        elif summary["ratio_metric_validity"] != "valid_for_ObjectText_ratio_metric":
            row["failure_reason"] = "invalid_ratio_denominator"; failures["invalid_ratio_denominator"] += 1
        else:
            scores, records = logits[program], matrix[program]
            choice = min(range(K), key=lambda index: (-scores[index], index))
            policy = policy45(scores, records); oracle = min(record["best_object_text_size_bytes"] for record in records); oz = int(summary["oz_object_text_size_bytes"])
            top1_correct += int(records[choice]["best_object_text_size_bytes"] == oracle)
            row.update({"valid": True, "selected_candidate_id": choice, "policy45_object_text_size_bytes": policy, "oracle_object_text_size_bytes": oracle, "oz_object_text_size_bytes": oz, "mean_over_oz": (oz - policy) / oz, "policy45_regret_bytes": policy - oracle})
        rows[program] = row
    return rows, {"model": METHOD, "seed": seed, "N_total": 4683, "N_primary_valid": 4679, "failure_count_by_reason": dict(failures), "top1_accuracy": top1_correct / len(eligible), "average_gate_alpha": alpha_total / (len(eligible) * K), "calibration_kl_final_to_nvp": kl_forward_total / len(eligible), "validation_kl_nvp_to_final": kl_reverse_total / len(eligible)}


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
    training_cfg = load_json(Path(cfg["gated_training_config"])); selected_steps = selected_steps_from_report(Path(cfg["gated_validation_report"]))
    programs, matrix, summaries = read_final_artifacts(Path(cfg["final_label_shards"])); eligible = [program for program in programs if program in matrix and summaries[program]["ratio_metric_validity"] == "valid_for_ObjectText_ratio_metric"]
    features = load_final_features(Path(cfg["final_feature_cache"]), eligible); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result_maps: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    for method in ("NVP", "Mamba", "MambaNVP"):
        for seed in cfg["final_seed_set"]:
            result_maps[(method, seed)] = read_results(Path(cfg["existing_model_results"][method]) / f"seed{seed}.jsonl.gz", method, seed)
    for seed in cfg["final_seed_set"]:
        result_maps[("CrossCandidateMambaNVP", seed)] = read_cross_candidate_rows(Path(cfg["existing_model_results"]["CrossCandidateMambaNVP"]), seed)
    gated_reports = []
    for seed in cfg["final_seed_set"]:
        rows, seed_report = infer_gated(load_gated(seed, cfg, training_cfg, selected_steps, device), seed, programs, matrix, summaries, features, device)
        result_maps[(METHOD, seed)] = rows; gated_reports.append(seed_report)
    args.output_dir.mkdir(parents=True); (args.output_dir / "config.json").write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n")
    combined = aggregate(cfg["comparison_family"], programs, summaries, result_maps); oracle = load_json(Path(cfg["existing_final_report"]))["offline_k50_oracle"]; mean = combined["dataset_macro"][METHOD]["three_seed_mean"]
    summary = {"three_seed_mean_over_oz": mean, "oracle_recovery": mean / oracle["dataset_macro"], "policy45_regret": regret_summary(METHOD, result_maps), "top1_accuracy_3seed": sum(item["top1_accuracy"] for item in gated_reports) / 3, "average_gate_alpha_3seed": sum(item["average_gate_alpha"] for item in gated_reports) / 3, "calibration_kl_final_to_nvp_3seed": sum(item["calibration_kl_final_to_nvp"] for item in gated_reports) / 3, "validation_kl_nvp_to_final_3seed": sum(item["validation_kl_nvp_to_final"] for item in gated_reports) / 3, "seed_results": gated_reports}
    (args.output_dir / "per_seed_results.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n"); (args.output_dir / "per_dataset_results.json").write_text(json.dumps(combined["per_dataset"], indent=2, sort_keys=True) + "\n")
    report = {"step_execution": "COMPLETE", "offline_only": True, "compiler_gym_initialized": False, "llvm_execution": False, "candidate_rollouts": 0, "objecttext_measurements": 0, "label_regeneration": False, "invalid_program_retry": False, "checkpoint_reselection": False, "model_training": False, "final_population": {"N_total": 4683, "N_complete_K50_valid": 4679, "N_invalid": 4}, "device": str(device), "common_cohort_invalid_statistics": invalid_summary(cfg["comparison_family"], result_maps), "combined_comparison": combined, "frozen_offline_k50_oracle": oracle, "gated_calibrated_mambanvp": summary, "oracle_recovery": {method: combined["dataset_macro"][method]["three_seed_mean"] / oracle["dataset_macro"] for method in cfg["comparison_family"]}, "policy45_regret": {method: regret_summary(method, result_maps) for method in cfg["comparison_family"]}, "gated_calibrated_minus": {method: mean - combined["dataset_macro"][method]["three_seed_mean"] for method in ("NVP", "Mamba", "MambaNVP", "CrossCandidateMambaNVP")}}
    (args.output_dir / "comparison_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n"); (args.output_dir / "experiment_report.json").write_text(json.dumps({"step_execution": "COMPLETE", "offline_only": True, "selected_steps": selected_steps}, indent=2, sort_keys=True) + "\n"); print(json.dumps(report, indent=2, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
