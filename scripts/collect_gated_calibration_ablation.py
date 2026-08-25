#!/usr/bin/env python3
"""Collect completed fixed Gate/KL ablation task reports without rerunning inference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

if __package__:
    from scripts.train_gated_calibrated_mambanvp import frozen_references
    from scripts.train_set_conditioned_mamba_ranker import load_json
else:
    from train_gated_calibrated_mambanvp import frozen_references
    from train_set_conditioned_mamba_ranker import load_json

VARIANTS = ("gated_full", "no_kl", "no_gate", "no_gate_no_kl")
ATTEMPTS = {
    "gated_full": {1: "parallel_reconstruction1", 2: "parallel1", 3: "parallel1"},
    "no_kl": {1: "parallel1", 2: "parallel1", 3: "parallel1"},
    "no_gate": {1: "parallel1", 2: "parallel1", 3: "parallel1"},
    "no_gate_no_kl": {1: "parallel1", 2: "parallel1", 3: "parallel1"},
}


def summarize(reports: list[Mapping[str, Any]], oracle: float) -> dict[str, Any]:
    selected = [row["selected"] for row in reports]
    mean = sum(float(row["ValidationFinalMeanOverOz"]) for row in selected) / 3
    return {
        "ValidationFinalMeanOverOz_3seed": mean,
        "oracle_recovery_3seed": mean / oracle,
        "policy45_regret_mean_bytes_3seed": sum(float(row["policy45_regret_mean_bytes"]) for row in selected) / 3,
        "top1_accuracy_3seed": sum(float(row["top1_accuracy"]) for row in selected) / 3,
        "validation_ce_3seed": sum(float(row["validation_ce"]) for row in selected) / 3,
        "calibration_kl_final_to_nvp_3seed": sum(float(row["calibration_kl_final_to_nvp"]) for row in selected) / 3,
        "validation_kl_nvp_to_final_3seed": sum(float(row["validation_kl_nvp_to_final"]) for row in selected) / 3,
        "average_gate_alpha_3seed": sum(float(row["average_gate_alpha"]) for row in selected) / 3,
        "trainable_parameters": reports[0]["trainable_parameters"],
        "seed_results": reports,
    }


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", type=Path, required=True); parser.add_argument("--output-dir", type=Path, required=True); args = parser.parse_args()
    cfg = load_json(args.config)
    if not args.output_dir.is_dir() or (args.output_dir / "comparison_report.json").exists():
        raise FileExistsError("output root missing or aggregate report already exists")
    base = load_json(Path(cfg["base_training_config"]))
    references = frozen_references({**base, "frozen_reference_reports": cfg["frozen_reference_reports"]})
    oracle = load_json(Path(cfg["frozen_reference_reports"]["stage_b"]))["fixed_route_a_oracle"]
    variants: dict[str, Any] = {}
    for variant in VARIANTS:
        directory = args.output_dir / variant
        training_path, comparison_path = directory / "training_report.json", directory / "comparison_report.json"
        if training_path.exists() or comparison_path.exists(): raise FileExistsError(directory)
        reports = []
        for seed in (1, 2, 3):
            path = directory / "checkpoints" / f"seed{seed}" / ATTEMPTS[variant][seed] / "task_report.json"
            report = load_json(path)
            if report["step_execution"] != "COMPLETE" or report["seed"] != seed or report["variant"] != cfg["variants"][variant]:
                raise ValueError(f"invalid completed task: {path}")
            reports.append(report)
        summary = summarize(reports, oracle)
        result = {"step_execution": "COMPLETE", "training_only": True, "final_test_accessed": False, "ood_accessed": False, "runtime_accessed": False, "compiler_gym_initialized": False, "llvm_execution": False, "candidate_rollouts": 0, "objecttext_measurements": 0, "label_regeneration": False, "validation_cohort": {"N_total": 4488, "N_primary_valid": 4488, "N_failed_or_invalid": 0}, "variant": cfg["variants"][variant], "summary": summary, "frozen_references": references, "differences": {name: summary["ValidationFinalMeanOverOz_3seed"] - value for name, value in references.items()}}
        training_path.write_text(json.dumps({"step_execution": "COMPLETE", "variant": cfg["variants"][variant], "tasks": reports}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        comparison_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        variants[variant] = result
    root = {"step_execution": "COMPLETE", "training_only": True, "final_test_accessed": False, "ood_accessed": False, "runtime_accessed": False, "compiler_gym_initialized": False, "llvm_execution": False, "candidate_rollouts": 0, "objecttext_measurements": 0, "label_regeneration": False, "validation_cohort": {"N_total": 4488, "N_primary_valid": 4488, "N_failed_or_invalid": 0}, "variants": variants, "frozen_references": references}
    (args.output_dir / "comparison_report.json").write_text(json.dumps(root, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({name: row["summary"]["ValidationFinalMeanOverOz_3seed"] for name, row in variants.items()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
