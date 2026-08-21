#!/usr/bin/env python3
"""Evaluate frozen Cross-Candidate MambaNVP checkpoints on final/OOD artifacts only."""
from __future__ import annotations

import argparse
import collections
import gzip
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

if __package__:
    from scripts.evaluate_mamba_nvp_final_objecttext import (
        K,
        aggregate,
        load_final_features,
        policy45,
        read_final_artifacts,
        regret_summary,
    )
    from scripts.train_controlled_nvp_stage_a import load_candidates
    from scripts.train_cross_candidate_mambanvp import CrossCandidateMambaNVP
    from scripts.train_mamba_nvp_objecttext import load_frozen_nvp, load_json
else:
    from evaluate_mamba_nvp_final_objecttext import K, aggregate, load_final_features, policy45, read_final_artifacts, regret_summary
    from train_controlled_nvp_stage_a import load_candidates
    from train_cross_candidate_mambanvp import CrossCandidateMambaNVP
    from train_mamba_nvp_objecttext import load_frozen_nvp, load_json


METHOD = "CrossCandidateMambaNVP"


def validate_config(cfg: Mapping[str, Any]) -> None:
    if cfg["final_seed_set"] != [1, 2, 3]:
        raise ValueError("final seed set must be exactly [1, 2, 3]")
    if cfg["final_population"] != {"total": 4683, "complete_k50": 4679, "invalid": 4}:
        raise ValueError("frozen final population mismatch")
    if cfg["candidate_representation"] != {"K": 50, "padded_length": 20, "pad_token_id": 124}:
        raise ValueError("frozen candidate representation mismatch")
    if cfg["inference"] != {
        "sampling": False,
        "ranking": "descending logits; candidate ID ascending tie break",
        "scored_pass_budget": 45,
    }:
        raise ValueError("frozen policy45 inference mismatch")
    if cfg["comparison_family"] != ["NVP", "Mamba", "MambaNVP", METHOD]:
        raise ValueError("final comparison family mismatch")
    if cfg["checkpoint_selection"]["selected_steps"] != {"1": 3400, "2": 600, "3": 1200}:
        raise ValueError("validation-selected Cross-Candidate checkpoints mismatch")


def read_results(path: Path, method: str, seed: int) -> dict[str, dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    if len(rows) != 4683 or len({row["program_id"] for row in rows}) != 4683:
        raise ValueError(f"frozen result population mismatch: {path}")
    if any(row.get("model") != method or row.get("seed") != seed for row in rows):
        raise ValueError(f"frozen result identity mismatch: {path}")
    return {row["program_id"]: row for row in rows}


def selected_steps_from_report(path: Path) -> dict[int, int]:
    report = load_json(path)
    entries = report["cross_candidate_mambanvp"]["seed_results"]
    selected = {int(entry["seed"]): int(entry["selected"]["step"]) for entry in entries}
    if selected != {1: 3400, 2: 600, 3: 1200}:
        raise ValueError("frozen Cross-Candidate validation selections mismatch")
    return selected


def load_cross_candidate(
    seed: int,
    cfg: Mapping[str, Any],
    training_cfg: Mapping[str, Any],
    controlled: Mapping[str, Any],
    selected_steps: Mapping[int, int],
    device: torch.device,
) -> CrossCandidateMambaNVP:
    path = Path(cfg["cross_candidate_checkpoint_root"]) / f"seed{seed}" / "model.pt"
    payload = torch.load(path, map_location="cpu")
    expected_fusion = training_cfg["fusion"]
    if (
        payload.get("stage") != "Route-A Cross-Candidate MambaNVP v1"
        or payload.get("architecture") != METHOD
        or payload.get("seed") != seed
        or payload.get("step") != selected_steps[seed]
        or payload.get("nvp_frozen") is not True
        or payload.get("fusion") != expected_fusion
    ):
        raise ValueError(f"not the validation-selected frozen Cross-Candidate checkpoint: {path}")
    tokens, lengths = load_candidates(
        Path(controlled["candidate_representation"]["candidate_sequences"]), pad_token_id=124, padded_length=20
    )
    model_cfg = {
        **controlled["candidate_representation"],
        **controlled["models"]["Mamba"],
        "candidate_interaction": training_cfg["architecture"]["candidate_interaction"],
    }
    nvp = load_frozen_nvp(Path(training_cfg["nvp_checkpoint_root"]) / f"seed{seed}" / "model.pt", seed)
    model = CrossCandidateMambaNVP(nvp, model_cfg, tokens, lengths)
    model.load_state_dict(payload["state_dict"], strict=True)
    if any(parameter.requires_grad for parameter in model.nvp.parameters()) or model.nvp.training:
        raise RuntimeError("frozen NVP invariant failed")
    return model.to(device).eval()


def infer_cross_candidate(
    model: CrossCandidateMambaNVP,
    seed: int,
    programs: Sequence[str],
    matrix: Mapping[str, Sequence[Mapping[str, Any]]],
    summaries: Mapping[str, Mapping[str, Any]],
    features: Mapping[str, Sequence[float]],
    device: torch.device,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    eligible = [
        program
        for program in programs
        if program in matrix and summaries[program]["ratio_metric_validity"] == "valid_for_ObjectText_ratio_metric"
    ]
    if len(eligible) != 4679:
        raise ValueError("frozen final eligible population mismatch")
    logits: dict[str, list[float]] = {}
    with torch.no_grad():
        for start in range(0, len(eligible), 128):
            current = eligible[start : start + 128]
            values = model(
                torch.tensor([features[program] for program in current], dtype=torch.float32, device=device)
            ).cpu().tolist()
            logits.update(zip(current, values))
    rows: dict[str, dict[str, Any]] = {}
    failures: collections.Counter[str] = collections.Counter()
    for program in programs:
        summary = summaries[program]
        row: dict[str, Any] = {
            "program_id": program,
            "dataset_id": summary["dataset_id"],
            "model": METHOD,
            "seed": seed,
            "valid": False,
        }
        if summary["oracle_K50_validity"] != "valid_complete_K50":
            row["failure_reason"] = "incomplete_K50"
            failures["incomplete_K50"] += 1
        elif summary["ratio_metric_validity"] != "valid_for_ObjectText_ratio_metric":
            row["failure_reason"] = "invalid_ratio_denominator"
            failures["invalid_ratio_denominator"] += 1
        else:
            policy = policy45(logits[program], matrix[program])
            oracle = min(record["best_object_text_size_bytes"] for record in matrix[program])
            oz = int(summary["oz_object_text_size_bytes"])
            row.update(
                {
                    "valid": True,
                    "policy45_object_text_size_bytes": policy,
                    "oracle_object_text_size_bytes": oracle,
                    "oz_object_text_size_bytes": oz,
                    "mean_over_oz": (oz - policy) / oz,
                    "policy45_regret_bytes": policy - oracle,
                }
            )
        rows[program] = row
    return rows, {"model": METHOD, "seed": seed, "failure_count_by_reason": dict(failures), "N_total": len(rows), "N_primary_valid": len(eligible)}


def invalid_summary(methods: Sequence[str], result_maps: Mapping[tuple[str, int], Mapping[str, Mapping[str, Any]]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for method in methods:
        per_seed: dict[str, Any] = {}
        for seed in (1, 2, 3):
            rows = result_maps[(method, seed)].values()
            failures = collections.Counter(row.get("failure_reason", "unknown") for row in rows if not row["valid"])
            per_seed[str(seed)] = {"N_total": 4683, "N_primary_valid": 4683 - sum(failures.values()), "N_invalid": sum(failures.values()), "failure_count_by_reason": dict(failures)}
        summary[method] = per_seed
    return summary


def write_rows(path: Path, methods: Sequence[str], programs: Sequence[str], result_maps: Mapping[tuple[str, int], Mapping[str, Mapping[str, Any]]]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for method in methods:
            for seed in (1, 2, 3):
                for program in programs:
                    handle.write(json.dumps(result_maps[(method, seed)][program], separators=(",", ":")) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {args.output_dir}")
    cfg = load_json(args.config)
    validate_config(cfg)
    training_cfg = load_json(Path(cfg["cross_candidate_training_config"]))
    controlled = load_json(Path(cfg["controlled_config"]))
    selected_steps = selected_steps_from_report(Path(cfg["checkpoint_selection"]["source"]))
    programs, matrix, summaries = read_final_artifacts(Path(cfg["final_label_shards"]))
    eligible = [program for program in programs if program in matrix and summaries[program]["ratio_metric_validity"] == "valid_for_ObjectText_ratio_metric"]
    features = load_final_features(Path(cfg["final_feature_cache"]), eligible)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result_maps: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    for method, root in cfg["existing_model_results"].items():
        for seed in cfg["final_seed_set"]:
            result_maps[(method, seed)] = read_results(Path(root) / f"seed{seed}.jsonl.gz", method, seed)
    cross_reports = []
    for seed in cfg["final_seed_set"]:
        model = load_cross_candidate(seed, cfg, training_cfg, controlled, selected_steps, device)
        rows, seed_report = infer_cross_candidate(model, seed, programs, matrix, summaries, features, device)
        result_maps[(METHOD, seed)] = rows
        cross_reports.append(seed_report)
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "config.json").write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_rows(args.output_dir / "per_program_results.jsonl.gz", cfg["comparison_family"], programs, result_maps)
    combined = aggregate(cfg["comparison_family"], programs, summaries, result_maps)
    oracle = load_json(Path(cfg["existing_final_report"]))["offline_k50_oracle"]
    recovery = {method: combined["dataset_macro"][method]["three_seed_mean"] / oracle["dataset_macro"] for method in cfg["comparison_family"]}
    report = {
        "step_execution": "COMPLETE",
        "offline_only": True,
        "compiler_gym_initialized": False,
        "llvm_execution": False,
        "candidate_rollouts": 0,
        "objecttext_measurements": 0,
        "label_regeneration": False,
        "checkpoint_reselection": False,
        "model_training": False,
        "final_population": {"N_total": 4683, "N_complete_K50_valid": 4679, "N_invalid": 4},
        "device": str(device),
        "cross_candidate_seed_results": cross_reports,
        "common_cohort_invalid_statistics": invalid_summary(cfg["comparison_family"], result_maps),
        "combined_comparison": combined,
        "frozen_offline_k50_oracle": oracle,
        "oracle_recovery": recovery,
        "policy45_regret": {method: regret_summary(method, result_maps) for method in cfg["comparison_family"]},
        "cross_candidate_minus": {
            method: combined["dataset_macro"][METHOD]["three_seed_mean"] - combined["dataset_macro"][method]["three_seed_mean"]
            for method in ("NVP", "Mamba", "MambaNVP")
        },
    }
    (args.output_dir / "comparison_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
