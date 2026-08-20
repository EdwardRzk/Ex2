#!/usr/bin/env python3
"""Evaluate frozen MambaNVP checkpoints on existing final/OOD artifacts only."""
from __future__ import annotations

import argparse
import collections
import gzip
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

if __package__:
    from scripts.train_controlled_nvp_stage_a import load_candidates
    from scripts.train_mamba_nvp_objecttext import MambaNVP, load_frozen_nvp, load_json
else:
    from train_controlled_nvp_stage_a import load_candidates
    from train_mamba_nvp_objecttext import MambaNVP, load_frozen_nvp, load_json


K = 50


def validate_config(cfg: Mapping[str, Any]) -> None:
    if cfg["final_seed_set"] != [1, 2, 3]:
        raise ValueError("final seed set must be exactly [1, 2, 3]")
    if cfg["final_population"] != {"total": 4683, "complete_k50": 4679, "invalid": 4}:
        raise ValueError("frozen final population mismatch")
    if cfg["candidate_representation"] != {"K": 50, "padded_length": 20, "pad_token_id": 124}:
        raise ValueError("frozen candidate representation mismatch")
    if cfg["inference"]["sampling"] is not False or cfg["inference"]["scored_pass_budget"] != 45:
        raise ValueError("frozen policy45 inference mismatch")
    if cfg["comparison_family"] != ["NVP", "Mamba", "MambaNVP"]:
        raise ValueError("final comparison family must remain frozen")


def read_final_artifacts(shards: Path) -> tuple[list[str], dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    programs, matrix, summaries = [], {}, {}
    for path in sorted(shards.glob("*.json.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        summary, records = payload["program_summary"], payload["records"]
        program = str(summary["program_id"])
        programs.append(program)
        summaries[program] = summary
        if summary.get("oracle_K50_validity") == "valid_complete_K50":
            ordered = sorted(records, key=lambda row: row["candidate_id"])
            if len(ordered) != K or [row["candidate_id"] for row in ordered] != list(range(K)):
                raise ValueError(f"invalid frozen K50 records: {path}")
            matrix[program] = ordered
    if len(programs) != 4683 or len(set(programs)) != 4683 or len(matrix) != 4679:
        raise ValueError("frozen final shard population mismatch")
    return programs, matrix, summaries


def load_final_features(path: Path, eligible: Sequence[str]) -> dict[str, list[float]]:
    rows: dict[str, list[float]] = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["split"] != "final" or len(row["raw_autophase"]) != 56 or len(row["normalized_autophase"]) != 56:
                raise ValueError("invalid final Autophase cache row")
            program = str(row["program_id"])
            if program in rows:
                raise ValueError(f"duplicate final cached feature: {program}")
            rows[program] = list(row["normalized_autophase"])
    if set(rows) != set(eligible):
        raise ValueError("final feature cache population mismatch")
    return rows


def load_mamba_nvp(seed: int, cfg: Mapping[str, Any], training_cfg: Mapping[str, Any], controlled: Mapping[str, Any], device: torch.device) -> MambaNVP:
    path = Path(cfg["mamba_nvp_checkpoint_root"]) / f"seed{seed}" / "model.pt"
    payload = torch.load(path, map_location="cpu")
    if payload.get("stage") != "Route-A MambaNVP v6" or payload.get("architecture") != "MambaNVP" or payload.get("seed") != seed or payload.get("nvp_frozen") is not True or payload.get("fusion") != training_cfg["fusion"]:
        raise ValueError(f"not frozen MambaNVP seed {seed}: {path}")
    tokens, lengths = load_candidates(Path(controlled["candidate_representation"]["candidate_sequences"]), pad_token_id=124, padded_length=20)
    nvp = load_frozen_nvp(Path(payload["nvp_checkpoint"]), seed)
    residual_cfg = {**controlled["candidate_representation"], **controlled["models"]["Mamba"]}
    model = MambaNVP(nvp, residual_cfg, tokens, lengths)
    model.residual.load_state_dict(payload["residual_state_dict"], strict=True)
    if any(parameter.requires_grad for parameter in model.nvp.parameters()) or model.nvp.training:
        raise RuntimeError("frozen NVP invariant failed")
    return model.to(device).eval()


def policy45(score: Sequence[float], records: Sequence[Mapping[str, Any]]) -> int:
    budget, observed = 45, []
    for candidate_id in sorted(range(K), key=lambda index: (-score[index], index)):
        prefix = records[candidate_id]["prefix_object_text_size_bytes"]
        take = min(budget, len(prefix))
        observed.extend(prefix[:take])
        budget -= take
        if budget == 0:
            break
    if budget != 0 or not observed:
        raise ValueError("policy45 did not consume exactly 45 frozen prefix measurements")
    return min(observed)


def evaluate_seed(model: MambaNVP, seed: int, programs: Sequence[str], matrix: Mapping[str, Sequence[Mapping[str, Any]]], summaries: Mapping[str, Mapping[str, Any]], features: Mapping[str, Sequence[float]], output: Path, device: torch.device) -> dict[str, Any]:
    eligible = [program for program in programs if program in matrix and summaries[program]["ratio_metric_validity"] == "valid_for_ObjectText_ratio_metric"]
    logits: dict[str, list[float]] = {}
    with torch.no_grad():
        for start in range(0, len(eligible), 128):
            current = eligible[start : start + 128]
            values = model(torch.tensor([features[program] for program in current], dtype=torch.float32, device=device)).cpu().tolist()
            logits.update(zip(current, values))
    output.parent.mkdir(parents=True, exist_ok=True)
    failures: collections.Counter[str] = collections.Counter()
    per_dataset: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    with gzip.open(output, "wt", encoding="utf-8") as handle:
        for program in programs:
            summary = summaries[program]
            dataset = summary["dataset_id"]
            row: dict[str, Any] = {"program_id": program, "dataset_id": dataset, "model": "MambaNVP", "seed": seed, "valid": False}
            if summary["oracle_K50_validity"] != "valid_complete_K50":
                reason = "incomplete_K50"
            elif summary["ratio_metric_validity"] != "valid_for_ObjectText_ratio_metric":
                reason = "invalid_ratio_denominator"
            else:
                policy = policy45(logits[program], matrix[program])
                oracle = min(record["best_object_text_size_bytes"] for record in matrix[program])
                oz = int(summary["oz_object_text_size_bytes"])
                row.update({"valid": True, "policy45_object_text_size_bytes": policy, "oracle_object_text_size_bytes": oracle, "oz_object_text_size_bytes": oz, "mean_over_oz": (oz - policy) / oz, "policy45_regret_bytes": policy - oracle})
                per_dataset[dataset]["N_primary_valid"] += 1
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")
                continue
            row["failure_reason"] = reason
            failures[reason] += 1
            per_dataset[dataset]["N_failed_or_invalid"] += 1
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    return {"model": "MambaNVP", "seed": seed, "result_file": str(output), "failure_count_by_reason": dict(failures), "per_dataset_method_validity": {dataset: dict(counts) for dataset, counts in sorted(per_dataset.items())}}


def read_results(path: Path) -> dict[str, dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return {row["program_id"]: row for row in (json.loads(line) for line in handle)}


def aggregate(methods: Sequence[str], programs: Sequence[str], summaries: Mapping[str, Mapping[str, Any]], result_maps: Mapping[tuple[str, int], Mapping[str, Mapping[str, Any]]]) -> dict[str, Any]:
    datasets = sorted({summary["dataset_id"] for summary in summaries.values()})
    seeds = [1, 2, 3]
    per_dataset: dict[str, Any] = {}
    for dataset in datasets:
        members = [program for program in programs if summaries[program]["dataset_id"] == dataset]
        common = [program for program in members if summaries[program]["ratio_metric_validity"] == "valid_for_ObjectText_ratio_metric" and all(result_maps[(method, seed)][program]["valid"] for method in methods for seed in seeds)]
        values: dict[str, Any] = {"N_total": len(members), "N_primary_valid": len(common), "N_failed_or_invalid": len(members) - len(common)}
        for method in methods:
            seed_values = {str(seed): (sum(result_maps[(method, seed)][program]["mean_over_oz"] for program in common) / len(common) if common else None) for seed in seeds}
            values[method] = {"per_seed": seed_values, "three_seed_mean": sum(seed_values.values()) / 3 if common else None}
        per_dataset[dataset] = values
    macro: dict[str, Any] = {}
    for method in methods:
        if any(per_dataset[dataset][method]["three_seed_mean"] is None for dataset in datasets):
            macro[method] = {"per_seed": {str(seed): None for seed in seeds}, "three_seed_mean": None}
        else:
            seed_values = {str(seed): sum(per_dataset[dataset][method]["per_seed"][str(seed)] for dataset in datasets) / len(datasets) for seed in seeds}
            macro[method] = {"per_seed": seed_values, "three_seed_mean": sum(seed_values.values()) / 3}
    return {"methods": list(methods), "per_dataset": per_dataset, "dataset_macro": macro}


def regret_summary(method: str, result_maps: Mapping[tuple[str, int], Mapping[str, Mapping[str, Any]]]) -> dict[str, Any]:
    values = {}
    for seed in (1, 2, 3):
        regrets = sorted(row["policy45_regret_bytes"] for row in result_maps[(method, seed)].values() if row["valid"])
        values[str(seed)] = {"mean": sum(regrets) / len(regrets), "median": regrets[len(regrets) // 2]}
    return {"per_seed": values, "mean_3seed": sum(row["mean"] for row in values.values()) / 3, "median_per_seed": [row["median"] for row in values.values()]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {args.output_dir}")
    cfg = load_json(args.config)
    validate_config(cfg)
    training_cfg, controlled = load_json(Path(cfg["mamba_nvp_training_config"])), load_json(Path(cfg["controlled_config"]))
    programs, matrix, summaries = read_final_artifacts(Path(cfg["final_label_shards"]))
    eligible = [program for program in programs if program in matrix and summaries[program]["ratio_metric_validity"] == "valid_for_ObjectText_ratio_metric"]
    if len(eligible) != 4679:
        raise ValueError("frozen final eligible population mismatch")
    features = load_final_features(Path(cfg["final_feature_cache"]), eligible)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "config.json").write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result_maps: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    mamba_nvp_reports = []
    for seed in cfg["final_seed_set"]:
        model = load_mamba_nvp(seed, cfg, training_cfg, controlled, device)
        result_path = args.output_dir / "model_results" / "mambanvp" / f"seed{seed}.jsonl.gz"
        mamba_nvp_reports.append(evaluate_seed(model, seed, programs, matrix, summaries, features, result_path, device))
        result_maps[("MambaNVP", seed)] = read_results(result_path)
    for method in ("NVP", "Mamba"):
        for seed in cfg["final_seed_set"]:
            result_maps[(method, seed)] = read_results(Path(cfg["existing_model_results"]) / method.lower() / f"seed{seed}.jsonl.gz")
    combined = aggregate(cfg["comparison_family"], programs, summaries, result_maps)
    frozen_oracle = load_json(Path(cfg["existing_final_report"]))["offline_k50_oracle"]
    mamba_nvp_mean = combined["dataset_macro"]["MambaNVP"]["three_seed_mean"]
    report = {
        "step_execution": "COMPLETE",
        "offline_only": True,
        "compiler_gym_initialized": False,
        "llvm_execution": False,
        "candidate_rollouts": 0,
        "objecttext_measurements": 0,
        "label_regeneration": False,
        "invalid_program_retry": False,
        "final_population": {"N_total": 4683, "N_complete_K50_valid": 4679, "N_invalid": 4},
        "mamba_nvp_seed_results": mamba_nvp_reports,
        "combined_comparison": combined,
        "frozen_offline_k50_oracle": frozen_oracle,
        "mamba_nvp_oracle_recovery": mamba_nvp_mean / frozen_oracle["dataset_macro"],
        "policy45_regret": {method: regret_summary(method, result_maps) for method in cfg["comparison_family"]},
        "mamba_nvp_minus_nvp": mamba_nvp_mean - combined["dataset_macro"]["NVP"]["three_seed_mean"],
    }
    (args.output_dir / "comparison_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
