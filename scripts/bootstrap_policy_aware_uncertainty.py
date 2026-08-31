#!/usr/bin/env python3
"""Paired dataset-stratified uncertainty for frozen PA-MambaNVP final results."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SEEDS = (1, 2, 3)
REPLICATES = 10_000
RNG_SEED = 20260831


def read_seed_rows(paths: Mapping[int, Path]) -> dict[int, dict[str, dict[str, Any]]]:
    result: dict[int, dict[str, dict[str, Any]]] = {}
    for seed, path in paths.items():
        rows: dict[str, dict[str, Any]] = {}
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row.get("valid") is not True:
                    continue
                program = str(row["program_id"])
                if program in rows:
                    raise ValueError(f"duplicate program record: seed={seed}, program={program}")
                if int(row["seed"]) != seed:
                    raise ValueError(f"seed mismatch in {path}: {program}")
                rows[program] = row
        result[seed] = rows
    return result


def macro(values: Mapping[str, np.ndarray]) -> float:
    return float(np.mean([float(np.mean(value)) for _, value in sorted(values.items())]))


def stratified_bootstrap(per_dataset: Mapping[str, Mapping[str, np.ndarray]], replicates: int, rng_seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(rng_seed)
    pa = np.zeros(replicates, dtype=np.float64)
    nvp = np.zeros(replicates, dtype=np.float64)
    datasets = sorted(per_dataset)
    for dataset in datasets:
        values = per_dataset[dataset]
        count = len(values["delta"])
        if count == 0:
            raise ValueError(f"empty paired cohort: {dataset}")
        for start in range(0, replicates, 1000):
            stop = min(start + 1000, replicates)
            index = rng.integers(0, count, size=(stop - start, count))
            pa[start:stop] += values["pa"][index].mean(axis=1) / len(datasets)
            nvp[start:stop] += values["nvp"][index].mean(axis=1) / len(datasets)
    return pa, nvp, pa - nvp


def seed_macro(rows: Mapping[int, Mapping[str, Mapping[str, Any]]], datasets: Sequence[str]) -> dict[str, float]:
    result: dict[str, float] = {}
    for seed in SEEDS:
        grouped: dict[str, list[float]] = {dataset: [] for dataset in datasets}
        for row in rows[seed].values():
            grouped[str(row["dataset_id"])].append(float(row["mean_over_oz"]))
        result[str(seed)] = macro({dataset: np.asarray(values, dtype=np.float64) for dataset, values in grouped.items()})
    return result


def analyze(root: Path, output: Path, replicates: int = REPLICATES, rng_seed: int = RNG_SEED) -> None:
    if output.exists():
        raise FileExistsError(output)
    pa_root = root / "outputs/policy_aware_mambanvp_v1"
    nvp_root = root / "outputs/route_a_final_objecttext_v6/model_results/nvp"
    pa_paths = {seed: pa_root / "final_results" / f"seed{seed}.jsonl.gz" for seed in SEEDS}
    nvp_paths = {seed: nvp_root / f"seed{seed}.jsonl.gz" for seed in SEEDS}
    if not all(path.is_file() for path in (*pa_paths.values(), *nvp_paths.values())):
        raise FileNotFoundError("missing frozen PA or NVP final records")
    pa_rows, nvp_rows = read_seed_rows(pa_paths), read_seed_rows(nvp_paths)
    pa_sets = [set(pa_rows[seed]) for seed in SEEDS]
    nvp_sets = [set(nvp_rows[seed]) for seed in SEEDS]
    paired = set.intersection(*pa_sets, *nvp_sets)
    if any(ids != paired for ids in (*pa_sets, *nvp_sets)):
        raise ValueError("final paired cohort differs across method/seed records")
    per_dataset_lists: dict[str, dict[str, list[float]]] = {}
    for program in sorted(paired):
        pa_seed = [pa_rows[seed][program] for seed in SEEDS]
        nvp_seed = [nvp_rows[seed][program] for seed in SEEDS]
        dataset = str(pa_seed[0]["dataset_id"])
        if any(str(row["dataset_id"]) != dataset for row in (*pa_seed, *nvp_seed)):
            raise ValueError(f"dataset pairing mismatch: {program}")
        target = per_dataset_lists.setdefault(dataset, {"pa": [], "nvp": [], "delta": []})
        pa_value, nvp_value = float(np.mean([float(row["mean_over_oz"]) for row in pa_seed])), float(np.mean([float(row["mean_over_oz"]) for row in nvp_seed]))
        target["pa"].append(pa_value); target["nvp"].append(nvp_value); target["delta"].append(pa_value - nvp_value)
    per_dataset = {dataset: {key: np.asarray(values, dtype=np.float64) for key, values in value.items()} for dataset, value in per_dataset_lists.items()}
    if len(per_dataset) != 14 or len(paired) != 4679:
        raise ValueError(f"frozen final cohort mismatch: datasets={len(per_dataset)}, paired={len(paired)}")
    observed_pa, observed_nvp = macro({dataset: values["pa"] for dataset, values in per_dataset.items()}), macro({dataset: values["nvp"] for dataset, values in per_dataset.items()})
    observed_delta = observed_pa - observed_nvp
    pa_report = json.loads((pa_root / "comparison_report.json").read_text(encoding="utf-8"))
    formal_delta = float(pa_report["final"]["delta_vs_nvp"])
    if not np.isclose(observed_delta, formal_delta, rtol=0.0, atol=1e-12):
        raise ValueError(f"formal PA-vs-NVP delta mismatch: {observed_delta} != {formal_delta}")
    bootstrap_pa, bootstrap_nvp, bootstrap_delta = stratified_bootstrap(per_dataset, replicates, rng_seed)
    datasets = sorted(per_dataset)
    dataset_deltas = {dataset: float(np.mean(per_dataset[dataset]["delta"])) for dataset in datasets}
    pa_seed_macro, nvp_seed_macro = seed_macro(pa_rows, datasets), seed_macro(nvp_rows, datasets)
    seed_deltas = {seed: pa_seed_macro[seed] - nvp_seed_macro[seed] for seed in pa_seed_macro}
    output.mkdir(parents=True)
    with (output / "bootstrap_replicates.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["replicate", "pa_dataset_macro_mean_over_oz", "nvp_dataset_macro_mean_over_oz", "paired_dataset_macro_delta"])
        writer.writeheader()
        for index, (pa_value, nvp_value, delta) in enumerate(zip(bootstrap_pa, bootstrap_nvp, bootstrap_delta)):
            writer.writerow({"replicate": index, "pa_dataset_macro_mean_over_oz": float(pa_value), "nvp_dataset_macro_mean_over_oz": float(nvp_value), "paired_dataset_macro_delta": float(delta)})
    with (output / "per_dataset_paired_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset_id", "N_paired_valid", "PA_MeanOverOz", "NVP_MeanOverOz", "paired_delta"])
        writer.writeheader()
        for dataset in datasets:
            values = per_dataset[dataset]
            writer.writerow({"dataset_id": dataset, "N_paired_valid": len(values["delta"]), "PA_MeanOverOz": float(np.mean(values["pa"])), "NVP_MeanOverOz": float(np.mean(values["nvp"])), "paired_delta": float(np.mean(values["delta"]))})
    summary = {
        "analysis": "frozen offline paired program-level bootstrap; no model inference or compiler execution",
        "comparison": "PA-MambaNVP minus NVP",
        "primary_estimand": "equal-weight 14-dataset macro MeanOverOz delta",
        "resampling": {"scheme": "within-dataset paired program resampling with replacement after three-seed program-level arithmetic averaging", "replicates": replicates, "rng_seed": rng_seed, "dataset_weights": "equal", "hierarchical_bootstrap": "not computed"},
        "cohort": {"N_total": int(pa_report["final_population"]["N_total"]), "N_paired_valid": len(paired), "N_invalid": int(pa_report["final_population"]["N_invalid"]), "dataset_count": len(datasets), "all_method_seed_program_sets_identical": True},
        "observed": {"PA_dataset_macro_mean_over_oz": observed_pa, "NVP_dataset_macro_mean_over_oz": observed_nvp, "PA_minus_NVP_dataset_macro_delta": observed_delta, "formal_report_delta_reproduced": True, "formal_report_delta": formal_delta, "median_dataset_delta": float(np.median(list(dataset_deltas.values()))), "positive_dataset_count": sum(value > 0.0 for value in dataset_deltas.values()), "negative_dataset_count": sum(value < 0.0 for value in dataset_deltas.values()), "leave_LLVM_Stress_out_13dataset_delta": float(np.mean([value for dataset, value in dataset_deltas.items() if dataset != "llvm-stress-v0"]))},
        "primary_bootstrap": {"mean": float(np.mean(bootstrap_delta)), "standard_error": float(np.std(bootstrap_delta, ddof=1)), "percentile_95_ci": [float(np.quantile(bootstrap_delta, 0.025)), float(np.quantile(bootstrap_delta, 0.975))], "probability_delta_gt_zero": float(np.mean(bootstrap_delta > 0.0))},
        "seed_variation_descriptive_only": {"PA_seed_dataset_macro": pa_seed_macro, "NVP_seed_dataset_macro": nvp_seed_macro, "paired_PA_minus_NVP_delta": seed_deltas, "note": "Three matched seeds are descriptive; no Gaussian seed confidence interval is claimed."},
        "source_artifacts": {"pa_final_records": [str(path.relative_to(root)) for path in pa_paths.values()], "nvp_final_records": [str(path.relative_to(root)) for path in nvp_paths.values()], "pa_formal_report": str((pa_root / "comparison_report.json").relative_to(root))},
        "forbidden_operations": {"model_training": False, "model_inference": False, "compiler_gym_initialized": False, "llvm_execution": False, "label_generation": False, "objecttext_measurement": False},
    }
    (output / "bootstrap_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=REPLICATES)
    parser.add_argument("--rng-seed", type=int, default=RNG_SEED)
    args = parser.parse_args()
    if args.replicates < 10_000:
        raise ValueError("primary analysis requires at least 10,000 bootstrap replicates")
    analyze(Path.cwd(), args.output_dir, args.replicates, args.rng_seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
