#!/usr/bin/env python3
"""Export frozen Autophase inputs without executing LLVM optimization actions."""
from __future__ import annotations

import argparse
import gzip
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


AUTOPHASE_DIM = 56
EXPECTED_COUNTS = {"train": 28159, "validation": 4488, "final": 4679}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def target_population(path: Path, split: str) -> list[dict[str, str]]:
    rows = read_jsonl(path)
    population = [
        {"program_id": str(row["program_id"]), "dataset_name": str(row["dataset_id"]), "split": split}
        for row in rows
        if row.get("training_target_validity") == "valid_complete_K50"
    ]
    validate_population(population, split)
    return population


def final_population(shards: Path) -> list[dict[str, str]]:
    population: list[dict[str, str]] = []
    for path in sorted(shards.glob("*.json.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            summary = json.load(handle)["program_summary"]
        if summary.get("oracle_K50_validity") == "valid_complete_K50":
            population.append(
                {
                    "program_id": str(summary["program_id"]),
                    "dataset_name": str(summary["dataset_id"]),
                    "split": "final",
                }
            )
    validate_population(population, "final")
    return population


def validate_population(population: Iterable[Mapping[str, str]], split: str) -> None:
    rows = list(population)
    if len(rows) != EXPECTED_COUNTS[split]:
        raise ValueError(f"{split} complete-K50 count mismatch: {len(rows)}")
    program_ids = [row["program_id"] for row in rows]
    if len(program_ids) != len(set(program_ids)):
        raise ValueError(f"{split} population contains duplicate program IDs")
    if any(row["split"] != split or not row["dataset_name"] for row in rows):
        raise ValueError(f"{split} population metadata is invalid")


_ENV: Any = None


def init_feature_worker() -> None:
    global _ENV
    import compiler_gym

    _ENV = compiler_gym.make("llvm-v0", reward_space=None)


def feature_row(row: Mapping[str, str]) -> dict[str, Any]:
    """Reset one frozen program and read only its Autophase observation."""
    _ENV.reset(benchmark=row["program_id"])
    raw = np.asarray(_ENV.observation["Autophase"], dtype=np.float32).reshape(-1)
    if raw.size != AUTOPHASE_DIM or not np.isfinite(raw).all() or raw[51] <= 0:
        raise ValueError(f"invalid Autophase feature for {row['program_id']}")
    normalized = raw / raw[51]
    if not np.isfinite(normalized).all():
        raise ValueError(f"non-finite normalized Autophase feature for {row['program_id']}")
    return {
        "program_id": row["program_id"],
        "dataset_name": row["dataset_name"],
        "split": row["split"],
        "raw_autophase": raw.tolist(),
        "normalized_autophase": normalized.tolist(),
    }


def export_split(population: list[dict[str, str]], output: Path, workers: int) -> int:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    failures = 0
    with gzip.open(output, "wt", encoding="utf-8") as handle:
        with ProcessPoolExecutor(max_workers=workers, initializer=init_feature_worker) as pool:
            for row in pool.map(feature_row, population, chunksize=16):
                if len(row["raw_autophase"]) != AUTOPHASE_DIM or len(row["normalized_autophase"]) != AUTOPHASE_DIM:
                    raise ValueError("exported feature dimension mismatch")
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    return failures


def load_config(output_dir: Path, workers: int) -> dict[str, Any]:
    return {
        "experiment_name": "autophase_feature_cache_v6",
        "purpose": "offline frozen Autophase cache for MambaNVP inputs only",
        "source_populations": {
            "train": "outputs/rlcompopt_route_a_nvp_targets_v6/train_targets.jsonl.gz",
            "validation": "outputs/rlcompopt_route_a_nvp_targets_v6/validation_targets.jsonl.gz",
            "final": "outputs/route_a_final_objecttext_v6/shards/final",
        },
        "output_dir": str(output_dir),
        "expected_complete_k50_counts": EXPECTED_COUNTS,
        "feature": {"observation": "Autophase", "dimension": AUTOPHASE_DIM, "normalization": "raw_autophase / raw_autophase[51]"},
        "allowed_environment_operations": ["env.reset(program)", "env.observation['Autophase']"],
        "compiler_actions_executed": 0,
        "candidate_rollouts": 0,
        "objecttext_measurements": 0,
        "model_training": False,
        "workers": workers,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {args.output_dir}")
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[variable] = "1"

    train = target_population(Path("outputs/rlcompopt_route_a_nvp_targets_v6/train_targets.jsonl.gz"), "train")
    validation = target_population(Path("outputs/rlcompopt_route_a_nvp_targets_v6/validation_targets.jsonl.gz"), "validation")
    final = final_population(Path("outputs/route_a_final_objecttext_v6/shards/final"))
    output = args.output_dir
    output.mkdir(parents=True)
    config = load_config(output, args.workers)
    (output / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    invalid_extraction_count = {
        "train": export_split(train, output / "train_autophase.jsonl.gz", args.workers),
        "validation": export_split(validation, output / "validation_autophase.jsonl.gz", args.workers),
        "final": export_split(final, output / "final_autophase.jsonl.gz", args.workers),
    }
    report = {
        "step_execution": "COMPLETE",
        "source_population_artifact_paths": config["source_populations"],
        "exported_counts": {"train": len(train), "validation": len(validation), "final": len(final)},
        "feature_dimension": AUTOPHASE_DIM,
        "normalization_formula": config["feature"]["normalization"],
        "invalid_extraction_count": invalid_extraction_count,
        "compiler_actions_executed": 0,
        "candidate_rollouts": 0,
        "ObjectText_measurements": 0,
        "model_training_executed": False,
    }
    (output / "experiment_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
