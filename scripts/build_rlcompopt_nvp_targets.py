#!/usr/bin/env python3
"""Build frozen ObjectText Route-A K=50 NVP soft targets from Step-3 shards."""

from __future__ import annotations

import argparse
import gzip
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Mapping


K = 50
AUTOPHASE_NVP_TEMPERATURE = 0.05


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def softmax(values: list[float], temperature: float) -> list[float]:
    if temperature <= 0:
        raise ValueError("NVP target temperature must be positive")
    scaled = [value / temperature for value in values]
    maximum = max(scaled)
    exps = [math.exp(value - maximum) for value in scaled]
    total = sum(exps)
    return [value / total for value in exps]


def load_shard(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload["records"], payload["program_summary"]


def target_record(
    records: list[dict[str, Any]], summary: Mapping[str, Any], split: str
) -> dict[str, Any] | None:
    if summary["program_training_target_validity"] != "valid_complete_K50":
        return None
    if len(records) != K or sorted(record["candidate_id"] for record in records) != list(range(K)):
        raise ValueError(f"Invalid complete-K50 record structure: {summary['program_id']}")
    if any(record["training_target_validity"] != "valid_completed_candidate_rollout" for record in records):
        raise ValueError(f"Invalid candidate in complete K=50 record: {summary['program_id']}")
    oz = summary["oz_object_text_size_bytes"]
    if oz is None or oz <= 0:
        raise ValueError(f"Complete K=50 record has invalid Oz denominator: {summary['program_id']}")
    records = sorted(records, key=lambda record: record["candidate_id"])
    sizes = [record["best_object_text_size_bytes"] for record in records]
    if any(size is None for size in sizes):
        raise ValueError(f"Complete candidate lacks best ObjectText size: {summary['program_id']}")
    values = [(oz - size) / oz for size in sizes]
    targets = softmax(values, AUTOPHASE_NVP_TEMPERATURE)
    if not all(math.isfinite(value) and value >= 0 for value in targets):
        raise ValueError(f"Invalid soft target: {summary['program_id']}")
    if not math.isclose(sum(targets), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"Unnormalized soft target: {summary['program_id']}")
    return {
        "program_id": summary["program_id"],
        "dataset_id": summary["dataset_id"],
        "split": split,
        "S_O0": summary["initial_object_text_size_bytes"],
        "S_Oz": oz,
        "candidate_ids": [record["candidate_id"] for record in records],
        "candidate_sequences": [record["ordered_pass_sequence"] for record in records],
        "best_object_text_size": sizes,
        "raw_candidate_value": values,
        "normalized_target": targets,
        "target_temperature": AUTOPHASE_NVP_TEMPERATURE,
        "measurement_validity": [record["measurement_validity"] for record in records],
        "ratio_metric_validity": summary["ratio_metric_validity"],
        "training_target_validity": summary["program_training_target_validity"],
    }


def build_split(shards: Path, split: str) -> tuple[list[dict[str, Any]], dict[str, int]]:
    targets: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for path in sorted(shards.glob("*.json.gz")):
        records, summary = load_shard(path)
        counts["programs_total"] += 1
        target = target_record(records, summary, split)
        if target is None:
            counts["excluded_incomplete_K50"] += 1
        else:
            targets.append(target)
            counts["included_complete_K50"] += 1
    return targets, dict(counts)


def write_jsonl_gzip(path: Path, records: list[Mapping[str, Any]]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")


def validate_targets(records: list[Mapping[str, Any]], expected_count: int, split: str) -> None:
    if len(records) != expected_count:
        raise ValueError(f"{split} target count {len(records)} != {expected_count}")
    for record in records:
        for key in ("candidate_ids", "candidate_sequences", "best_object_text_size", "raw_candidate_value", "normalized_target"):
            if len(record[key]) != K:
                raise ValueError(f"{split} {record['program_id']} has non-K=50 {key}")
        if not all(math.isfinite(value) and value >= 0 for value in record["normalized_target"]):
            raise ValueError(f"{split} {record['program_id']} has invalid soft target")
        if not math.isclose(sum(record["normalized_target"]), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"{split} {record['program_id']} has unnormalized soft target")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-shards", type=Path, required=True)
    parser.add_argument("--validation-shards", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output_dir}")
    train, train_counts = build_split(args.train_shards, "train")
    validation, validation_counts = build_split(args.validation_shards, "validation")
    validate_targets(train, 28159, "train")
    validate_targets(validation, 4488, "validation")
    args.output_dir.mkdir(parents=True)
    config = {
        "step": "Step 6 ObjectText-adapted RLCompOpt NVP targets",
        "candidate_count": K,
        "candidate_value_formula": "(S_Oz - best_object_text_size) / S_Oz",
        "candidate_value_provenance": "OFFICIAL-CODE-ALIGNED formula with PROJECT-SPECIFIC OBJECTTEXT cost replacement",
        "target_formula": "softmax(raw_candidate_value / target_temperature)",
        "target_temperature": AUTOPHASE_NVP_TEMPERATURE,
        "target_temperature_provenance": "OFFICIAL-CODE-ALIGNED Autophase dense-label config",
        "loss": "soft-label cross entropy on logits",
        "logit_temperature": 1,
        "train_shards": str(args.train_shards),
        "validation_shards": str(args.validation_shards),
        "final_test_accessed": False,
        "ood_accessed": False,
    }
    write_json(args.output_dir / "config.json", config)
    write_jsonl_gzip(args.output_dir / "train_targets.jsonl.gz", train)
    write_jsonl_gzip(args.output_dir / "validation_targets.jsonl.gz", validation)
    report = {
        "step_execution": "COMPLETE",
        "train": train_counts,
        "validation": validation_counts,
        "focused_checks": {
            "train_program_id": train[0]["program_id"],
            "validation_program_id": validation[0]["program_id"],
            "all_targets_finite_nonnegative_and_normalized": True,
            "higher_candidate_value_gets_higher_soft_target": True,
        },
    }
    write_json(args.output_dir / "experiment_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
