#!/usr/bin/env python3
"""Compute the frozen Route-A K=50 validation Oracle from Step-3 shards."""

from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


K = 50


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_program_shard(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    records = payload["records"]
    summary = payload["program_summary"]
    if len(records) != K:
        raise ValueError(f"Expected exactly {K} records in {path}, found {len(records)}")
    if any(record["program_id"] != summary["program_id"] for record in records):
        raise ValueError(f"Program ID mismatch in {path}")
    return records, summary


def oracle_audit_record(
    records: Sequence[Mapping[str, Any]], summary: Mapping[str, Any]
) -> dict[str, Any]:
    candidate_ids = sorted(record["candidate_id"] for record in records)
    complete = summary["oracle_K50_validity"] == "valid_complete_K50"
    if complete:
        if candidate_ids != list(range(K)):
            raise ValueError(f"Complete K=50 program has invalid candidate IDs: {summary['program_id']}")
        if any(
            record["training_target_validity"] != "valid_completed_candidate_rollout"
            for record in records
        ):
            raise ValueError(f"Complete K=50 program has an invalid candidate: {summary['program_id']}")
        best_sizes = [record["best_object_text_size_bytes"] for record in records]
        if any(size is None for size in best_sizes):
            raise ValueError(f"Complete K=50 program has no candidate best size: {summary['program_id']}")
    oz = summary["oz_object_text_size_bytes"]
    ratio_valid = oz is not None and oz > 0
    eligible = complete and ratio_valid
    oracle_size = min(record["best_object_text_size_bytes"] for record in records) if eligible else None
    reduction = (oz - oracle_size) / oz if eligible else None
    return {
        "program_id": summary["program_id"],
        "dataset_id": summary["dataset_id"],
        "S_Oz": oz,
        "S_oracle": oracle_size,
        "oracle_reduction_vs_Oz": reduction,
        "oracle_K50_validity": summary["oracle_K50_validity"],
        "ratio_metric_validity": summary["ratio_metric_validity"],
        "oracle_candidate_count": K if eligible else 0,
        "route_a_oracle_eligible": eligible,
    }


def compute_route_a_oracle(validation_shards: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    audit_records: list[dict[str, Any]] = []
    for path in sorted(validation_shards.glob("*.json.gz")):
        records, summary = load_program_shard(path)
        audit_records.append(oracle_audit_record(records, summary))
    if not audit_records:
        raise ValueError(f"No validation shards found: {validation_shards}")

    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in audit_records:
        by_dataset[record["dataset_id"]].append(record)

    per_dataset: dict[str, dict[str, Any]] = {}
    dataset_means: list[float] = []
    undefined = False
    for dataset_id in sorted(by_dataset):
        records = by_dataset[dataset_id]
        eligible = [record for record in records if record["route_a_oracle_eligible"]]
        complete = [record for record in records if record["oracle_K50_validity"] == "valid_complete_K50"]
        ratio_valid = [record for record in records if record["S_Oz"] is not None and record["S_Oz"] > 0]
        if eligible:
            mean = sum(record["oracle_reduction_vs_Oz"] for record in eligible) / len(eligible)
            dataset_means.append(mean)
        else:
            mean = None
            undefined = True
        per_dataset[dataset_id] = {
            "N_total": len(records),
            "N_complete_K50_oracle": len(complete),
            "N_ratio_valid": len(ratio_valid),
            "N_RouteA_oracle_valid": len(eligible),
            "N_failed_or_invalid": len(records) - len(eligible),
            "OracleMeanOverOz": mean,
        }

    if undefined:
        route_mean = None
        branch_status = "undefined_due_to_invalid_required_data"
        route_decision = "STOP_UNDEFINED"
    else:
        route_mean = sum(dataset_means) / len(dataset_means)
        branch_status = "defined"
        route_decision = "STAY_ROUTE_A" if route_mean > 0 else "ROUTE_B_AUTHORIZED"

    report = {
        "step": "Step 4 Route-A fixed-set Oracle",
        "step_execution": "COMPLETE",
        "candidate_count": K,
        "all_eligible_oracle_values_use_exactly_K50_candidates": True,
        "validation_population": {
            "N_total": len(audit_records),
            "N_complete_K50_oracle": sum(record["oracle_K50_validity"] == "valid_complete_K50" for record in audit_records),
            "N_ratio_valid": sum(record["S_Oz"] is not None and record["S_Oz"] > 0 for record in audit_records),
            "N_RouteA_oracle_valid": sum(record["route_a_oracle_eligible"] for record in audit_records),
            "N_failed_or_invalid": sum(not record["route_a_oracle_eligible"] for record in audit_records),
            "N_excluded_incomplete_K50": sum(record["oracle_K50_validity"] != "valid_complete_K50" for record in audit_records),
            "N_ratio_invalid": sum(record["S_Oz"] is None or record["S_Oz"] <= 0 for record in audit_records),
        },
        "per_dataset": per_dataset,
        "RouteAOracleMeanOverOz": route_mean,
        "branch_criterion_status": branch_status,
        "route_decision": route_decision,
    }
    return report, audit_records


def write_audit_records(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-shards", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output_dir}")
    report, audit_records = compute_route_a_oracle(args.validation_shards)
    args.output_dir.mkdir(parents=True)
    write_json(
        args.output_dir / "config.json",
        {
            "step": "Step 4 Route-A fixed-set Oracle",
            "validation_shards": str(args.validation_shards),
            "candidate_count": K,
            "final_test_accessed": False,
            "ood_accessed": False,
        },
    )
    write_audit_records(args.output_dir / "validation_oracle_programs.jsonl.gz", audit_records)
    write_json(args.output_dir / "experiment_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
