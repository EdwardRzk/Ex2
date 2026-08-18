#!/usr/bin/env python3
"""Generate v6 Route-A ObjectText labels from a frozen RLCompOpt K=50 set."""

from __future__ import annotations

import argparse
import ast
import gzip
import json
import math
import platform
import subprocess
import sys
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = 1
K = 50


def scalar_int(value: Any) -> int:
    """Convert a scalar CompilerGym observation, including shape-(1,) values."""
    try:
        import numpy as np

        array = np.asarray(value)
        if array.size != 1:
            raise ValueError(f"Expected one scalar observation, got shape {array.shape}")
        result = int(array.reshape(-1).item())
    except ImportError:
        result = int(value)
    if not math.isfinite(result):
        raise ValueError(f"ObjectText observation is not finite: {result}")
    return result


def load_candidates(path: Path) -> list[tuple[int, ...]]:
    candidates = [
        tuple(ast.literal_eval(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(candidates) != K or any(not candidate for candidate in candidates):
        raise ValueError(f"Expected exactly {K} non-empty Route-A candidates")
    return candidates


def load_split_programs(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    programs = [str(item["benchmark"]) for item in payload["samples"]]
    if len(programs) != len(set(programs)):
        raise ValueError(f"Split has duplicate program IDs: {path}")
    return programs


def failure_reason(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}"


def invalid_candidate_record(
    *,
    program_id: str,
    dataset_id: str,
    candidate_id: int,
    candidate: Sequence[int],
    action_names: Sequence[str],
    initial: int | None,
    oz: int | None,
    baseline_error: str | None,
    metadata: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "program_id": program_id,
        "dataset_id": dataset_id,
        "candidate_id": candidate_id,
        "ordered_pass_sequence": list(candidate),
        "ordered_pass_names": [action_names[action] for action in candidate],
        "candidate_length": len(candidate),
        "initial_object_text_size_bytes": initial,
        "oz_object_text_size_bytes": oz,
        "prefix_object_text_size_bytes": [],
        "best_prefix_index": None,
        "best_object_text_size_bytes": None,
        "final_object_text_size_bytes": None,
        "S_O0_measurement_validity": "invalid",
        "S_Oz_measurement_validity": "invalid",
        "measurement_validity": "invalid",
        "ratio_metric_validity": "invalid_missing_required_measurement",
        "training_target_validity": "invalid_incomplete_candidate_rollout",
        "failure_reason": baseline_error,
        **metadata,
    }


def label_program(
    env: Any,
    *,
    program_id: str,
    candidates: Sequence[Sequence[int]],
    action_names: Sequence[str],
    metadata: Mapping[str, str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dataset_id = program_id.split("://", 1)[1].split("/", 1)[0]
    try:
        env.reset(benchmark=program_id)
        initial = scalar_int(env.observation["ObjectTextSizeO0"])
        oz = scalar_int(env.observation["ObjectTextSizeOz"])
    except Exception as error:
        reason = failure_reason(error)
        records = [
            invalid_candidate_record(
                program_id=program_id,
                dataset_id=dataset_id,
                candidate_id=index,
                candidate=candidate,
                action_names=action_names,
                initial=None,
                oz=None,
                baseline_error=reason,
                metadata=metadata,
            )
            for index, candidate in enumerate(candidates)
        ]
        return records, program_summary(program_id, dataset_id, None, None, records)

    records: list[dict[str, Any]] = []
    for candidate_id, candidate in enumerate(candidates):
        prefix: list[int] = []
        reason: str | None = None
        try:
            env.reset(benchmark=program_id)
            for action in candidate:
                _, _, done, info = env.step(action)
                if done:
                    raise RuntimeError(f"Episode ended before candidate completion: {info}")
                prefix.append(scalar_int(env.observation["ObjectTextSizeBytes"]))
        except Exception as error:
            reason = failure_reason(error)

        complete = reason is None and len(prefix) == len(candidate)
        ratio_validity = (
            "valid_for_ObjectText_ratio_metric"
            if complete and oz > 0
            else "invalid_for_ObjectText_ratio_metric"
        )
        best_index = prefix.index(min(prefix)) if complete else None
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "program_id": program_id,
                "dataset_id": dataset_id,
                "candidate_id": candidate_id,
                "ordered_pass_sequence": list(candidate),
                "ordered_pass_names": [action_names[action] for action in candidate],
                "candidate_length": len(candidate),
                "initial_object_text_size_bytes": initial,
                "oz_object_text_size_bytes": oz,
                "prefix_object_text_size_bytes": prefix,
                "best_prefix_index": best_index,
                "best_object_text_size_bytes": min(prefix) if complete else None,
                "final_object_text_size_bytes": prefix[-1] if complete else None,
                "S_O0_measurement_validity": "valid",
                "S_Oz_measurement_validity": "valid",
                "measurement_validity": "valid" if complete else "invalid",
                "ratio_metric_validity": ratio_validity,
                "training_target_validity": (
                    "valid_completed_candidate_rollout"
                    if complete
                    else "invalid_incomplete_candidate_rollout"
                ),
                "failure_reason": reason,
                **metadata,
            }
        )

    summary = program_summary(program_id, dataset_id, initial, oz, records)
    for record in records:
        record["program_training_target_validity"] = summary[
            "program_training_target_validity"
        ]
        record["oracle_K50_validity"] = summary["oracle_K50_validity"]
    return records, summary


def program_summary(
    program_id: str,
    dataset_id: str,
    initial: int | None,
    oz: int | None,
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    complete = sum(
        record["training_target_validity"] == "valid_completed_candidate_rollout"
        for record in records
    )
    valid = complete == K
    failures = Counter(
        record["failure_reason"] for record in records if record["failure_reason"]
    )
    return {
        "program_id": program_id,
        "dataset_id": dataset_id,
        "initial_object_text_size_bytes": initial,
        "oz_object_text_size_bytes": oz,
        "S_O0_measurement_validity": "valid" if initial is not None else "invalid",
        "S_Oz_measurement_validity": "valid" if oz is not None else "invalid",
        "ratio_metric_validity": (
            "valid_for_ObjectText_ratio_metric" if oz is not None and oz > 0 else "invalid_for_ObjectText_ratio_metric"
        ),
        "complete_candidate_count": complete,
        "program_training_target_validity": (
            "valid_complete_K50" if valid else "invalid_incomplete_K50_target"
        ),
        "oracle_K50_validity": "valid_complete_K50" if valid else "invalid_incomplete_K50",
        "candidate_failure_count_by_reason": dict(failures),
    }


def environment_metadata() -> dict[str, str]:
    import compiler_gym

    target = subprocess.run(
        ["llvm-config", "--host-target"], capture_output=True, text=True, check=False
    ).stdout.strip()
    return {
        "compiler_gym_version": compiler_gym.__version__,
        "llvm_version": "10.0.0",
        "compiler_gym_fork": "generic CompilerGym 0.2.5 (PROJECT-SPECIFIC COMPATIBILITY ADAPTATION)",
        "target_triple": target or "unknown",
        "target_cpu_if_configured": "not_configured",
        "host_architecture": platform.machine(),
        "objecttext_observation_api": "ObjectTextSizeBytes/ObjectTextSizeO0/ObjectTextSizeOz",
    }


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_split(
    *, output_dir: Path, split_name: str, programs: Iterable[str], candidates: Sequence[Sequence[int]], metadata: Mapping[str, str]
) -> dict[str, Any]:
    import compiler_gym

    labels_path = output_dir / f"{split_name}_labels.jsonl.gz"
    summaries_path = output_dir / f"{split_name}_programs.jsonl.gz"
    counts = Counter()
    with compiler_gym.make("llvm-v0", reward_space=None) as env, gzip.open(labels_path, "wt", encoding="utf-8") as labels, gzip.open(summaries_path, "wt", encoding="utf-8") as summaries:
        action_names = list(env.action_space.names)
        for program_id in programs:
            records, summary = label_program(env, program_id=program_id, candidates=candidates, action_names=action_names, metadata=metadata)
            for record in records:
                labels.write(json.dumps(record, separators=(",", ":")) + "\n")
                if record["failure_reason"]:
                    counts[record["failure_reason"]] += 1
            summaries.write(json.dumps(summary, separators=(",", ":")) + "\n")
            counts["programs_total"] += 1
            counts["programs_complete_K50"] += summary["program_training_target_validity"] == "valid_complete_K50"
    counts["programs_incomplete_K50"] = counts["programs_total"] - counts["programs_complete_K50"]
    return dict(counts)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-split", type=Path, required=True)
    parser.add_argument("--validation-split", type=Path, required=True)
    parser.add_argument("--candidate-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output_dir}")
    candidates = load_candidates(args.candidate_file)
    train = load_split_programs(args.train_split)
    validation = load_split_programs(args.validation_split)
    if set(train) & set(validation):
        raise ValueError("Official train and validation program populations overlap")
    args.output_dir.mkdir(parents=True)
    metadata = environment_metadata()
    frozen_config = {
        "route": "Route A",
        "objective": "ObjectTextSizeBytes",
        "candidate_source": str(args.candidate_file),
        "candidate_count": K,
        "candidate_rollout": "independent reset from original benchmark",
        "reward_space": None,
        "automatic_retry_count": 0,
        "train_split_source": str(args.train_split),
        "validation_split_source": str(args.validation_split),
        "final_test_accessed": False,
        "environment": metadata,
    }
    write_json(args.output_dir / "config.json", frozen_config)
    report: dict[str, Any] = {"started_at_utc": datetime.now(timezone.utc).isoformat(), "decision": "INVALID"}
    try:
        report["train"] = run_split(output_dir=args.output_dir, split_name="train", programs=train, candidates=candidates, metadata=metadata)
        report["validation"] = run_split(output_dir=args.output_dir, split_name="validation", programs=validation, candidates=candidates, metadata=metadata)
        report["decision"] = "COMPLETE"
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
    report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(args.output_dir / "experiment_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["decision"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
