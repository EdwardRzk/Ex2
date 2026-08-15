#!/usr/bin/env python3
"""Generate the frozen MambaPO ObjectTextSize trajectory dataset."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import platform
import random
import shutil
import subprocess
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = 1
_WORKER_ENV: Any = None
_WORKER_ACTION_NAMES: list[str] = []
_WORKER_CONFIG: dict[str, Any] = {}


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        config = json.load(file)

    required_sections = {
        "experiment_name",
        "hypothesis",
        "dataset",
        "environment",
        "trajectory",
        "model",
        "objective",
        "metrics",
        "pass_fail_gate",
        "execution",
    }
    missing = required_sections - config.keys()
    if missing:
        raise ValueError(f"Config is missing required fields: {sorted(missing)}")

    dataset = config["dataset"]
    required_dataset_fields = {
        "name",
        "total_programs",
        "train_programs",
        "dev_programs",
        "final_test_dataset",
        "final_test_accessed",
        "trajectories_per_program",
        "selection_seed",
    }
    missing = required_dataset_fields - dataset.keys()
    if missing:
        raise ValueError(f"Dataset config is missing fields: {sorted(missing)}")
    if dataset["train_programs"] + dataset["dev_programs"] != dataset["total_programs"]:
        raise ValueError("train_programs + dev_programs must equal total_programs")
    if dataset["final_test_accessed"] is not False:
        raise ValueError("Dataset v0 must not access the sealed final test dataset")
    if dataset["trajectories_per_program"] < 1:
        raise ValueError("trajectories_per_program must be positive")

    trajectory = config["trajectory"]
    minimum = trajectory["minimum_sequence_length"]
    maximum = trajectory["maximum_sequence_length"]
    if minimum < 1 or maximum < minimum:
        raise ValueError("Invalid trajectory sequence-length bounds")
    if maximum != 32:
        raise ValueError("MambaPO v0 maximum sequence length must remain 32")
    if trajectory["action_sampling"] != "uniform_with_replacement":
        raise ValueError("Dataset v0 requires uniform action sampling with replacement")
    if config["execution"]["workers"] < 1:
        raise ValueError("workers must be positive")
    return config


def select_program_splits(
    benchmark_uris: Iterable[str],
    *,
    train_programs: int,
    dev_programs: int,
    seed: int,
) -> dict[str, list[str]]:
    uris = sorted(str(uri) for uri in benchmark_uris)
    total = train_programs + dev_programs
    if len(uris) < total:
        raise ValueError(f"Dataset has {len(uris)} programs, but {total} are required")
    selected = random.Random(seed).sample(uris, total)
    return {
        "train": selected[:train_programs],
        "dev": selected[train_programs:],
    }


def object_text_size_oz_reward(
    initial_size_bytes: int, oz_size_bytes: int, final_size_bytes: int
) -> float:
    denominator = max(initial_size_bytes - oz_size_bytes, 1)
    return (initial_size_bytes - final_size_bytes) / denominator


def validate_record(record: Mapping[str, Any], feature_dimension: int) -> None:
    length = record["sequence_length"]
    if length != len(record["action_indices"]):
        raise ValueError("sequence_length does not match action_indices")
    if length != len(record["action_names"]):
        raise ValueError("sequence_length does not match action_names")
    if len(record["states"]) != length + 1:
        raise ValueError("states must contain the initial state and every post-action state")
    if any(len(state) != feature_dimension for state in record["states"]):
        raise ValueError("Inconsistent feature dimension")
    if record["final_object_text_size_bytes"] < 0:
        raise ValueError("ObjectTextSizeBytes must be non-negative")


def _state_to_list(state: Any) -> list[int]:
    return [int(value) for value in state]


def generate_program_trajectories(task: Mapping[str, Any]) -> list[dict[str, Any]]:
    if _WORKER_ENV is None:
        raise RuntimeError("Worker environment is not initialized")

    program_uri = task["program_uri"]
    program_index = int(task["program_index"])
    split = task["split"]
    trajectories_per_program = int(task["trajectories_per_program"])
    minimum_length = int(task["minimum_sequence_length"])
    maximum_length = int(task["maximum_sequence_length"])
    rng = random.Random(int(task["seed"]) + program_index * 1_000_003)

    records: list[dict[str, Any]] = []
    initial_size_bytes: int | None = None
    oz_size_bytes: int | None = None

    for trajectory_index in range(trajectories_per_program):
        initial_state = _WORKER_ENV.reset(benchmark=program_uri)
        if initial_size_bytes is None:
            initial_size_bytes = int(
                _WORKER_ENV.observation[_WORKER_CONFIG["size_observation"]]
            )
            oz_size_bytes = int(
                _WORKER_ENV.observation[_WORKER_CONFIG["oz_baseline_observation"]]
            )

        length = rng.randint(minimum_length, maximum_length)
        action_indices = [rng.randrange(len(_WORKER_ACTION_NAMES)) for _ in range(length)]
        states = [_state_to_list(initial_state)]
        action_names: list[str] = []

        for action_index in action_indices:
            state, _, done, info = _WORKER_ENV.step(action_index)
            if done:
                raise RuntimeError(
                    f"LLVM episode ended for {program_uri} after action "
                    f"{_WORKER_ACTION_NAMES[action_index]}: {info}"
                )
            states.append(_state_to_list(state))
            action_names.append(_WORKER_ACTION_NAMES[action_index])

        final_size_bytes = int(
            _WORKER_ENV.observation[_WORKER_CONFIG["size_observation"]]
        )
        assert initial_size_bytes is not None
        assert oz_size_bytes is not None
        record = {
            "schema_version": SCHEMA_VERSION,
            "program_id": program_uri,
            "split": split,
            "trajectory_index": trajectory_index,
            "sequence_length": length,
            "action_indices": action_indices,
            "action_names": action_names,
            "states": states,
            "initial_object_text_size_bytes": initial_size_bytes,
            "oz_object_text_size_bytes": oz_size_bytes,
            "final_object_text_size_bytes": final_size_bytes,
            "object_text_size_oz_reward": object_text_size_oz_reward(
                initial_size_bytes, oz_size_bytes, final_size_bytes
            ),
            "size_reduction_vs_oz": (
                (oz_size_bytes - final_size_bytes) / oz_size_bytes
                if oz_size_bytes
                else 0.0
            ),
        }
        validate_record(record, len(states[0]))
        records.append(record)

    return records


def _initialize_worker(environment_config: Mapping[str, Any]) -> None:
    global _WORKER_ACTION_NAMES, _WORKER_CONFIG, _WORKER_ENV

    import compiler_gym

    _WORKER_CONFIG = dict(environment_config)
    _WORKER_ENV = compiler_gym.make(
        environment_config["id"],
        observation_space=environment_config["feature_space"],
    )
    _WORKER_ACTION_NAMES = list(_WORKER_ENV.action_space.names)


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2, sort_keys=True)
        file.write("\n")


def git_metadata(repo_root: Path) -> dict[str, Any]:
    def run_git(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    status = run_git("status", "--short")
    return {
        "commit": run_git("rev-parse", "HEAD"),
        "branch": run_git("branch", "--show-current"),
        "status_short": status.splitlines() if status else [],
    }


def dependency_metadata(config: Mapping[str, Any]) -> dict[str, str]:
    import compiler_gym
    import numpy

    actual = {
        "compiler_gym": compiler_gym.__version__,
        "numpy": numpy.__version__,
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
    }
    expected = {
        "compiler_gym": config["environment"]["compiler_gym_version"],
        "numpy": config["environment"]["numpy_version"],
        "python": config["environment"]["python_major_minor"],
    }
    if actual != expected:
        raise RuntimeError(f"Dependency mismatch: expected {expected}, got {actual}")
    return actual


def environment_metadata(config: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    import compiler_gym

    with compiler_gym.make(config["environment"]["id"]) as environment:
        dataset = environment.datasets[config["dataset"]["name"]]
        if not dataset.installed:
            raise RuntimeError(
                f"Dataset is not installed: {config['dataset']['name']}. "
                "Install it before running generation."
            )
        return list(dataset.benchmark_uris()), list(environment.action_space.names)


def build_tasks(
    splits: Mapping[str, Sequence[str]], config: Mapping[str, Any]
) -> list[dict[str, Any]]:
    dataset = config["dataset"]
    trajectory = config["trajectory"]
    tasks: list[dict[str, Any]] = []
    program_index = 0
    for split in ("train", "dev"):
        for program_uri in splits[split]:
            tasks.append(
                {
                    "program_uri": program_uri,
                    "program_index": program_index,
                    "split": split,
                    "trajectories_per_program": dataset["trajectories_per_program"],
                    "minimum_sequence_length": trajectory["minimum_sequence_length"],
                    "maximum_sequence_length": trajectory["maximum_sequence_length"],
                    "seed": dataset["selection_seed"],
                }
            )
            program_index += 1
    return tasks


def evaluate_result(
    *,
    report_counts: Mapping[str, int],
    splits: Mapping[str, Sequence[str]],
    config: Mapping[str, Any],
    observed_minimum_length: int,
    observed_maximum_length: int,
) -> dict[str, bool]:
    gate = config["pass_fail_gate"]
    overlap = set(splits["train"]) & set(splits["dev"])
    return {
        "trajectory_count": report_counts["trajectory_count"] == gate["trajectory_count"],
        "program_count": report_counts["program_count"] == gate["program_count"],
        "train_dev_program_overlap": len(overlap) == gate["train_dev_program_overlap"],
        "invalid_episode_count": report_counts["invalid_episode_count"]
        == gate["invalid_episode_count"],
        "sequence_lengths_within_bounds": (
            observed_minimum_length >= gate["sequence_length_minimum"]
            and observed_maximum_length <= gate["sequence_length_maximum"]
        ),
    }


def run_generation(
    *,
    config: Mapping[str, Any],
    output_dir: Path,
    splits: Mapping[str, Sequence[str]],
    action_names: Sequence[str],
    workers: int,
    formal: bool,
) -> dict[str, Any]:
    tasks = build_tasks(splits, config)
    output_paths = {
        "train": output_dir / "train.jsonl.gz",
        "dev": output_dir / "dev.jsonl.gz",
    }
    files = {
        split: gzip.open(path, "wt", encoding="utf-8")
        for split, path in output_paths.items()
    }
    counts = {"train": 0, "dev": 0}
    minimum_length = math.inf
    maximum_length = -math.inf
    feature_dimension: int | None = None

    try:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_initialize_worker,
            initargs=(config["environment"],),
        ) as executor:
            for completed_programs, records in enumerate(
                executor.map(generate_program_trajectories, tasks), start=1
            ):
                for record in records:
                    split = record["split"]
                    files[split].write(json.dumps(record, separators=(",", ":")))
                    files[split].write("\n")
                    counts[split] += 1
                    minimum_length = min(minimum_length, record["sequence_length"])
                    maximum_length = max(maximum_length, record["sequence_length"])
                    current_dimension = len(record["states"][0])
                    if feature_dimension is None:
                        feature_dimension = current_dimension
                    elif current_dimension != feature_dimension:
                        raise ValueError("Feature dimension changed across programs")
                print(
                    f"completed_programs={completed_programs}/{len(tasks)} "
                    f"trajectories={counts['train'] + counts['dev']}",
                    flush=True,
                )
    finally:
        for file in files.values():
            file.close()

    report_counts = {
        "trajectory_count": counts["train"] + counts["dev"],
        "train_trajectory_count": counts["train"],
        "dev_trajectory_count": counts["dev"],
        "program_count": len(tasks),
        "invalid_episode_count": 0,
    }
    checks = (
        evaluate_result(
            report_counts=report_counts,
            splits=splits,
            config=config,
            observed_minimum_length=int(minimum_length),
            observed_maximum_length=int(maximum_length),
        )
        if formal
        else {
            "trajectory_count": report_counts["trajectory_count"] == 4,
            "program_count": report_counts["program_count"] == 2,
            "train_dev_program_overlap": not (
                set(splits["train"]) & set(splits["dev"])
            ),
            "invalid_episode_count": True,
            "sequence_lengths_within_bounds": 1 <= minimum_length <= maximum_length <= 32,
        }
    )
    return {
        "counts": report_counts,
        "feature_dimension": feature_dimension,
        "action_space_size": len(action_names),
        "action_names": list(action_names),
        "observed_sequence_length": {
            "minimum": int(minimum_length),
            "maximum": int(maximum_length),
        },
        "checks": checks,
        "decision": "PASS" if all(checks.values()) else "FAIL",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    config = load_config(args.config.resolve())
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json(output_dir / "config.json", config)

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "experiment_name": config["experiment_name"],
        "mode": "smoke" if args.smoke else "formal",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": git_metadata(repo_root),
        "host": {"platform": platform.platform(), "python": sys.version},
        "decision": "FAIL",
    }

    try:
        report["dependencies"] = dependency_metadata(config)
        benchmark_uris, action_names = environment_metadata(config)
        dataset = config["dataset"]
        splits = select_program_splits(
            benchmark_uris,
            train_programs=dataset["train_programs"],
            dev_programs=dataset["dev_programs"],
            seed=dataset["selection_seed"],
        )
        workers = config["execution"]["workers"]
        if args.smoke:
            splits = {"train": splits["train"][:1], "dev": splits["dev"][:1]}
            smoke_config = json.loads(json.dumps(config))
            smoke_config["dataset"]["trajectories_per_program"] = 2
            config = smoke_config
            workers = 1

        write_json(output_dir / "program_splits.json", splits)
        report.update(
            run_generation(
                config=config,
                output_dir=output_dir,
                splits=splits,
                action_names=action_names,
                workers=workers,
                formal=not args.smoke,
            )
        )
    except Exception as error:
        report["error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
    finally:
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(output_dir / "experiment_report.json", report)

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["decision"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
