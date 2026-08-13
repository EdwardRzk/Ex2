#!/usr/bin/env python3
"""Run the frozen CompilerGym Gate 0 environment smoke experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        config = json.load(file)

    required_sections = {
        "experiment_name",
        "hypothesis",
        "dataset",
        "environment",
        "model",
        "objective",
        "metrics",
        "pass_fail_gate",
    }
    missing = required_sections - config.keys()
    if missing:
        raise ValueError(f"Config is missing required fields: {sorted(missing)}")

    environment = config["environment"]
    required_environment_fields = {
        "id",
        "python_major_minor",
        "compiler_gym_version",
        "numpy_version",
        "action_name",
        "runtime_warmup_runs",
        "runtime_measurement_runs",
    }
    missing_environment = required_environment_fields - environment.keys()
    if missing_environment:
        raise ValueError(
            "Config environment is missing required fields: "
            f"{sorted(missing_environment)}"
        )
    if environment["runtime_warmup_runs"] < 1:
        raise ValueError("runtime_warmup_runs must be at least 1")
    if environment["runtime_measurement_runs"] < 2:
        raise ValueError("runtime_measurement_runs must be at least 2")
    return config


def resolve_action(action_names: Sequence[str], action_name: str) -> int:
    try:
        return action_names.index(action_name)
    except ValueError as error:
        raise ValueError(f"LLVM action is unavailable: {action_name}") from error


def evaluate_gate(
    results: Mapping[str, Any], expected_runtime_sample_count: int
) -> dict[str, bool]:
    runtimes = results.get("runtime_samples_seconds", [])
    return {
        "reset_succeeded": results.get("reset_succeeded") is True,
        "action_had_effect": results.get("action_had_effect") is True,
        "is_buildable": results.get("is_buildable") is True,
        "is_runnable": results.get("is_runnable") is True,
        "runtime_sample_count": len(runtimes) == expected_runtime_sample_count,
        "all_runtime_samples_positive_and_finite": bool(runtimes)
        and all(math.isfinite(value) and value > 0 for value in runtimes),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def run_experiment(config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    import compiler_gym
    import numpy

    environment_config = config["environment"]
    actual_versions = {
        "compiler_gym": compiler_gym.__version__,
        "numpy": numpy.__version__,
    }
    expected_versions = {
        "compiler_gym": environment_config["compiler_gym_version"],
        "numpy": environment_config["numpy_version"],
    }
    if actual_versions != expected_versions:
        raise RuntimeError(
            f"Dependency version mismatch: expected {expected_versions}, "
            f"got {actual_versions}"
        )

    expected_python = environment_config["python_major_minor"]
    actual_python = f"{sys.version_info.major}.{sys.version_info.minor}"
    if actual_python != expected_python:
        raise RuntimeError(
            f"Python version mismatch: expected {expected_python}, got {actual_python}"
        )

    benchmark_uri = config["dataset"]["benchmark_uri"]
    results: dict[str, Any] = {
        "benchmark_uri": benchmark_uri,
        "reset_succeeded": False,
    }
    with compiler_gym.make(
        environment_config["id"], benchmark=benchmark_uri
    ) as environment:
        environment.runtime_warmup_runs_count = environment_config[
            "runtime_warmup_runs"
        ]
        environment.runtime_observation_count = environment_config[
            "runtime_measurement_runs"
        ]
        environment.reset()
        results["reset_succeeded"] = True
        results["action_space_size"] = environment.action_space.n

        action_index = resolve_action(
            environment.action_space.names, environment_config["action_name"]
        )
        _, _, done, info = environment.step(action_index)
        results.update(
            {
                "action_name": environment_config["action_name"],
                "action_index": action_index,
                "episode_done_after_action": bool(done),
                "action_had_effect": not info.get("action_had_no_effect", False),
                "is_buildable": bool(environment.observation["IsBuildable"]),
                "is_runnable": bool(environment.observation["IsRunnable"]),
            }
        )
        runtimes = [
            float(value) for value in environment.observation["Runtime"]
        ]
        results["runtime_samples_seconds"] = runtimes
        results["median_runtime_seconds"] = statistics.median(runtimes)

    return results, actual_versions


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2, sort_keys=True)
        file.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    config_path = args.config.resolve()
    output_dir = args.output_dir.resolve()
    config = load_config(config_path)

    output_dir.mkdir(parents=True, exist_ok=False)
    frozen_config_path = output_dir / "config.json"
    write_json(frozen_config_path, config)

    report: dict[str, Any] = {
        "experiment_name": config["experiment_name"],
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": sha256(frozen_config_path),
        "git": git_metadata(repo_root),
        "host": {
            "platform": platform.platform(),
            "python": sys.version,
        },
        "decision": "FAIL",
    }

    try:
        results, versions = run_experiment(config)
        checks = evaluate_gate(
            results, config["environment"]["runtime_measurement_runs"]
        )
        report.update(
            {
                "dependencies": versions,
                "results": results,
                "gate_checks": checks,
                "decision": "PASS" if all(checks.values()) else "FAIL",
            }
        )
    except Exception as error:  # Preserve a formal failure report.
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
