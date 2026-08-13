#!/usr/bin/env python3
"""Qualify Gate 1 runtime measurement using only identical LLVM -O3 baselines."""

from __future__ import annotations

import argparse
import json
import statistics
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

try:
    from scripts.run_gate1_search_headroom import (
        Toolchain,
        atomic_write_json,
        bootstrap_ratio_ci,
        correctness_hash,
        distribution_stats,
        git_metadata,
        load_config,
        measure_binary,
        qualification_checks,
        sha256_file,
    )
except ModuleNotFoundError:
    from run_gate1_search_headroom import (
        Toolchain,
        atomic_write_json,
        bootstrap_ratio_ci,
        correctness_hash,
        distribution_stats,
        git_metadata,
        load_config,
        measure_binary,
        qualification_checks,
        sha256_file,
    )


DATASET_ORDER = ("MEDIUM_DATASET", "LARGE_DATASET", "EXTRALARGE_DATASET")


def dataset_candidates(configured_macro: str) -> list[str]:
    if configured_macro not in DATASET_ORDER:
        raise ValueError(f"Unsupported starting dataset macro: {configured_macro}")
    return list(DATASET_ORDER[DATASET_ORDER.index(configured_macro) :])


def sanity_check(
    group_a: list[float],
    group_b: list[float],
    measurement: Mapping[str, Any],
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    ratio = statistics.median(group_a) / statistics.median(group_b)
    interval = bootstrap_ratio_ci(group_a, group_b, resamples, seed)
    checks = {
        "ratio_close_to_one": abs(ratio - 1.0)
        <= measurement["maximum_block_median_drift"],
        "bootstrap_95_ci_covers_one": interval[0] <= 1.0 <= interval[1],
    }
    return {
        "speedup_ratio_a_over_b": ratio,
        "bootstrap_95_ci": list(interval),
        "checks": checks,
        "pass": all(checks.values()),
    }


def measure_setting(
    toolchain: Toolchain,
    kernel: Mapping[str, str],
    setting_dir: Path,
    config: Mapping[str, Any],
    seed: int,
) -> dict[str, Any]:
    measurement = config["measurement"]
    input_bc = setting_dir / "input.bc"
    baseline_binary = setting_dir / "o3"
    toolchain.compile_unoptimized_bitcode(kernel, input_bc, dump=False)
    toolchain.build_executable(input_bc, [], baseline_binary, baseline=True)

    group_a, _ = measure_binary(
        baseline_binary,
        measurement["qualification_warmup_runs"],
        measurement["qualification_runs_per_block"],
        measurement["cpu_affinity"],
        minimum_warmup_seconds=5.0,
    )
    group_b, _ = measure_binary(
        baseline_binary,
        0,
        measurement["qualification_runs_per_block"],
        measurement["cpu_affinity"],
    )
    stats = [distribution_stats(group_a), distribution_stats(group_b)]
    checks = qualification_checks(stats[0], stats[1], measurement)
    sanity = sanity_check(
        group_a,
        group_b,
        measurement,
        resamples=config["metrics"]["bootstrap_resamples"],
        seed=seed,
    )
    return {
        "dataset_macro": kernel["dataset_macro"],
        "binary_sha256": sha256_file(baseline_binary),
        "raw_runtime_samples_seconds": [group_a, group_b],
        "block_stats": stats,
        "combined_stats": distribution_stats([*group_a, *group_b]),
        "qualification_checks": checks,
        "identical_baseline_sanity": sanity,
        "pass": all(checks.values()) and sanity["pass"],
    }


def verify_correctness(
    toolchain: Toolchain,
    kernel: Mapping[str, str],
    setting_dir: Path,
    cpu: int,
) -> dict[str, Any]:
    dump_bc = setting_dir / "dump_input.bc"
    dump_binary = setting_dir / "o3_dump"
    toolchain.compile_unoptimized_bitcode(kernel, dump_bc, dump=True)
    toolchain.build_executable(dump_bc, [], dump_binary, baseline=True)
    first_hash = correctness_hash(dump_binary, cpu)
    second_hash = correctness_hash(dump_binary, cpu)
    return {
        "method": "two deterministic POLYBENCH_DUMP_ARRAYS runs of the selected -O3 binary",
        "first_output_sha256": first_hash,
        "second_output_sha256": second_hash,
        "pass": first_hash == second_hash,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--polybench-root", type=Path, required=True)
    parser.add_argument("--polybench-archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    config_path = args.config.resolve()
    archive = args.polybench_archive.resolve()
    output_dir = args.output_dir.resolve()
    config = load_config(config_path)
    kernels = {
        kernel["name"]: kernel
        for kernel in config["dataset"]["kernels"]
        if kernel["name"] in {"2mm", "3mm"}
    }
    if set(kernels) != {"2mm", "3mm"}:
        raise ValueError("The Gate 1 config must contain 2mm and 3mm")

    output_dir.mkdir(parents=True, exist_ok=False)
    work_dir = output_dir / "work"
    work_dir.mkdir()
    report_path = output_dir / "report.json"
    report: dict[str, Any] = {
        "experiment_name": output_dir.name,
        "status": "RUNNING",
        "decision": "FAIL",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_config_path": str(config_path.relative_to(repo_root)),
        "base_config_sha256": sha256_file(config_path),
        "polybench_archive_sha256": sha256_file(archive),
        "git": git_metadata(repo_root),
        "protocol": {
            "scope": "LLVM -O3 baseline only; no pass search",
            "timer": "POLYBENCH_TIME kernel-only stdout",
            "cache_flush": "POLYBENCH_NO_FLUSH_CACHE",
            "cpu_binding": f"taskset -c {config['measurement']['cpu_affinity']}",
            "warmup_runs": config["measurement"]["qualification_warmup_runs"],
            "minimum_warmup_seconds": 5.0,
            "formal_runs_per_group": config["measurement"]["qualification_runs_per_block"],
            "groups": config["measurement"]["qualification_blocks"],
            "dataset_attempt_order": list(DATASET_ORDER),
            "thresholds_unchanged": {
                key: config["measurement"][key]
                for key in (
                    "minimum_runtime_seconds",
                    "maximum_cv",
                    "maximum_relative_mad",
                    "maximum_block_median_drift",
                )
            },
        },
        "kernels": [],
    }
    atomic_write_json(report_path, report)

    try:
        if report["polybench_archive_sha256"] != config["dataset"]["archive_sha256"]:
            raise RuntimeError("PolyBench archive SHA256 mismatch")
        toolchain = Toolchain(config, args.polybench_root.resolve(), work_dir)
        report["dependencies"] = toolchain.versions

        for kernel_index, name in enumerate(("2mm", "3mm")):
            configured = kernels[name]
            kernel_report: dict[str, Any] = {
                "program": name,
                "source": configured["source"],
                "attempts": [],
                "selected_dataset_macro": None,
            }
            report["kernels"].append(kernel_report)
            for attempt_index, dataset_macro in enumerate(
                dataset_candidates(configured["dataset_macro"])
            ):
                setting_dir = work_dir / name / dataset_macro.lower()
                setting_dir.mkdir(parents=True)
                kernel = {**configured, "dataset_macro": dataset_macro}
                attempt = measure_setting(
                    toolchain,
                    kernel,
                    setting_dir,
                    config,
                    config["search"]["seed"] + 100 * kernel_index + attempt_index,
                )
                kernel_report["attempts"].append(attempt)
                atomic_write_json(report_path, report)
                if attempt["pass"]:
                    correctness = verify_correctness(
                        toolchain,
                        kernel,
                        setting_dir,
                        config["measurement"]["cpu_affinity"],
                    )
                    attempt["correctness"] = correctness
                    attempt["pass"] = attempt["pass"] and correctness["pass"]
                    if attempt["pass"]:
                        kernel_report["selected_dataset_macro"] = dataset_macro
                        break
                atomic_write_json(report_path, report)

        report["decision"] = (
            "PASS"
            if all(kernel["selected_dataset_macro"] is not None for kernel in report["kernels"])
            else "FAIL"
        )
        report["status"] = "COMPLETE"
    except Exception as error:
        report["status"] = "INVALID"
        report["error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
    finally:
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_write_json(report_path, report)

    print(
        json.dumps(
            {"status": report["status"], "decision": report["decision"]},
            indent=2,
        )
    )
    return 0 if report["status"] == "COMPLETE" and report["decision"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
