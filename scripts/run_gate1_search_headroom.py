#!/usr/bin/env python3
"""Run the frozen Gate 1 PolyBench phase-ordering headroom experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {"experiment_name", "dataset", "toolchain", "search", "measurement", "metrics", "pass_fail_gate"}
    missing = required - config.keys()
    if missing:
        raise ValueError(f"Config is missing required fields: {sorted(missing)}")
    search = config["search"]
    expected_budget = search["max_sequence_length"] * search["greedy_candidates_per_step"]
    if expected_budget != search["evaluations_per_method_per_kernel"]:
        raise ValueError("Greedy steps and candidates do not equal the frozen evaluation budget")
    if search["random_sequence_count"] != search["evaluations_per_method_per_kernel"]:
        raise ValueError("Random sequence count does not equal the frozen evaluation budget")
    if len(set(search["action_subset"])) != len(search["action_subset"]):
        raise ValueError("action_subset contains duplicate actions")
    measurement = config["measurement"]
    if measurement["qualification_blocks"] != 2 or measurement["confirmation_blocks"] != 2:
        raise ValueError("This runner requires exactly two independent measurement blocks")
    return config


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sequence_hash(sequence: Sequence[str]) -> str:
    encoded = json.dumps(list(sequence), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def distribution_stats(values: Sequence[float]) -> dict[str, float]:
    if len(values) < 2:
        raise ValueError("At least two runtime samples are required")
    median = statistics.median(values)
    mean = statistics.mean(values)
    mad = statistics.median(abs(value - median) for value in values)
    return {
        "median_seconds": median,
        "mean_seconds": mean,
        "cv": statistics.stdev(values) / mean,
        "mad_seconds": mad,
        "relative_mad": mad / median,
        "min_seconds": min(values),
        "max_seconds": max(values),
    }


def qualification_checks(
    block_a: Mapping[str, float], block_b: Mapping[str, float], measurement: Mapping[str, Any]
) -> dict[str, bool]:
    combined_median = statistics.median(
        [block_a["median_seconds"], block_b["median_seconds"]]
    )
    drift = abs(block_a["median_seconds"] - block_b["median_seconds"]) / combined_median
    return {
        "minimum_runtime": min(block_a["median_seconds"], block_b["median_seconds"])
        >= measurement["minimum_runtime_seconds"],
        "maximum_cv": max(block_a["cv"], block_b["cv"]) <= measurement["maximum_cv"],
        "maximum_relative_mad": max(block_a["relative_mad"], block_b["relative_mad"])
        <= measurement["maximum_relative_mad"],
        "maximum_block_median_drift": drift <= measurement["maximum_block_median_drift"],
    }


def bootstrap_ratio_ci(
    baseline: Sequence[float], candidate: Sequence[float], resamples: int, seed: int
) -> tuple[float, float]:
    rng = random.Random(seed)
    ratios = []
    for _ in range(resamples):
        baseline_sample = [rng.choice(baseline) for _ in baseline]
        candidate_sample = [rng.choice(candidate) for _ in candidate]
        ratios.append(statistics.median(baseline_sample) / statistics.median(candidate_sample))
    ratios.sort()
    lower_index = int(0.025 * (resamples - 1))
    upper_index = int(0.975 * (resamples - 1))
    return ratios[lower_index], ratios[upper_index]


def bootstrap_geomean_ci(speedups: Sequence[float], resamples: int, seed: int) -> tuple[float, float]:
    rng = random.Random(seed)
    values = []
    for _ in range(resamples):
        sample = [rng.choice(speedups) for _ in speedups]
        values.append(math.exp(statistics.mean(math.log(value) for value in sample)))
    values.sort()
    return values[int(0.025 * (resamples - 1))], values[int(0.975 * (resamples - 1))]


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as file:
        json.dump(value, file, indent=2, sort_keys=True)
        file.write("\n")
        temporary = Path(file.name)
    os.replace(temporary, path)


def run_command(
    command: Sequence[str], *, timeout: int = 600, cpu: int | None = None
) -> subprocess.CompletedProcess[str]:
    def set_affinity() -> None:
        if cpu is not None:
            os.sched_setaffinity(0, {cpu})

    return subprocess.run(
        list(command),
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
        preexec_fn=set_affinity if cpu is not None else None,
    )


class Toolchain:
    def __init__(self, config: Mapping[str, Any], polybench_root: Path, work_dir: Path):
        import compiler_gym
        import numpy
        from compiler_gym.envs.llvm.llvm_benchmark import get_system_library_flags
        from compiler_gym.third_party.llvm import clang_path, llc_path, llvm_link_path, opt_path

        expected = config["toolchain"]
        actual_versions = {
            "compiler_gym": compiler_gym.__version__,
            "numpy": numpy.__version__,
        }
        if actual_versions["compiler_gym"] != expected["compiler_gym_version"]:
            raise RuntimeError(f"CompilerGym version mismatch: {actual_versions}")
        if actual_versions["numpy"] != expected["numpy_version"]:
            raise RuntimeError(f"NumPy version mismatch: {actual_versions}")

        self.versions = actual_versions
        self.polybench_root = polybench_root
        self.work_dir = work_dir
        self.clang = str(clang_path())
        self.opt = str(opt_path())
        self.llc = str(llc_path())
        self.llvm_link = str(llvm_link_path())
        self.system_flags = list(get_system_library_flags())
        version_text = run_command([self.opt, "--version"]).stdout
        if f"LLVM version {expected['llvm_version']}" not in version_text:
            raise RuntimeError(f"LLVM version mismatch: {version_text.splitlines()[:3]}")

    def validate_actions(self, actions: Sequence[str]) -> None:
        import compiler_gym

        with compiler_gym.make("llvm-v0", benchmark="generator://csmith-v0/1") as environment:
            available = set(environment.action_space.names)
        missing = sorted(set(actions) - available)
        if missing:
            raise RuntimeError(f"Configured actions are absent from CompilerGym: {missing}")

    def compile_unoptimized_bitcode(self, kernel: Mapping[str, str], destination: Path, dump: bool) -> None:
        source = self.polybench_root / kernel["source"]
        kernel_dir = source.parent
        utility_source = self.polybench_root / "utilities/polybench.c"
        defines = [f"-D{kernel['dataset_macro']}"]
        defines.append("-DPOLYBENCH_DUMP_ARRAYS" if dump else "-DPOLYBENCH_TIME")
        if not dump:
            defines.append("-DPOLYBENCH_NO_FLUSH_CACHE")
        common = [
            self.clang,
            "-O0",
            "-c",
            "-emit-llvm",
            "-Xclang",
            "-disable-llvm-passes",
            "-Xclang",
            "-disable-llvm-optzns",
            *defines,
            "-I",
            str(self.polybench_root / "utilities"),
            "-I",
            str(kernel_dir),
            *self.system_flags,
        ]
        utility_bc = destination.with_name(destination.stem + "_utility.bc")
        kernel_bc = destination.with_name(destination.stem + "_kernel.bc")
        run_command([*common, str(utility_source), "-o", str(utility_bc)])
        run_command([*common, str(source), "-o", str(kernel_bc)])
        run_command([self.llvm_link, str(utility_bc), str(kernel_bc), "-o", str(destination)])

    def build_executable(self, input_bc: Path, sequence: Sequence[str], destination: Path, baseline: bool) -> float:
        start = time.monotonic()
        optimized_bc = destination.with_suffix(".optimized.bc")
        object_path = destination.with_suffix(".o")
        pipeline = ["-O3"] if baseline else list(sequence)
        run_command([self.opt, str(input_bc), *pipeline, "-o", str(optimized_bc)])
        run_command([self.llc, "-O3", "-filetype=obj", str(optimized_bc), "-o", str(object_path)])
        run_command([self.clang, str(object_path), "-lm", "-o", str(destination)])
        return time.monotonic() - start


def measure_binary(binary: Path, warmup_runs: int, formal_runs: int, cpu: int) -> tuple[list[float], bool]:
    for _ in range(warmup_runs):
        run_command([str(binary)], timeout=600, cpu=cpu)
    values = []
    for _ in range(formal_runs):
        completed = run_command([str(binary)], timeout=600, cpu=cpu)
        try:
            value = float(completed.stdout.strip())
        except ValueError as error:
            raise RuntimeError(f"Invalid PolyBench timer output: {completed.stdout!r}") from error
        if not math.isfinite(value) or value <= 0:
            raise RuntimeError(f"Invalid runtime value: {value}")
        values.append(value)
    return values, True


def correctness_hash(binary: Path, cpu: int) -> str:
    completed = run_command([str(binary)], timeout=600, cpu=cpu)
    return hashlib.sha256(completed.stderr.encode("utf-8")).hexdigest()


def evaluate_sequence(
    toolchain: Toolchain,
    input_bc: Path,
    sequence: Sequence[str],
    method: str,
    evaluation_index: int,
    kernel_dir: Path,
    measurement: Mapping[str, Any],
    program: str,
    baseline_median_seconds: float,
) -> dict[str, Any]:
    digest = sequence_hash(sequence)
    binary = kernel_dir / f"{method}_{evaluation_index:03d}_{digest[:12]}"
    record: dict[str, Any] = {
        "program": program,
        "method": method,
        "evaluation_index": evaluation_index,
        "sequence": list(sequence),
        "sequence_hash": digest,
        "compile_ok": False,
        "run_ok": False,
        "runtime_median_ms": None,
        "speedup_vs_o3": None,
    }
    try:
        record["compile_seconds"] = toolchain.build_executable(input_bc, sequence, binary, baseline=False)
        record["compile_ok"] = True
        runtimes, run_ok = measure_binary(
            binary,
            measurement["search_warmup_runs"],
            measurement["search_formal_runs"],
            measurement["cpu_affinity"],
        )
        record.update(
            {
                "run_ok": run_ok,
                "runtime_samples_seconds": runtimes,
                "runtime_median_ms": statistics.median(runtimes) * 1000,
                "speedup_vs_o3": baseline_median_seconds / statistics.median(runtimes),
            }
        )
    except Exception as error:
        record["error"] = {"type": type(error).__name__, "message": str(error)}
    return record


def search_kernel(
    toolchain: Toolchain,
    input_bc: Path,
    kernel_dir: Path,
    config: Mapping[str, Any],
    seed: int,
    checkpoint: callable,
    program: str,
    baseline_median_seconds: float,
) -> dict[str, Any]:
    search = config["search"]
    measurement = config["measurement"]
    actions = search["action_subset"]
    rng = random.Random(seed)
    records: list[dict[str, Any]] = []

    for index in range(1, search["random_sequence_count"] + 1):
        length = rng.randint(1, search["max_sequence_length"])
        sequence = [rng.choice(actions) for _ in range(length)]
        records.append(
            evaluate_sequence(toolchain, input_bc, sequence, "random", index, kernel_dir,
                              measurement, program, baseline_median_seconds)
        )
        checkpoint(records)

    prefix: list[str] = []
    greedy_records: list[dict[str, Any]] = []
    evaluation_index = 0
    for _ in range(search["max_sequence_length"]):
        candidates = rng.sample(actions, search["greedy_candidates_per_step"])
        step_records = []
        for action in candidates:
            evaluation_index += 1
            record = evaluate_sequence(
                toolchain, input_bc, [*prefix, action], "greedy", evaluation_index, kernel_dir,
                measurement, program, baseline_median_seconds
            )
            records.append(record)
            greedy_records.append(record)
            step_records.append(record)
            checkpoint(records)
        valid = [record for record in step_records if record["run_ok"]]
        if not valid:
            break
        prefix = min(valid, key=lambda record: record["runtime_median_ms"])["sequence"]

    method_results = {}
    for method in ("random", "greedy"):
        method_records = [record for record in records if record["method"] == method]
        valid = [record for record in method_records if record["run_ok"]]
        method_results[method] = {
            "evaluations": len(method_records),
            "compile_failures": sum(not record["compile_ok"] for record in method_records),
            "run_failures": sum(record["compile_ok"] and not record["run_ok"] for record in method_records),
            "best_search_record": min(valid, key=lambda record: record["runtime_median_ms"]) if valid else None,
        }
    return {"evaluations": records, "methods": method_results}


def confirm_sequence(
    toolchain: Toolchain,
    input_bc: Path,
    sequence: Sequence[str],
    name: str,
    kernel_dir: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    measurement = config["measurement"]
    binary = kernel_dir / f"confirm_{name}"
    toolchain.build_executable(input_bc, sequence, binary, baseline=False)
    for _ in range(measurement["confirmation_warmup_runs"]):
        run_command([str(binary)], cpu=measurement["cpu_affinity"])
    blocks = []
    for _ in range(measurement["confirmation_blocks"]):
        values, _ = measure_binary(
            binary, 0, measurement["confirmation_runs_per_block"], measurement["cpu_affinity"]
        )
        blocks.append(values)
    return {"sequence": list(sequence), "sequence_hash": sequence_hash(sequence), "blocks": blocks}


def git_metadata(repo_root: Path) -> dict[str, Any]:
    def git(*args: str) -> str:
        return run_command(["git", *args], timeout=60).stdout.strip()

    return {
        "commit": git("-C", str(repo_root), "rev-parse", "HEAD"),
        "branch": git("-C", str(repo_root), "branch", "--show-current"),
        "status_short": git("-C", str(repo_root), "status", "--short").splitlines(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--polybench-root", type=Path, required=True)
    parser.add_argument("--polybench-archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    config_path = args.config.resolve()
    polybench_root = args.polybench_root.resolve()
    archive = args.polybench_archive.resolve()
    output_dir = args.output_dir.resolve()
    repo_root = Path(__file__).resolve().parents[1]
    config = load_config(config_path)

    output_dir.mkdir(parents=True, exist_ok=False)
    report_path = output_dir / "report.json"
    work_dir = output_dir / "work"
    work_dir.mkdir()
    report: dict[str, Any] = {
        "experiment_name": config["experiment_name"],
        "status": "RUNNING",
        "decision": "FAIL",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path.relative_to(repo_root)),
        "config_sha256": sha256_file(config_path),
        "polybench_archive_sha256": sha256_file(archive),
        "git": git_metadata(repo_root),
        "kernels": [],
    }
    atomic_write_json(report_path, report)

    try:
        if report["polybench_archive_sha256"] != config["dataset"]["archive_sha256"]:
            raise RuntimeError("PolyBench archive SHA256 mismatch")
        toolchain = Toolchain(config, polybench_root, work_dir)
        toolchain.validate_actions(config["search"]["action_subset"])
        report["dependencies"] = toolchain.versions

        all_qualified = True
        for kernel_index, kernel in enumerate(config["dataset"]["kernels"]):
            kernel_dir = work_dir / kernel["name"]
            kernel_dir.mkdir()
            input_bc = kernel_dir / "input.bc"
            toolchain.compile_unoptimized_bitcode(kernel, input_bc, dump=False)
            baseline_binary = kernel_dir / "o3"
            toolchain.build_executable(input_bc, [], baseline_binary, baseline=True)
            measurement = config["measurement"]
            for _ in range(measurement["qualification_warmup_runs"]):
                run_command([str(baseline_binary)], cpu=measurement["cpu_affinity"])
            qualification_blocks = []
            for _ in range(measurement["qualification_blocks"]):
                values, _ = measure_binary(
                    baseline_binary, 0, measurement["qualification_runs_per_block"], measurement["cpu_affinity"]
                )
                qualification_blocks.append(values)
            stats = [distribution_stats(block) for block in qualification_blocks]
            checks = qualification_checks(stats[0], stats[1], measurement)
            kernel_report: dict[str, Any] = {
                "program": kernel["name"],
                "source": kernel["source"],
                "dataset_macro": kernel["dataset_macro"],
                "measurement_qualification": {"blocks": qualification_blocks, "stats": stats, "checks": checks, "pass": all(checks.values())},
            }
            report["kernels"].append(kernel_report)
            atomic_write_json(report_path, report)
            if not all(checks.values()):
                all_qualified = False
                break

            def checkpoint(records: list[dict[str, Any]]) -> None:
                kernel_report["evaluations"] = records
                atomic_write_json(report_path, report)

            search_result = search_kernel(
                toolchain,
                input_bc,
                kernel_dir,
                config,
                config["search"]["seed"] + kernel_index,
                checkpoint,
                kernel["name"],
                statistics.median(value for block in qualification_blocks for value in block),
            )
            kernel_report.update(search_result)

            valid_best = [
                result["best_search_record"]
                for result in search_result["methods"].values()
                if result["best_search_record"] is not None
            ]
            if not valid_best:
                raise RuntimeError(f"No valid search result for {kernel['name']}")
            selected = min(valid_best, key=lambda record: record["runtime_median_ms"])
            confirmation = confirm_sequence(
                toolchain, input_bc, selected["sequence"], selected["method"], kernel_dir, config
            )

            for _ in range(measurement["confirmation_warmup_runs"]):
                run_command([str(baseline_binary)], cpu=measurement["cpu_affinity"])
            baseline_blocks = []
            for _ in range(measurement["confirmation_blocks"]):
                values, _ = measure_binary(
                    baseline_binary, 0, measurement["confirmation_runs_per_block"], measurement["cpu_affinity"]
                )
                baseline_blocks.append(values)
            baseline_values = [value for block in baseline_blocks for value in block]
            candidate_values = [value for block in confirmation["blocks"] for value in block]
            speedup = statistics.median(baseline_values) / statistics.median(candidate_values)
            ci = bootstrap_ratio_ci(
                baseline_values,
                candidate_values,
                config["metrics"]["bootstrap_resamples"],
                config["search"]["seed"] + 1000 + kernel_index,
            )

            dump_bc = kernel_dir / "dump_input.bc"
            toolchain.compile_unoptimized_bitcode(kernel, dump_bc, dump=True)
            o3_dump = kernel_dir / "o3_dump"
            candidate_dump = kernel_dir / "candidate_dump"
            toolchain.build_executable(dump_bc, [], o3_dump, baseline=True)
            toolchain.build_executable(dump_bc, selected["sequence"], candidate_dump, baseline=False)
            o3_hash = correctness_hash(o3_dump, measurement["cpu_affinity"])
            candidate_hash = correctness_hash(candidate_dump, measurement["cpu_affinity"])
            kernel_report["confirmation"] = {
                "selected_method": selected["method"],
                "selected_sequence": selected["sequence"],
                "selected_sequence_hash": selected["sequence_hash"],
                "o3_blocks": baseline_blocks,
                "candidate_blocks": confirmation["blocks"],
                "o3_median_seconds": statistics.median(baseline_values),
                "candidate_median_seconds": statistics.median(candidate_values),
                "speedup_vs_o3": speedup,
                "bootstrap_95_ci": list(ci),
                "stable_at_least_one_percent": ci[0] >= config["metrics"]["stable_kernel_ci_lower_bound"],
                "correctness": {"o3_output_sha256": o3_hash, "candidate_output_sha256": candidate_hash, "pass": o3_hash == candidate_hash},
            }
            atomic_write_json(report_path, report)

        if not all_qualified:
            report["failure_stage"] = "measurement_qualification"
        else:
            confirmations = [kernel["confirmation"] for kernel in report["kernels"]]
            speedups = [item["speedup_vs_o3"] for item in confirmations]
            geomean = math.exp(statistics.mean(math.log(value) for value in speedups))
            overall_ci = bootstrap_geomean_ci(
                speedups, config["metrics"]["bootstrap_resamples"], config["search"]["seed"] + 2000
            )
            threshold = config["metrics"]["classification_threshold"]
            improved = sum(value >= 1 + threshold for value in speedups)
            regressed = sum(value <= 1 - threshold for value in speedups)
            tied = len(speedups) - improved - regressed
            stable = sum(item["stable_at_least_one_percent"] for item in confirmations)
            correctness_failures = sum(not item["correctness"]["pass"] for item in confirmations)
            equal_budget = all(
                kernel["methods"]["random"]["evaluations"] == config["search"]["evaluations_per_method_per_kernel"]
                and kernel["methods"]["greedy"]["evaluations"] == config["search"]["evaluations_per_method_per_kernel"]
                for kernel in report["kernels"]
            )
            checks = {
                "measurement_qualification_all_kernels": True,
                "geometric_mean_speedup": geomean > 1.0,
                "bootstrap_ci_lower_bound": overall_ci[0] > 1.0,
                "minimum_stable_improved_kernel_fraction": stable / len(speedups) >= 0.25,
                "independent_confirmation": True,
                "correctness_failures": correctness_failures == 0,
                "equal_search_evaluation_budget": equal_budget,
            }
            report["summary"] = {
                "geometric_mean_speedup": geomean,
                "bootstrap_95_ci": list(overall_ci),
                "improved_kernels": improved,
                "regressed_kernels": regressed,
                "tied_kernels": tied,
                "stable_at_least_one_percent_kernels": stable,
                "correctness_failures": correctness_failures,
                "search_evaluations_per_method_per_kernel": config["search"]["evaluations_per_method_per_kernel"],
                "gate_checks": checks,
            }
            report["decision"] = "PASS" if all(checks.values()) else "FAIL"
        report["status"] = "COMPLETE"
    except Exception as error:
        report["status"] = "INVALID"
        report["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
    finally:
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_write_json(report_path, report)

    print(json.dumps({"status": report["status"], "decision": report["decision"], "failure_stage": report.get("failure_stage"), "error": report.get("error")}, indent=2))
    return 0 if report["status"] == "COMPLETE" and report["decision"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
