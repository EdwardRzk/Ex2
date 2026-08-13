#!/usr/bin/env python3
"""Run the frozen ACPO-style per-loop unroll headroom reproduction."""

from __future__ import annotations

import argparse
import math
import os
import re
import shutil
import statistics
import subprocess
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from scripts.run_gate1_search_headroom import (
        atomic_write_json,
        bootstrap_geomean_ci,
        correctness_hash,
        geometric_mean,
        measure_paired_sandwiches,
        paired_bootstrap_ci,
        sha256_file,
    )
except ModuleNotFoundError:
    from run_gate1_search_headroom import (  # type: ignore[no-redef]
        atomic_write_json,
        bootstrap_geomean_ci,
        correctness_hash,
        geometric_mean,
        measure_paired_sandwiches,
        paired_bootstrap_ci,
        sha256_file,
    )

FROZEN_UNROLL_COUNTS = [0, 2, 4, 8, 16, 32, 64]
UNROLL_COUNT_PATTERN = re.compile(r"UnrollCount:\s*['\"]?(\d+)['\"]?")


def load_config(path: Path) -> dict[str, Any]:
    import json

    config = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "experiment_name", "dataset", "sources", "toolchain", "tuning",
        "measurement", "pass_fail_gate",
    }
    missing = required - config.keys()
    if missing:
        raise ValueError(f"Config is missing required fields: {sorted(missing)}")
    programs = config["dataset"]["programs"]
    if len(programs) != 24 or len({item["name"] for item in programs}) != 24:
        raise ValueError("The frozen ACPO reproduction requires 24 unique programs")
    if config["tuning"]["values"] != FROZEN_UNROLL_COUNTS:
        raise ValueError(f"UnrollCount values must equal {FROZEN_UNROLL_COUNTS}")
    if config["tuning"]["configurations_per_program"] < 1:
        raise ValueError("configurations_per_program must be positive")
    if config["measurement"]["search_paired_repetitions"] < 1:
        raise ValueError("Search requires paired B-C-B measurement")
    if config["measurement"]["confirmation_paired_repetitions"] < 2:
        raise ValueError("Confirmation requires at least two B-C-B measurements")
    return config


def render_search_space(values: Sequence[int]) -> str:
    if list(values) != FROZEN_UNROLL_COUNTS:
        raise ValueError(f"Search space must equal {FROZEN_UNROLL_COUNTS}")
    rendered_values = ", ".join(str(value) for value in values)
    return (
        "# ACPO Loop-Unroll classes; factor 1 (peeling) is excluded.\n"
        "CodeRegion:\n"
        "  CodeRegionType: loop\n"
        "  Pass: loop-unroll\n"
        "  Args:\n"
        "    UnrollCount:\n"
        f"      Value: [{rendered_values}]\n"
        "      Type: enum\n"
    )


def parse_unroll_counts(config_yaml: Path) -> list[int]:
    values = [
        int(match.group(1))
        for match in UNROLL_COUNT_PATTERN.finditer(
            config_yaml.read_text(encoding="utf-8")
        )
    ]
    if not values:
        raise ValueError(f"No UnrollCount decisions in {config_yaml}")
    invalid = sorted(set(values) - set(FROZEN_UNROLL_COUNTS))
    if invalid:
        raise ValueError(f"Config contains values outside frozen action set: {invalid}")
    return values


def paired_minimization_objective(paired_measurement: Mapping[str, Any]) -> float:
    speedup = float(paired_measurement["aggregate_paired_speedup"])
    if speedup <= 0 or not math.isfinite(speedup):
        raise ValueError("Paired speedup must be positive and finite")
    return 1.0 / speedup


def is_stable_improvement(speedup: float, interval: Sequence[float]) -> bool:
    return speedup > 1.0 and interval[0] > 1.0


def find_winner_evaluation(
    evaluations: Sequence[Mapping[str, Any]], winner_sha256: str
) -> Mapping[str, Any]:
    matching = [
        item
        for item in evaluations
        if item["config"]["sha256"] == winner_sha256
    ]
    if len(matching) != 1:
        raise ValueError(
            "Final Autotuner winner must match exactly one evaluated configuration"
        )
    return matching[0]


def compile_command(
    clang: Path,
    polybench_root: Path,
    program: Mapping[str, str],
    dataset_macro: str,
    destination: Path,
    *,
    mode: str,
) -> list[str]:
    valid_modes = {
        "baseline", "candidate", "opportunities",
        "baseline_dump", "candidate_dump",
    }
    if mode not in valid_modes:
        raise ValueError(f"Unsupported compile mode: {mode}")
    source = polybench_root / program["source"]
    command = [
        str(clang),
        "-O3",
        "-g",
        f"-D{dataset_macro}",
        "-I",
        str(polybench_root / "utilities"),
        "-I",
        str(source.parent),
    ]
    if mode.endswith("dump"):
        command.append("-DPOLYBENCH_DUMP_ARRAYS")
    else:
        command.extend(["-DPOLYBENCH_TIME", "-DPOLYBENCH_NO_FLUSH_CACHE"])
    if mode == "opportunities":
        command.extend([
            "-fautotune-generate=Loop",
            "-mllvm",
            "-auto-tuning-pass-filter=loop-unroll",
        ])
    elif mode in {"candidate", "candidate_dump"}:
        command.append("-fautotune")
    command.extend([
        str(source),
        str(polybench_root / "utilities/polybench.c"),
        "-lm",
        "-o",
        str(destination),
    ])
    return command


def run_process(
    command: Sequence[str],
    *,
    env: Mapping[str, str] | None = None,
    timeout: int = 900,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=dict(env) if env is not None else None,
    )


def autotuner_command(
    python: Path, script: Path, action: str, args: Sequence[str] = ()
) -> list[str]:
    return [str(python), str(script), action, *args]


def warmup(binary: Path, runs: int, cpu: int) -> None:
    for _ in range(runs):
        subprocess.run(
            ["taskset", "-c", str(cpu), str(binary)],
            check=True,
            capture_output=True,
            text=True,
            timeout=900,
        )


def copy_and_describe_config(source: Path, destination: Path) -> dict[str, Any]:
    shutil.copy2(source, destination)
    values = parse_unroll_counts(destination)
    return {
        "path": str(destination),
        "sha256": sha256_file(destination),
        "decision_count": len(values),
        "unroll_count_histogram": {
            str(value): values.count(value) for value in FROZEN_UNROLL_COUNTS
        },
    }


def git_metadata(repo_root: Path) -> dict[str, Any]:
    def git(*args: str) -> str:
        return run_process(
            ["git", "-C", str(repo_root), *args], timeout=60
        ).stdout.strip()

    return {
        "commit": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "status_short": git("status", "--short").splitlines(),
    }


def toolchain_metadata(
    clang: Path, autotuner_python: Path, autotuner_script: Path
) -> dict[str, Any]:
    clang_version = run_process([str(clang), "--version"]).stdout.strip()
    autotuner_help = run_process(
        autotuner_command(autotuner_python, autotuner_script, "-h")
    ).stdout
    return {
        "clang_path": str(clang),
        "clang_version": clang_version,
        "clang_sha256": sha256_file(clang),
        "autotuner_python": str(autotuner_python),
        "autotuner_script": str(autotuner_script),
        "autotuner_available": (
            "minimize" in autotuner_help and "feedback" in autotuner_help
        ),
    }


def tuning_environment(data_dir: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment["AUTOTUNE_DATADIR"] = str(data_dir)
    environment["CONFIG_DB_DIR"] = str(data_dir)
    environment["PYTHONHASHSEED"] = "0"
    return environment


def run_program(
    program: Mapping[str, str],
    *,
    config: Mapping[str, Any],
    clang: Path,
    autotuner_python: Path,
    autotuner_script: Path,
    polybench_root: Path,
    program_dir: Path,
    report_program: dict[str, Any],
    checkpoint: Any,
) -> None:
    measurement = config["measurement"]
    tuning = config["tuning"]
    cpu = measurement["cpu_affinity"]
    data_dir = program_dir / "autotune_datadir"
    data_dir.mkdir()
    (data_dir / "opp").mkdir()
    search_configs = program_dir / "search_configs"
    search_configs.mkdir()
    environment = tuning_environment(data_dir)
    dataset_macro = config["dataset"]["dataset_macro"]

    opportunity_binary = program_dir / "opportunity_binary"
    run_process(
        compile_command(
            clang, polybench_root, program, dataset_macro,
            opportunity_binary, mode="opportunities",
        ),
        env=environment,
    )
    opportunity_files = sorted((data_dir / "opp").glob("*.yaml"))
    if not opportunity_files:
        raise RuntimeError(
            f"No Loop-Unroll opportunities generated for {program['name']}"
        )
    opportunity_archive = program_dir / "opportunities"
    shutil.copytree(data_dir / "opp", opportunity_archive)
    archived_opportunities = sorted(opportunity_archive.glob("*.yaml"))
    report_program["opportunities"] = {
        "files": [
            str(path.relative_to(program_dir)) for path in archived_opportunities
        ],
        "sha256": {
            path.name: sha256_file(path) for path in archived_opportunities
        },
    }

    search_space = program_dir / "search_space.yaml"
    search_space.write_text(
        render_search_space(tuning["values"]), encoding="utf-8"
    )
    source = (polybench_root / program["source"]).resolve()
    minimize_args = [
        "--trials", "1",
        "--search-space", str(search_space),
        "--file-name-filter", str(source),
        "--pass-filter", tuning["pass"],
        "--type-filter", tuning["code_region_type"],
        "--technique", tuning["search_technique"],
        "--deterministic", "True",
        "--seed", str(tuning["seed"]),
        "--use-optimal-configs", "none",
    ]
    if tuning["start_from_compiler_baseline"]:
        minimize_args.append("--use-baseline-config")
    run_process(
        autotuner_command(
            autotuner_python, autotuner_script, "minimize", minimize_args
        ),
        env=environment,
    )

    baseline = program_dir / "o3_baseline"
    run_process(
        compile_command(
            clang, polybench_root, program, dataset_macro,
            baseline, mode="baseline",
        ),
        env=environment,
    )
    candidate = program_dir / "search_candidate"
    evaluations: list[dict[str, Any]] = []
    report_program["evaluations"] = evaluations
    warmup(baseline, measurement["search_warmup_runs"], cpu)

    for index in range(1, tuning["configurations_per_program"] + 1):
        config_copy = search_configs / f"trial_{index:04d}.yaml"
        config_description = copy_and_describe_config(
            data_dir / "config.yaml", config_copy
        )
        config_description["path"] = str(config_copy.relative_to(program_dir))
        record: dict[str, Any] = {
            "evaluation_index": index,
            "compile_ok": False,
            "run_ok": False,
            "config": config_description,
        }
        try:
            run_process(
                compile_command(
                    clang, polybench_root, program, dataset_macro, candidate,
                    mode="candidate",
                ),
                env=environment,
            )
            record["compile_ok"] = True
            warmup(candidate, measurement["search_warmup_runs"], cpu)
            paired = measure_paired_sandwiches(
                baseline, candidate,
                measurement["search_paired_repetitions"], cpu,
            )
            record["run_ok"] = True
            record["paired_measurement"] = paired
            record["paired_speedup"] = paired["aggregate_paired_speedup"]
            record["minimization_objective"] = paired_minimization_objective(
                paired
            )
        except Exception as error:
            record["error"] = {
                "type": type(error).__name__,
                "message": str(error),
            }
            record["minimization_objective"] = 1.0e12
        evaluations.append(record)
        checkpoint()
        run_process(
            autotuner_command(
                autotuner_python, autotuner_script, "feedback",
                ["--trials", "1", str(record["minimization_objective"])],
            ),
            env=environment,
        )

    run_process(
        autotuner_command(autotuner_python, autotuner_script, "finalize"),
        env=environment,
    )
    final_config = program_dir / "winner.yaml"
    winner_description = copy_and_describe_config(
        data_dir / "config.yaml", final_config
    )
    winner_description["path"] = str(final_config.relative_to(program_dir))
    valid = [item for item in evaluations if item["run_ok"]]
    if not valid:
        raise RuntimeError(f"No valid candidate for {program['name']}")
    observed_best = max(valid, key=lambda item: item["paired_speedup"])
    winner_evaluation = find_winner_evaluation(
        evaluations, winner_description["sha256"]
    )
    if not winner_evaluation["run_ok"]:
        raise RuntimeError("Final Autotuner winner did not run successfully")
    report_program["search"] = {
        "evaluations": len(evaluations),
        "compile_failures": sum(
            not item["compile_ok"] for item in evaluations
        ),
        "run_failures": sum(
            item["compile_ok"] and not item["run_ok"] for item in evaluations
        ),
        "best_observed_speedup": observed_best["paired_speedup"],
        "best_observed_evaluation_index": observed_best["evaluation_index"],
        "winner_observed_speedup": winner_evaluation["paired_speedup"],
        "winner_evaluation_index": winner_evaluation["evaluation_index"],
        "winner": winner_description,
    }

    confirmation_binary = program_dir / "confirmation_candidate"
    run_process(
        compile_command(
            clang, polybench_root, program, dataset_macro, confirmation_binary,
            mode="candidate",
        ),
        env=environment,
    )
    warmup(baseline, measurement["confirmation_warmup_runs"], cpu)
    warmup(confirmation_binary, measurement["confirmation_warmup_runs"], cpu)
    confirmation = measure_paired_sandwiches(
        baseline, confirmation_binary,
        measurement["confirmation_paired_repetitions"], cpu,
    )
    speedup = confirmation["aggregate_paired_speedup"]
    interval = paired_bootstrap_ci(
        confirmation["ratios"],
        measurement["bootstrap_resamples"],
        tuning["seed"] + 1000,
    )

    baseline_dump = program_dir / "o3_dump"
    candidate_dump = program_dir / "candidate_dump"
    run_process(
        compile_command(
            clang, polybench_root, program, dataset_macro, baseline_dump,
            mode="baseline_dump",
        ),
        env=environment,
    )
    run_process(
        compile_command(
            clang, polybench_root, program, dataset_macro, candidate_dump,
            mode="candidate_dump",
        ),
        env=environment,
    )
    baseline_hash = correctness_hash(baseline_dump, cpu)
    candidate_hash = correctness_hash(candidate_dump, cpu)
    report_program["confirmation"] = {
        "search_observed_speedup": winner_evaluation["paired_speedup"],
        "independent_paired_measurement": confirmation,
        "independent_paired_speedup": speedup,
        "paired_95_ci": list(interval),
        "stable_improvement": is_stable_improvement(speedup, interval),
        "correctness": {
            "o3_output_sha256": baseline_hash,
            "candidate_output_sha256": candidate_hash,
            "pass": baseline_hash == candidate_hash,
        },
    }
    checkpoint()


def main() -> int:
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--polybench-root", type=Path, required=True)
    parser.add_argument("--polybench-archive", type=Path, required=True)
    parser.add_argument("--clang", type=Path, required=True)
    parser.add_argument("--autotuner-python", type=Path, required=True)
    parser.add_argument("--autotuner-script", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    config_path = args.config.resolve()
    config = load_config(config_path)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(config_path, output_dir / "config.json")
    report_path = output_dir / "experiment_report.json"
    report: dict[str, Any] = {
        "experiment_name": config["experiment_name"],
        "experiment_kind": config["experiment_kind"],
        "claim_scope": config["claim_scope"],
        "status": "RUNNING",
        "decision": "FAIL",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": sha256_file(config_path),
        "polybench_archive_sha256": sha256_file(
            args.polybench_archive.resolve()
        ),
        "git": git_metadata(repo_root),
        "programs": [],
    }
    atomic_write_json(report_path, report)

    try:
        if report["polybench_archive_sha256"] != config["dataset"]["archive_sha256"]:
            raise RuntimeError("PolyBench archive SHA256 mismatch")
        toolchain = toolchain_metadata(
            args.clang.resolve(),
            args.autotuner_python.absolute(),
            args.autotuner_script.resolve(),
        )
        if config["toolchain"]["clang_version"] not in toolchain["clang_version"]:
            raise RuntimeError("Clang version mismatch")
        expected_commit = config["sources"]["openeuler_llvm"]["commit"]
        if expected_commit not in toolchain["clang_version"]:
            raise RuntimeError("OpenEuler LLVM commit mismatch")
        if not toolchain["autotuner_available"]:
            raise RuntimeError("Official llvm-autotune CLI is unavailable")
        report["toolchain"] = toolchain

        for program in config["dataset"]["programs"]:
            program_dir = output_dir / "work" / program["name"]
            program_dir.mkdir(parents=True)
            program_report: dict[str, Any] = {
                "program": program["name"],
                "source": program["source"],
                "status": "RUNNING",
            }
            report["programs"].append(program_report)

            def checkpoint() -> None:
                atomic_write_json(report_path, report)

            run_program(
                program,
                config=config,
                clang=args.clang.resolve(),
                autotuner_python=args.autotuner_python.absolute(),
                autotuner_script=args.autotuner_script.resolve(),
                polybench_root=args.polybench_root.resolve(),
                program_dir=program_dir,
                report_program=program_report,
                checkpoint=checkpoint,
            )
            program_report["status"] = "COMPLETE"
            checkpoint()

        confirmations = [
            item["confirmation"] for item in report["programs"]
        ]
        speedups = [
            item["independent_paired_speedup"] for item in confirmations
        ]
        overall_interval = bootstrap_geomean_ci(
            speedups,
            config["measurement"]["bootstrap_resamples"],
            config["tuning"]["seed"] + 2000,
        )
        correctness_failures = sum(
            not item["correctness"]["pass"] for item in confirmations
        )
        checks = {
            "all_24_programs_completed": len(speedups) == 24,
            "geometric_mean_independent_speedup": geometric_mean(speedups) > 1.0,
            "bootstrap_95_percent_ci_lower_bound": overall_interval[0] > 1.0,
            "correctness_failures": correctness_failures == 0,
        }
        report["summary"] = {
            "geometric_mean_independent_speedup": geometric_mean(speedups),
            "median_independent_speedup": statistics.median(speedups),
            "bootstrap_95_ci": list(overall_interval),
            "stable_improved_programs": sum(
                item["stable_improvement"] for item in confirmations
            ),
            "correctness_failures": correctness_failures,
            "paper_reported_geomean_speedup": config["sources"]["paper"]["reported_polybench_geomean_speedup"],
            "gate_checks": checks,
        }
        report["decision"] = "PASS" if all(checks.values()) else "FAIL"
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

    print(json.dumps({
        "status": report["status"],
        "decision": report["decision"],
        "error": report.get("error"),
    }, indent=2))
    return 0 if report["status"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
