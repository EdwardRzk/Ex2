import json
import math
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.run_gate1_search_headroom import (
    confirm_sequence,
    bootstrap_geomean_ci,
    bootstrap_ratio_ci,
    distribution_stats,
    geometric_mean,
    evaluate_sequence,
    is_confirmed_improvement,
    load_config,
    measure_paired_sandwiches,
    measurement_qualification_report,
    paired_bootstrap_ci,
    paired_speedup,
    qualification_checks,
    sequence_hash,
    search_kernel,
    validate_paired_config,
)


def test_sequence_hash_is_stable_and_order_sensitive() -> None:
    assert sequence_hash(["-mem2reg", "-gvn"]) == sequence_hash(["-mem2reg", "-gvn"])
    assert sequence_hash(["-mem2reg", "-gvn"]) != sequence_hash(["-gvn", "-mem2reg"])


def test_distribution_stats_and_qualification_checks() -> None:
    values_a = [0.1000, 0.1001, 0.0999, 0.1000]
    values_b = [0.1001, 0.1000, 0.0999, 0.1000]
    stats_a = distribution_stats(values_a)
    stats_b = distribution_stats(values_b)
    measurement = {
        "minimum_runtime_seconds": 0.02,
        "maximum_cv": 0.01,
        "maximum_relative_mad": 0.005,
        "maximum_block_median_drift": 0.01,
    }
    assert all(qualification_checks(stats_a, stats_b, measurement).values())


def test_bootstrap_intervals_detect_clear_improvement() -> None:
    baseline = [1.00, 1.01, 0.99, 1.00, 1.00]
    candidate = [0.90, 0.91, 0.89, 0.90, 0.90]
    ratio_ci = bootstrap_ratio_ci(baseline, candidate, resamples=1000, seed=7)
    assert ratio_ci[0] > 1.0

    geomean_ci = bootstrap_geomean_ci([1.05, 1.10, 1.08], resamples=1000, seed=7)
    assert geomean_ci[0] > 1.0


def test_high_noise_qualification_is_diagnostic_and_search_is_allowed() -> None:
    measurement = {
        "minimum_runtime_seconds": 0.02,
        "maximum_cv": 0.01,
        "maximum_relative_mad": 0.005,
        "maximum_block_median_drift": 0.01,
        "maximum_paired_ratio_cv": 0.01,
        "maximum_paired_ratio_relative_mad": 0.005,
    }
    baseline_blocks = [
        [0.050, 0.060, 0.070, 0.055],
        [0.052, 0.062, 0.072, 0.057],
    ]
    ratios = [0.97, 1.03, 0.98, 1.02]
    paired = {
        "sandwiches": [
            {
                "baseline_before_seconds": 1.0,
                "candidate_seconds": 1.0 / ratio,
                "baseline_after_seconds": 1.0,
                "local_baseline_seconds": 1.0,
                "paired_speedup": ratio,
            }
            for ratio in ratios
        ],
        "ratios": ratios,
        "aggregate_paired_speedup": geometric_mean(ratios),
    }

    report = measurement_qualification_report(
        baseline_blocks, paired, measurement
    )

    assert report["search_allowed"] is True
    assert report["hard_stop"] is False
    baseline = report["absolute_baseline_diagnostics"]
    assert baseline["diagnostic_only"] is True
    assert baseline["hard_stop"] is False
    assert baseline["pass"] is False
    assert baseline["high_noise"] is True
    assert baseline["checks"]["maximum_cv"] is False
    assert baseline["checks"]["maximum_relative_mad"] is False
    paired_report = report["paired_ratio"]
    assert paired_report["diagnostic_only"] is True
    assert paired_report["hard_stop"] is False
    assert paired_report["pass"] is False
    assert paired_report["high_noise"] is True
    assert paired_report["checks"] == {
        "maximum_paired_ratio_cv": False,
        "maximum_paired_ratio_relative_mad": False,
    }
    assert paired_report["stats"]["cv"] > measurement["maximum_paired_ratio_cv"]
    assert (
        paired_report["stats"]["relative_mad"]
        > measurement["maximum_paired_ratio_relative_mad"]
    )


def test_confirmation_requires_speedup_and_ci_lower_bound_above_one() -> None:
    assert is_confirmed_improvement(1.02, (1.001, 1.04))
    assert not is_confirmed_improvement(1.02, (1.0, 1.04))
    assert not is_confirmed_improvement(1.02, (0.999, 1.04))
    assert not is_confirmed_improvement(1.0, (1.001, 1.04))


def test_load_config_rejects_unequal_budget(tmp_path: Path) -> None:
    config = json.loads(Path("configs/gate1_search_headroom_v1.json").read_text(encoding="utf-8"))
    config["search"]["random_sequence_count"] -= 1
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="Random sequence count"):
        load_config(path)


def test_gate1_config_has_no_cbench_and_frozen_speedup_gate() -> None:
    config = load_config(Path("configs/gate1_search_headroom_v1.json"))
    assert config["dataset"]["cbench_accessed"] is False
    assert all("cbench" not in kernel["source"].lower() for kernel in config["dataset"]["kernels"])
    assert math.isclose(
        config["pass_fail_gate"]["geometric_mean_speedup_strictly_greater_than"], 1.0
    )


def test_paired_speedup_uses_local_geometric_baseline() -> None:
    expected = math.sqrt(100.0 * 102.0) / 90.0
    assert math.isclose(paired_speedup(100.0, 90.0, 102.0), expected)


def test_paired_speedup_cancels_common_drift() -> None:
    original = paired_speedup(100.0, 95.0, 100.0)
    slowed = paired_speedup(120.0, 114.0, 120.0)
    assert math.isclose(original, slowed)


def test_paired_report_preserves_all_bcb_samples_and_ratios() -> None:
    values = iter([100.0, 90.0, 102.0, 120.0, 108.0, 122.4])

    def completed(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess([], 0, stdout=f"{next(values)}\n", stderr="")

    with patch(
        "scripts.run_gate1_search_headroom.run_command",
        side_effect=completed,
    ) as mocked:
        result = measure_paired_sandwiches(
            Path("o3"),
            Path("candidate"),
            repetitions=2,
            cpu=24,
        )

    first = result["sandwiches"][0]
    assert first == {
        "baseline_before_seconds": 100.0,
        "candidate_seconds": 90.0,
        "baseline_after_seconds": 102.0,
        "local_baseline_seconds": math.sqrt(100.0 * 102.0),
        "paired_speedup": math.sqrt(100.0 * 102.0) / 90.0,
    }
    expected_ratios = [
        math.sqrt(100.0 * 102.0) / 90.0,
        math.sqrt(120.0 * 122.4) / 108.0,
    ]
    assert result["ratios"] == expected_ratios
    assert math.isclose(
        result["aggregate_paired_speedup"],
        geometric_mean(expected_ratios),
    )
    assert [call.args[0][0] for call in mocked.call_args_list] == [
        "o3", "candidate", "o3", "o3", "candidate", "o3"
    ]


def test_v2_config_freezes_paired_budget_without_changing_candidate_budget() -> None:
    config = load_config(Path("configs/gate1_search_headroom_v2.json"))
    original = load_config(Path("configs/gate1_search_headroom_v1.json"))
    validate_paired_config(config)
    assert config["search"]["random_sequence_count"] == 128
    assert config["search"]["evaluations_per_method_per_kernel"] == 128
    assert config["search"] == original["search"]
    assert config["dataset"] == original["dataset"]
    assert config["toolchain"] == original["toolchain"]
    assert config["pass_fail_gate"] == original["pass_fail_gate"]

    assert config["measurement"]["paired"] == {
        "search_initial_repetitions": 2,
        "search_top_k_per_method": 8,
        "search_additional_repetitions": 3,
        "greedy_top_k_per_step": 2,
        "qualification_repetitions": 10,
        "confirmation_repetitions": 10,
    }

def paired_record(
    method: str,
    evaluation_index: int,
    sequence: list[str],
    score: float,
) -> dict[str, object]:
    return {
        "program": "kernel",
        "method": method,
        "evaluation_index": evaluation_index,
        "sequence": sequence,
        "sequence_hash": f"hash-{method}-{evaluation_index}",
        "compile_ok": True,
        "run_ok": True,
        "measurement_stage": "initial",
        "paired_measurement": {
            "sandwiches": [],
            "ratios": [score],
            "aggregate_paired_speedup": score,
        },
        "paired_speedup": score,
    }


def test_search_method_winners_use_only_refined_paired_scores() -> None:
    config = load_config(Path("configs/gate1_search_headroom_v2.json"))
    config["search"].update(
        {
            "random_sequence_count": 2,
            "evaluations_per_method_per_kernel": 2,
            "max_sequence_length": 1,
            "greedy_candidates_per_step": 2,
        }
    )
    config["measurement"]["paired"].update(
        {
            "search_top_k_per_method": 1,
            "greedy_top_k_per_step": 1,
        }
    )
    scores = iter([1.50, 1.10, 1.20, 1.05])

    def evaluate(
        toolchain: object,
        input_bc: Path,
        sequence: list[str],
        method: str,
        evaluation_index: int,
        kernel_dir: Path,
        measurement: dict[str, object],
        program: str,
        baseline_binary: Path,
    ) -> dict[str, object]:
        return paired_record(method, evaluation_index, sequence, next(scores))

    def refine(
        record: dict[str, object],
        baseline_binary: Path,
        kernel_dir: Path,
        measurement: dict[str, object],
    ) -> None:
        record["measurement_stage"] = "refined"
        record["paired_speedup"] = float(record["paired_speedup"]) - 0.40

    with (
        patch("scripts.run_gate1_search_headroom.evaluate_sequence", side_effect=evaluate),
        patch("scripts.run_gate1_search_headroom.refine_record", side_effect=refine),
    ):
        result = search_kernel(
            object(),
            Path("input.bc"),
            Path("."),
            config,
            seed=41,
            checkpoint=lambda records: None,
            program="kernel",
            baseline_binary=Path("o3"),
        )

    assert result["methods"]["random"]["best_search_record"]["paired_speedup"] == 1.10
    assert math.isclose(
        result["methods"]["greedy"]["best_search_record"]["paired_speedup"], 0.80
    )
    assert all(
        result["methods"][method]["best_search_record"]["measurement_stage"] == "refined"
        for method in ("random", "greedy")
    )


def test_confirmation_uses_new_binary_and_independent_paired_measurement() -> None:
    class FakeToolchain:
        def build_executable(
            self,
            input_bc: Path,
            sequence: list[str],
            destination: Path,
            baseline: bool,
        ) -> float:
            assert destination == Path("work/confirm_random")
            assert not baseline
            return 0.1

    config = load_config(Path("configs/gate1_search_headroom_v2.json"))
    paired_result = {
        "sandwiches": [
            {
                "baseline_before_seconds": 100.0,
                "candidate_seconds": 95.0,
                "baseline_after_seconds": 100.0,
                "local_baseline_seconds": 100.0,
                "paired_speedup": 100.0 / 95.0,
            }
        ],
        "ratios": [100.0 / 95.0],
        "aggregate_paired_speedup": 100.0 / 95.0,
    }
    with (
        patch(
            "scripts.run_gate1_search_headroom.measure_binary",
            return_value=([1.0], True),
        ) as warmup,
        patch(
            "scripts.run_gate1_search_headroom.measure_paired_sandwiches",
            return_value=paired_result,
        ) as paired,
    ):
        result = confirm_sequence(
            FakeToolchain(),
            Path("input.bc"),
            ["-dse"],
            "random",
            Path("work"),
            config,
            Path("o3"),
        )

    assert warmup.call_count == 2
    paired.assert_called_once_with(
        Path("o3"),
        Path("work/confirm_random"),
        config["measurement"]["paired"]["confirmation_repetitions"],
        config["measurement"]["cpu_affinity"],
    )
    assert result["paired_measurement"] == paired_result


def test_paired_bootstrap_ci_resamples_ratios_not_absolute_runtimes() -> None:
    interval = paired_bootstrap_ci([1.04, 1.05, 1.06], resamples=1000, seed=41)
    assert interval[0] > 1.0


def test_2mm_dse_candidate_record_has_no_historical_baseline_speedup() -> None:
    class FakeToolchain:
        def build_executable(
            self,
            input_bc: Path,
            sequence: list[str],
            destination: Path,
            baseline: bool,
        ) -> float:
            assert sequence == ["-dse"]
            assert not baseline
            return 0.1

    config = load_config(Path("configs/gate1_search_headroom_v2.json"))
    paired_result = {
        "sandwiches": [
            {
                "baseline_before_seconds": 100.0,
                "candidate_seconds": 98.0,
                "baseline_after_seconds": 101.0,
                "local_baseline_seconds": math.sqrt(10100.0),
                "paired_speedup": math.sqrt(10100.0) / 98.0,
            },
            {
                "baseline_before_seconds": 120.0,
                "candidate_seconds": 117.6,
                "baseline_after_seconds": 121.2,
                "local_baseline_seconds": math.sqrt(14544.0),
                "paired_speedup": math.sqrt(14544.0) / 117.6,
            },
        ],
        "ratios": [
            math.sqrt(10100.0) / 98.0,
            math.sqrt(14544.0) / 117.6,
        ],
        "aggregate_paired_speedup": geometric_mean(
            [
                math.sqrt(10100.0) / 98.0,
                math.sqrt(14544.0) / 117.6,
            ]
        ),
    }
    with (
        patch(
            "scripts.run_gate1_search_headroom.measure_binary",
            return_value=([1.0], True),
        ),
        patch(
            "scripts.run_gate1_search_headroom.measure_paired_sandwiches",
            return_value=paired_result,
        ),
    ):
        record = evaluate_sequence(
            FakeToolchain(),
            Path("input.bc"),
            ["-dse"],
            "greedy",
            1,
            Path("work"),
            config["measurement"],
            "2mm",
            Path("work/o3"),
        )

    assert record["paired_speedup"] == paired_result["aggregate_paired_speedup"]
    assert record["paired_measurement"] == paired_result
    assert "speedup_vs_o3" not in record
    assert "runtime_median_ms" not in record
