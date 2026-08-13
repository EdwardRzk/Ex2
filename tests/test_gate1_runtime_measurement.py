import math
import subprocess
from pathlib import Path
from unittest.mock import patch

from scripts.qualify_gate1_runtime_measurement import dataset_candidates, sanity_check
from scripts.run_gate1_search_headroom import (
    distribution_stats,
    measure_binary,
    run_command,
)


def test_distribution_stats_includes_standard_deviation() -> None:
    values = [0.1000, 0.1001, 0.0999, 0.1000]
    stats = distribution_stats(values)
    assert math.isclose(stats["std_seconds"], 8.16496580927726e-05)
    assert math.isclose(stats["cv"], stats["std_seconds"] / stats["mean_seconds"])


def test_dataset_candidates_only_scale_up() -> None:
    assert dataset_candidates("MEDIUM_DATASET") == [
        "MEDIUM_DATASET",
        "LARGE_DATASET",
        "EXTRALARGE_DATASET",
    ]
    assert dataset_candidates("LARGE_DATASET") == [
        "LARGE_DATASET",
        "EXTRALARGE_DATASET",
    ]


def test_identical_baseline_sanity_covers_one() -> None:
    group_a = [0.1000, 0.1001, 0.0999, 0.1000, 0.1001]
    group_b = [0.1001, 0.1000, 0.0999, 0.1000, 0.1000]
    result = sanity_check(
        group_a,
        group_b,
        {"maximum_block_median_drift": 0.01},
        resamples=1000,
        seed=41,
    )
    assert result["pass"]
    assert result["bootstrap_95_ci"][0] <= 1.0 <= result["bootstrap_95_ci"][1]


def test_identical_baseline_sanity_rejects_fake_improvement() -> None:
    group_a = [0.1050, 0.1051, 0.1049, 0.1050, 0.1051]
    group_b = [0.1001, 0.1000, 0.0999, 0.1000, 0.1000]
    result = sanity_check(
        group_a,
        group_b,
        {"maximum_block_median_drift": 0.01},
        resamples=1000,
        seed=41,
    )
    assert not result["pass"]
    assert result["bootstrap_95_ci"][0] > 1.0


def test_run_command_uses_taskset_for_child_only() -> None:
    completed = subprocess.CompletedProcess([], 0, stdout="", stderr="")
    with patch(
        "scripts.run_gate1_search_headroom.subprocess.run",
        return_value=completed,
    ) as mocked:
        assert run_command(["binary", "--flag"], cpu=24) is completed
    assert mocked.call_args.args[0] == [
        "taskset", "-c", "24", "binary", "--flag"
    ]
    assert "preexec_fn" not in mocked.call_args.kwargs


def test_measure_binary_respects_run_and_time_warmup_minima() -> None:
    completed = subprocess.CompletedProcess(
        [],
        0,
        stdout="0.1\n",
        stderr="",
    )
    monotonic_values = iter([0.0, 0.5, 1.0, 1.5, 2.0])
    with (
        patch(
            "scripts.run_gate1_search_headroom.run_command",
            return_value=completed,
        ) as mocked,
        patch(
            "scripts.run_gate1_search_headroom.time.monotonic",
            side_effect=monotonic_values,
        ),
    ):
        values, run_ok = measure_binary(Path("binary"), 2, 1, 24, 1.5)
    assert run_ok and values == [0.1]
    assert mocked.call_count == 5
