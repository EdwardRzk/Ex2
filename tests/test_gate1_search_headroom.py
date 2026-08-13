import json
import math
from pathlib import Path

import pytest

from scripts.run_gate1_search_headroom import (
    bootstrap_geomean_ci,
    bootstrap_ratio_ci,
    distribution_stats,
    load_config,
    qualification_checks,
    sequence_hash,
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
