import math
from pathlib import Path

import pytest

from scripts.run_acpo_loop_unroll_reproduction import (
    FROZEN_UNROLL_COUNTS,
    compile_command,
    copy_and_describe_config,
    find_winner_evaluation,
    is_stable_improvement,
    load_config,
    paired_minimization_objective,
    parse_unroll_counts,
    render_search_space,
)


def test_frozen_config_matches_acpo_problem_definition() -> None:
    config = load_config(Path("configs/acpo_loop_unroll_reproduction_v1.json"))
    assert len(config["dataset"]["programs"]) == 24
    assert config["dataset"]["cbench_accessed"] is False
    assert config["tuning"]["values"] == [0, 2, 4, 8, 16, 32, 64]
    assert config["tuning"]["configurations_per_program"] == 128
    assert config["toolchain"]["baseline"] == "clang -O3"
    assert "not bit-for-bit" in config["claim_scope"]


def test_search_space_is_exact_and_excludes_peeling() -> None:
    text = render_search_space(FROZEN_UNROLL_COUNTS)
    assert "Value: [0, 2, 4, 8, 16, 32, 64]" in text
    assert "factor 1 (peeling) is excluded" in text
    with pytest.raises(ValueError, match="Search space must equal"):
        render_search_space([0, 1, 2, 4, 8, 16, 32, 64])


def test_paired_feedback_is_reciprocal_speedup() -> None:
    paired = {"aggregate_paired_speedup": math.sqrt(100.0 * 102.0) / 90.0}
    assert math.isclose(
        paired_minimization_objective(paired),
        90.0 / math.sqrt(100.0 * 102.0),
    )


def test_compile_commands_keep_o3_and_use_only_per_loop_yaml(
    tmp_path: Path,
) -> None:
    program = {
        "name": "2mm",
        "source": "linear-algebra/kernels/2mm/2mm.c",
    }
    common = dict(
        clang=Path("clang"),
        polybench_root=Path("polybench"),
        program=program,
        dataset_macro="MEDIUM_DATASET",
        destination=tmp_path / "binary",
    )
    baseline = compile_command(**common, mode="baseline")
    candidate = compile_command(**common, mode="candidate")
    opportunities = compile_command(**common, mode="opportunities")

    assert "-O3" in baseline and "-fautotune" not in baseline
    assert "-O3" in candidate and "-fautotune" in candidate
    assert all("unroll-count" not in argument for argument in candidate)
    assert "-fautotune-generate=Loop" in opportunities
    assert "-auto-tuning-pass-filter=loop-unroll" in opportunities


def test_config_report_preserves_full_decisions_and_rejects_other_values(
    tmp_path: Path,
) -> None:
    source = tmp_path / "config.yaml"
    source.write_text(
        "!AutoTuning {Args: [{UnrollCount: '0'}]}\n"
        "--- !AutoTuning {Args: [{UnrollCount: '4'}]}\n"
        "--- !AutoTuning {Args: [{UnrollCount: '64'}]}\n",
        encoding="utf-8",
    )
    destination = tmp_path / "saved.yaml"
    description = copy_and_describe_config(source, destination)

    assert parse_unroll_counts(destination) == [0, 4, 64]
    assert description["decision_count"] == 3
    assert description["unroll_count_histogram"]["0"] == 1
    assert description["unroll_count_histogram"]["4"] == 1
    assert description["unroll_count_histogram"]["64"] == 1
    assert description["sha256"]

    source.write_text(
        "!AutoTuning {Args: [{UnrollCount: '1'}]}\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="outside frozen action set"):
        parse_unroll_counts(source)


def test_confirmation_requires_speedup_and_ci_lower_above_one() -> None:
    assert is_stable_improvement(1.02, (1.001, 1.04))
    assert not is_stable_improvement(1.02, (1.0, 1.04))
    assert not is_stable_improvement(1.0, (1.001, 1.04))


def test_final_winner_observed_score_is_bound_by_config_hash() -> None:
    evaluations = [
        {"evaluation_index": 1, "config": {"sha256": "winner"}, "paired_speedup": 1.02},
        {"evaluation_index": 2, "config": {"sha256": "noisy-best"}, "paired_speedup": 1.30},
    ]
    selected = find_winner_evaluation(evaluations, "winner")
    assert selected["evaluation_index"] == 1
    assert selected["paired_speedup"] == 1.02
