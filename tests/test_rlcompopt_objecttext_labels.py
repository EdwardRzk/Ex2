from scripts.generate_rlcompopt_objecttext_labels import K, load_candidates, program_summary, scalar_int


def test_scalar_int_serializes_shape_one_values() -> None:
    assert scalar_int([123]) == 123


def test_load_candidates_requires_complete_k50(tmp_path) -> None:
    path = tmp_path / "candidates.txt"
    path.write_text("(1, 2)\n", encoding="utf-8")
    try:
        load_candidates(path)
    except ValueError as error:
        assert "exactly" in str(error)
    else:
        raise AssertionError("Expected incomplete candidate set to fail")


def test_program_summary_requires_all_k_candidates() -> None:
    valid = {"training_target_validity": "valid_completed_candidate_rollout", "failure_reason": None}
    records = [valid] * (K - 1) + [{"training_target_validity": "invalid_incomplete_candidate_rollout", "failure_reason": "done"}]
    summary = program_summary("benchmark://x-v0/a", "x-v0", 100, 90, records)
    assert summary["program_training_target_validity"] == "invalid_incomplete_K50_target"
    assert summary["oracle_K50_validity"] == "invalid_incomplete_K50"
    assert summary["candidate_failure_count_by_reason"] == {"done": 1}
