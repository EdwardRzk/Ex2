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


def test_program_shard_is_atomically_committed_after_complete_payload(tmp_path) -> None:
    import gzip
    import json

    from scripts.generate_rlcompopt_objecttext_labels import _shard_path, _write_program_shard

    path = _shard_path(tmp_path, "train", 7)
    records = [{"candidate_id": index} for index in range(K)]
    summary = {"program_id": "benchmark://x-v0/a", "complete_candidate_count": K}
    _write_program_shard(path, records, summary)

    assert path.exists()
    assert not path.with_suffix(path.suffix + ".tmp").exists()
    with gzip.open(path, "rt", encoding="utf-8") as file:
        payload = json.load(file)
    assert payload["program_summary"] == summary
    assert len(payload["records"]) == K
