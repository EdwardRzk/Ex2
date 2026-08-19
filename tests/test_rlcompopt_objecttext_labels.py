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


def test_resume_requires_exact_frozen_config_and_no_prior_report(tmp_path) -> None:
    from scripts.generate_rlcompopt_objecttext_labels import prepare_output_directory

    output_dir = tmp_path / "experiment"
    frozen_config = {"candidate_count": K, "workers": 12}
    prepare_output_directory(output_dir, frozen_config, resume=False)
    prepare_output_directory(output_dir, frozen_config, resume=True)

    try:
        prepare_output_directory(output_dir, {"candidate_count": K, "workers": 8}, resume=True)
    except ValueError as error:
        assert "exactly match" in str(error)
    else:
        raise AssertionError("Expected mismatched resume config to fail")

    (output_dir / "experiment_report.json").write_text("{}\n", encoding="utf-8")
    try:
        prepare_output_directory(output_dir, frozen_config, resume=True)
    except FileExistsError as error:
        assert "completed or failed" in str(error)
    else:
        raise AssertionError("Expected reported experiment resume to fail")


def test_completed_shard_is_counted_when_resuming(tmp_path) -> None:
    from collections import Counter

    from scripts.generate_rlcompopt_objecttext_labels import (
        _accumulate_program_result,
        _load_program_shard,
        _write_program_shard,
    )

    path = tmp_path / "complete.json.gz"
    program_id = "benchmark://x-v0/a"
    records = [
        {"training_target_validity": "valid_completed_candidate_rollout", "failure_reason": None}
        for _ in range(K)
    ]
    summary = {"program_id": program_id, "program_training_target_validity": "valid_complete_K50"}
    _write_program_shard(path, records, summary)
    loaded_records, loaded_summary = _load_program_shard(path, program_id)
    counts = Counter()
    _accumulate_program_result(counts, loaded_records, loaded_summary)
    assert counts == {"programs_total": 1, "programs_complete_K50": 1}
