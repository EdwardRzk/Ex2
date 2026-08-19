from scripts.build_rlcompopt_nvp_targets import K, target_record


def test_objecttext_nvp_target_preserves_official_higher_is_better_direction() -> None:
    program_id = "benchmark://x-v0/a"
    records = [
        {
            "program_id": program_id,
            "candidate_id": index,
            "training_target_validity": "valid_completed_candidate_rollout",
            "best_object_text_size_bytes": 100 if index == 0 else 90 if index == 1 else 95,
            "ordered_pass_sequence": [index],
            "measurement_validity": "valid",
        }
        for index in range(K)
    ]
    summary = {
        "program_id": program_id,
        "dataset_id": "x-v0",
        "program_training_target_validity": "valid_complete_K50",
        "oz_object_text_size_bytes": 100,
        "initial_object_text_size_bytes": 110,
        "ratio_metric_validity": "valid_for_ObjectText_ratio_metric",
    }

    target = target_record(records, summary, "train")

    assert len(target["raw_candidate_value"]) == K
    assert len(target["normalized_target"]) == K
    assert target["raw_candidate_value"][1] > target["raw_candidate_value"][0]
    assert target["normalized_target"][1] > target["normalized_target"][0]
    assert abs(sum(target["normalized_target"]) - 1.0) < 1e-12


def test_incomplete_program_is_excluded() -> None:
    summary = {
        "program_id": "benchmark://x-v0/a",
        "program_training_target_validity": "invalid_incomplete_K50_target",
    }
    assert target_record([], summary, "train") is None
