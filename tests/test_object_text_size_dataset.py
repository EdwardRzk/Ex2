import json
from pathlib import Path

import pytest

from scripts.generate_object_text_size_dataset import (
    evaluate_result,
    load_config,
    object_text_size_oz_reward,
    select_program_splits,
    validate_record,
)


def test_select_program_splits_is_deterministic_and_disjoint() -> None:
    uris = [f"benchmark://jotaibench-v0/{index}" for index in range(20)]
    first = select_program_splits(
        uris, train_programs=8, dev_programs=2, seed=41
    )
    second = select_program_splits(
        reversed(uris), train_programs=8, dev_programs=2, seed=41
    )

    assert first == second
    assert len(first["train"]) == 8
    assert len(first["dev"]) == 2
    assert not (set(first["train"]) & set(first["dev"]))


def test_select_program_splits_excludes_frozen_programs() -> None:
    uris = [f"benchmark://jotaibench-v0/{index}" for index in range(20)]
    excluded = set(uris[:10])
    splits = select_program_splits(
        uris, train_programs=8, dev_programs=2, seed=42, excluded_uris=excluded
    )
    assert not ((set(splits["train"]) | set(splits["dev"])) & excluded)


def test_object_text_size_oz_reward_matches_compiler_gym_definition() -> None:
    assert object_text_size_oz_reward(120, 100, 90) == pytest.approx(1.5)
    assert object_text_size_oz_reward(100, 100, 90) == pytest.approx(10.0)


def test_validate_record_requires_state_for_every_prefix() -> None:
    record = {
        "sequence_length": 2,
        "action_indices": [1, 2],
        "action_names": ["-a", "-b"],
        "states": [[1, 2], [2, 3], [3, 4]],
        "final_object_text_size_bytes": 10,
    }
    validate_record(record, feature_dimension=2)

    record["states"] = record["states"][:-1]
    with pytest.raises(ValueError, match="states must contain"):
        validate_record(record, feature_dimension=2)


def test_frozen_config_and_formal_result_gate() -> None:
    config_path = Path("configs/object_text_size_dataset_v0_seed41.json")
    config = load_config(config_path)
    assert config["trajectory"]["maximum_sequence_length"] == 32
    assert config["dataset"]["final_test_accessed"] is False

    splits = {
        "train": [f"train-{index}" for index in range(900)],
        "dev": [f"dev-{index}" for index in range(100)],
    }
    counts = {
        "trajectory_count": 16000,
        "program_count": 1000,
        "invalid_episode_count": 0,
    }
    assert all(
        evaluate_result(
            report_counts=counts,
            splits=splits,
            config=config,
            observed_minimum_length=1,
            observed_maximum_length=32,
        ).values()
    )


def test_load_config_rejects_non_v0_sequence_length(tmp_path: Path) -> None:
    config = json.loads(
        Path("configs/object_text_size_dataset_v0_seed41.json").read_text(
            encoding="utf-8"
        )
    )
    config["trajectory"]["maximum_sequence_length"] = 64
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="must remain 32"):
        load_config(path)
