import hashlib
import json

import pytest

from scripts.evaluate_route_a_final_objecttext import (
    aggregate_family,
    assert_split_integrity,
    validate_checkpoint_inventory,
    load_final_manifest,
    policy45,
    validate_config,
)


def _config() -> dict:
    return {
        "candidate_space": {"K": 50},
        "inference": {"learned_scored_pass_budget": 45, "sampling": False},
        "target": {"temperature": 0.05},
        "models": {"names": ["NVP", "MLP", "LSTM", "Transformer", "Mamba"], "seeds": [1, 2, 3], "stage_a_checkpoint_allowed": False, "final_checkpoint_selection": False},
        "comparison_families": {"H1": ["Mamba", "Oz"], "H2a": ["Mamba", "NVP"], "H2b": ["MLP", "LSTM", "Transformer", "Mamba"]},
        "program_manifest": {},
    }


def test_manifest_requires_exact_identity_count_and_dataset_membership(tmp_path) -> None:
    path = tmp_path / "final.json"
    payload = {"benchmarks": ["benchmark://x-v0/a", "benchmark://y-v0/b"]}
    path.write_text(json.dumps(payload), encoding="utf-8")
    cfg = _config()
    cfg["program_manifest"] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "expected_program_count": 2, "expected_dataset_counts": {"x-v0": 1, "y-v0": 1}}
    assert load_final_manifest(path, cfg) == payload["benchmarks"]
    cfg["program_manifest"]["expected_dataset_counts"] = {"x-v0": 2}
    with pytest.raises(ValueError, match="dataset membership"):
        load_final_manifest(path, cfg)


def test_frozen_config_rejects_stage_a_or_final_checkpoint_selection() -> None:
    cfg = _config()
    validate_config(cfg)
    cfg["models"]["stage_a_checkpoint_allowed"] = True
    with pytest.raises(ValueError, match="Stage-A"):
        validate_config(cfg)


def test_split_integrity_rejects_train_or_validation_leakage(tmp_path) -> None:
    train, validation = tmp_path / "train.json", tmp_path / "validation.json"
    train.write_text(json.dumps({"samples": [{"benchmark": "benchmark://x-v0/a"}]}), encoding="utf-8")
    validation.write_text(json.dumps({"samples": [{"benchmark": "benchmark://y-v0/b"}]}), encoding="utf-8")
    cfg = {"split_integrity": {"train_source": str(train), "validation_source": str(validation), "train_expected_count": 1, "validation_expected_count": 1}}
    assert_split_integrity(["benchmark://z-v0/c"], cfg)
    with pytest.raises(ValueError, match="leakage"):
        assert_split_integrity(["benchmark://x-v0/a"], cfg)


def test_actual_checkpoint_inventory_is_exact_stage_b_five_by_three() -> None:
    cfg = _config()
    validate_checkpoint_inventory(__import__("pathlib").Path("outputs/route_a_stage_b_v6"), cfg)


def test_policy45_matches_exact_prefix_budget_and_tie_break() -> None:
    scores = [0.0] * 50
    records = [{"prefix_object_text_size_bytes": [100 - index] * 20} for index in range(50)]
    assert policy45(scores, records) == 98  # Tie order is 0, 1, 2; budget is 20+20+5.
    with pytest.raises(ValueError, match="exactly 45"):
        policy45(scores, [{"prefix_object_text_size_bytes": []} for _ in range(50)])


def test_family_empty_cohort_is_undefined_and_not_dropped() -> None:
    programs = ["benchmark://x-v0/a", "benchmark://y-v0/b"]
    summaries = {program: {"dataset_id": program.split("://", 1)[1].split("/", 1)[0], "ratio_metric_validity": "valid_for_ObjectText_ratio_metric"} for program in programs}
    results = {}
    for method in ("Mamba", "NVP"):
        for seed in (1, 2, 3):
            results[(method, seed)] = {
                programs[0]: {"valid": True, "mean_over_oz": 0.1},
                programs[1]: {"valid": method != "NVP", "mean_over_oz": 0.2},
            }
    aggregate = aggregate_family("H2a", ["Mamba", "NVP"], ["x-v0", "y-v0"], programs, results, summaries)
    assert aggregate["per_dataset"]["x-v0"]["N_primary_valid"] == 1
    assert aggregate["per_dataset"]["y-v0"]["N_primary_valid"] == 0
    assert aggregate["per_dataset"]["y-v0"]["Mamba"]["three_seed_mean"] is None
    assert aggregate["dataset_macro"]["Mamba"]["three_seed_mean"] is None
