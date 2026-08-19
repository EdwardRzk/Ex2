import copy


import pytest
from scripts.train_route_a_stage_b import (
    aggregate_model,
    checkpoint_payload,
    run_plan,
    seeded_config,
    validate_stage_b_config,
)


STAGE = {
    "final_seed_set": [1, 2, 3],
    "model_order": ["NVP", "MLP", "LSTM", "Transformer", "Mamba"],
    "stage_a_checkpoint_reuse": False,
}
NVP = {"seed": 0, "total_steps": 10000, "validation_cadence_steps": 100, "nvp_target_temperature": 0.05}
CONTROLLED = {
    "training": {"seed": 0, "total_steps": 10000, "checkpoint_evaluation_cadence_steps": 100},
    "target_and_objective": {"target_temperature": 0.05},
    "candidate_representation": {"K": 50, "padded_length": 20, "pad_token_id": 124},
    "models": {"MLP": {"d_model": 64}},
}


def test_stage_b_plan_uses_every_model_with_exact_new_seeds() -> None:
    validate_stage_b_config(STAGE, NVP, CONTROLLED)
    plan = run_plan(STAGE)
    assert len(plan) == 15
    assert {seed for _, seed in plan} == {1, 2, 3}
    assert all(seed != 0 for _, seed in plan)
    assert [item for item in plan if item[0] == "Mamba"] == [("Mamba", 1), ("Mamba", 2), ("Mamba", 3)]


def test_seeded_config_only_changes_the_seed_and_does_not_mutate_stage_a_config() -> None:
    before = copy.deepcopy(CONTROLLED)
    seeded = seeded_config(CONTROLLED, seed=2, kind="MLP", architecture="MLP")
    assert CONTROLLED == before
    assert seeded["training"]["seed"] == 2
    assert seeded["models"] == {"MLP": {"d_model": 64}}
    assert seeded_config(NVP, seed=3, kind="NVP")["seed"] == 3


def test_aggregate_uses_exact_three_seed_arithmetic_mean_and_preserves_per_dataset() -> None:
    reports = []
    for seed, value in ((1, 0.1), (2, 0.2), (3, 0.3)):
        reports.append({"architecture": "MLP", "seed": seed, "selected": {"ValidationFinalMeanOverOz": value, "per_dataset": {"a": value, "b": value + 0.1}, "policy45_regret_mean_bytes": float(seed), "policy45_regret_median_bytes": 0.0, "positive_program_count_vs_Oz": seed, "N_total": 4488, "N_primary_valid": 4488, "N_failed_or_invalid": 0}})
    aggregate = aggregate_model(reports, oracle=0.5)
    assert aggregate["ValidationFinalMeanOverOz_3seed"] == pytest.approx(0.2)
    assert aggregate["per_dataset_3seed"] == pytest.approx({"a": 0.2, "b": 0.3})
    assert aggregate["oracle_opportunity_recovered"] == pytest.approx(0.4)


def test_stage_b_checkpoint_metadata_never_claims_stage_a_reuse() -> None:
    payload = checkpoint_payload(__import__("torch").nn.Linear(1, 1), name="MLP", seed=1, model_config={}, step=100, metrics={})
    assert payload["seed"] == 1
    assert payload["stage_a_checkpoint_reused"] is False
