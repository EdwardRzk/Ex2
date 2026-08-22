import torch

from scripts.train_gated_calibrated_mambanvp import K, kl_final_to_nvp, kl_nvp_to_final, validate_config


def test_calibration_kls_are_zero_for_identical_logits():
    logits = torch.randn(3, K)
    assert torch.allclose(kl_final_to_nvp(logits, logits), torch.zeros(()))
    assert torch.allclose(kl_nvp_to_final(logits, logits), torch.zeros(()))


def test_frozen_config_contract():
    controlled = {"candidate_representation": {"K": 50, "padded_length": 20, "pad_token_id": 124}}
    cfg = {"final_seed_set": [1, 2, 3], "frozen_data_population": {"train_complete_k50": 28159, "validation_complete_k50": 4488}, "target_and_objective": {"target_temperature": 0.05, "lambda_kl": 0.1}, "training": {"total_steps": 10000, "checkpoint_evaluation_cadence_steps": 100, "early_stopping": False}, "validation": {"sampling": False, "ranking": "descending final logits; candidate ID ascending tie break", "scored_pass_budget": 45, "selection_metric": "ValidationFinalMeanOverOz policy-45 dataset macro mean", "final_test_accessed": False, "ood_accessed": False, "runtime_accessed": False}}
    validate_config(cfg, controlled)
