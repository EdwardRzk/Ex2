import torch

from scripts.train_autophase_nvp_objecttext import AutophaseNVP, K, policy_metrics


def test_anchor_has_official_autophase_to_k50_shape() -> None:
    model = AutophaseNVP()
    assert model(torch.zeros(3, 56)).shape == (3, K)


def test_policy_metric_uses_ranked_prefixes_under_45_pass_budget() -> None:
    program_id = "benchmark://x-v0/a"
    best_sizes = [99] * K
    best_sizes[1] = 80
    records = [{"program_id": program_id, "dataset_id": "x-v0", "S_Oz": 100, "best_object_text_size": best_sizes}]
    matrix = {
        program_id: [
            {"candidate_id": index, "prefix_object_text_size_bytes": [80] if index == 1 else [99]}
            for index in range(K)
        ]
    }
    logits = torch.zeros(1, K)
    logits[0, 1] = 1
    metrics = policy_metrics(logits, records, matrix)
    assert metrics["ValidationFinalMeanOverOz"] == 0.2
    assert metrics["policy45_regret_mean_bytes"] == 0
