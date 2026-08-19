import torch

from scripts.train_controlled_nvp_stage_a import (
    ControlledCandidateModel,
    K,
    load_candidates,
    policy_metrics,
)


COMMON = {
    "d_model": 8,
    "padded_length": 3,
    "vocabulary_size": 6,
    "pad_token_id": 5,
}


def test_candidates_preserve_order_and_right_padding(tmp_path) -> None:
    path = tmp_path / "candidates.txt"
    path.write_text("(3, 1, 2)\n(2,)\n" + "(0,)\n" * 48)
    tokens, lengths = load_candidates(path, pad_token_id=5, padded_length=3)
    assert tokens[0].tolist() == [3, 1, 2]
    assert tokens[1].tolist() == [2, 5, 5]
    assert lengths[:2].tolist() == [3, 1]


def test_mlp_and_transformer_score_every_candidate() -> None:
    tokens = torch.tensor([[1, 2, 5]] * K)
    lengths = torch.tensor([2] * K)
    for name, extra in (("MLP", {}), ("Transformer", {"layers": 1, "attention_heads": 2, "feedforward_dimension": 16})):
        model = ControlledCandidateModel(name, {**COMMON, **extra}, tokens, lengths)
        assert model(torch.zeros(2, 56)).shape == (2, K)


def test_policy_metric_consumes_ranked_prefixes_under_exact_budget() -> None:
    program_id = "benchmark://x-v0/a"
    records = [{"program_id": program_id, "dataset_id": "x-v0", "S_Oz": 100, "best_object_text_size": [99, 80] + [99] * 48}]
    matrix = {program_id: [{"candidate_id": index, "prefix_object_text_size_bytes": [80] if index == 1 else [99]} for index in range(K)]}
    logits = torch.zeros(1, K); logits[0, 1] = 1
    metric = policy_metrics(logits, records, matrix)
    assert metric["ValidationFinalMeanOverOz"] == 0.2
    assert metric["policy45_regret_mean_bytes"] == 0
    assert metric["positive_program_count_vs_Oz"] == 1
