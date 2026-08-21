import ast
from pathlib import Path

import torch

from scripts.train_set_conditioned_mamba_ranker import K, listmle_loss, ranking_permutation, sampled_pairwise_loss


def test_listmle_prefers_the_frozen_reward_order_and_tie_breaks_by_candidate_id():
    rewards = torch.tensor([[1.0, 3.0, 3.0] + [0.0] * (K - 3)])
    permutation = ranking_permutation(rewards)
    assert permutation[0, :3].tolist() == [1, 2, 0]
    good = torch.tensor([[1.0, 4.0, 3.0] + [0.0] * (K - 3)])
    bad = torch.tensor([[4.0, 1.0, 3.0] + [0.0] * (K - 3)])
    assert listmle_loss(good, permutation) < listmle_loss(bad, permutation)


def test_pairwise_loss_only_penalizes_wrong_strict_reward_order():
    rewards = torch.tensor([[2.0, 1.0] + [0.0] * (K - 2)])
    scores = torch.tensor([[3.0, 1.0] + [0.0] * (K - 2)])
    torch.manual_seed(3)
    loss = sampled_pairwise_loss(scores, rewards, pairs_per_program=256)
    assert torch.isfinite(loss)
    assert loss >= 0


def test_ranker_source_never_imports_or_uses_compiler_environment():
    source = Path("scripts/train_set_conditioned_mamba_ranker.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_modules = [
        alias.name
        for node in ast.walk(tree)
        for alias in (node.names if isinstance(node, ast.Import) else [])
    ]
    assert "compiler_gym" not in imported_modules
    assert "env.step" not in source
    assert "ObjectTextSizeBytes" not in source
