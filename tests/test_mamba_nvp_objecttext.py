import gzip
import json
from pathlib import Path

import pytest
import torch

from scripts.train_autophase_nvp_objecttext import AutophaseNVP
from scripts.train_mamba_nvp_objecttext import MambaNVP, load_feature_cache


def small_model() -> MambaNVP:
    tokens = torch.tensor([[1, 2, 4]] * 50)
    lengths = torch.tensor([2] * 50)
    return MambaNVP(
        AutophaseNVP(),
        {"d_model": 8, "padded_length": 3, "vocabulary_size": 5, "pad_token_id": 4, "layers": 1, "d_state": 4, "d_conv": 2, "expand": 2, "use_fast_path": True},
        tokens,
        lengths,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Mamba test requires CUDA")
def test_zero_residual_reproduces_frozen_nvp_logits_and_ranking() -> None:
    model = small_model().cuda().train()
    features = torch.randn(3, 56, device="cuda")
    with torch.no_grad():
        expected = model.nvp(features)
        actual = model(features)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.argsort(actual, dim=1, descending=True).tolist() == torch.argsort(expected, dim=1, descending=True).tolist()
    assert not model.nvp.training


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Mamba test requires CUDA")
def test_nvp_is_frozen_and_residual_scores_k50() -> None:
    model = small_model().cuda()
    features, target = torch.randn(2, 56, device="cuda"), torch.full((2, 50), 1 / 50, device="cuda")
    assert all(not parameter.requires_grad for parameter in model.nvp.parameters())
    assert model(features).shape == (2, 50)
    (-(target * torch.log_softmax(model(features), dim=1)).sum(dim=1).mean()).backward()
    assert all(parameter.grad is None for parameter in model.nvp.parameters())
    assert any(parameter.grad is not None for parameter in model.residual.parameters())


def test_feature_cache_loader_requires_exact_population(tmp_path: Path) -> None:
    source = tmp_path / "features.jsonl.gz"
    row = {"program_id": "benchmark://x-v0/a", "dataset_name": "x-v0", "split": "train", "raw_autophase": [1.0] * 56, "normalized_autophase": [1.0] * 56}
    with gzip.open(source, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")
    assert load_feature_cache(source, "train", [row["program_id"]]) == {row["program_id"]: [1.0] * 56}
    with pytest.raises(ValueError, match="population mismatch"):
        load_feature_cache(source, "train", ["benchmark://x-v0/missing"])


def test_runner_has_no_compilergym_or_llvm_execution_path() -> None:
    source = Path("scripts/train_mamba_nvp_objecttext.py").read_text(encoding="utf-8")
    assert "import compiler_gym" not in source
    assert "env.step(" not in source
    assert "ObjectTextSize" not in source
