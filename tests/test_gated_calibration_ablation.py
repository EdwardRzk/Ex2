from pathlib import Path

import torch

from scripts.train_gated_calibration_ablation import AblatedGatedCalibratedMambaNVP, VARIANTS, validate_config
from scripts.train_set_conditioned_mamba_ranker import load_json


def test_ablation_contract_is_fixed():
    cfg = load_json(Path("configs/gated_calibration_ablation_v1.json"))
    controlled = load_json(Path(cfg["candidate_representation_source"]))
    validate_config(cfg, controlled)
    assert VARIANTS == ("gated_full", "no_kl", "no_gate", "no_gate_no_kl")


def test_no_gate_removes_gate_parameters():
    source = Path("scripts/train_gated_calibration_ablation.py").read_text(encoding="utf-8")
    assert "del self.gate" in source
    assert "alpha = torch.ones_like(residual)" in source
