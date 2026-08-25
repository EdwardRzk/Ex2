import ast
from pathlib import Path

from scripts.run_gated_calibrated_mambanvp_runtime import METHODS, MODELS, validate_config


def test_runtime_contract_has_required_frozen_methods():
    assert MODELS == ("NVP", "MambaNVP", "GatedCalibratedMambaNVP")
    assert len(METHODS) == 10
    cfg = {
        "experiment_name": "gated_calibrated_mambanvp_runtime_v1",
        "protocol_class": "post_hoc_exploratory_runtime",
        "population": {"included_program_count": 9},
        "inference": {"sampling": False, "ranking": "descending frozen logits; candidate_id ascending ties", "policy": {"scored_pass_budget": 45}},
        "methods": {"learned": [{"model": model, "seed": seed} for model in MODELS for seed in (1, 2, 3)]},
        "gated_selected_steps": {"1": 3400, "2": 500, "3": 3500},
    }
    validate_config(cfg)


def test_runtime_runner_has_no_compilergym_or_phase_application():
    source = Path("scripts/run_gated_calibrated_mambanvp_runtime.py").read_text(encoding="utf-8")
    ast.parse(source)
    assert "import compiler_gym" not in source
    assert ".step(" not in source
