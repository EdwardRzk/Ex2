import ast
from pathlib import Path

from scripts.evaluate_gated_calibrated_mambanvp_final import METHOD, validate_config


def test_final_config_contract_uses_frozen_validation_selections():
    cfg = {"final_seed_set": [1, 2, 3], "final_population": {"total": 4683, "complete_k50": 4679, "invalid": 4}, "candidate_representation": {"K": 50, "padded_length": 20, "pad_token_id": 124}, "inference": {"sampling": False, "ranking": "descending final logits; candidate ID ascending tie break", "scored_pass_budget": 45}, "comparison_family": ["NVP", "Mamba", "MambaNVP", "CrossCandidateMambaNVP", METHOD], "checkpoint_selection": {"selected_steps": {"1": 3400, "2": 500, "3": 3500}}}
    validate_config(cfg)


def test_final_evaluator_never_imports_or_calls_a_compiler_environment():
    source = Path("scripts/evaluate_gated_calibrated_mambanvp_final.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    assert "compiler_gym" not in [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
    assert "env.step" not in source
    assert "ObjectTextSizeBytes" not in source
