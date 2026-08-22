import ast
from pathlib import Path

from scripts.evaluate_preference_mambanvp_final import METHOD, validate_config


def test_final_config_is_frozen_and_uses_validation_selected_checkpoints():
    cfg = {"final_seed_set": [1, 2, 3], "final_population": {"total": 4683, "complete_k50": 4679, "invalid": 4}, "candidate_representation": {"K": 50, "padded_length": 20, "pad_token_id": 124}, "inference": {"sampling": False, "ranking": "descending value logits; candidate ID ascending tie break", "scored_pass_budget": 45}, "comparison_family": ["NVP", "Mamba", "MambaNVP", "CrossCandidateMambaNVP", METHOD], "checkpoint_selection": {"selected_steps": {"1": 5800, "2": 7500, "3": 6800}}, "preference_diagnostic": {"enabled": True, "definition": "strict final K=50 pair accuracy of the frozen preference head; a smaller best ObjectText size is preferred", "affects_ranking": False}}
    validate_config(cfg)


def test_final_evaluator_never_imports_or_calls_a_compiler_environment():
    source = Path("scripts/evaluate_preference_mambanvp_final.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    assert "compiler_gym" not in [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
    assert "env.step" not in source
    assert "ObjectTextSizeBytes" not in source
