import ast
from pathlib import Path

from scripts.evaluate_cross_candidate_mambanvp_final import METHOD, validate_config


def test_cross_candidate_final_config_is_frozen_and_complete():
    config = {
        "final_seed_set": [1, 2, 3],
        "final_population": {"total": 4683, "complete_k50": 4679, "invalid": 4},
        "candidate_representation": {"K": 50, "padded_length": 20, "pad_token_id": 124},
        "inference": {"sampling": False, "ranking": "descending logits; candidate ID ascending tie break", "scored_pass_budget": 45},
        "comparison_family": ["NVP", "Mamba", "MambaNVP", METHOD],
        "checkpoint_selection": {"selected_steps": {"1": 3400, "2": 600, "3": 1200}},
    }
    validate_config(config)


def test_final_evaluator_never_imports_compilergym_or_executes_environment():
    source = Path("scripts/evaluate_cross_candidate_mambanvp_final.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in (node.names if isinstance(node, ast.Import) else [])
    }
    assert "compiler_gym" not in imported
    assert "env.step" not in source
    assert "ObjectTextSizeBytes" not in source
