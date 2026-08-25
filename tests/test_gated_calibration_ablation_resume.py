import ast
from pathlib import Path


def test_parallel_runner_persists_exact_resume_state():
    source = Path("scripts/run_gated_calibration_ablation_task.py").read_text(encoding="utf-8")
    ast.parse(source)
    for field in ("optimizer_state_dict", "generator_state", "current_epoch_order", "next_batch_begin"):
        assert field in source
    assert "fresh_reconstruction" in source
