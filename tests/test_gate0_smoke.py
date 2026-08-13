import json
from pathlib import Path

import pytest

from scripts.run_gate0_smoke import evaluate_gate, load_config, resolve_action


def test_resolve_action_uses_environment_names() -> None:
    assert resolve_action(["-adce", "-mem2reg", "-sroa"], "-mem2reg") == 1


def test_resolve_action_rejects_missing_action() -> None:
    with pytest.raises(ValueError, match="LLVM action is unavailable"):
        resolve_action(["-adce"], "-mem2reg")


def test_evaluate_gate_requires_every_frozen_condition() -> None:
    results = {
        "reset_succeeded": True,
        "action_had_effect": True,
        "is_buildable": True,
        "is_runnable": True,
        "runtime_samples_seconds": [0.3, 0.2, 0.4],
    }
    assert all(evaluate_gate(results, expected_runtime_sample_count=3).values())

    results["runtime_samples_seconds"] = [0.3, 0.0, 0.4]
    checks = evaluate_gate(results, expected_runtime_sample_count=3)
    assert checks["runtime_sample_count"]
    assert not checks["all_runtime_samples_positive_and_finite"]


def test_load_config_rejects_single_runtime_measurement(tmp_path: Path) -> None:
    config_path = Path("configs/gate0_compilergym_smoke_v1.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["environment"]["runtime_measurement_runs"] = 1
    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="runtime_measurement_runs"):
        load_config(invalid_path)
