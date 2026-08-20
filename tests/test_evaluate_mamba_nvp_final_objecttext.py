import gzip
import json
from pathlib import Path

import pytest

from scripts.evaluate_mamba_nvp_final_objecttext import aggregate, policy45


def test_policy45_uses_descending_logits_and_candidate_id_ties() -> None:
    records = [{"prefix_object_text_size_bytes": [index + 10]} for index in range(50)]
    score = [0.0] * 50
    score[2] = 1.0
    assert policy45(score, records) == 10


def test_aggregate_uses_predeclared_three_method_common_cohort() -> None:
    programs = ["a", "b"]
    summaries = {program: {"dataset_id": "d", "ratio_metric_validity": "valid_for_ObjectText_ratio_metric"} for program in programs}
    results = {}
    for method in ("NVP", "Mamba", "MambaNVP"):
        for seed in (1, 2, 3):
            results[(method, seed)] = {"a": {"valid": True, "mean_over_oz": 0.1}, "b": {"valid": method != "MambaNVP" or seed != 3, "mean_over_oz": 0.2}}
    report = aggregate(["NVP", "Mamba", "MambaNVP"], programs, summaries, results)
    assert report["per_dataset"]["d"]["N_primary_valid"] == 1
    assert report["dataset_macro"]["MambaNVP"]["three_seed_mean"] == pytest.approx(0.1)


def test_final_runner_has_no_compilergym_or_measurement_path() -> None:
    source = Path("scripts/evaluate_mamba_nvp_final_objecttext.py").read_text(encoding="utf-8")
    assert "import compiler_gym" not in source
    assert "env.step(" not in source
    assert "ObjectTextSize" not in source
