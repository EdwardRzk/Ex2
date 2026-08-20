import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.export_autophase_feature_cache import (
    AUTOPHASE_DIM,
    EXPECTED_COUNTS,
    feature_row,
    final_population,
    target_population,
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_target_population_requires_complete_k50_and_exact_membership(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(EXPECTED_COUNTS, "train", 1)
    source = tmp_path / "train.jsonl.gz"
    write_jsonl(source, [
        {"program_id": "benchmark://x/one", "dataset_id": "x", "training_target_validity": "valid_complete_K50"},
        {"program_id": "benchmark://x/two", "dataset_id": "x", "training_target_validity": "invalid_incomplete_K50_target"},
    ])
    assert target_population(source, "train") == [{"program_id": "benchmark://x/one", "dataset_name": "x", "split": "train"}]


def test_final_population_excludes_incomplete_k50(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(EXPECTED_COUNTS, "final", 1)
    complete = {"program_summary": {"program_id": "benchmark://x/one", "dataset_id": "x", "oracle_K50_validity": "valid_complete_K50"}}
    invalid = {"program_summary": {"program_id": "benchmark://x/two", "dataset_id": "x", "oracle_K50_validity": "invalid_incomplete_K50"}}
    for index, payload in enumerate((complete, invalid)):
        with gzip.open(tmp_path / f"{index:06d}.json.gz", "wt", encoding="utf-8") as handle:
            json.dump(payload, handle)
    assert final_population(tmp_path) == [{"program_id": "benchmark://x/one", "dataset_name": "x", "split": "final"}]


def test_feature_row_uses_only_raw_index_51_normalization(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeEnvironment:
        def __init__(self) -> None:
            self.observation = {"Autophase": np.arange(1, AUTOPHASE_DIM + 1, dtype=np.float32)}
            self.resets: list[str] = []

        def reset(self, *, benchmark: str) -> None:
            self.resets.append(benchmark)

    fake = FakeEnvironment()
    monkeypatch.setattr("scripts.export_autophase_feature_cache._ENV", fake)
    row = feature_row({"program_id": "benchmark://x/one", "dataset_name": "x", "split": "train"})
    assert fake.resets == ["benchmark://x/one"]
    assert len(row["raw_autophase"]) == AUTOPHASE_DIM
    assert row["normalized_autophase"] == pytest.approx((np.arange(1, AUTOPHASE_DIM + 1) / 52).tolist())


def test_exporter_has_no_candidate_or_objecttext_execution_path() -> None:
    source = Path("scripts/export_autophase_feature_cache.py").read_text(encoding="utf-8")
    assert ".step(" not in source
    assert "ObjectTextSize" not in source
