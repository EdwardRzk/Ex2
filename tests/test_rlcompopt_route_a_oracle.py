import gzip
import json

from scripts.compute_rlcompopt_route_a_oracle import K, compute_route_a_oracle


def write_shard(path, *, program_id, dataset_id, oz, valid, best_sizes):
    records = [
        {
            "program_id": program_id,
            "candidate_id": index,
            "training_target_validity": "valid_completed_candidate_rollout" if valid else "invalid_incomplete_candidate_rollout",
            "best_object_text_size_bytes": best_sizes[index] if valid else None,
        }
        for index in range(K)
    ]
    summary = {
        "program_id": program_id,
        "dataset_id": dataset_id,
        "oz_object_text_size_bytes": oz,
        "oracle_K50_validity": "valid_complete_K50" if valid else "invalid_incomplete_K50",
        "ratio_metric_validity": "valid_for_ObjectText_ratio_metric" if oz and oz > 0 else "invalid_for_ObjectText_ratio_metric",
    }
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump({"records": records, "program_summary": summary}, handle)


def test_route_a_oracle_uses_all_k50_and_dataset_macro_mean(tmp_path) -> None:
    write_shard(
        tmp_path / "000000.json.gz",
        program_id="benchmark://a-v0/one",
        dataset_id="a-v0",
        oz=100,
        valid=True,
        best_sizes=[80] + [90] * (K - 1),
    )
    write_shard(
        tmp_path / "000001.json.gz",
        program_id="benchmark://a-v0/two",
        dataset_id="a-v0",
        oz=100,
        valid=False,
        best_sizes=[0] * K,
    )
    write_shard(
        tmp_path / "000002.json.gz",
        program_id="benchmark://b-v0/one",
        dataset_id="b-v0",
        oz=200,
        valid=True,
        best_sizes=[180] + [190] * (K - 1),
    )

    report, audit = compute_route_a_oracle(tmp_path)

    assert report["per_dataset"]["a-v0"] == {
        "N_total": 2,
        "N_complete_K50_oracle": 1,
        "N_ratio_valid": 2,
        "N_RouteA_oracle_valid": 1,
        "N_failed_or_invalid": 1,
        "OracleMeanOverOz": 0.2,
    }
    assert report["per_dataset"]["b-v0"]["OracleMeanOverOz"] == 0.1
    assert report["RouteAOracleMeanOverOz"] == 0.15000000000000002
    assert report["route_decision"] == "STAY_ROUTE_A"
    assert audit[0]["S_oracle"] == 80
    assert audit[0]["oracle_candidate_count"] == K
    assert audit[1]["S_oracle"] is None


def test_route_a_oracle_is_undefined_for_empty_dataset_cohort(tmp_path) -> None:
    write_shard(
        tmp_path / "000000.json.gz",
        program_id="benchmark://a-v0/one",
        dataset_id="a-v0",
        oz=100,
        valid=False,
        best_sizes=[0] * K,
    )

    report, _ = compute_route_a_oracle(tmp_path)

    assert report["per_dataset"]["a-v0"]["N_RouteA_oracle_valid"] == 0
    assert report["per_dataset"]["a-v0"]["OracleMeanOverOz"] is None
    assert report["RouteAOracleMeanOverOz"] is None
    assert report["branch_criterion_status"] == "undefined_due_to_invalid_required_data"
    assert report["route_decision"] == "STOP_UNDEFINED"
