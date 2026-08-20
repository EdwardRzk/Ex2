from scripts.run_mambanvp_nvp_runtime import METHODS, gmean, method_id, selected_prefix


def test_policy45_uses_first_minimum_in_deterministic_rank_order() -> None:
    scores = [0.0] * 50
    records = [{"candidate_id": candidate_id, "ordered_pass_sequence": [candidate_id], "prefix_object_text_size_bytes": [100]} for candidate_id in range(50)]
    records[2]["prefix_object_text_size_bytes"] = [50]
    selected = selected_prefix(scores, records)
    assert selected["candidate_id"] == 2
    assert selected["candidate_rank"] == 2
    assert selected["action_ids"] == [2]


def test_runtime_runner_has_no_compilergym_rollout() -> None:
    source = __import__("pathlib").Path("scripts/run_mambanvp_nvp_runtime.py").read_text(encoding="utf-8")
    assert "import compiler_gym" not in source
    assert ".step(" not in source


def test_method_inventory_and_geomean_are_frozen() -> None:
    assert METHODS == ("oz", "nvp_seed1", "nvp_seed2", "nvp_seed3", "mambanvp_seed1", "mambanvp_seed2", "mambanvp_seed3")
    assert method_id("MambaNVP", 3) == "mambanvp_seed3"
    assert gmean([1.0, 4.0]) == 2.0
