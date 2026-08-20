from scripts.recover_route_a_posthoc_runtime_prefixes import selected_prefix


def test_selected_prefix_preserves_policy45_rank_and_first_minimum() -> None:
    scores = [0.0] * 50
    records = []
    for candidate_id in range(50):
        records.append(
            {
                "candidate_id": candidate_id,
                "ordered_pass_sequence": list(range(20)),
                "prefix_object_text_size_bytes": [10 - candidate_id] * 20,
            }
        )
    records[2]["prefix_object_text_size_bytes"] = [8] * 20
    selected = selected_prefix(scores, records)
    assert selected["candidate_id"] == 2
    assert selected["candidate_rank"] == 2
    assert selected["prefix_index"] == 0
    assert selected["pass_count"] == 1
    assert selected["object_text_size_bytes"] == 8
