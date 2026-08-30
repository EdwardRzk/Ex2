import numpy as np

from scripts.analyze_candidate_ranking_error_decomposition import K, average_rank_desc, kendall_tau_b, policy_details, true_top_inversions
from scripts.evaluate_mamba_nvp_final_objecttext import policy45


def _records():
    return [{"candidate_length": 5 if index < 9 else 4, "prefix_object_text_size_bytes": [100 - index] * (5 if index < 9 else 4)} for index in range(K)]


def test_policy_details_reuses_exact_policy45_result_and_budget():
    scores = np.arange(K, dtype=np.float64); records = _records(); values = scores.copy()
    details = policy_details(scores, records, values)
    assert details["policy45_object_text_size_bytes"] == policy45(scores.tolist(), records)
    assert sum(row["observed_prefix_count"] for row in details["admitted"]) == 45
    assert details["ordered"][0] == 49


def test_oracle_tie_rank_and_top_region_inversions_are_deterministic():
    values = np.zeros(K); values[1] = values[2] = 4
    assert average_rank_desc(values)[1] == average_rank_desc(values)[2]
    order = list(range(K))
    assert true_top_inversions(order, values, 5) >= 0
    assert -1 <= kendall_tau_b(values, values) <= 1
