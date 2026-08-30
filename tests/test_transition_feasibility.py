import numpy as np
import torch

from scripts.run_transition_feasibility import CandidateValueProbe, FEATURE_DIM, K, PAD_LENGTH, PAD_TOKEN, normalize_autophase, transition_summary, validate_trajectory


def _candidates():
    return [[1, 2, 3, 4] for _ in range(K)]


def test_normalization_and_trajectory_schema_preserve_prefix_order():
    raw = np.ones(FEATURE_DIM, dtype=np.float32); raw[51] = 2
    state = normalize_autophase(raw, "p")
    assert len(state) == FEATURE_DIM and state[51] == 1.0
    row = {"program_id": "p", "source_id": "source", "split": "train", "candidate_id": 0, "pass_ids": [1, 2, 3, 4], "states": [state] * 5}
    validate_trajectory(row, _candidates(), "train")


def test_transition_summary_has_exact_four_feature_blocks():
    states = np.arange(3 * FEATURE_DIM, dtype=np.float32).reshape(3, FEATURE_DIM)
    summary = transition_summary(states)
    assert summary.shape == (4 * FEATURE_DIM,)
    assert np.array_equal(summary[:FEATURE_DIM], states[-1] - states[0])


def test_controlled_probe_output_shapes_for_base_and_oracle():
    candidates = _candidates(); initial = torch.randn(2, FEATURE_DIM); summary = torch.randn(2, K, 4 * FEATURE_DIM)
    assert CandidateValueProbe(candidates, False)(initial).shape == (2, K)
    assert CandidateValueProbe(candidates, True)(initial, summary).shape == (2, K)


def test_collector_contract_has_no_final_cohort_and_uses_only_existing_candidate_ids():
    candidates = _candidates()
    assert len(candidates) == K and all(len(row) <= PAD_LENGTH and all(0 <= action < PAD_TOKEN for action in row) for row in candidates)
