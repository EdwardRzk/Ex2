import unittest

import numpy as np

from scripts.train_mambapo_value import build_sequence_tensors, state_statistics


class MambaPoValueTest(unittest.TestCase):
    def setUp(self) -> None:
        self.records = [
            {
                "program_id": "p",
                "trajectory_index": 0,
                "sequence_length": 2,
                "states": [[1, 2], [3, 4], [5, 6]],
                "action_indices": [2, 1],
                "size_reduction_vs_oz": 0.25,
            },
            {
                "program_id": "q",
                "trajectory_index": 0,
                "sequence_length": 1,
                "states": [[2, 3], [4, 5]],
                "action_indices": [0],
                "size_reduction_vs_oz": -0.5,
            },
        ]

    def test_state_statistics_cover_every_prefix_state(self) -> None:
        mean, std = state_statistics(self.records, state_dimension=2)
        np.testing.assert_allclose(mean, [3.0, 4.0])
        np.testing.assert_allclose(std, np.sqrt([2.0, 2.0]))

    def test_sequence_tokens_follow_state_then_previous_pass_schema(self) -> None:
        mean = np.asarray([0, 0], dtype=np.float32)
        std = np.asarray([1, 1], dtype=np.float32)
        states, actions, lengths, targets, programs = build_sequence_tensors(
            self.records,
            state_mean=mean,
            state_std=std,
            action_count=3,
            start_action_index=3,
            max_sequence_length=2,
            target_name="size_reduction_vs_oz",
        )
        self.assertEqual(tuple(states.shape), (2, 3, 2))
        self.assertEqual(actions[0].tolist(), [3, 2, 1])
        self.assertEqual(actions[1].tolist(), [3, 0, 3])
        self.assertEqual(lengths.tolist(), [3, 2])
        self.assertEqual(programs, ["p", "q"])
        np.testing.assert_allclose(targets.numpy(), [0.25, -0.5])


if __name__ == "__main__":
    unittest.main()
