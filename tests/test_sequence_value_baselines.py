import unittest

import torch

from scripts.train_sequence_value_baseline import SequenceValueModel


REPRESENTATION = {
    "state_dimension": 2,
    "action_count": 3,
    "start_action_index": 3,
    "max_sequence_length": 2,
}


class SequenceValueBaselinesTest(unittest.TestCase):
    def _inputs(self):
        return (
            torch.randn(2, 3, 2),
            torch.tensor([[3, 0, 1], [3, 2, 3]]),
            torch.tensor([3, 2]),
        )

    def test_lstm_forward_returns_one_value_per_trajectory(self) -> None:
        model = SequenceValueModel(
            REPRESENTATION,
            {"type": "LSTM", "d_model": 8, "layers": 2, "dropout": 0.0},
        )
        self.assertEqual(tuple(model(*self._inputs()).shape), (2,))

    def test_transformer_forward_masks_padding_and_returns_values(self) -> None:
        model = SequenceValueModel(
            REPRESENTATION,
            {
                "type": "Transformer",
                "d_model": 8,
                "layers": 2,
                "attention_heads": 2,
                "feedforward_dimension": 16,
                "dropout": 0.0,
                "activation": "gelu",
            },
        )
        states, actions, lengths = self._inputs()
        first = model(states, actions, lengths)
        states[1, 2] = 1000
        second = model(states, actions, lengths)
        self.assertEqual(tuple(first.shape), (2,))
        self.assertAlmostEqual(float(first[1]), float(second[1]), places=5)


if __name__ == "__main__":
    unittest.main()
