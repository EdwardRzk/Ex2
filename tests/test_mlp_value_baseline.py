import gzip
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from scripts.train_mlp_value_baseline import (
    build_feature,
    pairwise_accuracy,
    pairwise_ranking_loss,
    program_slices,
    read_records,
)


class MlpValueBaselineTest(unittest.TestCase):
    def test_build_feature_uses_final_state_histogram_and_length(self) -> None:
        record = {
            "states": [[1, 2], [3, 4], [5, 6]],
            "action_indices": [1, 1],
            "sequence_length": 2,
        }
        feature = build_feature(record, action_count=3, max_length=4)
        np.testing.assert_allclose(feature, [5, 6, 0, 1, 0, 0.5])

    def test_program_slices_preserve_program_groups(self) -> None:
        self.assertEqual(
            program_slices(["a", "a", "b", "b", "b"]),
            [slice(0, 2), slice(2, 5)],
        )

    def test_pairwise_loss_prefers_correct_order(self) -> None:
        targets = torch.tensor([0.0, 1.0, 2.0])
        correct = pairwise_ranking_loss(torch.tensor([0.0, 1.0, 2.0]), targets)
        reversed_loss = pairwise_ranking_loss(torch.tensor([2.0, 1.0, 0.0]), targets)
        self.assertLess(float(correct), float(reversed_loss))

    def test_pairwise_accuracy_is_group_local(self) -> None:
        predictions = np.asarray([0.0, 1.0, 10.0, 9.0])
        targets = np.asarray([0.0, 1.0, 0.0, 1.0])
        self.assertEqual(
            pairwise_accuracy(
                predictions, targets, [slice(0, 2), slice(2, 4)]
            ),
            0.5,
        )

    def test_read_records_combines_multiple_gzip_files_in_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = [
                Path(directory) / "first.jsonl.gz",
                Path(directory) / "second.jsonl.gz",
            ]
            for path, value in zip(paths, (1, 2)):
                with gzip.open(path, "wt", encoding="utf-8") as file:
                    json.dump({"value": value}, file)
                    file.write("\n")

            self.assertEqual(
                read_records([str(path) for path in paths]),
                [{"value": 1}, {"value": 2}],
            )


if __name__ == "__main__":
    unittest.main()
