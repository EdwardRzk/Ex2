import unittest

from scripts.run_same_budget_search import (
    aggregate_program_results,
    generate_random_candidates,
)


class SameBudgetSearchTest(unittest.TestCase):
    def test_random_candidates_are_unique_and_bounded(self) -> None:
        candidates = generate_random_candidates(
            action_count=4,
            budget=20,
            minimum_length=1,
            maximum_length=3,
            seed=41,
        )
        sequences = [tuple(candidate["actions"]) for candidate in candidates]
        self.assertEqual(len(sequences), len(set(sequences)))
        self.assertTrue(all(1 <= len(sequence) <= 3 for sequence in sequences))

    def test_aggregate_uses_per_program_oz_reductions(self) -> None:
        program_results = []
        for reduction in (0.1, -0.1):
            methods = {
                method: {
                    "budget_curve": {
                        "8": {"size_reduction_vs_oz": reduction}
                    }
                }
                for method in ("random", "mlp", "lstm", "transformer", "mamba")
            }
            program_results.append({"methods": methods})
        aggregate = aggregate_program_results(program_results, [8])
        random_result = aggregate["random"]["budget_curve"]["8"]
        self.assertAlmostEqual(random_result["mean_size_reduction_vs_oz"], 0.0)
        self.assertEqual(random_result["positive_program_count"], 1)
        self.assertAlmostEqual(random_result["geomean_size_ratio_vs_oz"], (0.99) ** 0.5)


if __name__ == "__main__":
    unittest.main()
