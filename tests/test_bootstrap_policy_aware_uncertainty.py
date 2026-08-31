import inspect
import unittest

import numpy as np

import scripts.bootstrap_policy_aware_uncertainty as implementation


class BootstrapProtocolTest(unittest.TestCase):
    def test_paired_stratified_bootstrap_preserves_equal_dataset_weight(self):
        values = {
            "large": {"pa": np.asarray([1.0, 1.0, 1.0]), "nvp": np.asarray([0.0, 0.0, 0.0]), "delta": np.asarray([1.0, 1.0, 1.0])},
            "small": {"pa": np.asarray([0.0]), "nvp": np.asarray([1.0]), "delta": np.asarray([-1.0])},
        }
        pa, nvp, delta = implementation.stratified_bootstrap(values, 10_000, 7)
        self.assertTrue(np.allclose(pa, 0.5))
        self.assertTrue(np.allclose(nvp, 0.5))
        self.assertTrue(np.allclose(delta, 0.0))

    def test_analysis_is_offline_only(self):
        source = inspect.getsource(implementation)
        self.assertNotIn("import torch", source)
        self.assertNotIn("import compiler_gym", source)
        self.assertNotIn("subprocess", source)
        self.assertNotIn("env.step", source)


if __name__ == "__main__":
    unittest.main()
