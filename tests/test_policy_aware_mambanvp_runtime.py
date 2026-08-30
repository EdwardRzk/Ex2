"""Focused guardrails for the PA-only supplementary runtime runner."""
import inspect
import unittest

import scripts.run_policy_aware_mambanvp_runtime as runtime


class PolicyAwareRuntimeGuardTest(unittest.TestCase):
    def test_only_pa_methods_are_newly_timed(self):
        self.assertEqual(runtime.METHODS, ("pa_mambanvp_seed1", "pa_mambanvp_seed2", "pa_mambanvp_seed3"))

    def test_existing_formal_runtime_is_baseline_source(self):
        self.assertEqual(str(runtime.BASELINE), "outputs/gated_calibrated_mambanvp_runtime_v1_retry1")

    def test_no_compilergym_or_phase_application(self):
        source = inspect.getsource(runtime)
        self.assertNotIn("import compiler_gym", source)
        self.assertNotIn("env.step", source)
        self.assertIn("no exact legacy action provenance", source)


if __name__ == "__main__":
    unittest.main()
