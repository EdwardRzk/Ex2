from scripts.collect_gated_calibration_ablation import ATTEMPTS, VARIANTS


def test_all_fixed_ablation_tasks_are_collected_once():
    assert VARIANTS == ("gated_full", "no_kl", "no_gate", "no_gate_no_kl")
    assert {seed for attempts in ATTEMPTS.values() for seed in attempts} == {1, 2, 3}
    assert ATTEMPTS["gated_full"][1] == "parallel_reconstruction1"
