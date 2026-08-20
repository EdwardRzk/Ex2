from scripts.validate_route_a_posthoc_runtime_correctness import method_status


def test_cohorts_require_all_seven_and_never_upgrade_unverified() -> None:
    passed = [{"correctness_status": "semantic_validated_pass"} for _ in range(7)]
    assert method_status(passed) == (True, True, None)
    mixed = passed[:-1] + [{"correctness_status": "execution_only_unverified"}]
    assert method_status(mixed)[0] is False
    assert method_status(mixed)[1] is True
    assert method_status(mixed)[2] == "execution_only_unverified; semantic_validated_pass"
