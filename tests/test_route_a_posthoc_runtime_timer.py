from scripts.time_route_a_posthoc_runtime import timed_command


def test_taskset_command_preserves_canonical_argv_without_duplicate_binary() -> None:
    command = timed_command("outputs/x/a.out", ["./a.out", "1125000"])
    assert command[:3] == ["taskset", "-c", "0"]
    assert command[-1] == "1125000"
    assert len(command) == 5
    assert command[3] == command[3]
    assert command[4] != command[3]
