from scripts.build_route_a_posthoc_runtime_binaries import method_id, prefix_key


def test_frozen_model_method_and_prefix_keys_are_distinct_and_stable() -> None:
    nvp = {"model": "NVP", "seed": 2}
    mamba = {"model": "Mamba", "seed": 3}
    assert method_id(nvp) == "nvp_seed2"
    assert method_id(mamba) == "mamba_seed3"
    assert prefix_key(nvp) == "NVP_seed2"
    assert prefix_key(mamba) == "Mamba_seed3"
