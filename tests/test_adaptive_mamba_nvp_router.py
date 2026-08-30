import json
from pathlib import Path

import torch

from scripts.train_adaptive_mamba_nvp_router import ProgramAdaptiveRouter, SourceBalancedSampler, distribution_features, load_frozen_mamba, policy45, policy45_utility, thresholded_mixture
from scripts.train_controlled_nvp_stage_a import load_candidates
from scripts.train_mamba_nvp_objecttext import load_frozen_nvp


def test_router_input_and_output_shapes():
    x=torch.randn(3,56); m=torch.randn(3,50); n=torch.randn(3,50)
    inputs,pm,pn=distribution_features(x,m,n)
    assert inputs.shape == (3,215)
    alpha=ProgramAdaptiveRouter()(inputs)
    assert alpha.shape == (3,) and torch.all((0 <= alpha) & (alpha <= 1))
    assert pm.shape == pn.shape == (3,50)


def test_tau_one_exactly_reproduces_mamba_probabilities():
    pm=torch.softmax(torch.randn(4,50),dim=1); pn=torch.softmax(torch.randn(4,50),dim=1); alpha=torch.rand(4)
    mixed,effective=thresholded_mixture(pm,pn,alpha,1.0)
    assert torch.equal(mixed,pm)
    assert torch.equal(effective,torch.zeros_like(alpha))
    records=[{"prefix_object_text_size_bytes":[100-index]} for index in range(50)]
    assert policy45(mixed[0].tolist(),records) == policy45(pm[0].tolist(),records)


def test_policy45_advantage_uses_existing_prefix_labels_only():
    records=[]
    for candidate in range(50): records.append({"prefix_object_text_size_bytes":[100-candidate]})
    matrix={"p":records}; row={"program_id":"p","S_Oz":100}
    nvp=[0.0]*50; mamba=[0.0]*50; nvp[49]=1.0; mamba[0]=1.0
    assert policy45(nvp,records)==51
    assert policy45_utility(nvp,row,matrix) > policy45_utility(mamba,row,matrix)


def test_source_balanced_sampler_draws_all_explicit_sources():
    sampler=SourceBalancedSampler(["large"]*100+["small"],seed=3)
    sample=sampler.sample(1000,torch.device("cpu")).tolist()
    assert any(index == 100 for index in sample)
    assert any(index < 100 for index in sample)



def test_frozen_seed_matched_expert_loaders_are_eval_only():
    root = Path(__file__).resolve().parents[1]
    controlled = json.loads((root / "configs/controlled_nvp_stage_a_v6.json").read_text())
    tokens, lengths = load_candidates(root / "configs/rlcompopt_action_seq_50.txt", pad_token_id=124, padded_length=20)
    nvp = load_frozen_nvp(root / "outputs/route_a_stage_b_v6/nvp/seed1/model.pt", 1)
    mamba = load_frozen_mamba(root / "outputs/route_a_stage_b_v6/mamba/seed1/model.pt", 1, controlled, tokens, lengths)
    assert not nvp.training and not mamba.training
    assert not any(parameter.requires_grad for parameter in nvp.parameters())
    assert not any(parameter.requires_grad for parameter in mamba.parameters())
