import torch
import pytest
from scripts.train_autophase_nvp_objecttext import AutophaseNVP
from scripts.train_cross_candidate_mambanvp import CrossCandidateMambaNVP

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Mamba CUDA kernel required")
def test_cross_candidate_model_scores_k50_and_keeps_nvp_frozen():
 cfg={"d_model":8,"padded_length":3,"vocabulary_size":5,"pad_token_id":4,"layers":1,"d_state":4,"d_conv":2,"expand":2,"use_fast_path":True,"candidate_interaction":{"layers":2,"num_heads":4,"dropout":0.0}}
 model=CrossCandidateMambaNVP(AutophaseNVP(),cfg,torch.tensor([[1,2,4]]*50),torch.tensor([2]*50)).cuda()
 assert model(torch.randn(2,56,device="cuda")).shape==(2,50)
 assert all(not p.requires_grad for p in model.nvp.parameters())
 assert len(model.attention)==2
