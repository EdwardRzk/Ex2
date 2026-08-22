import torch
from scripts.train_preference_mambanvp import K, sample_balanced_pairs, strict_eligible

def test_tied_program_is_retained_but_has_no_preference_pairs():
    values=torch.zeros(2,K); values[1,0]=1
    eligible=strict_eligible(values)
    assert eligible.tolist()==[False,True]

def test_strict_pair_sampler_is_exactly_balanced_and_never_samples_ties():
    values=torch.zeros(2,K); values[:,0]=2; values[:,1]=1
    first,second,labels,active=sample_balanced_pairs(values,strict_eligible(values))
    assert active.all() and labels.sum().item()==10
    assert torch.all(values.gather(1,first[:,:5])>values.gather(1,second[:,:5]))
    assert torch.all(values.gather(1,first[:,5:])<values.gather(1,second[:,5:]))
