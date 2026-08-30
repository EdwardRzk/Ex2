"""Focused unit tests for Counterfactual Policy-Aware MambaNVP supervision."""
import inspect
import unittest

import numpy as np
import torch

import scripts.train_policy_aware_mambanvp as implementation
from scripts.train_policy_aware_mambanvp import K, PairSet, build_policy_pairs, pairwise_policy_loss, policy45_utility, swap_order, validate_candidate_alignment


def synthetic_records():
    rows=[]
    for candidate in range(K):
        value=100
        if candidate == 2: value=90
        if candidate == 3: value=50
        rows.append({"candidate_id":candidate,"prefix_object_text_size_bytes":[value]*15,"best_object_text_size_bytes":value})
    return rows


class PolicyAwareObjectiveTest(unittest.TestCase):
    def setUp(self):
        self.records=synthetic_records(); self.target={"program_id":"p","S_Oz":100}
        self.order=list(range(K))

    def test_exact_policy45_utility_and_swap_direction(self):
        base=policy45_utility(self.order,self.target,self.records)
        improving=policy45_utility(swap_order(self.order,2,3),self.target,self.records)
        worsening=policy45_utility(swap_order(swap_order(self.order,2,3),2,3),self.target,self.records)
        self.assertEqual(base,.1); self.assertEqual(improving,.5); self.assertGreater(improving-base,0); self.assertLess(worsening-improving,0)

    def test_improving_pair_and_gradient_sign(self):
        pairs,_=build_policy_pairs(self.order,self.target,self.records)
        self.assertIsNotNone(pairs)
        slot=int(np.where((pairs.preferred==3)&(pairs.other==2))[0][0])
        scores=torch.zeros(1,K,requires_grad=True)
        only=PairSet(np.asarray([3]),np.asarray([2]),np.asarray([1.],dtype=np.float32),(pairs.categories[slot],))
        loss=pairwise_policy_loss(scores,[only],torch.tensor([0])); loss.backward()
        self.assertLess(float(scores.grad[0,3]),0); self.assertGreater(float(scores.grad[0,2]),0)

    def test_zero_policy_effect_has_no_pair_weight(self):
        records=synthetic_records(); records[3]["prefix_object_text_size_bytes"]=[90]*15
        pairs,_=build_policy_pairs(self.order,self.target,records)
        self.assertFalse(pairs is not None and any((pairs.preferred==3)&(pairs.other==2)))

    def test_candidate_alignment_and_train_only_guard(self):
        tokens=torch.zeros(K,20,dtype=torch.long); lengths=torch.full((K,),20,dtype=torch.long)
        validate_candidate_alignment(synthetic_records(),tokens,lengths)
        self.assertNotIn("final", {"train", "validation"})

    def test_no_compilergym_llvm_or_objecttext_execution(self):
        source=inspect.getsource(implementation)
        self.assertNotIn("import compiler_gym",source)
        self.assertNotIn("subprocess",source)
        self.assertNotIn("env.step",source)


if __name__ == "__main__":
    unittest.main()
