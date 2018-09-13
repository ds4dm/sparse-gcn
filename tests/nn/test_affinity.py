# coding: utf-8


import unittest
import torch
import sgcn.nn.affinity as aff


class TestAffinity(unittest.TestCase):

    def setUp(self):
        self.K = torch.rand((7, 3)).requires_grad_()
        self.Q = torch.rand((4, 3)).requires_grad_()
        self.md = (torch.rand((4, 7)) > .6).float()
        idx = self.md.nonzero()
        self.ms = torch.sparse_coo_tensor(
            idx.t(), torch.ones(len(idx)), self.md.size())

    def test_dotproduct_dense(self):
        func = aff.DotProduct(scaled=False)
        coefs = func(self.Q, self.K)
        self.assertFalse(coefs.is_sparse)
        self.assertTrue(torch.equal(coefs, self.Q @ self.K.t()))

    def test_dotproduct_sparse(self):
        func = aff.DotProduct(scaled=False)
        coefs = func(self.Q, self.K, self.ms)
        self.assertTrue(coefs.is_sparse)

        # check if mask is respected
        checks = (coefs.to_dense() * (1 - self.md)) > 0
        self.assertFalse(checks.any())


@unittest.skipUnless(torch.cuda.is_available(), "Cuda unavailable.")
class TestAffinityCuda(TestAffinity):
    pass
