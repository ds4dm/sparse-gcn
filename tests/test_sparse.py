# coding: utf-8

"""Test for sparse module."""

import unittest
import torch
from torch.autograd import Variable, grad
import sgat.sparse as sp


class TestSparse(unittest.TestCase):

    def setUp(self):
        self.i = torch.LongTensor([[1, 4, 9]])
        self.v = Variable(torch.rand(3), requires_grad=True)

    def test_sparse(self):
        raise NotImplementedError()

    def test_values(self):
        raise NotImplementedError()

    def test_sparse_values(self):
        v_hat = sp.values(sp.sparse(self.i, self.v))
        g, = grad(v_hat.sum(), self.v)
        diff = g == torch.ones_like(g)
        self.assertTrue(bool(diff.all()))

    @unittest.skipIf(not torch.cuda.is_available(), "Cuda unavailable.")
    def test_sparse_values_cuda(self):
        raise NotImplementedError()
