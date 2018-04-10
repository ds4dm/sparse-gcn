# coding: utf-8

"""Test for sparse module."""

import unittest
import torch
from torch.autograd import Variable, grad
import sgat.sparse as sp

import numpy as np

class TestSparse(unittest.TestCase):

    def setUp(self):
        self.i = torch.LongTensor([[1, 4, 9]])
        self.v = Variable(torch.rand(3), requires_grad=True)

    @unittest.skip("Not implemented")
    def test_sparse(self):
        raise NotImplementedError()

    @unittest.skip("Not implemented")
    def test_values(self):
        raise NotImplementedError()

    def test_sparse_values(self):
        v_hat = sp.values(sp.build(self.i, self.v))
        g, = grad(v_hat.sum(), self.v)
        diff = g == torch.ones_like(g)
        self.assertTrue(bool(diff.all()))

    @unittest.skip("Not implemented")
    @unittest.skipUnless(torch.cuda.is_available(), "Cuda unavailable.")
    def test_sparse_values_cuda(self):
        raise NotImplementedError()

    def test_sum(self):
        i = torch.LongTensor([[1, 4, 5], [0, 1, 0], [1, 7, 1]])
        v = torch.FloatTensor([1, 2, 3])
        s = (10, 10, 10)
        w = Variable(torch.FloatTensor([11, 15, 12]), requires_grad=True)

        # dims=None
        x = sp.build(i, v*w, (10, 10, 10))
        o = sp.sum(x)
        
        od = x.to_dense().sum()
        self.assertTrue(np.all(o.to_dense().data == od.data))
        
        g, = grad(sp.values(o).sum(), w)
        self.assertTrue(np.all(g.data == v.data))

        # dims=(0,)
        x = sp.build(i, v*w, s)
        o = sp.sum(x, (0,))
        
        od = x.to_dense().sum(0, keepdim=True)
        self.assertTrue(np.all(o.to_dense().data == od.data))
        
        g, = grad(sp.values(o).sum(), w)
        self.assertTrue(np.all(g.data == v.data))

        # dims=(1,)
        x = sp.build(i, v*w, s)
        o = sp.sum(x, (1,))
        
        od = x.to_dense().sum(1, keepdim=True)
        self.assertTrue(np.all(o.to_dense().data == od.data))
        
        g, = grad(sp.values(o).sum(), w)
        self.assertTrue(np.all(g.data == v.data))

        # dims=(2,)
        x = sp.build(i, v*w, s)
        o = sp.sum(x, (2,))
        
        od = x.to_dense().sum(2, keepdim=True)
        self.assertTrue(np.all(o.to_dense().data == od.data))
        
        g, = grad(sp.values(o).sum(), w)
        self.assertTrue(np.all(g.data == v.data))

        # dims=(0, 2)
        x = sp.build(i, v*w, s)
        o = sp.sum(x, (0, 2))
        
        od = x.to_dense().sum(0, keepdim=True).sum(2, keepdim=True)
        self.assertTrue(np.all(o.to_dense().data == od.data))
        
        g, = grad(sp.values(o).sum(), w)
        self.assertTrue(np.all(g.data == v.data))

