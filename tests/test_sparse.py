# coding: utf-8

"""Test for sparse module."""

import unittest
import torch
import numpy as np
from torch.autograd import Variable, grad
import sgat.sparse as sp


class TestSparse(unittest.TestCase):

    def assertEpsilonEqual(self, A, B, epsilon):
        self.assertTupleEqual(A.size(), B.size())
        diff = (A - B) < epsilon
        self.assertTrue(bool(diff.all()))

    def setUp(self):
        self.i = torch.LongTensor([[1, 2, 0], [1, 4, 6]])
        self.v = Variable(torch.rand(3), requires_grad=True)
        self.A = Variable(torch.rand((3, 7)), requires_grad=True)
        self.B = Variable(torch.rand((7, 5)), requires_grad=True)
        m = torch.rand((3, 5)) > .5
        m_i = m.nonzero()
        self.m = torch.sparse.FloatTensor(
            m_i.t(), torch.ones(len(m_i)), m.size()
        )

    @unittest.skip("Not implemented")
    def test_build(self):
        raise NotImplementedError()

    @unittest.skip("Not implemented")
    def test_values(self):
        raise NotImplementedError()

    def test_sparse_values(self):
        v_hat = sp.values(sp.build(self.i, self.v))
        self.assertTrue(torch.equal(v_hat.sum(), self.v.sum()))
        g, = grad(v_hat.sum(), self.v)
        self.assertTrue(torch.equal(g, torch.ones_like(g)))

    def test_sum(self):
        i = torch.LongTensor([[1, 4, 5], [0, 1, 0], [1, 7, 1]])
        v = torch.FloatTensor([1, 2, 3])
        s = (10, 10, 10)
        w = Variable(torch.FloatTensor([11, 15, 12]), requires_grad=True)

        # dims=None
        x = sp.build(i, v*w, (10, 10, 10))
        o = sp.sum(x)
        od = x.to_dense().sum()
        self.assertTrue(np.all(o.to_dense() == od))

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

    def test_matmulmasked(self):
        Ms = sp.matmulmasked(self.A, self.B, self.m)

        # Check type
        self.assertTrue(Ms.is_sparse)

        # Check forward
        Md = (self.A @ self.B) * self.m.to_dense()
        self.assertEpsilonEqual(Ms.to_dense(), Md, 1e-4)

        # Check grad B
        grad_Bs, = grad(sp.values(Ms).sum(), self.B, retain_graph=True)
        grad_Bd, = grad(Md.sum(), self.B, retain_graph=True)
        self.assertEpsilonEqual(grad_Bs, grad_Bd, 1e-4)

        # Check grad A
        grad_As, = grad(sp.values(Ms).sum(), self.A)
        grad_Ad, = grad(Md.sum(), self.A)
        self.assertEpsilonEqual(grad_As, grad_Ad, 1e-4)


@unittest.skipUnless(torch.cuda.is_available(), "Cuda unavailable.")
class TestSparseCuda(TestSparse):

    def setUp(self):
        raise NotImplementedError()
