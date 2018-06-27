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
        self.i = torch.LongTensor([[0, 1, 2], [1, 4, 6]])
        self.v = Variable(torch.rand(3), requires_grad=True)
        self.s = (3, 7)
        self.A = Variable(torch.rand((3, 7)), requires_grad=True)
        self.B = Variable(torch.rand((7, 5)), requires_grad=True)
        m = torch.rand((3, 5)) > .5
        m_i = m.nonzero()
        self.m = torch.sparse.FloatTensor(
            m_i.t(), torch.ones(len(m_i)), m.size()
        )
        self.i3 = torch.LongTensor([[1, 4, 5], [0, 1, 0], [1, 7, 1]])
        self.s3 = (10, 10, 10)
        self.w = Variable(torch.rand(3), requires_grad=True)
        
        self.sum_Ai = torch.LongTensor([[0, 0, 1, 1, 2], [1, 3, 0, 0, 2], [5, 2, 3, 0, 2]])
        self.sum_Av = Variable(torch.FloatTensor([
            [-4.5,  8.0,  2.8],
            [ 7.0, -0.5,  4.0],
            [ 2.7,  3.1, -8.1],
            [-1.0, -1.2,  5.4],
            [ 0.4, -2.2,  1.5]]), requires_grad=True)
        self.sum_As = (4, 5, 6, 3)

    def test_masked_mm(self):
        A = self.A
        B = self.B
        Mi = self.m._indices()
        M = self.m.to_dense()

        # Check forward
        Cv = sp.masked_mm(A, B, Mi)
        Cd = (A @ B) * M
        self.assertEpsilonEqual(Cv, Cd[Mi[0], Mi[1]], 1e-4)

        # Check grad B
        grad_Bs, = grad(Cv.sum(), B, retain_graph=True)
        grad_Bd, = grad(Cd.sum(), B, retain_graph=True)
        self.assertEpsilonEqual(grad_Bs, grad_Bd, 1e-4)

        # Check grad A
        grad_As, = grad(Cv.sum(), A)
        grad_Ad, = grad(Cd.sum(), A)
        self.assertEpsilonEqual(grad_As, grad_Ad, 1e-4)

    def test_mm(self):
        Ai = self.i
        Av = self.v
        As = self.s

        A = sp.mask(Ai, Av, As)
        Ad = Variable(A.to_dense(), requires_grad=True)

        # Check forward
        Ms = A.mm(self.B)
        Md = Ad @ self.B
        self.assertEpsilonEqual(Ms, Md, 1e-4)

        # Check grad B
        grad_Bs, = grad(Ms.sum(), self.B, retain_graph=True)
        grad_Bd, = grad(Md.sum(), self.B, retain_graph=True)
        self.assertEpsilonEqual(grad_Bs, grad_Bd, 1e-4)

        # Check grad A
        grad_Av, = grad(Ms.sum(), Av)
        grad_Ad, = grad(Md.sum(), Ad)
        self.assertEpsilonEqual(grad_Av, grad_Ad[Ai[0], Ai[1]], 1e-4)

    def test_sum(self):
        Ai = self.sum_Ai
        Av = self.sum_Av
        As = self.sum_As

        A = sp.mask(Ai, Av, As)
        Ad = Variable(A.to_dense(), requires_grad=True)

        for keepdims in (True, False):
            for dims in (None, (0, ), (1, ), (2, ),
                         (1, 2), (2, 0), (0, 1, 2), ()):

                # Check forward
                Ss = A.sum(dims, keepdims)
                Sd = Ad
                if dims is None:
                    dims = range(Ai.size(0))
                dims = np.flip(np.sort(dims), 0)
                for i, d in enumerate(dims):
                    Sd = Sd.sum(int(d), keepdim=keepdims)

                self.assertEpsilonEqual(Ss, Sd, 1e-4)

                # Check grad
                grad_Av, = grad(Ss.sum(), Av, retain_graph=True)
                grad_Ad, = grad(Sd.sum(), Ad, retain_graph=True)

                self.assertEpsilonEqual(
                    grad_Av, grad_Ad[[idxs for idxs in Ai]], 1e-4)

    def test_to_dense(self):
        for Ai, Av, As in zip(
                (self.sum_Ai, self.i),
                (self.sum_Av, self.v),
                (self.sum_As, self.s)):

            A = sp.mask(Ai, Av, As)
            Ad = Variable(A.to_sparse().to_dense(), requires_grad=True)
            
            # check forward
            self.assertEpsilonEqual(A.to_dense(), Ad, 1e-4)

            # check backward
            grad_Av, = grad(A.to_dense().sum(), Av, retain_graph=True)
            grad_Ad, = grad(Ad.sum(), Ad, retain_graph=True)

            self.assertEpsilonEqual(
                grad_Av, grad_Ad[[idxs for idxs in Ai]], 1e-4)


@unittest.skipUnless(torch.cuda.is_available(), "Cuda unavailable.")
class TestSparseCuda(TestSparse):

    def setUp(self):
        super().setUp()
        for k, v in vars(self).items():
            if isinstance(v, torch.Tensor):
                setattr(self, k, v.cuda())
