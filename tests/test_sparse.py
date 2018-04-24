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
        Ai = self.i
        Av = self.v
        As = self.s

        A = sp.mask(Ai, Av, As)
        Ad = Variable(A.to_dense(), requires_grad=True)

        for dim in (None, 0, 1):
            for keepdim in (True, False):
                # Check forward
                sum_kwargs = {} if dim is None else {
                    'dim': dim,
                    'keepdim': keepdim,
                }
                Ss = A.sum(**sum_kwargs)
                Sd = Ad.sum(**sum_kwargs)
                self.assertEpsilonEqual(Ss, Sd, 1e-4)

                # Check grad
                grad_Av, = grad(Ss.sum(), Av, retain_graph=True)
                grad_Ad, = grad(Sd.sum(), Ad, retain_graph=True)
                self.assertEpsilonEqual(grad_Av, grad_Ad[Ai[0], Ai[1]], 1e-4)


@unittest.skipUnless(torch.cuda.is_available(), "Cuda unavailable.")
class TestSparseCuda(TestSparse):

    def setUp(self):
        super().setUp()
        for k, v in vars(self).items():
            if isinstance(v, torch.Tensor):
                setattr(self, k, v.cuda())
