# coding: utf-8


import unittest
import unittest.mock as mock

import torch
import numpy as np

import sgcn


class TestAttention(unittest.TestCase):

    def setUp(self):
        self.K = torch.rand((7, 3)).requires_grad_()
        self.Q = torch.rand((4, 3)).requires_grad_()
        self.V = torch.rand((7, 4)).requires_grad_()
        self.md = (torch.rand((4, 7)) > .6).float()
        idx = self.md.nonzero()
        self.ms = torch.sparse_coo_tensor(
            idx.t(), torch.ones(len(idx)), self.md.size())

    def test_attention(self):
        # Mock module forward
        aff = type(
            "Affinity",
            (sgcn.nn.Affinity, ),
            {"forward": mock.MagicMock(return_value=self.ms)}
        )()
        norm = type(
            "Normalization",
            (sgcn.nn.Normalization, ),
            {"forward": mock.MagicMock(return_value=self.ms)}
        )()

        attention = sgcn.nn.Attention(affinity=aff, normalization=norm)
        attention(self.K, self.V, self.Q, m=self.ms)

        aff.forward.assert_called_once_with(self.Q, self.K, self.ms)
        norm.forward.assert_called_once_with(self.ms)

    def test_multi_head_attention(self):
        att = sgcn.nn.MultiHeadAttention(
            in_key=3, in_value=4, in_query=3,
            n_head=13, head_qk=9, head_v=11
        )
        out_s = att(self.K, self.V, self.Q, m=self.ms)
        out_d = att(self.K, self.V, self.Q, m=self.md)

        self.assertTupleEqual(out_s.shape, (4, 13*11))
        np.testing.assert_allclose(
            out_s.detach().cpu(), out_d.detach().cpu(), atol=1e-5
        )

    def test_multi_head_attention_backward(self):
        att = sgcn.nn.MultiHeadAttention(
            in_key=3, in_value=4, in_query=3,
            n_head=13, head_qk=9, head_v=11
        )
        out_s = att(self.K, self.V, self.Q, m=self.ms)

        out_s.sum().backward()


@unittest.skipUnless(torch.cuda.is_available(), "Cuda unavailable.")
class TestAttentionCuda(TestAttention):
    pass
