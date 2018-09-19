# coding: utf-8

import mock
import pytest
import numpy as np
import torch

import sgcn
from sgcn.masked.tensor import MaskedTensor


def _allclose(A, B):
    return np.allclose(A.detach().cpu(), B.detach().cpu(), 1e-4, 1e-7)


@pytest.fixture(params=["dense", "masked"])
def data(device, request):
    K = torch.rand(7, 3, device=device, requires_grad=True)
    V = torch.rand(7, 4, device=device, requires_grad=True)
    Q = torch.rand(4, 3, device=device, requires_grad=True)
    md = (torch.rand(4, 7, device=device, requires_grad=True) > .6).float()

    if request.param == "dense":
        return K, V, Q, md
    else:
        idx = md.nonzero()
        mm = MaskedTensor(idx.t(), torch.ones(len(idx), device=device), (4, 7))
        return K, V, Q, mm


def test_attention(data):
    K, V, Q, m = data
    # Mock module forward
    aff = type(
        "Affinity",
        (sgcn.nn.Affinity, ),
        {"forward": mock.MagicMock(return_value=m)}
    )()
    norm = type(
        "Normalization",
        (sgcn.nn.Normalization, ),
        {"forward": mock.MagicMock(return_value=m)}
    )()

    attention = sgcn.nn.Attention(affinity=aff, normalization=norm)
    attention(K, V, Q, m=m)

    aff.forward.assert_called_once_with(Q, K, m)
    norm.forward.assert_called_once_with(m)


def test_multi_head_attention(data, device):
    K, V, Q, m = data
    att = sgcn.nn.MultiHeadAttention(
        in_key=3, in_value=4, in_query=3,
        n_head=13, head_qk=9, head_v=11
    ).to(device)
    out = att(K, V, Q, m=m)

    assert out.shape == (4, 13*11)
    if isinstance(m, MaskedTensor):
        out_d = att(K, V, Q, m=m.to_sparse().to_dense())
        assert _allclose(out, out_d)


def test_multi_head_attention_backward(data, device):
    K, V, Q, m = data
    att = sgcn.nn.MultiHeadAttention(
        in_key=3, in_value=4, in_query=3,
        n_head=13, head_qk=9, head_v=11
    ).to(device)
    out = att(K, V, Q, m=m)

    out.sum().backward()
