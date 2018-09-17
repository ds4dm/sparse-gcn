# coding: utf-8

import numpy as np
import pytest
import torch

from sgcn.masked.tensor import MaskedTensor
import sgcn.nn.affinity as aff


def _allclose(A, B):
    return np.allclose(A.detach().cpu(), B.detach().cpu())


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


def test_dotproduct_no_mask(data):
    K, _, Q, _ = data
    func = aff.DotProduct(scaled=False)
    coefs = func(Q, K)
    assert torch.is_tensor(coefs)
    assert _allclose(coefs, Q @ K.t())


def test_dotproduct(data):
    K, _, Q, m = data
    func = aff.DotProduct(scaled=False)
    coefs = func(Q, K, m)
    assert isinstance(coefs, m.__class__)

    # check if mask is respected
    if isinstance(m, MaskedTensor):
        dense_mask = m.to_sparse().to_dense()
        dense_coefs = coefs.to_sparse().to_dense()
    else:
        dense_mask = m
        dense_coefs = coefs

    checks = (dense_coefs * (1 - dense_mask)) > 0
    assert not checks.any()
