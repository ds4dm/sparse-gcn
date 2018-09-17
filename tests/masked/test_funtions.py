# coding: utf-8

import numpy as np
import pytest
import torch
from torch.autograd import gradcheck

import sgcn.masked.functions as F


def _allclose(A, B):
    return np.allclose(A.detach().cpu(), B.detach().cpu())


@pytest.fixture
def sparse_matrix_data(device):
    idx = torch.tensor(
        [[1, 1, 9, 4], [2, 3, 0, 4]], dtype=torch.int64, device=device)
    values = torch.rand(4, device=device)
    return idx, values, (13, 7)


def test_matmul_forward(sparse_matrix_data, device):
    A = torch.sparse_coo_tensor(*sparse_matrix_data)
    B = torch.rand(7, 5, device=device)
    AB_tested = F.matmul(*sparse_matrix_data, B)
    AB_expected = A @ B

    assert _allclose(AB_tested, AB_expected)


def test_matmul_grad(sparse_matrix_data, device):
    idx, values, size = sparse_matrix_data
    B = torch.rand(7, 5, device=device)

    # gradcheck requires double precision
    values = values.double().requires_grad_()
    B = B.double().requires_grad_()

    assert gradcheck(F.matmul, (idx, values, size, B))


def test_matmulmasked_forward(sparse_matrix_data, device):
    A = torch.rand(13, 5, device=device)
    B = torch.rand(5, 7, device=device)
    indices, _, _ = sparse_matrix_data

    AB_values_tested = F.matmulmasked(A, B, indices)
    AB_alues_expected = (A @ B)[tuple(indices)]
    assert _allclose(AB_values_tested, AB_alues_expected)


def test_matmulmasked_grad(sparse_matrix_data, device):
    # gradcheck requires double precision
    A = torch.rand(13, 5, device=device).double().requires_grad_()
    B = torch.rand(5, 7, device=device).double().requires_grad_()
    indices, _, _ = sparse_matrix_data

    assert gradcheck(F.matmulmasked, (A, B, indices))
