# coding: utf-8

import numpy as np
import pytest
import torch
from mock import MagicMock

from sgcn.masked.tensor import MaskedTensor


def _allclose(A, B):
    return np.allclose(A.detach().cpu(), B.detach().cpu())


@pytest.fixture
def data(device):
    idx = torch.tensor(
        [[1, 1, 9, 4], [2, 3, 0, 4]], dtype=torch.int64, device=device)
    values = torch.rand(4, 3, 9, device=device)
    return idx, values, (13, 7, 3, 9)


@pytest.fixture
def maskedtensor(data):
    return MaskedTensor(*data)


def test_initialization(data):
    idx, val, shape = data

    # shape
    assert MaskedTensor(idx, val).shape == (10, 5, 3, 9)
    assert MaskedTensor(idx, val, shape).shape == shape
    assert isinstance(MaskedTensor(idx, val).shape, torch.Size)
    # dtype
    assert MaskedTensor(idx, val).dtype == val.dtype
    assert MaskedTensor(idx, val, dtype=torch.int).dtype == torch.int
    # device
    assert isinstance(
        MaskedTensor(idx, val, device="cpu").device, torch.device)
    # These is a test when fixture device is cuda.
    assert MaskedTensor(idx, val, device="cpu").device.type == "cpu"
    assert MaskedTensor(idx.cpu(), val).device == val.device
    # indices
    assert MaskedTensor(idx.float(), val).indices.dtype == idx.dtype
    with pytest.raises(ValueError):
        MaskedTensor(idx.t(), val)
    # values
    with pytest.raises(ValueError):
        MaskedTensor(idx, val[1:])


def test_to_sparse(data):
    m = MaskedTensor(*data)
    s = torch.sparse_coo_tensor(*data)

    assert torch.is_tensor(m.to_sparse())
    assert m.to_sparse().is_sparse
    assert _allclose(s.to_dense(), m.to_sparse().to_dense())


def test_from_sparse(data):
    s = torch.sparse_coo_tensor(*data)
    m = MaskedTensor.from_sparse(s)

    assert isinstance(m, MaskedTensor)
    # Should suceed is test_to_sparse succeed
    assert _allclose(s.to_dense(), m.to_sparse().to_dense())


def test_size(maskedtensor):
    assert maskedtensor.size() == maskedtensor.shape
    assert maskedtensor.size(3) == maskedtensor.shape[3]


def test_dims(maskedtensor):
    assert maskedtensor.sparse_dims == 2
    assert maskedtensor.dense_dims == 2
    assert maskedtensor.dims == 4


def test_with_values(maskedtensor):
    v = torch.rand(4, 13).to(maskedtensor.device)
    m = maskedtensor.with_values(v)

    assert isinstance(m, MaskedTensor)
    assert (m.values == v).all().item()
    assert m.shape == (13, 7, 13)
    assert m.dtype == maskedtensor.dtype
    assert m.device == maskedtensor.device
    assert m.indices is maskedtensor.indices


def test_apply(maskedtensor):
    v = torch.rand(4, 13).to(maskedtensor.device)
    func = MagicMock()
    func.return_value = v
    m = maskedtensor.apply(func)

    assert isinstance(m, MaskedTensor)
    func.assert_called_once_with(maskedtensor.values)
    assert (m.values == v).all().item()
    assert m.indices is maskedtensor.indices
    assert (m.shape[:m.sparse_dims]
            == maskedtensor.shape[:maskedtensor.sparse_dims])


def test_sum(maskedtensor):
    assert (maskedtensor.sum() == maskedtensor.values.sum()).item()
    # sum on dense dim
    m = maskedtensor.sum(2)
    assert (m.values == maskedtensor.values.sum(1)).all().item()
    assert m.shape == (13, 7, 9)
    # keepdim
    assert maskedtensor.sum(2, True).shape == (13, 7, 1, 9)
    # not implemented yet
    with pytest.raises(NotImplementedError):
        maskedtensor.sum(1)


def test_mm(maskedtensor, device):
    # mm only works for sparse matrices (2 sparse dims)
    maskedtensor = maskedtensor.with_values(maskedtensor.values[:, 0, 0])
    other = torch.rand(7, 5, device=device)

    result = maskedtensor.mm(other)
    expected_result = maskedtensor.to_sparse() @ other
    assert torch.is_tensor(result)
    assert not result.is_sparse
    assert _allclose(result, expected_result)


def test_mv(maskedtensor, device):
    # mv only works for sparse matrices (2 sparse dims)
    maskedtensor = maskedtensor.with_values(maskedtensor.values[:, 0, 0])
    other = torch.rand(7, device=device)

    result = maskedtensor.mv(other)
    expected_result = (maskedtensor.to_sparse().to_dense() @ other)
    assert torch.is_tensor(result)
    assert not result.is_sparse
    assert _allclose(result, expected_result)


def test_mask_mm(maskedtensor, device):
    A = torch.rand(13, 5, device=device)
    B = torch.rand(5, 7, device=device)
    dense_mask = maskedtensor.with_values(torch.ones(4, device=device)) \
                             .to_sparse().to_dense()

    result = maskedtensor.mask_mm(A, B)
    expected_result = (A @ B) * dense_mask
    assert isinstance(result, MaskedTensor)
    assert result.shape == (13, 7)
    assert _allclose(result.to_sparse().to_dense(), expected_result)


def test_transpose(maskedtensor):
    # sparse dims
    t = maskedtensor.transpose(0, 1)
    assert t.shape == (7, 13, 3, 9)
    assert (t.indices[0] == maskedtensor.indices[1]).all().item()
    assert (t.indices[1] == maskedtensor.indices[0]).all().item()

    # dense dims
    t = maskedtensor.transpose(2, 3)
    assert t.shape == (13, 7, 9, 3)
    assert (t.values == maskedtensor.values.transpose(1, 2)).all().item()

    # raises
    with pytest.raises(RuntimeError):
        maskedtensor.transpose(0, 2)


def test_t(maskedtensor):
    with pytest.raises(RuntimeError):
        maskedtensor.t()

    # Make 2 dimensional
    maskedtensor = maskedtensor.with_values(maskedtensor.values[:, 0, 0])
    assert maskedtensor.t().shape == (7, 13)
    assert (maskedtensor.t().indices
            == maskedtensor.transpose(0, 1).indices).all().item()
