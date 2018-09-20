# coding: utf-8


"""Module for the MaskedTensor class."""

# FIXME: Pyflakes flaging unquoted forward references
from __future__ import annotations

from typing import Optional, Union, Callable

import attr
import torch

from . import functions as func


@attr.s(auto_attribs=True, frozen=True, cmp=False)
class MaskedTensor:
    """MaskedTensor class.

    A class define a sparse Tensor where the sparsity pattern represent
    the sparsity patter represent the only existing element.
    This would be similar to have a mask (all the indices not provided) of
    zeros applied after each operations.

    For instance, the gradient with reagrd to this tensor would not have the
    same sparsity pattern, even though the gradient wrt a sparse tensor isn't
    necessarily sparse.

    This class is represented as a hybrid COO sparse tensor. Sparse dimesnions
    are the firs ones and dense dimensions are the last ones.

    The tensor must always be caolesced, i.e. it should not contain duplicate
    coordinates, otherwise many opertations would not give the right results.
    This is not enforsed explicitly.

    Why not differentiate through PyTorch sparse tensors?
    As of PyTorch version 0.4.1, it is difficlut to enforce that the values
    stay in the same order. More especially, coalescing doesn't always put the
    indices in the same order (example on cpu, using `transpose` then
    `coalesce`). Being transparent on the indices ensure that no permuation is
    done without explicit knowledge.
    """

    indices: torch.Tensor = attr.ib(converter=lambda x: x.long())
    values: torch.Tensor = attr.ib()
    shape: torch.Size = attr.ib(converter=torch.Size)
    dtype: torch.dtype = attr.ib()
    device: torch.device = attr.ib(converter=torch.device)

    @shape.default
    def _shape_default(self):
        sparse_dims, _ = self.indices.max(1)
        dense_dims = self.values.size()[1:]
        return torch.Size(1 + sparse_dims) + dense_dims

    @dtype.default
    def _dtype_default(self):
        return self.values.dtype

    @device.default
    def _device_default(self):
        return self.values.device

    @indices.validator
    def _check_indices(self, attribute, val):
        if len(val.size()) != 2:
            raise ValueError("Indices must have two dimensions.")

    @values.validator
    def _check_values(self, attribute, val):
        if self.indices.size(1) != val.size(0):
            raise ValueError("Indices and values must have same nnz.")

    def __attrs_post_init__(self):
        """Initialize after attr.

        Moves `indices` and `values` to the correct `device` and convert
        `values` to the desired type.
        """
        # Actualize the device and dtype of indices and values
        indices = self.indices.to(device=self.device)
        values = self.values.to(device=self.device, dtype=self.dtype)
        # Necessary to use this to overcome the frozen aspect of the class
        object.__setattr__(self, "indices", indices)
        object.__setattr__(self, "values", values)

    @classmethod
    def from_sparse(ctx, tensor: torch.Tensor) -> "MaskedTensor":
        """Build from a torch sparse tensor.

        This function is not diffrentiable.
        """
        return ctx(
            indices=tensor._indices(),
            values=tensor._values(),
            shape=tensor.shape,
            dtype=tensor.dtype,
            device=tensor.device
        )

    def to_sparse(self) -> torch.Tensor:
        """Return the object as a torch sparse Tensor.

        This method is not diffrentiable.
        """
        return torch.sparse_coo_tensor(
            indices=self.indices,
            values=self.values,
            size=self.shape,
            dtype=self.dtype,
            device=self.device
        )

    def size(self, dim: Optional[int] = None) -> Union[torch.Size, int]:
        """Shape of the MaskedTensor, or one of the specic dimension."""
        if dim is None:
            return self.shape
        else:
            return self.shape[dim]

    @property
    def sparse_dims(self) -> int:
        """Return the number of sparse dimensions."""
        return self.indices.size(0)

    @property
    def dense_dims(self) -> int:
        """Return the number of dense dimensions."""
        return len(self.values.shape) - 1

    @property
    def dims(self) -> int:
        """Return the total number of dimensions."""
        return len(self.shape)

    def with_values(self, values: torch.Tensor) -> "MaskedTensor":
        """Return a new MaskedTensor with different values.

        The sparsity pattern, device, and dtype are preserved.

        Parameters
        ----------
        values:
            New values to use. Must have the same dimensions across the
            first dimension as the previous values.

        Returns
        -------
        output:
            The output of the Tensor is wrapped in a new MaskedTensor, hence
            preserving the sparsity pattern.

        """
        # We have to compute the shape here because new values may not have the
        # same dimensions, and sparse dimensions might have been set to
        # something else than the default.
        # Shape is shape sparse + new shae dense.
        shape = self.shape[:self.sparse_dims] + values.size()[1:]
        return MaskedTensor(
            indices=self.indices,
            values=values,
            shape=shape,
            dtype=self.dtype,
            device=self.device
        )

    def apply(
        self, func: Callable[[torch.Tensor], torch.Tensor]
    ) -> "MaskedTensor":
        """Apply a function on all the values.

        Does not change the sparsity pattern.

        Parameters
        ----------
        func:
            Takes as input the values and returns a new Tensor. The function
            must preseve the first axis.

        Returns
        -------
        output:
            The output of the Tensor is wrapped in a new MaskedTensor, hence
            preserving the sparsity pattern.

        """
        return self.with_values(func(self.values))

    def transpose(self, dim1: int, dim2: int) -> "MaskedTensor":
        """Transpose two dimensions."""
        # Transposing two sparse dimensions by permuting axis
        if dim1 < self.sparse_dims and dim2 < self.sparse_dims:
            idx = torch.arange(self.indices.size(0))
            # Using item() because this gets a view of the tensor.
            idx[dim1], idx[dim2] = idx[dim2].item(), idx[dim1].item()
            # Using tensor to permute indexes of size
            shape = torch.Size(torch.tensor(self.shape).index_select(0, idx))
            shape += self.shape[self.sparse_dims:]
            return MaskedTensor(
                indices=self.indices.index_select(0, idx.to(self.device)),
                values=self.values,
                shape=shape,
                dtype=self.dtype,
                device=self.device
            )

        # Transposing two dense dimensions by simple transposition
        elif dim1 >= self.sparse_dims and dim2 >= self.sparse_dims:
            offset = self.sparse_dims - 1
            values = self.values.transpose(dim1 - offset, dim2 - offset)
            return self.with_values(values)

        else:
            raise RuntimeError("Transposing sparse and dense dims.")

    def t(self) -> "MaskedTensor":
        """Transpose a matrix."""
        if self.dims == 2:
            return self.transpose(0, 1)
        else:
            raise RuntimeError("t() is valid for matrices only.")

    def sum(
        self, dim: Optional[int] = None, keepdim: bool = False
    ) -> Union[torch.Tensor, "MaskedTensor"]:
        """Sum across one or all dimensions.

        Parameters
        ----------
        dim:
            Dimension to sum across, or None for all dimensions.
        keepdim:
            Whether or not to keep the original number of dimensions (the
            dimension summed over gets length one). Does nothing if no
            axis is specified.

        Returns
        -------
        output
            A new tensor, dense or masked depending on the sumation dimensions.

        """
        if dim is None:
            return self.values.sum()
        elif dim < self.sparse_dims:
            # Cannot remove dim and use coalesce because it may permutes
            # the values
            raise NotImplementedError("Summation over sparse dimension.")
        else:
            return self.with_values(
                self.values.sum(dim - self.sparse_dims + 1, keepdim=keepdim))

    def mm(self, other: torch.Tensor) -> torch.Tensor:
        """Perform matrix matrix multiplication.

        Supports self as sparse matrix (2 dense dims) and other as dense matrix
        (2 dims) with matching shapes.

        Parameters
        ----------
        other:
            Dense matrix left multiplied by this tensor.

        Returns
        -------
        output:
            The dense result of the `self` `@` `other`.

        """
        return func.matmul(self.indices, self.values, self.shape, other)

    def mv(self, other: torch.Tensor) -> torch.Tensor:
        """Perform matrix vector multiplication.

        Supports self as sparse matrix (2 dense dims) and other as dense vector
        (1 dim) with matching shapes.

        Parameters
        ----------
        other:
            Dense vector left multiplied by this tensor.

        Returns
        -------
        output:
            The dense result of the `self` `@` `other`.

        """
        return self.mm(other.unsqueeze(-1)).squeeze(-1)

    def mask_mm(self, A: torch.Tensor, B: torch.Tensor) -> "MaskedTensor":
        """Compute the masked matrix multiplication.

        Computes `A @ B` on;y for the indices provided in the current
        MaskedTensor.

        Parameters
        ----------
        A:
            Left dense matrix (two dimensions only).
        B:
            Right dense matrix (two dimensions only).

        Returns
        -------
        ouput:
            The masked result of `A @ B`.

        """
        return self.with_values(func.matmulmasked(A, B, self.indices))
