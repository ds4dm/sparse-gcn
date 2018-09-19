# coding: utf-8

"""Functions for automatic differentiation of masked tensors."""

from typing import Tuple, Optional

import torch
from torch.autograd import Function


class MatMul(Function):
    """See matmul()."""

    @staticmethod
    def forward(
        ctx,
        indices: torch.Tensor,
        values: torch.Tensor,
        size: torch.Size,
        B: torch.Tensor
    ) -> torch.Tensor:
        """Forward computation.

        Computes the matrix multiplication of `A` and `B`, where `A` is defined
        by the sparse tensor (`indices`, `values`, `shape`)

        Parameters
        ----------
        indices
            Indices to define `A` according to Pytorch sparse tensor.
        values:
            Values to define `A` according to Pytorch sparse tensor.
        size:
            Size to define `A` according to Pytorch sparse tensor.
        B:
            Other matrix to multiply by.

        Returns
        -------
        output
            The result of the sparse multiplication of `A` and `B`.

        """
        A = torch.sparse_coo_tensor(indices, values, size)
        # Save indices because we don't want to rely on `A._indices()`
        ctx.save_for_backward(A, indices, B)
        return A @ B

    @ staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> Tuple[None, Optional[torch.Tensor], None, Optional[torch.Tensor]]:
        """Backward computation.

        Parameters
        ----------
        grad_output:
            The gradient of some quantity with regard to the output of this
            function.

        Returns
        -------
        outputs:
            The gradient wrt to each of the inputs of forward. None if does
            not exist or not required.

        """
        A, indices, B = ctx.saved_tensors
        grad_A = grad_B = None

        if ctx.needs_input_grad[1]:
            grad_A = matmulmasked(grad_output, B.t(), indices)
        if ctx.needs_input_grad[3]:
            grad_B = A.t() @ grad_output

        return None, grad_A, None, grad_B


# define function aliases, useful since Function.apply()
# does not support named arguments
def matmul(
    indices: torch.Tensor,
    values: torch.Tensor,
    size: torch.Size,
    B: torch.Tensor
) -> torch.Tensor:
    """Matrix multiplication with a sparse tensor.

    Computes the matrix multiplication of `A` and `B`, where `A` is defined
    by the sparse tensor (`indices`, `values`, `shape`)

    Parameters
    ----------
    indices
        Indices to define `A` according to Pytorch sparse tensor.
    values:
        Values to define `A` according to Pytorch sparse tensor.
    size:
        Size to define `A` according to Pytorch sparse tensor.
    B:
        Other matrix to multiply by.

    Returns
    -------
    output
        The result of the sparse multiplication of `A` and `B`.

    """
    return MatMul.apply(indices, values, size, B)


def matmulmasked(
    A: torch.Tensor, B: torch.Tensor, indices: torch.Tensor
) -> torch.Tensor:
    """Matrix multiplication and mask.

    This function computes `(A @ B) * m` in a masked way. Only the values with
    a positive mask are computed by levraging sparse computations.
    Note that this function yields a small numeric difference from its
    equivalent dense version.

    Parameters
    ----------
    A
        Tensor of size (n, p)
    B
        Tensor of size (p, m)
    indices
        The mask defining the computation to do. Has to be given according to
        Pytorch indices convention for sparse tensors.

    Returns
    -------
    values
        The values of `A @ B` associated with the `indices` passed.

    """
    idx, jdx = indices
    A_values = A.index_select(0, idx)
    B_values = B.t().index_select(0, jdx)

    AB_values = torch.bmm(
        A_values.view(A_values.size(0), 1, -1),
        B_values.view(B_values.size(0), -1, 1)
    ).view(-1)

    return AB_values
