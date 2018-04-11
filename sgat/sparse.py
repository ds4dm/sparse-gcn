# coding: utf-8

"""PyTorch sparse differention.

In this module, we implement various functions using the sparse representation
available in PyTorch.
We implement the differentiation of these function on their support. More
precisely: the gradient of a function with regard to a sparse input is not
necessarily sparse. Here we actually use the PyTorch sparse representation
as a mask. This means we don't want computation to occure on values not defined
by the mask.
"""

import torch
from torch.autograd import Function

import numpy as np


class Build(Function):
    """Build a sparse Tensor.

    This class defines a function that construct a pytorch sparse tensor (on
    the same device as the input).
    The differentiation is implemented with regard to the values (as given to
    a sparse tensor).

    This is similar to calling `sparse.FloatTensor(i, v)`` except that this
    differentiable with regard to `v`.
    """

    @staticmethod
    def forward(ctx, i, v, *args, **kwargs):
        """Compute the function.

        Parameters
        ----------
        i : LongTensor
            Indices tensor as given to `sparse.FloatTensor`.
        v : FloatTensor
            Values tensor as given to `sparse.FloatTensor`.
        *args, **kwargs
            Additional options given to `sparse.FloatTensor`.

        Returns
        -------
        sparse.FloatTensor
            A new sparse tensor.

        """
        ctx.n_options = len(args) + len(kwargs)
        _torch = torch.cuda if v.is_cuda else torch

        return _torch.sparse.FloatTensor(i, v, *args, **kwargs)

    @staticmethod
    def backward(ctx, output_grad):
        """Compute the backpropagation.

        Parameters
        ----------
        output_grad : sparse.FloatTensor
            The gradient of some quantity with regard to the output of this
            function.

        Returns
        -------
        None
            The gradient with regard to the indices is not computed.
        FloatTensor
            The gradient of the same quantity with regard to the values of the
            sparse tensor.
        None
            Gradients with regard to additional options are not computed.

        """
        v_grad = None
        if ctx.needs_input_grad[1]:
            v_grad = output_grad._values()

        return (None, v_grad) + (None, ) * ctx.n_options


class Values(Function):
    """Extract the values of a sparse tensor.

    This class defines a function that extract the values pof a sparse tensor.
    The values are the values actually kept in memory.
    This is equivalent to calling `my_sparse_tensor._values()` expcept that the
    function is differentiable.
    """

    @staticmethod
    def forward(ctx, A):
        """Compute the function.

        Parameters
        ----------
        A : sparse.FloatTensor
            The sparse tensor.

        Returns
        -------
        FloatTensor
            The values of the sparse tensor i.e. the ones actually stored
            in the representation

        """
        ctx.save_for_backward(A._indices())
        ctx.size = A.size()
        return A._values()

    @staticmethod
    def backward(ctx, grad_output):
        """Compute the backpropagation.

        Parameters
        ----------
        output_grad : FloatTensor
            The gradient of some quantity with regard to the output of this
            function.

        Returns
        -------
        sparse.FloatTensor
            The gradient of the same quantity with regard to the input sparse
            tensor. Note that this is not the true gradient but only the
            gradient for the values defined in the sparse tensor.

        """
        i, = ctx.saved_tensors
        size = ctx.size

        A_grad = None
        if ctx.needs_input_grad[0]:
            _torch = torch.cuda if grad_output.is_cuda else torch
            A_grad = _torch.sparse.FloatTensor(i, grad_output, size)

        return A_grad


class Sum(Function):

    @staticmethod
    def forward(ctx, input, dims=None):
        _torch = torch.cuda if input.is_cuda else torch

        ndims = len(input.size())
        nvals = input._values().size(0)

        if dims is None:
            dims=list(range(ndims))
        else:
            # safety checks on dims
            dims = np.asarray(dims)
            assert np.all(dims >= 0)
            assert np.all(dims < ndims)
            assert len(np.unique(dims)) == len(dims)
            dims = np.sort(dims).tolist()

        zero_idx = _torch.LongTensor(1).zero_().expand(nvals)

        indices = []
        size = []
        for d in range(ndims):
            if d not in dims:
                indices.append(input._indices()[d])
                size.append(input.size(d))
            else:
                indices.append(zero_idx)
                size.append(1)

        indices = torch.stack(indices)

        ctx.save_for_backward(input._indices(), indices)
        ctx.input_nvals = nvals
        ctx.input_shape = input.shape

        output = _torch.sparse.FloatTensor(
            indices,
            input._values(),
            size,
        ).coalesce()

        return output

    @staticmethod
    def backward(ctx, grad_output):
        _torch = torch.cuda if grad_output.is_cuda else torch
        
        input_indices, coalesced_indices = ctx.saved_tensors

        grad_input = None
        if ctx.needs_input_grad[0]:
            output_indices = grad_output._indices()

            # assume input is coalesced and indices are sorted from first to last dimension
            count = _torch.sparse.FloatTensor(
                coalesced_indices,
                _torch.FloatTensor(ctx.input_nvals).fill_(1),
                ctx.input_shape).coalesce()._values()

            out_to_in_map = _torch.LongTensor(np.repeat(
                np.arange(grad_output._values().size(0)),
                count))

            grad_input = _torch.sparse.FloatTensor(
                input_indices,
                grad_output._values()[out_to_in_map],
                ctx.input_shape)

        return grad_input, None


class MatMul(Function):
    """Matrix multiplication with a sparse tensor.

    This class defines a function compute `A @ B` where `A` is sparse and `B`
    is dense. This is differentiable with regard to `B` and to the values of
    `A`.
    """

    @staticmethod
    def forward(ctx, A, B):
        """Compute the function.

        Parameters
        ----------
        A : sparse.FloatTensor
            A sparse matrix.
        B : FloatTensor
            A dense matrix

        Returns
        -------
        FloatTensor
            The computation of `A @ B`

        """
        ctx.save_for_backward(A, B)
        return A @ B

    @ staticmethod
    def backward(ctx, grad_output):
        """Compute the backpropagation.

        Parameters
        ----------
        output_grad : FloatTensor
            The gradient of some quantity with regard to the output of this
            function.

        Returns
        -------
        sparse.FloatTensor
            The gradient of the same quantity with regard to the input sparse
            tensor `A`. Note that this is not the true gradient but only the
            gradient for the values defined in the sparse tensor.
        FloatTensor
            The gradient of the same quantity with regard to `B`.

        """
        A, B = ctx.saved_tensors
        grad_A = grad_B = None

        if ctx.needs_input_grad[0]:
            # values of A are not used, only its indexes
            grad_A = matmulmasked(grad_output, B.t(), A)
        if ctx.needs_input_grad[1]:
            grad_B = A.t() @ grad_output

        return grad_A, grad_B


# We alias the function applications
build = Build.apply
values = Values.apply
sum = Sum.apply
matmul = MatMul.apply


def matmulmasked(A, B, m):
    """Matrix multiplication and mask.

    This function computes `(A @ B) * m` in a sparse way. Only the values with
    a positive mask are computed by levraging sparse computations.
    Note that this function yields a small numeric difference from its
    equivalent dense version.

    Parameters
    ----------
    A : FloatTensor
        FloatTensor of size (n, p)
    B : FloatTensor
        FloatTensor of size (p, m)
    m : sparse.FloatTensor of size (n, m)
        The mask defining the computation to do.

    Returns
    -------
    sparse.FloatTensor
        The sparse equivalent of `(A @ B) * m`.

    """
    idx, jdx = m._indices()
    Av = A.index_select(0, idx)
    Bv = B.t().index_select(0, jdx)

    ABv = torch.bmm(
        Av.view(Av.size(0), 1, -1),
        Bv.view(Bv.size(0), -1, 1)
    ).view(-1)

    return build(m._indices(), ABv, (A.size(0), B.size(1)))
