# coding: utf-8

"""PyTorch sparse differention.

In this module, we implement various functions using the sparse representation
available in PyTorch.
We implement the differentiation of these function on their support. More
precisely: the gradient of a function with regard to a sparse input is not
necessarily sparse. Here we actually use the PyTorch sparse representation
as a mask. This means we don't want computation to occur on values not defined
by the mask.
"""

import torch
from torch.autograd import Function

import numpy as np


class Build(Function):
    """See build() function."""

    @staticmethod
    def forward(ctx, i, v, skip_check, *args):
        """Forward computation.

        Parameters
        ----------
        i : LongTensor
            Indices tensor as given to `sparse.FloatTensor`.
        v : FloatTensor
            Values tensor as given to `sparse.FloatTensor`.
        *args
            Additional options given to `sparse.FloatTensor`.
        skip_check: boolean
            Decides wether to skip the check for coalesced indices
            on the input. May be useful to save computations in the
            forward pass.

        Returns
        -------
        sparse.FloatTensor
            A new sparse tensor.

        """
        ctx.n_options = len(args)

        _torch = torch.cuda if v.is_cuda else torch
        output = _torch.sparse.FloatTensor(i, v, *args).coalesce()

        if not skip_check and (
                not i.shape == output._indices().shape or
                not i.eq(output._indices()).all()):
            raise Exception("Input indices must be in coalesced form.")

        return output

    @staticmethod
    def backward(ctx, output_grad):
        """Backward computation.

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
            v_grad = output_grad.coalesce()._values()

        return (None, v_grad, None) + (None, ) * ctx.n_options


class Values(Function):
    """See values()."""

    @staticmethod
    def forward(ctx, A):
        """Forward computation.

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
        """Backward computation.

        Parameters
        ----------
        output_grad : FloatTensor
            The gradient of something w.r.t the output of forward().

        Returns
        -------
        sparse.FloatTensor
            The gradient of the same thing w.r.t to the input sparse
            tensor. Note that this is not the true gradient but only the
            gradient w.r.t the values defined in the sparse tensor.

        """
        i, = ctx.saved_tensors
        size = ctx.size

        A_grad = None
        if ctx.needs_input_grad[0]:
            _torch = torch.cuda if grad_output.is_cuda else torch
            A_grad = _torch.sparse.FloatTensor(i, grad_output, size)

        return A_grad


class Sum(Function):
    """See sum()."""

    @staticmethod
    def forward(ctx, input, dims):
        """Forward computation.

        Parameters
        ----------
        input : sparse.FloatTensor
            The sparse tensor, must be coalesced.
        dims: tuple of ints
            The dimensions over which to sum, or `None` to
            sum over all dimensions.

        Returns
        -------
        sparse.FloatTensor
            The sparse tensor summed over dimensions in `dims`.
            Note that the original dimensions are kept but
            shrinked to size 1.

        """
        if not input.is_coalesced():
            raise Exception("Sparse input must be coalesced.")

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
        """Backward pass.

        Parameters
        ----------
        output_grad : FloatTensor
            The gradient of something w.r.t the output of forward().

        Returns
        -------
        sparse.FloatTensor
            The gradient of the same thing w.r.t to the input sparse
            tensor. Note that this is not the true gradient but only the
            gradient w.r.t the values defined in the sparse tensor.

        """
        _torch = torch.cuda if grad_output.is_cuda else torch
        
        input_indices, coalesced_indices = ctx.saved_tensors

        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_output = grad_output.coalesce()

            output_indices = grad_output._indices()

            # assumes input is coalesced, which implies the indices are
            # sorted from first to last dimension
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
    """See matmul()."""

    @staticmethod
    def forward(ctx, A, B):
        """Forward computation.

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
        """Backward computation.

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


# define function aliases, useful since Function.apply()
# does not support named arguments
def build(i, v, *args, skip_check=False):
    """Builds a sparse Tensor.

    Constructs a pytorch sparse tensor (on the same device as
    the input). The derivatives are computed w.r.t the values
    (as given to a sparse tensor). This is similar to calling
    `sparse.FloatTensor(i, v, *args)` with differentiation
    support.

    Parameters
    ----------
    i : LongTensor
        Indices tensor as given to `sparse.FloatTensor`.
    v : FloatTensor
        Values tensor as given to `sparse.FloatTensor`.
    *args
        Additional options given to `sparse.FloatTensor`.
    skip_check: boolean
        Decides wether to skip the check for coalesced indices
        on the input. May be useful to save computations in the
        forward pass.

    Returns
    -------
    sparse.FloatTensor
        A new sparse tensor.

    """
    return Build.apply(i, v, skip_check, *args)


def values(t):
    """Extracts the values stored in a sparse tensor.

    The values are the values actually kept in memory.
    This is equivalent to calling `t._values()` with
    differentiation support.

    Parameters
    ----------
    t : sparse.FloatTensor
        The sparse tensor.

    Returns
    -------
    FloatTensor
        One-dimensional FloatTensor
    """
    return Values.apply(t)


def sum(t, dims=None):
    """
    Sums a sparse tensor over some specific or all dimensions.

    Parameters
    ----------
    t : sparse.FloatTensor
        The sparse tensor, must be coalesced.
    dims: tuple of ints
        The dimensions over which to sum, or `None` to
        sum over all dimensions.

    Returns
    -------
    sparse.FloatTensor
        The sparse tensor summed over dimensions in `dims`.
        Note that the original dimensions are kept but
        shrinked to size 1.

    """
    return Sum.apply(t, dims)


def matmul(A, B):
    """
    Matrix multiplication with a sparse tensor.

    This is equivalent to calling `A @ B` where `A` is
    sparse and `B` is dense, with differentiation support
    w.r.t both `A` and `B`.

    Parameters
    ----------
    A : sparse.FloatTensor
        A sparse matrix.
    B : FloatTensor
        A dense matrix

    Returns
    -------
    FloatTensor
        The dense matrix resulting from `A @ B`

    """
    return MatMul.apply(A, B)


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
