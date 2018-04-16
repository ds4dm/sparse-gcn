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
        skip_check : boolean
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
            The gradient of something w.r.t the output of forward().

        Returns
        -------
        None
            The gradient w.r.t the indices is not computed.
        FloatTensor
            The gradient w.r.t the values of the sparse tensor.
        None
            Gradients w.r.t additional options are not computed.

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
            in the internal representation.

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
            The gradient w.r.t to the input sparse tensor.

        """
        i, = ctx.saved_tensors
        size = ctx.size

        A_grad = None
        if ctx.needs_input_grad[0]:
            _torch = torch.cuda if grad_output.is_cuda else torch
            A_grad = _torch.sparse.FloatTensor(i, grad_output, size)

        return A_grad


class MaskedSum(Function):
    """
    Sum of a masked matrix `A=A_*M`.
    
    """

    @staticmethod
    def forward(ctx, input, dims):
        """Forward computation.

        Parameters
        ----------
        input : sparse.FloatTensor
            A sparse tensor representing `A_*M`, must be coalesced.
        dims : tuple of ints
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
        """Backward computation.

        Parameters
        ----------
        output_grad : FloatTensor
            The gradient of something w.r.t the output of forward().

        Returns
        -------
        sparse.FloatTensor
            The gradient of the same thing w.r.t to `A_`.

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


class MaskedMatMul(Function):
    """
    Masked matrix multiplication between a masked matrix `A=A_*M` and a dense
    matrix `B`.
    
    """

    @staticmethod
    def forward(ctx, A, B):
        """Forward computation.

        Parameters
        ----------
        A : sparse.FloatTensor
            A sparse matrix representing `A_*M`.
        B : FloatTensor
            A dense matrix

        Returns
        -------
        FloatTensor
            The result of `(A_*M) @ B`

        """
        ctx.save_for_backward(A, B)
        return A @ B

    @ staticmethod
    def backward(ctx, grad_output):
        """Backward computation.

        Parameters
        ----------
        output_grad : FloatTensor
            The gradient of something w.r.t the output of forward().

        Returns
        -------
        sparse.FloatTensor
            The gradient of the same thing w.r.t `A_`.
        FloatTensor
            The gradient of the same thing w.r.t `B`.

        """
        A, B = ctx.saved_tensors
        grad_A = grad_B = None

        if ctx.needs_input_grad[0]:
            # values of A are not used, only its indexes
            grad_A = matmulmasked(grad_output, B.t(), A._indices())
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
    skip_check : boolean
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
    """Extracts the values stored in a sparse tensor. The values are those
    actually kept in memory. This is equivalent to calling `t._values()` with
    differentiation support.

    Parameters
    ----------
    t : sparse.FloatTensor
        A sparse tensor.

    Returns
    -------
    FloatTensor
        One-dimensional FloatTensor

    """
    return Values.apply(t)


def matmulmasked(A, B, mask_idxs):
    """Matrix multiplication and mask.

    This function computes `(A @ B) * M` efficitently by leveraging sparse
    computations, where `M` is an implicit constant binary matrix with 1
    values at indices in `mask_idxs`. Only the values of non masked-out
    elements are computed and returned in the form of a sparse tensor. Note
    that as a result one will observe a small numeric difference compared to
    the equivalent dense operation.

    Parameters
    ----------
    A : FloatTensor
        Matrix of size (n, p)
    B : FloatTensor
        Matrix of size (p, m)
    mask_idxs : LongTensor
        Matrix of size (2, n_ones) whose columns correspond to the indices of
        the 1 values in the binary mask `M`.

    Returns
    -------
    SparseTensor
        The sparse equivalent of `(A @ B) * M`.

    """
    idx, jdx = mask_idxs
    Av = A.index_select(0, idx)
    Bv = B.t().index_select(0, jdx)

    ABv = torch.bmm(
        Av.view(Av.size(0), 1, -1),
        Bv.view(Bv.size(0), -1, 1)
    ).view(-1)

    return build(mask_idxs, ABv, (A.size(0), B.size(1)))


def mask(A_):
    return MaskedTensor(A_)


class MaskedTensor:
    """
    Represents a tensor `A` resulting from the application of an implicit
    (constant) binary mask `M` to an initial tensor `A_`, i.e. `A=A_*M`.
    Internally, only the non-masked entries of `A_` and their indices are
    stored in the form of a sparse tensor. As a result, derivatives w.r.t the
    original tensor `A_` are sparse since masked-out elements have zero
    gradient.

    """

    def __init__(self, A_):
        """
        Builds the masked tensor `A=A_*M` from the sparse tensor `A_`, where
        the binary mask `M` is implicitely defined so that it retains only the
        non-sparse elements of `A_`. Due to implementation details, `A_` is
        required to be coalesced.

        Parameters
        ----------
        A_: sparse.FloatTensor
            A coalesced sparse tensor.

        """
        if not A_.is_coalesced():
            raise Exception("MaskedTensor requires a coalesced sparse tensor.")
        self.A_ = A_

    def clone(self):
        return mask(self.A_.clone())

    def dim(self):
        return self.A_.dim()

    def size(self):
        return self.A_.size()

    def t(self):
        # mask can be propagated harmlessly
        return mask(self.A_.t())

    def unmask(self):
        """
        Removes the implicit mask on `A`.

        Returns
        -------
        sparse.FloatTensor
            The original sparse matrix `A_`.

        """
        return self.A_

    def values(self):
        """
        Extracts the values of the non-masked elements in the form of a
        flattened tensor.

        Returns
        -------
        FloatTensor
            One-dimensional tensor of length `n`, where `n` is the number of
            non-maked elements in `A`.

        """
        return values(self.A_)

    def indices(self):
        """
        Extracts the indices of the non-masked elements in `A`.

        Returns
        -------
        LongTensor
            Matrix tensor whose `i`-th colum is the index of the `i`-th
            non-masked element.

        """
        return self.A_._indices()

    def to_dense(self):
        return self.A_.to_dense()  # non-differentiable

    def sum(self, dims=None):
        """
        Sums the masked tensor over some specific or all dimensions.

        Parameters
        ----------
        dims: tuple of ints
            The dimensions over which to sum, or `None` to sum over all
            dimensions (default).

        Returns
        -------
        sparse.FloatTensor
            The sparse tensor summed over dimensions in `dims`. Note that the
            original dimensions are kept but shrinked to size 1.

        """
        return MaskedSum.apply(self.A_, dims)

    def mm(self, B):
        """
        Matrix multiplication with a dense tensor. Returns a dense tensor.

        This is equivalent to calling `A @ B` where `A` is sparse and `B` is
        dense, with differentiation support w.r.t both `A_` and `B`.

        Parameters
        ----------
        B: FloatTensor
            A dense matrix.

        Returns
        -------
        FloatTensor
            The dense matrix resulting from `A @ B`

        """
        return MaskedMatMul.apply(self.A_, B)

    def add_m(self, c):
        """Adds c then re-applies the mask."""
        return mask(build(
            self.A_._indices(),
            self.values() + c,
            self.A_.size(),
            skip_check=True))

    def exp_m(self):
        """Exponentiates elements then re-applies the mask."""
        return mask(build(
            self.A_._indices(),
            self.values().exp(),
            self.A_.size(),
            skip_check=True))

    def softmax_m(self, dim=1):
        """Exponentiates elements, re-applies the mask, then normalize."""
        raise Exception("Not implemented yet.")
        # x = self.values()
        # x = x - x.max().detach()
        # x = x.exp()
        # x = build(
        #     self.A_._indices(),
        #     x,
        #     self.A_.size(),
        #     skip_check=True)
        # x = x / masked_sum(x)...
        return mask(x)

    def type(self):
        return self.__module__ + "." + self.__class__.__name__

    def __repr__(self):
        return self.type() + " represented as a " + repr(self.A_)

