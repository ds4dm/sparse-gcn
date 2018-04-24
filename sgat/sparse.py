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


def ivs_to_sparse(i, v, s, check_coalesced=True):
    _torch = torch.cuda if v.is_cuda else torch
    output = _torch.sparse.FloatTensor(i, v, s)

    if check_coalesced:
        output = output.coalesce()
        if not (i.shape == output._indices().shape and
                i.eq(output._indices()).all()):
            raise Exception("Input indices must be in coalesced form.")

    return output


class MaskedMatMul(Function):
    """
    Masked matrix multiplication between a masked matrix `A=A_*M` and a dense
    matrix `B`.
    
    """

    @staticmethod
    def forward(ctx, Ai, Av, As, B):
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
        A = ivs_to_sparse(Ai, Av, As, check_coalesced=False)
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
        grad_Av = grad_B = None

        if ctx.needs_input_grad[1]:
            grad_Av = masked_mm(grad_output, B.t(), A._indices())
        if ctx.needs_input_grad[3]:
            grad_B = A.t() @ grad_output

        return None, grad_Av, None, grad_B


# define function aliases, useful since Function.apply()
# does not support named arguments
def masked_mm(A, B, mask_idxs):
    """Masked matrix multiplication.

    This function computes `(A @ B) * M` efficitently by leveraging sparse
    computations, where `M` is an implicit constant binary matrix with 1
    values at indices in `mask_idxs`. Only the values of non masked-out
    elements are computed and returned in the form of a sparse tensor. Note
    that as a result one may observe a small numeric difference compared to
    the equivalent dense operation.

    Parameters
    ----------
    A : FloatTensor
        Matrix of size (n, p)
    B : FloatTensor
        Matrix of size (p, m)
    mask_idxs : LongTensor
        Matrix of size (2, nb_ones) whose columns correspond to the indices of
        the 1 values in the binary mask `M`.

    Returns
    -------
<<<<<<< HEAD
    FloatTensor
        The non-masked elements of `(A @ B) * M`, i.e. those at indices
        `mask_idxs`.
=======
    sparse.FloatTensor
        The sparse equivalent of `(A @ B) * M`.
>>>>>>> 1c98646b7a126c162ffde92204d3b81f85928fda

    """
    idx, jdx = mask_idxs
    Av = A.index_select(0, idx)
    Bv = B.t().index_select(0, jdx)

    ABv = torch.bmm(
        Av.view(Av.size(0), 1, -1),
        Bv.view(Bv.size(0), -1, 1)
    ).view(-1)

    return ABv


def mask(*args, **kwargs):
    return MaskedTensor(*args, **kwargs)


class MaskedTensor:
    """
    Represents the implicit tensor `A` that results from the application of a
    binary mask `M` to an initial tensor `A_`, i.e. `A=A_*M`, and provides a
    series of operations on `A` with differentiation support w.r.t `A_`. Due
    to the masking operation, both the resulting tensor `A` and its
    derivatives w.r.t the original tensor `A_` are sparse (masked-out elements
    have zero gradient).

    """

    def __init__(self, i, v, s):
        """
        Builds the masked tensor `A=A_*M` of size `s`, whose non-masked-out
        elements lie at indices `i` and have values `v` (similarly to
        `torch.sparse.FloatTensor`).

        Parameters
        ----------
        i: LongTensor
            Matrix of size `(ndims, nvals)`, the position of non-masked-out
            elements.
        v: FloatTensor
            Vector of size `nvals`, the value of non-masked-out elements.
        s: tuple
            Tuple of size `ndims`, the size of the tensor `A`.

        """
        self.i = i
        self.v = v
        self.s = s

    def clone(self):
        return mask(self.i.clone(), self.v.clone(), self.s)

    def dim(self):
        return len(self.s)

    def size(self, d=None):
        if d is None:
            return self.s
        
        return self.s[d]

    def t(self):
        assert self.dim() == 2
        return mask(self.i[[1, 0], :], self.v, self.s[::-1])

    def values(self):
        return self.v

    def indices(self):
        return self.i

    def to_sparse(self):
        """Looses differentiation."""
        return ivs_to_sparse(self.i, self.v, self.s, check_coalesced=False)

    def to_dense(self):
        """Looses differentiation."""
        return self.to_sparse().to_dense()

    def sum(self, dim=None, keepdim=False):
        if dim is None:
            return self.v.sum()

        assert dim >= 0 and dim < self.dim(), \
                "sum(): invalid dimension {}.".format(dim)

        assert self.dim() <= 2, \
                "sum(): tensors with more than 2 dimensions are currently" + \
                "not supported."

        _torch = torch.cuda if self.v.is_cuda else torch
        ones = _torch.FloatTensor(self.s[dim], 1).fill_(1)
        if dim == 0:
            output = self.t().mm(ones).t()
        else:
            output = self.mm(ones)

        if not keepdim:
            output = output.squeeze(dim)

        return output

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
        return MaskedMatMul.apply(self.i, self.v, self.s, B)

    def type(self):
        return self.__module__ + "." + self.__class__.__name__

    def __repr__(self):
        return self.type() + " of size " + repr(self.s) + \
            " with indices " + repr(self.i) + \
            " and values " + repr(self.v)

