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


class Sum(Function):
    """
    Summation over a masked tensor `A=A_*M`. Returns a dense tensor.
    """

    @staticmethod
    def forward(ctx, Ai, Av, As, dims=None, keepdims=False):
        n_sdims = Ai.size(0)  # sparse dims
        n_ddims = Av.dim() - 1  # dense dims
        sshape = As[:n_sdims]
        dshape = As[n_sdims:]
        
        if dims is not None:
            # safety check on dims
            dims = np.asarray(dims)
            
            assert np.all(dims >= 0)
            assert np.all(dims < n_sdims)
            assert len(np.unique(dims)) == len(dims)

            if len(dims) == n_sdims:
                dims = None
            else:
                dims = np.sort(dims).tolist()

        ctx.dims = dims

        if dims is None:
            # sum over all sparse dimensions
            ctx.Av_size = Av.size()
            output = Av.sum(dim=0)
            
            if keepdims:
                new_shape = (1, ) * n_sdims + dshape
                output = output.view(new_shape)

        else:
            # sum over dims
            sdim_is_summed = np.isin(
                np.arange(n_sdims), dims, assume_unique=True)
            remaining_sdims = np.where(sdim_is_summed == False)[0].tolist()

            sparse_indices = Ai[remaining_sdims]
            squeezed_shape = tuple(As[d] for d in remaining_sdims) + dshape

            ctx.save_for_backward(sparse_indices)
            ctx.squeezed_shape = squeezed_shape
            output = ivs_to_sparse(sparse_indices, Av, squeezed_shape,
                                   check_coalesced=False).to_dense()

            if keepdims:
                new_shape = tuple(1 if summed else sshape[i]
                    for i, summed in enumerate(sdim_is_summed)) + dshape
                output = output.view(new_shape)

        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None

        if ctx.needs_input_grad[1]:

            if ctx.dims is None:
                grad_input = grad_output.squeeze().expand(*(ctx.Av_size))
            
            else:
                sparse_indices, = ctx.saved_tensors
                grad_input = grad_output.view(ctx.squeezed_shape
                    )[[idxs for idxs in sparse_indices]]

        return None, grad_input, None, None, None


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
    FloatTensor
        The non-masked elements of `(A @ B) * M`, i.e. those at indices
        `mask_idxs`.

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
        assert self.i.size(0) == 2
        return mask(self.i[[1, 0], :], self.v, self.s[::-1])

    def values(self):
        return self.v

    def indices(self):
        return self.i

    def to_sparse(self):
        """Looses differentiation."""
        return ivs_to_sparse(self.i, self.v, self.s, check_coalesced=False)

    def to_dense(self):
        return self.sum(dims=())

    def masked_softmax(self, dim):
        n_sdims = self.i.size(0)  # number of sparse dims
        n_ddims = self.v.dim() - 1  # number of dense dims
        s_sdims = self.s[:n_sdims]  # sparse shape
        s_ddims = self.s[n_sdims:]  # dense shape

        # softmax over sparse dimension
        if dim < n_sdims:
            v = self.v

            # exponentiate
            v = v - v.max().detach()  # (global) overflow trick. Can it be made local ?
            v = v.exp()

            # normalize over softmax dimension
            v_norm = Sum.apply(self.i, v, self.s, (dim, ), False)
            v = v / v_norm[[idxs for d, idxs in enumerate(self.i) if d != dim]]

        # softmax over dense dimension
        else:
            v = torch.nn.functional.softmax(self.v, dim=1+dim-n_sdims)

        return mask(self.i, v, self.s)

    def sum(self, dims=None, keepdims=False):
        """Returns a dense tensor."""
        return Sum.apply(self.i, self.v, self.s, dims, keepdims)

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

