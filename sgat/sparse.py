# coding: utf-8

""""""

import torch
from torch.autograd import Function

import numpy as np


class Build(Function):

    @staticmethod
    def forward(ctx, i, v, *args, **kwargs):
        ctx.n_options = len(args) + len(kwargs)
        _torch = torch.cuda if v.is_cuda else torch

        return _torch.sparse.FloatTensor(i, v, *args, **kwargs)

    @staticmethod
    def backward(ctx, output_grad):
        v_grad = None
        if ctx.needs_input_grad[1]:
            v_grad = output_grad._values()

        return (None, v_grad) + (None, ) * ctx.n_options


class Values(Function):

    @staticmethod
    def forward(ctx, A):
        ctx.save_for_backward(A._indices())
        ctx.size = A.size()
        return A._values()

    @staticmethod
    def backward(ctx, grad_output):
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

        SparseTensorType = eval(input.type())
        IdxTensorType = eval(input._indices().type())

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

        zero_idx = IdxTensorType(1).zero_().expand(nvals)

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
        ctx.input_type = SparseTensorType
        ctx.input_nvals = nvals
        ctx.input_shape = input.shape

        output = SparseTensorType(
            indices,
            input._values(),
            size,
        ).coalesce()

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_indices, coalesced_indices = ctx.saved_tensors

        grad_input = None
        if ctx.needs_input_grad[0]:
            output_indices = grad_output._indices()

            output_indices_dict = {
                tuple(k): i for i, k in enumerate(
                    output_indices.numpy().transpose())}

            out_to_in_map = [output_indices_dict[
                tuple(k)] for k in
                    coalesced_indices.numpy().transpose()]

            grad_input = ctx.input_type(
                input_indices,
                grad_output._values()[out_to_in_map],
                ctx.input_shape
            )

        return grad_input, None


build = Build.apply
values = Values.apply
sum = Sum.apply


def matmulmasked(A, B, m):
    idx, jdx = m._indices()
    Av = A.index_select(0, idx)
    Bv = B.t().index_select(0, jdx)

    ABv = torch.bmm(
        Av.view(Av.size(0), 1, -1),
        Bv.view(Bv.size(0), -1, 1)
    ).view(-1)

    return build(m._indices(), ABv, (A.size(0), B.size(1)))
