# coding: utf-8

""""""

import torch
from torch.autograd import Function


class Sparse(Function):
    @staticmethod
    def forward(ctx, i, v, *args, **kwargs):
        ctx.n_options = len(args) + len(kwargs)
        _torch = torch.cuda if v.is_cuda else torch
        return _torch.sparse.FloatTensor(i, v, *args, **kwargs)

    @staticmethod
    def backward(ctx, output_grad):
        return (None, output_grad._values()) + (None, ) * ctx.n_options


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
        _torch = torch.cuda if grad_output.is_cuda else torch
        return _torch.sparse.FloatTensor(i, grad_output, size)


sparse = Sparse.apply
values = Values.apply
