# coding: utf-8

""""""

import torch
from torch.autograd import Function


class Sparse(Function):
    @staticmethod
    def forward(ctx, i, v, *args, **kwargs):
        return torch.sparse.FloatTensor(i, v, *args, **kwargs)

    @staticmethod
    def backward(ctx, output_grad):
        return None, output_grad._values()


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
        return torch.sparse.FloatTensor(i, grad_output, size)


sparse = Sparse.apply
values = Values.apply
