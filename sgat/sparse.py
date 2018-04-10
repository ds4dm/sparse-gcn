# coding: utf-8

""""""

import torch
from torch.autograd import Function

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

build = Build.apply
values = Values.apply
