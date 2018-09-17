# coding: utf-8

"""Normalization classes."""

from typing import Union

import torch
import torch.nn as nn

from sgcn.masked.tensor import MaskedTensor


class Normalization(nn.Module):
    """Normalization base class.

    A normalization class implements a function `forward` with one parameter:
    the unnormalized attention weights for every query over every values.
    """

    pass


class NoNorm(Normalization):
    """No normalization class."""

    def forward(
        self, QKt: Union[torch.Tensor, MaskedTensor]
    ) -> Union[torch.Tensor, MaskedTensor]:
        """Identity function."""
        return QKt
