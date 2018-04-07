# coding: utf-8


"""Normalization classes."""

import torch.nn as nn


class Normalization(nn.Module):
    """Normalization base class.

    A normalization class implements a function `forward` with one parameter:
    the unnormalized attention weights for every query over every values.
    """

    pass


class NoNorm(Normalization):
    """No normalization class."""

    def forward(self, QKt):
        """Identity function."""
        return QKt
