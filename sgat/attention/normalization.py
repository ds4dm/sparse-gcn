# coding: utf-8


"""Normalization classes."""

from .utils import Callable


class Normalization(object, metaclass=Callable):
    """Normalization base class.

    A normalization class implements a function `__call__` with one parameter:
    the unnormalized attention weights for every query over every values.
    """

    pass


class NoNorm(Normalization):
    """No normalization class."""

    def __call__(self, QKt):
        """Identity function."""
        return QKt
