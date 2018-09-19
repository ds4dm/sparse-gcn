# coding: utf-8

"""Affinities classes."""

from typing import Optional, Union

import torch
import torch.nn as nn
from math import sqrt

from sgcn.masked.tensor import MaskedTensor


class Affinity(nn.Module):
    """Affinity base class.

    An affinity class implements a function `forward` with three parameters:
    the attention keys, the attention queries and some optional query-keys
    specifics (e.g a mask).
    """

    pass


class DotProduct(Affinity):
    """Dot product attention.

    The attetnion between a key and a value are computed using their dot
    product.
    """

    def __init__(self, scaled: bool = True) -> None:
        """Initialize affinity.

        Parameters
        ----------
        scaled:
            Wether or nor to scale the attention weights by the inverse
            of the square root of the key dimension.

        """
        super().__init__()
        self.scaled = scaled

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        m: Optional[Union[torch.Tensor, MaskedTensor]] = None
    ) -> Union[torch.Tensor, MaskedTensor]:
        """Compute dot-product affinity.

        Parameters
        ----------
        Q
            Queries tensor with dimension (n_queries, feat_dim).
        K
            Keys tensor with dimensions (n_keys, feat_dim).
        m
            Optional mask. If the mask is given in dense form it is applied
            by elementwise multiplication. If given in sparse form, only
            the non zero values will be computed.

        Returns
        -------
        affinity
            The matrice of attention weights first dimension is query
            index, second dimension is key index. If a mask is given by the
            type MaskedTensor, the results is also MaskedTensor, otherwise
            the result is dense.

        """
        if isinstance(m, MaskedTensor):
            QKt = m.mask_mm(Q, K.t())
            if self.scaled:
                QKt = QKt.apply(lambda x: x / sqrt(K.size(1)))

        else:
            QKt = Q @ K.t()
            QKt = QKt if m is None else QKt * m
            if self.scaled:
                QKt = QKt / sqrt(K.size(1))

        return QKt
