# coding: utf-8

"""Affinities classes."""

import torch.nn as nn
from math import sqrt
from .. import sparse as sp


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

    def __init__(self, scaled: bool=True):
        """Initialize affinity.

        Parameters
        ----------
            scaled : bool
                Wether or nor to scale the attention weights by the inverse
                of the square root of the key dimension.

        """
        super().__init__()
        self.scaled = scaled

    def forward(self, Q, K, m=None):
        """Compute dot-product affinity.

        Parameters
        ----------
            Q : FloatTensor
                Queries tensor with dimension (n_queries, feat_dim).
            K : FloatTensor
                Keys tensor with dimensions (n_keys, feat_dim).
            m : FloatTensor or sparse.Tensor
                Optional mask. If the mask is given in dense form it is applied
                by elementwise multiplication. If given in sparse form, only
                the non zero values will be computed.

        Returns
        -------
            FloatTensor or sparse.FloatTensor
                The matrice of attention weights first dimension is query
                index, second dimension is key index. If a sparse mask is
                given, the resulting matrice is also sparse (with the same
                support).

        """
        if (m is None) or (not m.is_sparse):
            QKt = Q @ K.t()
            QKt = QKt if m is None else QKt * m
        else:
            QKt = sp.matmulmasked(Q, K.t(), m)

        if self.scaled:  # Apply scaling if specified
            QKt = QKt / sqrt(K.size(1))

        return QKt
