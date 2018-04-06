# coding: utf-8

"""Affinities classes."""

import torch
from math import sqrt
from .utils import Callable


class Affinity(object, metaclass=Callable):
    """Affinity base class.

    An affinity class implements a function `__call__` with three parameters:
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
        self.scaled = scaled

    def __call__(self, K, Q, m=None):
        """Compute dot-product affinity.

        Parameters
        ----------
            K : FloatTensor
                Keys tensor with dimensions (n_keys, feat_dim).
            Q : FloatTensor
                Queries tensor with dimension (n_queries, feat_dim).
            m : FloatTensor or sparse.Tensor
                Optional mask. If the mask is given in dense form it is applied
                by elementwise multiplication. If given in sparse form, only
                the non zero values will be computed.

        Returns
        -------
            FloatTensor or sparse.FloatTensor
                The matrice of attention weights first dimension is query index,
                second dimension is key index. If a sparse mask is given, the
                resulting matrice is also sparse (with the same support).

        """
        if (m is None) or (not m.is_sparse):
            QKt = Q @ K.t()
            QKt = QKt if m is None else QKt * m
        else:
            # Allowed attentions
            Q_i, K_i = m._indices()

            # Computing allowed values in QKt
            Q_v = Q.index_select(0, Q_i)
            K_v = K.index_select(0, K_i)
            QKt_v = torch.bmm(
                Q_v.view(len(Q_v), 1, -1),
                K_v.view(len(K_v), -1, 1)
            ).view(-1)

            # Shaping as sparse matrix
            _torch = torch.cuda if QKt_v.is_cuda else torch
            QKt = _torch.sparse.FloatTensor(m._indices(), QKt_v, m.size())

        if self.scaled:  # Apply scaling if specified
            QKt = QKt / sqrt(K.size(1))

        return QKt
