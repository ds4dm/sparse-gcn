# conding: utf-8

"""Attention module.

Attention as defined in Attention is All you Need.
https://arxiv.org/abs/1706.03762
"""

import torch.nn as nn
from .affinity import Affinity
from .normalization import Normalization


class Attention(nn.Module):
    """Attention.

    TODO: if necessary make a batched version of this. Note batch of different
    sizes can already be done using a block diagonal mask.
    """

    def __init__(self, affinity: Affinity, norm: Normalization):
        """Initialize the Attention.

        Parameters
        ----------
            affinity : Affinity
                Object of type Affinity to compute the affinity between keys
                and attentions queries.
            norm : Norm
                Object of type Normalization to apply a correction to the
                attention  weights.

        """
        super().__init__()
        self.affinity = affinity
        self.norm = norm

    def forward(self, K, V, Q, m=None):
        """Compute attention.

        Accoring to _Attention is All you Need_:
        > An attention function can be described as mapping a query and a set of
        > key-value pairs to an output, where the query, keys, values, and
        > output are all vectors. The output is computed as a weighted sum of
        > the values, where the weight assigned to each value is computed by a
        > compatibility function of the query with the corresponding key.
        https://arxiv.org/abs/1706.03762

        Parameters
        ----------
            K : FloatTensor
                Attention keys. First dimension is key index, other are feature
                values.
            V : FloatTensor
                Attention values. First dimension is the value index. There
                should be as many attention values as their are keys.
            Q : FloatTensor
                Queries to make on attention keys.
            m : FloatTensor
                A matrix of dimension number of queries per number of keys.
                Passed to the affinity function. Can be used to make a mask
                or to pass additional queries data (e.g. edge information for
                a graph).

        Returns
        -------
            FloatTensor
            First dimension is align with queries indexes. Other dimensions are
            similar to the value ones.

        """
        QKt = self.affinity(Q, K, m)
        QKt_n = self.norm(QKt)
        return QKt_n @ V
