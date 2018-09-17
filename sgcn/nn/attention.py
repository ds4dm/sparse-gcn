# coding: utf-8

"""Attention module.

Attention as defined in Attention is All you Need.
https://arxiv.org/abs/1706.03762
"""

from typing import Union, Optional

import torch
import torch.nn as nn

from sgcn.masked.tensor import MaskedTensor
from . import affinity as aff
from . import normalization as norm


class Attention(nn.Module):
    """Attention.

    TODO: if necessary make a batched version of this. Note batch of different
    sizes can already be done using a block diagonal mask.
    """

    def __init__(
        self,
        affinity: aff.Affinity,
        normalization: norm.Normalization
    ) -> None:
        """Initialize the Attention.

        Parameters
        ----------
        affinity
            Object of type Affinity to compute the affinity between keys
            and attentions queries.
        normalization
            Object of type Normalization to apply a correction to the
            attention  weights.

        """
        super().__init__()
        self.affinity = affinity
        self.normalization = normalization

    def forward(
        self,
        K: torch.Tensor,
        V: torch.Tensor,
        Q: torch.Tensor,
        m: Optional[Union[torch.Tensor, MaskedTensor]] = None
    ) -> torch.Tensor:
        """Compute attention.

        Accoring to _Attention is All you Need_:
        > An attention function can be described as mapping a query and a set
        > of key-value pairs to an output, where the query, keys, values, and
        > output are all vectors. The output is computed as a weighted sum of
        > the values, where the weight assigned to each value is computed by a
        > compatibility function of the query with the corresponding key.
        https://arxiv.org/abs/1706.03762

        Parameters
        ----------
        K:
            Attention keys. First dimension is key index, other are feature
            values.
        V:
            Attention values. First dimension is the value index. There
            should be as many attention values as their are keys.
        Q:
            Queries to make on attention keys.
        m:
            A matrix of dimension number of queries per number of keys.
            Passed to the affinity function. Can be used to make a mask
            or to pass additional queries data (e.g. edge information for
            a graph).

        Returns
        -------
        attention:
            First dimension is align with queries indexes. Other dimensions are
            similar to the value ones.

        """
        QKt = self.affinity(Q, K, m)
        QKt_n = self.normalization(QKt)
        if isinstance(QKt_n, MaskedTensor):
            return QKt_n.mm(V)
        else:
            return QKt_n @ V


class MultiHeadAttention(Attention):
    """Dot product attention with multiple heads.

    Linearly project the keys, values, and queries and applies dot product
    attention to the result. This process is repeated as many times as there
    are heads, and the results are concatenated together.
    """

    def __init__(
        self,
        in_key: int,
        in_value: int,
        in_query: int,
        n_head: int,
        head_qk: int,
        head_v: int
    ) -> None:
        """Initialize multi head attention.

        Parameters
        ----------
        in_key:
            Dimension of input keys.
        in_value:
            Dimension of input values.
        in_query:
            Dimension of input queries.
        n_head:
            Number of heads to use.
        head_qk:
            Dimension every projected head for queries and keys. They share the
            Same dimension as the affinity is computed through dot product.
        head_v:
            Dimension every projected head for values.

        """
        super().__init__(
            affinity=aff.DotProduct(), normalization=norm.NoNorm()
        )
        self.lin_k = nn.Linear(in_key, head_qk * n_head)
        self.lin_v = nn.Linear(in_value, head_v * n_head)
        self.lin_q = nn.Linear(in_query, head_qk * n_head)
        self._n_head = n_head

    def _view_heads(self, X: torch.Tensor) -> torch.Tensor:
        """Reshape output of Linear by number of heads."""
        if X.dim() == 2:
            out_dim = X.size(1)
            return X.view(-1, self._n_head, out_dim // self._n_head)
        else:
            raise RuntimeError(
                f"Only dimension 2 supported, recieved: {X.dim()}"
            )

    def forward(
        self,
        K: torch.Tensor,
        V: torch.Tensor,
        Q: torch.Tensor,
        m: Optional[Union[torch.Tensor, MaskedTensor]] = None
    ) -> torch.Tensor:
        """Compute attention.

        Parameters
        ----------
        K:
            Attention keys. First dimension is key index, other are feature
            values.
        V:
            Attention values. First dimension is the value index. There
            should be as many attention values as their are keys.
        Q:
            Queries to make on attention keys.
        m:
            A matrix of dimension number of queries per number of keys.
            Passed to the affinity function. Can be used to make a mask
            or to pass additional queries data (e.g. edge information for
            a graph).

        Returns
        -------
        attention:
            First dimension is align with queries indexes. Second dimension is
            the number of heads times the output dimension of one value head
            (`head_v`).

        """
        K_proj = self._view_heads(self.lin_k(K))
        V_proj = self._view_heads(self.lin_v(V))
        Q_proj = self._view_heads(self.lin_q(Q))

        V_out = []
        for k in range(self._n_head):
            V_out.append(super().forward(
                K=K_proj[:, k], V=V_proj[:, k], Q=Q_proj[:, k], m=m
            ))

        return torch.cat(V_out, dim=1)
