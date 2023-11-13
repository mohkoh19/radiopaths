from torch import nn
import torch
import math
import numpy as np
from copy import deepcopy
from einops import rearrange
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        """
        Source: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html
        
        Inputs
            d_model - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] represe`nting the positional encoding for max_len inputs
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=None, pe=None):
        """
        Implementation of multi-head attention layer of the original transformer model
        modified from: https://theaisummer.com/transformer/.
        einsum and einops.rearrange is used whenever possible
        Args:
            dim: token's dimension, i.e. word embedding vector size
            heads: the number of distinct representations to learn
            dim_head: the dim of the head. In general dim_head<dim.
            However, it may not necessary be (dim/heads)
        """
        super().__init__()
        self.dim_head = (int(dim / heads)) if dim_head is None else dim_head
        _dim = self.dim_head * heads
        self.heads = heads
        self.to_kv = nn.Linear(dim, _dim * 2, bias=False)
        self.to_q = nn.Linear(dim, _dim, bias=False)
        self.W_0 = nn.Linear(_dim, dim, bias=False)
        self.scale_factor = self.dim_head**-0.5
        self.pe = pe if pe is not None else None

    def compute_mha(self, q, k, v, mask=None, qr=None):
        # resulted shape will be: [batch, heads, tokens, tokens]
        dot_prod = torch.einsum("... i d , ... j d -> ... i j", q, k)

        if qr is not None:
            dot_prod += qr

        scaled_dot_prod = dot_prod * self.scale_factor

        if mask is not None:
            assert mask.shape == scaled_dot_prod.shape
            scaled_dot_prod = scaled_dot_prod.masked_fill(mask, -np.inf)

        attention = torch.softmax(scaled_dot_prod, dim=-1)
        attention = torch.nan_to_num(attention)
        # calc result per head
        return torch.einsum("... i j , ... j d -> ... i d", attention, v)

    def forward(self, q, kv, mask=None):
        assert kv.dim() == 3
        assert q.dim() == 3
        kv = self.to_kv(kv)  # [batch, tokens, dim*3*heads ]
        q = self.to_q(q)  # [batch, tokens, dim*3*heads ]

        # decomposition to q,v,k and cast to tuple
        # the resulted shape before casting to tuple will be: [3, batch, heads, tokens, dim_head]
        k, v = tuple(rearrange(kv, "b t (d k h ) -> k b h t d ", k=2, h=self.heads))
        q = rearrange(q, "b t (d h) -> b h t d ", h=self.heads)

        qr = self.pe(q) if self.pe is not None else None

        out = self.compute_mha(q, k, v, mask=mask, qr=qr)

        # re-compose: merge heads with dim_head
        out = rearrange(out, "b h t d -> b t (h d)")
        # Apply final linear transformation layer
        return self.W_0(out)


class EncoderLayer(nn.Module):
    """
    Vanilla transformer encoder layer from the original paper "Attention is all you need"
    Modified implementation from: https://theaisummer.com/transformer/
    """

    def __init__(
        self,
        dim,
        heads=8,
        dim_head=None,
        dim_linear_block=1024,
        dropout=0.1,
        activation=nn.ReLU,
        mha=None,
        prenorm=False,
        pe=None,
    ):
        """
        Args:
            dim: token's vector length
            heads: number of heads
            dim_head: if none dim/heads is used
            dim_linear_block: the inner projection dim
            dropout: probability of droppping values
            mha: if provided you can change the vanilla self-attention block
            prenorm: if the layer norm will be applied before the mha or after
        """
        super().__init__()

        self.mha = mha if mha is not None else MultiHeadAttention(dim=dim, heads=heads, dim_head=dim_head, pe=pe)
        self.prenorm = prenorm
        self.drop = nn.Dropout(dropout)
        self.norm_1 = nn.LayerNorm(dim)
        self.norm_2 = nn.LayerNorm(dim)

        self.linear = nn.Sequential(
            nn.Linear(dim, dim_linear_block),
            activation(),  # nn.ReLU or nn.GELU
            nn.Dropout(dropout),
            nn.Linear(dim_linear_block, dim),
            nn.Dropout(dropout),
        )

    def forward(self, q, kv, mask=None, dense=False):
        if self.prenorm:
            y = self.drop(self.mha(q=self.norm_1(q), kv=self.norm_1(kv), mask=mask)) + q
            out = self.linear(self.norm_2(y)) + y
        else:
            y = self.norm_1(self.drop(self.mha(q=q, kv=kv, mask=mask)) + q)
            out = self.norm_2(self.linear(y) + y)
        return out


class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        heads=8,
        dim_head=None,
        dim_linear_block=1024,
        dropout=0.1,
        activation=nn.ReLU,
        mha=None,
        prenorm=False,
        pe=None,
    ):
        """ Applies the cross-attention as described in the paper.

        Args:
            dim (int): Hidden dimension.
            heads (int, optional): Number of heads of the MHA. Defaults to 8.
            dim_head (int, optional): If None, dim/heads is used. Defaults to None.
            dim_linear_block (int, optional): Inner projective dimension. Defaults to 1024.
            dropout (float, optional): Dopout rate. Defaults to 0.1.
            activation (func, optional): Activation function. Defaults to nn.ReLU.
            mha (func, optional): Custom MHA function. Defaults to None.
            prenorm (bool, optional): Applies linear norm before MHA if enabled. Defaults to False.
            pe (func, optional): Positional encoding/embedding function. Defaults to None.
        """
        super().__init__()

        # Input Encoder
        self.input_encoder = EncoderLayer(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            dim_linear_block=dim_linear_block,
            dropout=dropout,
            activation=activation,
            mha=mha,
            prenorm=prenorm,
            pe=pe,
        )

        self.aux_encoder = deepcopy(self.input_encoder)

        self.norm = nn.LayerNorm(dim)

    def forward(self, X, Y, mask1=None, mask2=None):
        # MHA(X, Y) --> Z
        Z = self.aux_encoder(q=X, kv=Y, mask=mask1)

        # MHA(Z, X) --> X'
        X_prime = self.input_encoder(q=Z, kv=X, mask=mask2)

        # norm(X' + Z) --> out
        out = self.norm(X_prime + Z)

        return out
