import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from .types import ModelParams


class LayerNorm(nn.Module):
    """
    Layer normalization.
    """

    weight: nn.Parameter
    bias: nn.Parameter | None

    def __init__(self, n_dim: int, bias: bool) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_dim))  # (n_dim)
        self.bias = nn.Parameter(torch.zeros(n_dim)) if bias else None  # (n_dim)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            input=input,  # (B, T, n_dim)
            normalized_shape=self.weight.shape,  # (n_dim)
            weight=self.weight,  # (n_dim)
            bias=self.bias,  # (n_dim)
            eps=1e-5,
        )  # (B, T, n_dim)


class CausalSelfAttention(nn.Module):
    """
    Causal self-attention.
    """

    c_attn: nn.Linear
    c_proj: nn.Linear
    attn_dropout: nn.Dropout
    resid_dropout: nn.Dropout
    n_head: int
    n_embd: int
    dropout_p: float

    def __init__(self, params: ModelParams) -> None:
        super().__init__()
        assert params.n_embd % params.n_head == 0
        # Key, query, and value projections for all heads in a batch
        self.c_attn = nn.Linear(
            params.n_embd, 3 * params.n_embd, bias=params.bias
        )  # (n_embd, 3 * n_embd)
        # Output projection
        self.c_proj = nn.Linear(params.n_embd, params.n_embd, bias=params.bias)  # (n_embd, n_embd)
        # Regularization
        self.attn_dropout = nn.Dropout(params.dropout_p)  # (n_embd)
        self.resid_dropout = nn.Dropout(params.dropout_p)  # (n_embd)
        self.n_head = params.n_head
        self.n_embd = params.n_embd
        self.dropout_p = params.dropout_p
        # Try to use flash attention if available
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            params.logger.warning("Using slow attention. Flash Attention not available")
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(params.block_size, params.block_size)).view(
                    1, 1, params.block_size, params.block_size
                ),
            )  # (1, 1, block_size, block_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()

        # Calculate query, key, and values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(
            self.n_embd, dim=2
        )  # (B, T, 3 * n_embd) -> (B, T, n_embd), (B, T, n_embd), (B, T, n_embd)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # Causal self-attention (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # Efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q,  # (B, nh, T, hs)
                k,  # (B, nh, T, hs)
                v,  # (B, nh, T, hs)
                attn_mask=None,  # (1, 1, block_size, block_size)
                dropout_p=self.dropout_p if self.training else 0,  # (n_embd)
                is_causal=True,
            )  # (B, nh, T, hs)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  # (B, nh, T, T)
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))  # (B, nh, T, T)
            att = F.softmax(att, dim=-1)  # (B, nh, T, T)
            att = self.attn_dropout(att)  # (B, nh, T, T)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        # Re-assemble all head outputs side by side
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # (B, nh, T, hs) -> (B, T, nh * hs) = (B, T, n_embd)

        # Output projection
        y = self.resid_dropout(self.c_proj(y))  # (B, T, n_embd) -> (B, T, n_embd)
        return y


class MLP(nn.Module):
    def __init__(self, params: ModelParams) -> None:
        super().__init__()
        self.c_fc = nn.Linear(
            params.n_embd, 4 * params.n_embd, bias=params.bias
        )  # (n_embd, 4 * n_embd)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(
            4 * params.n_embd, params.n_embd, bias=params.bias
        )  # (4 * n_embd, n_embd)
        self.dropout = nn.Dropout(params.dropout_p)  # (n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)  # (B, T, n_embd) -> (B, T, 4 * n_embd)
        x = self.gelu(x)  # (B, T, 4 * n_embd)
        x = self.c_proj(x)  # (B, T, 4 * n_embd) -> (B, T, n_embd)
        x = self.dropout(x)  # (B, T, n_embd)
        return x


class Block(nn.Module):
    def __init__(self, params: ModelParams) -> None:
        super().__init__()
        self.ln_1 = LayerNorm(params.n_embd, bias=params.bias)
        self.attn = CausalSelfAttention(params)
        self.ln_2 = LayerNorm(params.n_embd, bias=params.bias)
        self.mlp = MLP(params)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))  # (B, T, n_embd)
        x = x + self.mlp(self.ln_2(x))  # (B, T, n_embd)
        return x
