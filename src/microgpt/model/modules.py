import math
from contextlib import nullcontext

import torch
import torch.nn as nn
from torch.nn import functional as F

from microgpt.common.logger import _new_logger

from .model_types import _ModelParams

logger = _new_logger(__name__)


def _layer_norm(
    normalized_shape: int,
    eps: float = 1e-6,
    elementwise_affine: bool = True,
    bias: bool = True,
) -> nn.LayerNorm:
    ln = nn.LayerNorm(normalized_shape, eps=eps, elementwise_affine=elementwise_affine, bias=bias)
    return ln


def _embedding(num_embeddings: int, embedding_dim: int, init_std: float | None) -> nn.Embedding:
    embedding = nn.Embedding(num_embeddings, embedding_dim)
    if init_std is None:
        init_std = 1 / math.sqrt(embedding_dim)
    torch.nn.init.normal_(embedding.weight, mean=0.0, std=init_std)
    return embedding


def _linear(
    in_features: int,
    out_features: int,
    n_layers: int,
    init_std: float | None,
    init_residual_scaled_factor: float | None = None,
    bias: bool = True,
) -> nn.Linear:
    linear = nn.Linear(in_features, out_features, bias=bias)
    if init_std is None:
        init_std = 1 / math.sqrt(in_features)
    if init_residual_scaled_factor is not None:
        # Residual layers are scaled down by a factor of sqrt(init_residual_scaled_factor * n_layers)
        # to prevent the residual layers from exploding in value
        init_std /= math.sqrt(init_residual_scaled_factor * n_layers)
    torch.nn.init.normal_(
        linear.weight,
        mean=0.0,
        std=init_std,
    )
    if bias:
        torch.nn.init.zeros_(linear.bias)
    return linear


class _RoPE(nn.Module):
    """
    Rotary positional embeddings.
    See: https://arxiv.org/abs/2104.09864
    """

    hs: int
    rope_theta: float
    is_rope_full_precision: bool

    def __init__(self, params: _ModelParams, device: str):
        super().__init__()
        self.hs = params.d_model // params.n_heads
        self.rope_theta = params.rope_theta
        self.is_rope_full_precision = params.is_rope_full_precision
        self.register_buffer("positions_sin", None)
        self.register_buffer("positions_cos", None)
        self._get_position_buffers(
            seq_len=params.max_seq_len, torch_device=torch.device(device) if device is not None else None
        )

    def _get_position_buffers(
        self, seq_len: int, torch_device: torch.device | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            self.positions_sin is not None  # type: ignore
            and self.positions_cos is not None  # type: ignore
            and seq_len <= self.positions_sin.size(2)  # type: ignore
        ):
            if torch_device is not None and self.positions_sin.device != torch_device:  # type: ignore
                self.positions_sin = self.positions_sin.to(torch_device)  # type: ignore
                self.positions_cos = self.positions_cos.to(torch_device)  # type: ignore
            return self.positions_sin[:, :, :seq_len, :], self.positions_cos[:, :, :seq_len, :]

        ctx = (
            torch.autocast(device_type=torch_device.type, enabled=False) if torch_device is not None else nullcontext()
        )
        with ctx:
            inv_freq = 1.0 / (
                self.rope_theta ** (torch.arange(0, self.hs, 2, device=torch_device, dtype=torch.float) / self.hs)
            )  # (hs // 2)
            seq = torch.arange(seq_len, device=torch_device, dtype=torch.float)  # (seq_len)
            freqs = torch.einsum("i,j -> i j", seq, inv_freq)  # (seq_len, hs // 2)
            positions = torch.cat((freqs, freqs), dim=-1)  # (seq_len, hs)
            self.positions_sin = positions.sin()[None, None, :, :]  # (1, 1, seq_len, hs)  # type: ignore
            self.positions_cos = positions.cos()[None, None, :, :]  # (1, 1, seq_len, hs)  # type: ignore
        return self.positions_sin, self.positions_cos

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        B, nh, T, hs = x.size()
        x = x.view(B, nh, T, 2, hs // 2)  # (B, nh, T, 2, hs // 2)
        x1, x2 = x.unbind(dim=-2)  # (B, nh, T, hs // 2), (B, nh, T, hs // 2)
        return torch.cat((-x2, x1), dim=-1)  # (B, nh, T, hs)

    def _apply_rotary_pos_emb(
        self,
        t: torch.Tensor,  # (B, nh, T, hs)
        positions_sin: torch.Tensor,  # (1, 1, T, hs)
        positions_cos: torch.Tensor,  # (1, 1, T, hs)
    ) -> torch.Tensor:  # (B, nh, T, hs)
        return ((t * positions_cos) + (self._rotate_half(t) * positions_sin)).to(t.dtype)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.is_rope_full_precision:
            q_, k_ = q.float(), k.float()
        else:
            q_, k_ = q, k

        with torch.autocast(q.device.type, enabled=False):
            # These might be different in the future
            query_len, key_len = q_.shape[-2], k_.shape[-2]
            positions_sin, positions_cos = self._get_position_buffers(seq_len=key_len, torch_device=q_.device)
            positions_sin = positions_sin.type_as(q_)  # (1, 1, max_seq_len, hs)
            positions_cos = positions_cos.type_as(q_)  # (1, 1, max_seq_len, hs)
            q_ = self._apply_rotary_pos_emb(
                q_,  # (B, nh, T, hs)s
                positions_sin[:, :, key_len - query_len : key_len, :],  # (1, 1, T, hs)
                positions_cos[:, :, key_len - query_len : key_len, :],  # (1, 1, T, hs)
            )
            k_ = self._apply_rotary_pos_emb(
                k_,  # (B, nh, T, hs)
                positions_sin[:, :, 0:key_len, :],  # (1, 1, T, hs)
                positions_cos[:, :, 0:key_len, :],  # (1, 1, T, hs)
            )
        return q_.type_as(q), k_.type_as(k)


class _CausalSelfAttention(nn.Module):
    """
    Causal self-attention.
    """

    _layer_id: int
    _n_heads: int
    _d_model: int
    _use_rope: bool
    _attn_dropout_p: float

    c_attn: nn.Linear
    rope: _RoPE | None
    c_proj: nn.Linear
    attn_dropout: nn.Dropout
    resid_dropout: nn.Dropout

    def __init__(self, params: _ModelParams, layer_id: int, device: str) -> None:
        super().__init__()
        assert params.d_model % params.n_heads == 0

        self._layer_id = layer_id
        self._n_heads = params.n_heads
        self._d_model = params.d_model
        self._use_rope = params.use_rope
        self._attn_dropout_p = params.attn_dropout_p

        # Key, query, and value projections for all heads in a batch
        self.c_attn = _linear(
            params.d_model, 3 * params.d_model, params.n_layers, params.init_std
        )  # (d_model, 3 * d_model)
        if params.use_rope:
            self.rope = _RoPE(params, device)
        # Output projection
        self.c_proj = _linear(
            params.d_model,
            params.d_model,
            params.n_layers,
            params.init_std,
            init_residual_scaled_factor=params.init_residual_scaled_factor,
        )  # (d_model, d_model)
        # Regularization
        self.attn_dropout = nn.Dropout(params.attn_dropout_p)  # (d_model)
        self.resid_dropout = nn.Dropout(params.residual_dropout_p)  # (d_model)
        # Try to use flash attention if available
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            logger.warning("Using slow attention. Flash Attention not available")
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(params.max_seq_len, params.max_seq_len)).view(
                    1, 1, params.max_seq_len, params.max_seq_len
                ),
            )  # (1, 1, max_seq_len, max_seq_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()

        # Calculate query, key, and values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(
            self._d_model, dim=2
        )  # (B, T, 3 * d_model) -> (B, T, d_model), (B, T, d_model), (B, T, d_model)
        k = k.view(B, T, self._n_heads, C // self._n_heads).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self._n_heads, C // self._n_heads).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self._n_heads, C // self._n_heads).transpose(1, 2)  # (B, nh, T, hs)

        if self.rope is not None:
            q, k = self.rope(q, k)

        # Causal self-attention (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # Efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q,  # (B, nh, T, hs)
                k,  # (B, nh, T, hs)
                v,  # (B, nh, T, hs)
                attn_mask=None,  # (1, 1, max_seq_len, max_seq_len)
                dropout_p=self._attn_dropout_p if self.training else 0,  # (d_model)
                is_causal=True,
            )  # (B, nh, T, hs)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  # (B, nh, T, T)
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))  # type: ignore # (B, nh, T, T)
            att = F.softmax(att, dim=-1)  # (B, nh, T, T)
            att = self.attn_dropout(att)  # (B, nh, T, T)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        # Re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # (B, nh, T, hs) -> (B, T, nh * hs) = (B, T, d_model)

        # Output projection
        y = self.resid_dropout(self.c_proj(y))  # (B, T, d_model) -> (B, T, d_model)
        return y  # type: ignore


class _MLP(nn.Module):
    """
    Multi-layer perceptron.
    """

    _layer_id: int

    c_fc: nn.Linear
    gelu: nn.GELU
    c_proj: nn.Linear
    dropout: nn.Dropout

    def __init__(self, params: _ModelParams, layer_id: int) -> None:
        super().__init__()
        self._layer_id = layer_id
        self.c_fc = _linear(
            params.d_model, 4 * params.d_model, params.n_layers, params.init_std
        )  # (d_model, 4 * d_model)
        self.gelu = nn.GELU()
        self.c_proj = _linear(
            4 * params.d_model,
            params.d_model,
            params.n_layers,
            params.init_std,
            init_residual_scaled_factor=params.init_residual_scaled_factor,
        )  # (4 * d_model, d_model)
        self.dropout = nn.Dropout(params.residual_dropout_p)  # (d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)  # (B, T, d_model) -> (B, T, 4 * d_model)
        x = self.gelu(x)  # (B, T, 4 * d_model)
        x = self.c_proj(x)  # (B, T, 4 * d_model) -> (B, T, d_model)
        x = self.dropout(x)  # (B, T, d_model)
        return x


class _Block(nn.Module):
    """
    Transformer block.
    """

    _layer_id: int

    ln_1: nn.LayerNorm
    attn: _CausalSelfAttention
    ln_2: nn.LayerNorm
    mlp: _MLP

    def __init__(self, params: _ModelParams, layer_id: int, device: str) -> None:
        super().__init__()
        self._layer_id = layer_id
        self.ln_1 = _layer_norm(params.d_model)
        self.attn = _CausalSelfAttention(params, layer_id, device)
        self.ln_2 = _layer_norm(params.d_model)
        self.mlp = _MLP(params, layer_id)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))  # (B, T, d_model)
        x = x + self.mlp(self.ln_2(x))  # (B, T, d_model)
        return x
