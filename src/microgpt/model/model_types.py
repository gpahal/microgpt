from pydantic import BaseModel, Field


class _ModelParams(BaseModel):
    max_seq_len: int = 512
    d_model: int = 384
    n_layers: int = 6
    n_heads: int = 6
    use_padded_vocab_size: bool = True
    use_rope: bool = True
    rope_theta: float = Field(default=10000.0, ge=100.0, le=1000000.0)
    is_rope_full_precision: bool = True
    embd_dropout_p: float = Field(default=0.1, ge=0.0, le=1.0)
    attn_dropout_p: float = Field(default=0.1, ge=0.0, le=1.0)
    residual_dropout_p: float = Field(default=0.1, ge=0.0, le=1.0)
    init_std: float | None = Field(default=None, ge=0.01, le=1.0)
    init_residual_scaled_factor: float = Field(default=2.0, ge=1.0, le=16.0)
