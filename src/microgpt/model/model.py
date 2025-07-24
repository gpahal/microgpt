"""
Micro GPT model.

See: https://github.com/openai/gpt-2/blob/master/src/model.py
See: https://github.com/karpathy/nanoGPT/blob/master/model.py
"""

import enum
import inspect
import logging
import os
from collections import OrderedDict
from typing import Annotated, Literal, no_type_check

import torch
import torch.nn as nn
from pydantic import BaseModel, Field
from torch.nn import functional as F

from microgpt.common.device import DeviceType, _get_device, _get_device_type
from microgpt.common.logger import _new_logger
from microgpt.tokenizer import CustomTrainedTokenizerConfig, PretrainedGPTTokenizerConfig, Tokenizer, TokenizerConfig

from .model_types import _ModelParams
from .modules import _Block, _embedding, _layer_norm, _linear


class UntrainedModelConfig(BaseModel):
    """
    Untrained model config.
    """

    type: Literal["untrained"] = "untrained"
    tokenizer_config: TokenizerConfig
    max_seq_len: int = 512
    d_model: int = 384
    n_layers: int = 6
    n_heads: int = 6
    use_padded_vocab_size: bool = True

    # Rope
    use_rope: bool = True
    rope_theta: float = Field(default=10000.0, ge=100.0, le=1000000.0)
    is_rope_full_precision: bool = True

    # Dropout probabilities
    embd_dropout_p: float = Field(default=0.0, ge=0.0, le=1.0)
    attn_dropout_p: float = Field(default=0.0, ge=0.0, le=1.0)
    residual_dropout_p: float = Field(default=0.0, ge=0.0, le=1.0)

    # Initialization standard deviation
    init_std: float | None = Field(default=None, ge=0.01, le=1.0)
    # Residual layers are scaled down by a factor of sqrt(2 * n_layers)
    # to prevent the residual layers from exploding in value
    init_residual_scaled_factor: float = Field(default=2.0, ge=1.0, le=16.0)


class CustomTrainedModelConfig(BaseModel):
    """
    Custom trained model config. Loads the model from the `scripts/model/trained_model/output` directory if dir_path is
    not provided.
    """

    type: Literal["custom_trained"] = "custom_trained"
    dir_path: str | None = None


class PretrainedModelConfig(BaseModel):
    """
    Pretrained model config. Loads the model from the `pretrained/model` directory.
    """

    type: Literal["pretrained"] = "pretrained"


class PretrainedGPT2ModelType(enum.StrEnum):
    GPT_2 = "gpt-2"
    GPT_2_MEDIUM = "gpt-2-medium"
    GPT_2_LARGE = "gpt-2-large"
    GPT_2_XL = "gpt-2-xl"

    def is_valid(self) -> bool:
        return self in {self.GPT_2, self.GPT_2_MEDIUM, self.GPT_2_LARGE, self.GPT_2_XL}

    def huggingface_model_name(self) -> str:
        return {
            self.GPT_2: "gpt2",
            self.GPT_2_MEDIUM: "gpt2-medium",
            self.GPT_2_LARGE: "gpt2-large",
            self.GPT_2_XL: "gpt2-xl",
        }[self]

    def tokenizer_config(self) -> PretrainedGPTTokenizerConfig:
        return PretrainedGPTTokenizerConfig(
            encoding_or_model_name="gpt-2",
        )

    def _get_model_params(
        self,
        embd_dropout_p: float | None = None,
        attn_dropout_p: float | None = None,
        residual_dropout_p: float | None = None,
    ) -> _ModelParams:
        params = {
            self.GPT_2: _ModelParams(max_seq_len=1024, d_model=768, n_layers=12, n_heads=12),
            self.GPT_2_MEDIUM: _ModelParams(max_seq_len=1024, d_model=768, nn_layers=24, n_heads=16),
            self.GPT_2_LARGE: _ModelParams(max_seq_len=1024, d_model=768, nn_layers=36, n_heads=20),
            self.GPT_2_XL: _ModelParams(max_seq_len=1024, d_model=768, nn_layers=48, n_heads=25),
        }[self]
        params.use_padded_vocab_size = False
        params.use_rope = False
        if embd_dropout_p is not None:
            params.embd_dropout_p = embd_dropout_p
        if attn_dropout_p is not None:
            params.attn_dropout_p = attn_dropout_p
        if residual_dropout_p is not None:
            params.residual_dropout_p = residual_dropout_p
        return params


class PretrainedGPT2ModelConfig(BaseModel):
    """
    Pretrained GPT-2 model config.
    """

    type: Literal["pretrained_gpt_2"] = "pretrained_gpt_2"
    model_type: PretrainedGPT2ModelType
    embd_dropout_p: float | None = None
    attn_dropout_p: float | None = None
    residual_dropout_p: float | None = None


ModelConfig = Annotated[
    UntrainedModelConfig | CustomTrainedModelConfig | PretrainedModelConfig | PretrainedGPT2ModelConfig,
    Field(discriminator="type"),
]


_MODEL_CREATE_KEY = object()


class Model(nn.Module):
    """
    Micro GPT model.

    See: https://github.com/openai/gpt-2/blob/master/src/model.py
    See: https://github.com/karpathy/nanoGPT/blob/master/model.py
    """

    _logger: logging.Logger
    _device: str
    _device_type: DeviceType
    _params: _ModelParams
    _tokenizer: Tokenizer
    _vocab_size: int
    _padded_vocab_size: int

    transformer: nn.ModuleDict
    lm_head: nn.Linear

    def __init__(
        self,
        create_key: object,
        logger: logging.Logger,
        device: str,
        params: _ModelParams,
        tokenizer: Tokenizer,
    ) -> None:
        """
        Initialize the model. Do not call this constructor directly.
        Instead, use Model.load.

        Args:
            create_key: A key to prevent instantiating the model directly
            logger: The logger to use
            device: The device to use
            params: The params to use
            tokenizer: The tokenizer to use
        """
        if create_key != _MODEL_CREATE_KEY:
            raise ValueError("Model cannot be instantiated directly. Use Model.load")

        super().__init__()
        self._logger = logger
        self._device = device
        self._device_type = _get_device_type(self._device)
        self._params = params
        self._tokenizer = tokenizer
        self._vocab_size = tokenizer.vocab_size
        # Padded vocab size to be divisible by 64
        self._padded_vocab_size = (
            ((tokenizer.vocab_size + 63) // 64) * 64 if params.use_padded_vocab_size else tokenizer.vocab_size
        )

        # Modules
        self.transformer = nn.ModuleDict(
            {
                "wte": _embedding(self._padded_vocab_size, params.d_model, params.init_std),
                "drop": nn.Dropout(params.embd_dropout_p),
                "h": nn.ModuleList([_Block(params, layer_id, self._device) for layer_id in range(params.n_layers)]),
                "ln_f": _layer_norm(params.d_model),
            }
        )
        if not params.use_rope:
            self.transformer.update({"wpe": _embedding(params.max_seq_len, params.d_model, params.init_std)})
        self.lm_head = _linear(params.d_model, self._padded_vocab_size, params.n_layers, params.init_std, bias=False)

        # Weight tying.
        # See: https://paperswithcode.com/method/weight-tying
        self.lm_head.weight = self.transformer.wte.weight  # type: ignore

        self.to(self._device)

        # Report number of parameters
        self._logger.info("No. of parameters: %.2fM" % (self.get_num_params() / 1e6))

    def __str__(self) -> str:
        return (
            "Model(\n"
            f"  device={self._device}\n"
            f"  params={self._params.model_dump()}\n"
            f"  tokenizer={self._tokenizer}\n"
            f"  vocab_size={self._vocab_size}\n"
            f"  padded_vocab_size={self._padded_vocab_size}\n"
            ")"
        )

    def __repr__(self) -> str:
        return str(self)

    @property
    def device(self) -> str:
        return self._device

    @property
    def device_type(self) -> DeviceType:
        return self._device_type

    @property
    def params(self) -> _ModelParams:
        return self._params

    @property
    def tokenizer(self) -> Tokenizer:
        return self._tokenizer

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def padded_vocab_size(self) -> int:
        return self._padded_vocab_size

    @no_type_check
    def forward(
        self,
        ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        return_all_logits: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        device = ids.device
        B, T = ids.size()
        assert T <= self._params.max_seq_len, (
            f"Cannot forward sequence of length {T}, maximum sequence length is only {self._params.max_seq_len}"
        )

        x = self.transformer.wte(ids)  # (B, T, d_model)
        if not self._params.use_rope:
            pos = torch.arange(0, T, dtype=torch.long, device=device)  # shape (t)
            pos_emb = self.transformer.wpe(pos)  # (T, d_model)
            x = x + pos_emb  # (B, T, d_model)

        x = self.transformer.drop(x)  # (B, T, d_model)
        for block in self.transformer.h:
            x = block(x)  # (B, T, d_model)
        x = self.transformer.ln_f(x)  # (B, T, d_model)

        if targets is not None:
            # If we are given some desired targets then calculate the loss
            logits = self.lm_head(x)  # (B, T, vocab_size)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            if return_all_logits:
                logits = self.lm_head(x)
            else:
                # Inference-time mini-optimization: only forward the lm_head on the very last position
                logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """
        Take a conditioning sequence of indices ids (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time

        Args:
            ids: The conditioning sequence of indices
            max_new_tokens: The maximum number of new tokens to generate
            temperature: The temperature of the softmax
            top_k: The top k options

        Returns:
            The generated sequence
        """
        assert temperature >= 0.0 and temperature <= 1.0

        self.eval()
        for _ in range(max_new_tokens):
            # If the sequence context is growing too long we must crop it at max_seq_len
            ids_cond = ids if ids.size(1) <= self._params.max_seq_len else ids[:, -self._params.max_seq_len :]
            # Forward the model to get the logits for the index in the sequence
            logits: torch.Tensor
            logits, _ = self(ids_cond)  # (B, T, vocab_size)

            # If the logits are padded, crop them to the actual vocabulary size
            if logits.size(-1) > self._vocab_size:
                logits = logits[:, :, : self._vocab_size]

            # Pluck the logits at the final step and scale by desired temperature
            if temperature > 0.0:
                logits = logits[:, -1, :] / temperature
            else:
                # Argmax sampling
                logits = logits[:, -1, :]
                max_indices = torch.argmax(logits, dim=-1, keepdim=True)
                mask = torch.ones_like(logits, dtype=torch.bool)
                mask.scatter_(dim=-1, index=max_indices, value=False)
                logits.masked_fill_(mask, -float("inf"))

            # Optionally crop the logits to only the top k options
            if top_k is not None and top_k < logits.size(-1):
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("inf")

            # Apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # Sample from the distribution
            id_next = torch.multinomial(probs, num_samples=1)
            # Append sampled index to the running sequence and continue
            ids = torch.cat((ids, id_next), dim=1)

        return ids

    @torch.no_grad()
    def generate_text(
        self,
        text: str,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> str:
        ids = self._tokenizer.encode(text)
        ids_tensor = torch.tensor(ids, dtype=torch.long, device=self._device).unsqueeze(0)
        ids_tensor = self.generate(ids_tensor, max_new_tokens, temperature, top_k)
        return self._tokenizer.decode(ids_tensor[0].tolist())

    def crop_seq_len(self, max_seq_len: int) -> None:
        """
        Model surgery to decrease the maximum sequence length if necessary.
        For example: we may load the GPT2 pretrained model checkpoint (max_seq_len 1024)
        but want to use a smaller max_seq_len for some smaller, simpler model.

        Args:
            max_seq_len: The new maximum sequence length
        """
        assert max_seq_len <= self._params.max_seq_len
        self._params.max_seq_len = max_seq_len
        if not self._params.use_rope:
            self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:max_seq_len])  # type: ignore
        for block in self.transformer.h:  # type: ignore
            if hasattr(block.attn, "bias"):
                block.attn.bias = block.attn.bias[:, :, :max_seq_len, :max_seq_len]

    def configure_optimizer(
        self,
        weight_decay: float | None,
        learning_rate: float,
        betas: tuple[float, float],
    ) -> torch.optim.Optimizer:
        if weight_decay is None:
            weight_decay = 0.01

        # Start with all of the candidate parameters
        param_dict = dict(self.named_parameters())
        # Filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # Create optimizer groups. Any parameters that is 2D will be weight decayed, otherwise not.
        # So all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        self._logger.info(f"Decayed parameter tensors: len={len(decay_params)} parameters={num_decay_params}")
        self._logger.info(f"Non-decayed parameter tensors: len={len(nodecay_params)} parameters={num_nodecay_params}")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and self._device_type == "cuda"
        extra_args = {"fused": True} if use_fused else {}
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, eps=1e-8, **extra_args)
        self._logger.info(f"AdamW optimizer: use_fused={use_fused}")

        return optimizer

    def get_num_params(self, include_embedding_params: bool = False) -> int:
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.

        Args:
            include_embedding: Whether to include the embedding parameters in the count

        Returns:
            The number of parameters in the model
        """
        n_params = sum(p.numel() for p in self.parameters())
        if include_embedding_params and not self._params.use_rope:
            n_params -= self.transformer.wpe.weight.numel()  # type: ignore
        return n_params

    @no_type_check
    def get_num_params_dict(self) -> dict[str, int]:
        """
        Return the number of parameters in the model by type, layer, and total.

        Returns:
            A dictionary of the number of parameters in the model by type, layer, and total
        """

        n_params = OrderedDict()
        n_params["total"] = sum(p.numel() for p in self.parameters())
        n_params["total/without_embedding"] = n_params["total"] - self.transformer.wte.weight.numel()
        n_params["embedding/token"] = self.transformer.wte.weight.numel()
        n_params["embedding/position"] = self.transformer.wpe.weight.numel() if not self._params.use_rope else 0
        n_params["embedding/total"] = n_params["embedding/token"] + n_params["embedding/position"]
        n_params["block/single/ln_1"] = self.transformer.h[0].ln_1.weight.numel() + (
            self.transformer.h[0].ln_1.bias.numel() if self.transformer.h[0].ln_1.bias is not None else 0
        )
        n_params["block/all/ln_1"] = n_params["block/single/ln_1"] * self._params.n_layers
        n_params["block/single/ln_2"] = self.transformer.h[0].ln_2.weight.numel() + (
            self.transformer.h[0].ln_2.bias.numel() if self.transformer.h[0].ln_2.bias is not None else 0
        )
        n_params["block/all/ln_2"] = n_params["block/single/ln_2"] * self._params.n_layers
        n_params["block/single/attn"] = sum(p.numel() for p in self.transformer.h[0].attn.parameters())
        n_params["block/all/attn"] = n_params["block/single/attn"] * self._params.n_layers
        n_params["block/single/mlp"] = sum(p.numel() for p in self.transformer.h[0].mlp.parameters())
        n_params["block/all/mlp"] = n_params["block/single/mlp"] * self._params.n_layers
        n_params["block/single/total"] = (
            n_params["block/single/ln_1"]
            + n_params["block/single/ln_2"]
            + n_params["block/single/attn"]
            + n_params["block/single/mlp"]
        )
        n_params["block/all/total"] = n_params["block/single/total"] * self._params.n_layers
        n_params["ln_f"] = self.transformer.ln_f.weight.numel() + (
            self.transformer.ln_f.bias.numel() if self.transformer.ln_f.bias is not None else 0
        )
        n_params["lm_head"] = 0
        return n_params

    def estimate_flops_per_fwdbwd(self) -> float:
        """
        Estimate model flops per forward and backward pass.

        Returns:
            The number of flops per iteration
        """
        # First estimate the number of flops we do per iteration.
        # See: Appendix B: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        params = self._params
        max_seq_len, dh, n_layers, n_heads = (
            params.max_seq_len,
            params.d_model // params.n_heads,
            params.n_layers,
            params.n_heads,
        )
        flops_per_token = 6 * N + 12 * n_layers * n_heads * dh * max_seq_len
        flops_per_fwdbwd = flops_per_token * max_seq_len
        return flops_per_fwdbwd

    def estimate_flops_per_fwdbwd_dict(self) -> dict[str, float]:
        """
        Estimate model flops per forward and backward pass by type, layer, and total.

        Returns:
            A dictionary of the number of flops per iteration by type, layer, and total
        """
        flops_per_fwdbwd_dict = OrderedDict()
        flops_per_fwdbwd_dict["total"] = self.estimate_flops_per_fwdbwd()
        flops_per_fwdbwd_dict["embedding/token"] = self._padded_vocab_size * self._params.d_model
        flops_per_fwdbwd_dict["embedding/position"] = self._params.max_seq_len * self._params.d_model

        flops_per_fwdbwd_dict["block/single/attn/kqv"] = (
            2 * self._params.max_seq_len * (self._params.d_model * 3 * self._params.d_model)
        )
        flops_per_fwdbwd_dict["block/all/attn/kqv"] = (
            flops_per_fwdbwd_dict["block/single/attn/kqv"] * self._params.n_layers
        )
        flops_per_fwdbwd_dict["block/single/attn/scores"] = (
            2 * self._params.max_seq_len * self._params.max_seq_len * self._params.d_model
            + (
                self._params.max_seq_len * self._params.max_seq_len * self._params.n_heads
            )  # Division by d_head = d_model / n_heads
        )
        flops_per_fwdbwd_dict["block/all/attn/scores"] = (
            flops_per_fwdbwd_dict["block/single/attn/scores"] * self._params.n_layers
        )
        flops_per_fwdbwd_dict["block/single/attn/reduce"] = (
            2 * self._params.n_heads * (self._params.max_seq_len * self._params.max_seq_len * self._params.d_model)
        )
        flops_per_fwdbwd_dict["block/all/attn/reduce"] = (
            flops_per_fwdbwd_dict["block/single/attn/reduce"] * self._params.n_layers
        )
        flops_per_fwdbwd_dict["block/single/attn/proj"] = (
            2 * self._params.max_seq_len * (self._params.d_model * self._params.d_model)
        )
        flops_per_fwdbwd_dict["block/single/attn"] = sum(
            flops_per_fwdbwd_dict["block/single/attn/" + k] for k in ["kqv", "scores", "reduce", "proj"]
        )
        flops_per_fwdbwd_dict["block/all/attn"] = flops_per_fwdbwd_dict["block/single/attn"] * self._params.n_layers
        ffw_size = 4 * self._params.d_model
        flops_per_fwdbwd_dict["block/single/mlp/fc"] = 2 * self._params.max_seq_len * (self._params.d_model * ffw_size)
        flops_per_fwdbwd_dict["block/all/mlp/fc"] = flops_per_fwdbwd_dict["block/single/mlp/fc"] * self._params.n_layers
        flops_per_fwdbwd_dict["block/single/mlp/proj"] = (
            2 * self._params.max_seq_len * (ffw_size * self._params.d_model)
        )
        flops_per_fwdbwd_dict["block/single/mlp"] = sum(
            flops_per_fwdbwd_dict["block/single/mlp/" + k] for k in ["fc", "proj"]
        )
        flops_per_fwdbwd_dict["block/all/mlp"] = flops_per_fwdbwd_dict["block/single/mlp"] * self._params.n_layers
        flops_per_fwdbwd_dict["block/single/total"] = (
            flops_per_fwdbwd_dict["block/single/attn"] + flops_per_fwdbwd_dict["block/single/mlp"]
        )
        flops_per_fwdbwd_dict["block/all/total"] = flops_per_fwdbwd_dict["block/single/total"] * self._params.n_layers
        flops_per_fwdbwd_dict["lm_head"] = (
            2 * self._params.max_seq_len * (self._params.d_model * self._padded_vocab_size)
        )
        flops_per_fwdbwd_dict["fwd/total"] = flops_per_fwdbwd_dict["block/all/total"] + flops_per_fwdbwd_dict["lm_head"]
        flops_per_fwdbwd_dict["bwd/total"] = 2 * flops_per_fwdbwd_dict["fwd/total"]
        flops_per_fwdbwd_dict["total/rough"] = flops_per_fwdbwd_dict["fwd/total"] + flops_per_fwdbwd_dict["bwd/total"]
        flops_per_fwdbwd_dict["total/rough/accuracy"] = (
            flops_per_fwdbwd_dict["total/rough"] / flops_per_fwdbwd_dict["total"]
        )
        return flops_per_fwdbwd_dict

    def estimate_flops_per_iter(self, fwdbwd_per_iter: int = 1) -> float:
        """
        Estimate model flops per iteration.
        An iteration is a forward and backward pass repeated multiple times.

        Args:
            fwdbwd_per_iter: The number of forward and backward passes per iteration

        Returns:
            The number of flops per iteration
        """
        flops_per_fwdbwd = self.estimate_flops_per_fwdbwd()
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        return flops_per_iter

    def estimate_flops_per_iter_per_sec(self, time_per_iter: float, fwdbwd_per_iter: int = 1) -> float:
        """
        Estimate model flops per iteration per second.
        An iteration is a forward and backward pass repeated multiple times.

        Args:
            time_per_iter: The time taken for an iteration
            fwdbwd_per_iter: The number of forward and backward passes per iteration

        Returns:
            The number of flops per iteration
        """
        flops_per_iter = self.estimate_flops_per_iter(fwdbwd_per_iter)
        flops_per_iter_per_sec = flops_per_iter * (1.0 / time_per_iter)
        return flops_per_iter_per_sec

    def estimate_mfu(self, flops_promised: int, time_per_iter: float, fwdbwd_per_iter: int = 1) -> float:
        """
        Estimate model flops utilization (MFU) in units of flops_promised peak FLOPS

        Args:
            flops_promised: The number of flops promised by the device
            time_per_iter: The time taken for an iteration
            fwdbwd_per_iter: The number of forward and backward passes per iteration

        Returns:
            The model flops utilization (MFU)
        """
        flops_per_iter_per_sec = self.estimate_flops_per_iter_per_sec(time_per_iter, fwdbwd_per_iter)
        mfu = flops_per_iter_per_sec / flops_promised
        return mfu

    async def save(self, dir_path: str) -> None:
        """
        Save the model to two files: dir_path/model.json and dir_path/model.pt.
        - model.json file contains the model parameters
        - model.pt file contains the model weights

        Save the tokenizer to dir_path/tokenizer.json and human-readable dir_path/tokenizer_vocab.json.
        See Tokenizer.save() for more details.

        Args:
            dir_path: The directory to save the model to. If the directory already exists, files in it might be
                overwritten
        """
        await self._tokenizer.save(dir_path)

        import aiofiles

        params_file_path = os.path.join(dir_path, "model.json")
        self._logger.info(f"Saving model params: file_path={params_file_path}")
        async with aiofiles.open(params_file_path, "w") as f:
            await f.write(self._params.model_dump_json(indent=2))
        self._logger.info("Saved model params")

        weights_file_path = os.path.join(dir_path, "model.pt")
        self._logger.info(f"Saving model weights: file_path={weights_file_path}")
        torch.save(self.state_dict(), weights_file_path)
        self._logger.info("Saved model weights")

    async def _validate_matches_config(self, config: ModelConfig) -> None:
        if config.type == "untrained" or config.type == "pretrained":
            other_model = await Model.load(config=config)
            if self._params != other_model._params:
                raise ValueError("Model params do not match config")
            return
        elif config.type == "pretrained_gpt_2":
            if not config.model_type.is_valid():
                raise ValueError(f"Invalid model type: {config.model_type}")

            config_params = config.model_type._get_model_params()
            if self._params != config_params:
                raise ValueError("Model params do not match config")
        else:
            raise ValueError(f"Unknown model config type: {config.type}")

    @classmethod
    async def _load_untrained(
        cls,
        config: UntrainedModelConfig,
        logger: logging.Logger,
        device: str,
    ) -> "Model":
        """
        Load an untrained model.

        Args:
            config: The untrained model config to use
            logger: The logger to use
            device: The device to use

        Returns:
            A model
        """
        logger.info(f"Loading untrained model: config={config} device={device}")
        tokenizer = await Tokenizer.load(config=config.tokenizer_config)
        params = _ModelParams(
            max_seq_len=config.max_seq_len,
            d_model=config.d_model,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            use_padded_vocab_size=config.use_padded_vocab_size,
            use_rope=config.use_rope,
            rope_theta=config.rope_theta,
            is_rope_full_precision=config.is_rope_full_precision,
            embd_dropout_p=config.embd_dropout_p,
            attn_dropout_p=config.attn_dropout_p,
            residual_dropout_p=config.residual_dropout_p,
            init_std=config.init_std,
            init_residual_scaled_factor=config.init_residual_scaled_factor,
        )
        model = Model(
            create_key=_MODEL_CREATE_KEY,
            logger=logger,
            device=device,
            params=params,
            tokenizer=tokenizer,
        )
        logger.info(f"Loaded untrained model: model={model}")
        return model

    @classmethod
    async def _load_custom_trained(
        cls,
        config: CustomTrainedModelConfig,
        logger: logging.Logger,
        device: str,
        name: Literal["custom_trained", "pretrained"] = "custom_trained",
    ) -> "Model":
        """
        Load a custom trained model that was saved to config.dir_path.

        Args:
            config: The custom trained model config to use
            logger: The logger to use
            device: The device to use
            name: The name to display in the logger

        Returns:
            A model
        """
        if config.dir_path is None:
            config.dir_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..", "..", "scripts", "model", "trained_model", "output")
            )

        logger.info(f"Loading {name} model: config={config}")
        tokenizer = await Tokenizer.load(config=CustomTrainedTokenizerConfig(dir_path=config.dir_path))

        import json

        import aiofiles

        params_file_path = os.path.join(config.dir_path, "model.json")
        logger.info(f"Loading {name} model params: file_path={params_file_path}")
        async with aiofiles.open(params_file_path, encoding="utf-8") as f:
            params_json = json.loads(await f.read())
        params = _ModelParams(**params_json)
        logger.info(f"Loaded {name} model params: params={params.model_dump()}")

        weights_file_path = os.path.join(config.dir_path, "model.pt")
        logger.info(f"Loading {name} model weights: file_path={weights_file_path}")
        model = Model(
            create_key=_MODEL_CREATE_KEY,
            logger=logger,
            device=device,
            tokenizer=tokenizer,
            params=params,
        )
        model.load_state_dict(torch.load(weights_file_path, map_location=device, weights_only=True))
        logger.info(f"Loaded {name} model weights")
        logger.info(f"Loaded {name} model: model={model}")
        return model

    @classmethod
    async def _load_pretrained_gpt_2(
        cls,
        config: PretrainedGPT2ModelConfig,
        logger: logging.Logger,
        device: str,
    ) -> "Model":
        """
        Load a pretrained GPT-2 model.

        Args:
            config: The pretrained GPT-2 model config to use
            logger: The logger to use
            device: The device to use

        Returns:
            A pretrained GPT-2 model
        """
        logger.info(f"Loading pretrained GPT-2 model: config={config}")
        assert config.model_type.is_valid()

        from transformers import GPT2LMHeadModel

        tokenizer_config = config.model_type.tokenizer_config()
        tokenizer = await Tokenizer.load(config=tokenizer_config)

        params = config.model_type._get_model_params(
            embd_dropout_p=config.embd_dropout_p,
            attn_dropout_p=config.attn_dropout_p,
            residual_dropout_p=config.residual_dropout_p,
        )
        logger.info(f"Loaded model params: params={params.model_dump()}")

        model = Model(
            create_key=_MODEL_CREATE_KEY,
            logger=logger,
            # Load the model on the CPU. Move it to device later when weights are loaded
            device="cpu",
            tokenizer=tokenizer,
            params=params,
        )
        sd = model.state_dict()
        # Discard the attention bias buffer
        sd_keys = [k for k in sd.keys() if not k.endswith(".attn.bias")]

        # Initialize a huggingface/transformers model
        huggingface_model_name = config.model_type.huggingface_model_name()
        logger.info(f"Loading Huggingface pretrained GPT-2 model: huggingface_model_name={huggingface_model_name}")
        model_hf = GPT2LMHeadModel.from_pretrained(huggingface_model_name)
        logger.info("Loaded Huggingface pretrained GPT-2 model")
        sd_hf = model_hf.state_dict()

        # Copy while ensuring all of the parameters are aligned and match in names and shapes
        # Discard the attention bias buffer
        sd_keys_hf = [k for k in sd_hf.keys() if not k.endswith(".attn.masked_bias") and not k.endswith(".attn.bias")]
        assert len(sd_keys_hf) == len(sd_keys), f"Mismatched keys length: {len(sd_keys_hf)} != {len(sd_keys)}"

        # The GPT checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear.
        # This means that we have to transpose these weights when we import them
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # Special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # Vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        model.to(device)
        logger.info(f"Loaded pretrained GPT-2 model: model={model}")
        return model

    @classmethod
    async def load(cls, config: ModelConfig, device: str | None = None) -> "Model":
        """
        Load a model.

        Args:
            config: The model config to use
            device: The device to use

        Returns:
            A model
        """
        logger = _new_logger(__name__)
        device = _get_device(device)
        if config.type == "untrained":
            return await Model._load_untrained(
                config=config,
                logger=logger,
                device=device,
            )
        elif config.type == "custom_trained":
            return await Model._load_custom_trained(
                config=config,
                logger=logger,
                device=device,
                name="custom_trained",
            )
        elif config.type == "pretrained":
            dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "pretrained", "model"))
            model_weights_file_path = os.path.join(dir_path, "model.pt")
            if not os.path.exists(model_weights_file_path):
                raise FileNotFoundError(
                    f"Pretrained model file at {model_weights_file_path} not found. "
                    "Please run `python scripts/download_pretrained.py` to download the model files."
                )

            file_size = os.path.getsize(model_weights_file_path)
            if file_size < 100 * 1024:
                raise ValueError(
                    f"Invalid pretrained model file at {model_weights_file_path}. "
                    "Please run `python scripts/download_pretrained.py` to download the model files."
                )

            logger.info(f"Pretrained model file size: {file_size} bytes")
            return await Model._load_custom_trained(
                config=CustomTrainedModelConfig(dir_path=dir_path),
                logger=logger,
                device=device,
                name="pretrained",
            )
        elif config.type == "pretrained_gpt_2":
            return await Model._load_pretrained_gpt_2(
                config=config,
                logger=logger,
                device=device,
            )
        else:
            raise ValueError(f"Unknown model config type: {config.type}")
