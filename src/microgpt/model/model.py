"""
Micro GPT model.

See: https://github.com/openai/gpt-2/blob/master/src/model.py
See: https://github.com/karpathy/nanoGPT/blob/master/model.py
"""

import inspect
import logging
import math
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn import functional as F

from microgpt.logger import _new_logger
from microgpt.tokenizer import PretrainedTokenizerConfig, Tokenizer, load_tokenizer
from microgpt.types import TextSource

from .model_utils import _get_device, _get_device_type
from .modules import Block, LayerNorm
from .types import ModelParams, PretrainedGPT2ModelType


def _get_logger(logger: logging.Logger | None = None) -> logging.Logger:
    if not logger:
        logger = _new_logger(__name__)
    return logger


_MODEL_CREATE_KEY = object()


class Model(nn.Module):
    """
    Micro GPT model.
    """

    _logger: logging.Logger
    device: str
    device_type: str
    tokenizer: Tokenizer
    vocab_size: int
    params: ModelParams
    transformer: nn.ModuleDict
    lm_head: nn.Linear

    def __init__(
        self,
        create_key: object,
        tokenizer: Tokenizer,
        params: ModelParams,
        device: str | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        """
        Initialize the model. Do not call this constructor directly.
        Instead, use microgpt.model.load_model.

        Args:
            create_key: A key to prevent instantiating the model directly
            tokenizer: The tokenizer to use
            params: The params to use
            device: The device to use
            logger: The logger to use
        """
        if create_key != _MODEL_CREATE_KEY:
            raise ValueError("Model cannot be instantiated directly. Use microgpt.model.load_model")

        super().__init__()
        self._logger = _get_logger(logger)
        self.device = _get_device(device)
        self.device_type = _get_device_type(self.device)
        self.tokenizer = tokenizer
        # Padded vocab size to be divisible by 128
        self.vocab_size = ((tokenizer.vocab_size + 127) // 128) * 128
        self.params = params
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(self.vocab_size, params.n_embd),
                "wpe": nn.Embedding(params.block_size, params.n_embd),
                "drop": nn.Dropout(params.dropout_p),
                "h": nn.ModuleList([Block(params) for _ in range(params.n_layer)]),
                "ln_f": LayerNorm(params.n_embd, bias=params.bias),
            }
        )
        self.lm_head = nn.Linear(params.n_embd, self.vocab_size, bias=False)

        # See: https://paperswithcode.com/method/weight-tying
        self.transformer.wte.weight = self.lm_head.weight

        # Initialize all weights
        self.apply(self._init_weights)

        # Apply special scaled init to the residual projections
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * params.n_layer))

        self.to(self.device)

        # Report number of parameters
        self._logger.info("Number of parameters: %.2fM" % (self.get_num_params() / 1e6))

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        device = idx.device
        B, T = idx.size()
        assert T <= self.params.block_size, (
            f"Cannot forward sequence of length {T}, block size is only {self.params.block_size}"
        )
        pos = torch.arange(0, T, dtype=torch.long, device=device)  # shape (t)

        tok_emb = self.transformer.wte(idx)  # (B, T, n_embd)
        pos_emb = self.transformer.wpe(pos)  # (T, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)  # (B, T, n_embd)
        for block in self.transformer.h:
            x = block(x)  # (B, T, n_embd)
        x = self.transformer.ln_f(x)  # (B, T, n_embd)

        if targets is not None:
            # If we are given some desired targets then calculate the loss
            logits = self.lm_head(x)  # (B, T, vocab_size)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            # Inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    def crop_block_size(self, block_size: int) -> None:
        """
        Model surgery to decrease the block size if necessary.
        For example: we may load the GPT2 pretrained model checkpoint (block size 1024)
        but want to use a smaller block size for some smaller, simpler model.

        Args:
            block_size: The new block size
        """
        assert block_size <= self.params.block_size
        self.params.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, "bias"):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]

    def configure_optimizer(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: tuple[float, float],
    ) -> torch.optim.Optimizer:
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
        self._logger.info(
            f"Num. of decayed parameter tensors: {len(decay_params)}, with {num_decay_params} parameters"
        )
        self._logger.info(
            f"Num. of non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params} parameters"
        )
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and self.device_type == "cuda"
        extra_args = {"fused": True} if use_fused else {}
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        self._logger.info(f"Using fused AdamW: {use_fused}")

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
        if include_embedding_params:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def get_num_params_dict(self) -> dict[str, int]:
        """
        Return the number of parameters in the model by type, layer, and total.

        Returns:
            A dictionary of the number of parameters in the model by type, layer, and total
        """

        n_params = OrderedDict()
        n_params["total"] = sum(p.numel() for p in self.parameters())
        n_params["total/without_embedding"] = (
            n_params["total"] - self.transformer.wte.weight.numel()
        )
        n_params["embedding/token"] = self.transformer.wte.weight.numel()
        n_params["embedding/position"] = self.transformer.wpe.weight.numel()
        n_params["embedding/total"] = n_params["embedding/token"] + n_params["embedding/position"]
        n_params["block/single/ln_1"] = self.transformer.h[0].ln_1.weight.numel() + (
            self.transformer.h[0].ln_1.bias.numel()
            if self.transformer.h[0].ln_1.bias is not None
            else 0
        )
        n_params["block/all/ln_1"] = n_params["block/single/ln_1"] * self.params.n_layer
        n_params["block/single/ln_2"] = self.transformer.h[0].ln_2.weight.numel() + (
            self.transformer.h[0].ln_2.bias.numel()
            if self.transformer.h[0].ln_2.bias is not None
            else 0
        )
        n_params["block/all/ln_2"] = n_params["block/single/ln_2"] * self.params.n_layer
        n_params["block/single/attn"] = sum(
            p.numel() for p in self.transformer.h[0].attn.parameters()
        )
        n_params["block/all/attn"] = n_params["block/single/attn"] * self.params.n_layer
        n_params["block/single/mlp"] = sum(
            p.numel() for p in self.transformer.h[0].mlp.parameters()
        )
        n_params["block/all/mlp"] = n_params["block/single/mlp"] * self.params.n_layer
        n_params["block/single/total"] = (
            n_params["block/single/ln_1"]
            + n_params["block/single/ln_2"]
            + n_params["block/single/attn"]
            + n_params["block/single/mlp"]
        )
        n_params["block/all/total"] = n_params["block/single/total"] * self.params.n_layer
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
        params = self.params
        L, H, Q, T = (
            params.n_layer,
            params.n_head,
            params.n_embd // params.n_head,
            params.block_size,
        )
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        return flops_per_fwdbwd

    def estimate_flops_per_fwdbwd_dict(self) -> dict[str, float]:
        """
        Estimate model flops per forward and backward pass by type, layer, and total.

        Returns:
            A dictionary of the number of flops per iteration by type, layer, and total
        """
        flops_per_fwdbwd_dict = OrderedDict()
        flops_per_fwdbwd_dict["total"] = self.estimate_flops_per_fwdbwd()
        flops_per_fwdbwd_dict["embedding/token"] = self.vocab_size * self.params.n_embd
        flops_per_fwdbwd_dict["embedding/position"] = self.params.block_size * self.params.n_embd

        flops_per_fwdbwd_dict["block/single/attn/kqv"] = (
            2 * self.params.block_size * (self.params.n_embd * 3 * self.params.n_embd)
        )
        flops_per_fwdbwd_dict["block/all/attn/kqv"] = (
            flops_per_fwdbwd_dict["block/single/attn/kqv"] * self.params.n_layer
        )
        flops_per_fwdbwd_dict["block/single/attn/scores"] = (
            2 * self.params.block_size * self.params.block_size * self.params.n_embd
            + (
                self.params.block_size * self.params.block_size * self.params.n_head
            )  # Division by dh = n_embd / n_head
        )
        flops_per_fwdbwd_dict["block/all/attn/scores"] = (
            flops_per_fwdbwd_dict["block/single/attn/scores"] * self.params.n_layer
        )
        flops_per_fwdbwd_dict["block/single/attn/reduce"] = (
            2
            * self.params.n_head
            * (self.params.block_size * self.params.block_size * self.params.n_embd)
        )
        flops_per_fwdbwd_dict["block/all/attn/reduce"] = (
            flops_per_fwdbwd_dict["block/single/attn/reduce"] * self.params.n_layer
        )
        flops_per_fwdbwd_dict["block/single/attn/proj"] = (
            2 * self.params.block_size * (self.params.n_embd * self.params.n_embd)
        )
        flops_per_fwdbwd_dict["block/single/attn"] = sum(
            flops_per_fwdbwd_dict["block/single/attn/" + k]
            for k in ["kqv", "scores", "reduce", "proj"]
        )
        flops_per_fwdbwd_dict["block/all/attn"] = (
            flops_per_fwdbwd_dict["block/single/attn"] * self.params.n_layer
        )
        ffw_size = 4 * self.params.n_embd
        flops_per_fwdbwd_dict["block/single/mlp/fc"] = (
            2 * self.params.block_size * (self.params.n_embd * ffw_size)
        )
        flops_per_fwdbwd_dict["block/all/mlp/fc"] = (
            flops_per_fwdbwd_dict["block/single/mlp/fc"] * self.params.n_layer
        )
        flops_per_fwdbwd_dict["block/single/mlp/proj"] = (
            2 * self.params.block_size * (ffw_size * self.params.n_embd)
        )
        flops_per_fwdbwd_dict["block/single/mlp"] = sum(
            flops_per_fwdbwd_dict["block/single/mlp/" + k] for k in ["fc", "proj"]
        )
        flops_per_fwdbwd_dict["block/all/mlp"] = (
            flops_per_fwdbwd_dict["block/single/mlp"] * self.params.n_layer
        )
        flops_per_fwdbwd_dict["block/single/total"] = (
            flops_per_fwdbwd_dict["block/single/attn"] + flops_per_fwdbwd_dict["block/single/mlp"]
        )
        flops_per_fwdbwd_dict["block/all/total"] = (
            flops_per_fwdbwd_dict["block/single/total"] * self.params.n_layer
        )
        flops_per_fwdbwd_dict["lm_head"] = (
            2 * self.params.block_size * (self.params.n_embd * self.vocab_size)
        )
        flops_per_fwdbwd_dict["fwd/total"] = (
            flops_per_fwdbwd_dict["block/all/total"] + flops_per_fwdbwd_dict["lm_head"]
        )
        flops_per_fwdbwd_dict["bwd/total"] = 2 * flops_per_fwdbwd_dict["fwd/total"]
        flops_per_fwdbwd_dict["total/rough"] = (
            flops_per_fwdbwd_dict["fwd/total"] + flops_per_fwdbwd_dict["bwd/total"]
        )
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

    def estimate_flops_per_iter_per_sec(
        self, time_per_iter: float, fwdbwd_per_iter: int = 1
    ) -> float:
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

    def estimate_mfu(
        self, flops_promised: int, time_per_iter: float, fwdbwd_per_iter: int = 1
    ) -> float:
        """
        Estimate model flops utilization (MFU) in units of flops_promised peak FLOPS

        Args:
            flops_promised: The number of flops promised by the device
            time_per_iter: The time taken for an iteration
            fwdbwd_per_iter: The number of forward and backward passes per iteration

        Returns:
            The model flops utilization (MFU)
        """
        flops_per_iter_per_sec = self.estimate_flops_per_iter_per_sec(
            time_per_iter, fwdbwd_per_iter
        )
        mfu = flops_per_iter_per_sec / flops_promised
        return mfu

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
            idx: The conditioning sequence of indices
            max_new_tokens: The maximum number of new tokens to generate
            temperature: The temperature of the softmax
            top_k: The top k options

        Returns:
            The generated sequence
        """
        assert temperature >= 0.0 and temperature <= 1.0

        self.eval()
        for _ in range(max_new_tokens):
            # If the sequence context is growing too long we must crop it at block_size
            ids_cond = (
                ids if ids.size(1) <= self.params.block_size else ids[:, -self.params.block_size :]
            )
            # Forward the model to get the logits for the index in the sequence
            logits: torch.Tensor
            logits, _ = self(ids_cond)  # (B, T, vocab_size)
            if self.tokenizer.vocab_size != logits.size(-1):
                logits = logits[:, :, : self.tokenizer.vocab_size]
            # Pluck the logits at the final step and scale by desired temperature
            if temperature > 0.0:
                logits = logits[:, -1, :] / temperature
            else:
                # Argmax sampling
                logits = logits[:, -1, :]
                max_indices = torch.argmax(logits, dim=-1, keepdim=True)
                mask = torch.ones_like(logits, dtype=torch.bool)
                mask.scatter_(dim=-1, index=max_indices, value=False)
                logits.masked_fill_(mask, -float("Inf"))

            # Optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            # Apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # Append sampled index to the running sequence and continue
            ids = torch.cat((ids, idx_next), dim=1)

        return ids

    @torch.no_grad()
    def generate_text(
        self,
        text: str,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> str:
        ids = self.tokenizer.encode(text)
        ids = torch.tensor(ids, dtype=torch.long, device=self.device).unsqueeze(0)
        ids = self.generate(ids, max_new_tokens, temperature, top_k)
        return self.tokenizer.decode(ids[0].tolist())

    async def save_tokenized_text_sources(
        self, out_dir_path: str, text_sources: TextSource | list[TextSource]
    ) -> None:
        """
        Save the tokenized text sources to out_dir_path in shards.

        Args:
            out_dir_path: The path to save the tokenized text sources to
            text_sources: The text sources to tokenize and save
        """
        await self.tokenizer.save_tokenized_text_sources(out_dir_path, text_sources)

    @classmethod
    def _load_untrained(
        cls,
        tokenizer: Tokenizer,
        params: ModelParams,
        device: str | None = None,
        logger: logging.Logger | None = None,
    ) -> "Model":
        """
        Load an untrained model.

        Args:
            tokenizer: The tokenizer to use for the model
            params: The parameters to use for the model
            device: The device to use for the model
            logger: The logger to use for the model

        Returns:
            A model
        """
        logger = _get_logger(logger)

        logger.info("Loading untrained model")
        device = _get_device(device)
        model = Model(
            create_key=_MODEL_CREATE_KEY,
            tokenizer=tokenizer,
            params=params,
            device=device,
            logger=logger,
        )
        logger.info("Loaded untrained model")
        return model

    def save(self, file_path_prefix: str) -> None:
        """
        Save the model to file_path_prefix.model.pt.
        Save the params to file_path_prefix.model_params.pt.
        Save the tokenizer to file_path_prefix.tokenizer.model and human-readable file_path_prefix.tokenizer.vocab.

        Args:
            file_path_prefix: The prefix of the file path to save the model to. If the file
                already exists, it will be overwritten
        """
        self.tokenizer.save(file_path_prefix)

        params_file_path = file_path_prefix + ".model_params.pt"
        self._logger.info(f"Saving model params to {params_file_path}")
        torch.save(self.params.model_dump(), params_file_path)
        self._logger.info(f"Saved model params to {params_file_path}")

        file_path = file_path_prefix + ".model.pt"
        self._logger.info(f"Saving model to {file_path}")
        torch.save(self.state_dict(), file_path)
        self._logger.info(f"Saved model to {file_path}")

    @classmethod
    async def _load_pretrained(
        cls,
        file_path_prefix: str,
        device: str | None = None,
        logger: logging.Logger | None = None,
    ) -> "Model":
        """
        Load a pretrained model that was saved to file_path_prefix.

        Args:
            file_path_prefix: The prefix of the file path to load the model from
            device: The device to use for the model
            logger: The logger to use for the model

        Returns:
            A model
        """
        logger = _get_logger(logger)

        tokenizer = await load_tokenizer(
            PretrainedTokenizerConfig(file_path_prefix=file_path_prefix), logger=logger
        )

        params_file_path = file_path_prefix + ".model_params.pt"
        logger.info(f"Loading model params from {params_file_path}")
        params_dump = torch.load(params_file_path)
        params = ModelParams(**params_dump)
        logger.info(f"Loaded model params: {params}")

        model_file_path = file_path_prefix + ".model.pt"
        logger.info(f"Loading model from {model_file_path}")
        device = _get_device(device)
        model = Model(
            create_key=_MODEL_CREATE_KEY,
            tokenizer=tokenizer,
            params=params,
            device=device,
            logger=logger,
        )
        model.load_state_dict(torch.load(model_file_path, map_location=device, weights_only=True))
        logger.info(f"Loaded model from {model_file_path}")
        return model

    @classmethod
    async def _load_pretrained_gpt_2(
        cls,
        model_type: PretrainedGPT2ModelType,
        dropout_p: float | None = None,
        bias: bool | None = None,
        device: str | None = None,
        logger: logging.Logger | None = None,
    ) -> "Model":
        """
        Load a pretrained GPT-2 model.

        Args:
            model_type: The type of pretrained GPT-2 model to load
            dropout_p: The dropout probability
            bias: Whether to use bias in the model
            device: The device to use for the model
            logger: The logger to use for the model

        Returns:
            A pretrained GPT-2 model
        """
        assert model_type.is_valid()

        from transformers import GPT2LMHeadModel

        logger = _get_logger(logger)

        logger.info(f"Loading tokenizer for pretrained gpt 2 model: {model_type}")
        tokenizer_config = model_type.tokenizer_config()
        tokenizer = await load_tokenizer(tokenizer_config, logger=logger)
        logger.info(f"Loaded tokenizer: {tokenizer}")

        params = model_type.params(dropout_p=dropout_p, bias=bias)
        logger.info(f"Loaded params: {params}")

        model = Model(
            create_key=_MODEL_CREATE_KEY,
            tokenizer=tokenizer,
            params=params,
            # Load the model on the CPU. Move it to device later when weights are loaded
            device="cpu",
            logger=logger,
        )
        sd = model.state_dict()
        sd_keys = sd.keys()
        # Discard the attention bias buffer
        sd_keys = [k for k in sd_keys if not k.endswith(".attn.bias")]

        # Initialize a huggingface/transformers model
        huggingface_model_name = model_type.huggingface_model_name()
        logger.info(f"Loading Huggingface pretrained gpt 2 model: {huggingface_model_name}")
        model_hf = GPT2LMHeadModel.from_pretrained(huggingface_model_name)
        sd_hf = model_hf.state_dict()

        # Copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        # Discard the attention bias buffer
        sd_keys_hf = [
            k
            for k in sd_keys_hf
            if not k.endswith(".attn.masked_bias") and not k.endswith(".attn.bias")
        ]
        assert len(sd_keys_hf) == len(sd_keys), (
            f"Mismatched keys length: {len(sd_keys_hf)} != {len(sd_keys)}"
        )

        # The openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear.
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
                # For wte and lm_head, the vocab size might be padded to be divisible by 128
                if k.endswith("wte.weight") or k.endswith("lm_head.weight"):
                    assert (
                        sd_hf[k].shape[1] == sd[k].shape[1] and sd_hf[k].shape[0] <= sd[k].shape[0]
                    )
                    extra_rows = sd[k].shape[0] - sd_hf[k].shape[0]
                    sd_hf_k = sd_hf[k]
                    if extra_rows > 0:
                        sd_k_delta = torch.zeros(
                            (extra_rows, sd_hf[k].shape[1]),
                            dtype=sd_hf[k].dtype,
                            device=sd_hf[k].device,
                        )
                        sd_hf_k = torch.cat((sd_hf[k], sd_k_delta), dim=0)
                    with torch.no_grad():
                        sd[k].copy_(sd_hf_k)
                else:
                    assert sd_hf[k].shape == sd[k].shape
                    with torch.no_grad():
                        sd[k].copy_(sd_hf[k])

        model.to(device)
        return model
