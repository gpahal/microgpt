import logging
import math
import os
import shutil
import time
from contextlib import nullcontext
from typing import Literal

import torch
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from microgpt.logger import _new_logger
from microgpt.model import Model, PretrainedModelConfig, load_model
from microgpt.model.model_utils import _get_dtype
from microgpt.tokenizer.data_utils import _DataLoader

from .types import TrainerParams


def _get_logger(logger: logging.Logger | None = None) -> logging.Logger:
    if not logger:
        logger = _new_logger(__name__)
    return logger


_TRAINER_CREATE_KEY = object()


class Trainer:
    _logger: logging.Logger
    _model: Model
    _params: TrainerParams
    _device: str
    _device_type: str
    _dtype: str
    _ptdtype: torch.dtype
    _backend: str
    _iteration: int
    _local_iterations: int
    _min_val_loss: float
    _optimizer: torch.optim.Optimizer
    _scaler: torch.amp.GradScaler

    def __init__(
        self,
        create_key: object,
        model: Model,
        params: TrainerParams,
        optimizer: torch.optim.Optimizer | None = None,
        iteration: int | None = None,
        min_val_loss: float | None = None,
        logger: logging.Logger | None = None,
    ):
        """
        Initialize the model. Do not call this constructor directly.
        Instead, use microgpt.model.load_model.

        Args:
            create_key: A key to prevent instantiating the trainer directly
            model: The model to use
            params: The params to use
            logger: The logger to use
        """
        if create_key != _TRAINER_CREATE_KEY:
            raise ValueError(
                "Trainer cannot be instantiated directly. Use microgpt.trainer.load_trainer"
            )

        self._logger = _get_logger(logger)
        self._model = model
        self._params = params
        self._device = model.device
        self._device_type = model.device_type
        self._dtype = _get_dtype(self._device_type)
        self._ptdtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[self._dtype]
        self._backend = "nccl" if self._device_type == "cuda" else "gloo"
        self._iteration = 0 if iteration is None else iteration
        self._local_iterations = 0
        self._min_val_loss = 1e9 if min_val_loss is None else min_val_loss
        self._is_ddp = self._device_type == "cuda" and int(os.environ.get("RANK", -1)) != -1
        self._is_compiled_model = False
        self._is_ddp_model = False

        self._optimizer = (
            model.configure_optimizer(
                weight_decay=params.weight_decay,
                learning_rate=params.max_learning_rate,
                betas=params.betas,
            )
            if optimizer is None
            else optimizer
        )
        self._scaler = torch.amp.GradScaler(device=self._device, enabled=(self._dtype == "float16"))

    def train(self) -> None:
        gradient_accumulation_steps = self._params.gradient_accumulation_steps
        if self._is_ddp:
            init_process_group(backend=self._backend)
            ddp_rank = int(os.environ["RANK"])
            ddp_local_rank = int(os.environ["LOCAL_RANK"])
            ddp_world_size = int(os.environ["WORLD_SIZE"])
            device = f"cuda:{ddp_local_rank}"
            torch.cuda.set_device(device)
            master_process = ddp_rank == 0
            seed_offset = ddp_rank
            # world_size number of processes will be training simultaneously, so we can scale down the
            # desired gradient accumulation iterations per process proportionally
            assert gradient_accumulation_steps % ddp_world_size == 0
            gradient_accumulation_steps //= ddp_world_size
        else:
            master_process = True
            seed_offset = 0
            ddp_world_size = 1

        if self._device_type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        tokens_per_iter = (
            gradient_accumulation_steps
            * ddp_world_size
            * self._params.batch_size
            * self._model.params.block_size
        )
        self._logger.info(f"Tokens per iteration: {tokens_per_iter}")

        enable_checkpointing = (
            self._params.checkpoint_eval_interval is not None
            and self._params.checkpoint_dir_path is not None
        )
        if enable_checkpointing:
            os.makedirs(self._params.checkpoint_dir_path, exist_ok=True)

        torch.manual_seed(self._params.manual_seed + seed_offset)

        ctx = (
            nullcontext()
            if self._device_type == "cpu"
            else torch.amp.autocast(device_type=self._device_type, dtype=self._ptdtype)
        )

        # Compile the model
        if self._params.should_compile and not self._is_compiled_model:
            self._logger.info("Compiling model")
            self._model = torch.compile(self._model)
            self._logger.info("Model compiled")
            self._is_compiled_model = True

        raw_model = self._model
        # Wrap the model in a DDP container
        if self._is_ddp:
            self._model = DDP(self._model, device_ids=[ddp_local_rank])
            self._is_ddp_model = True

        def _get_data_loader(split: Literal["train", "val"]) -> _DataLoader:
            if split == "train":
                return _DataLoader(
                    out_dir_path=self._params.train_out_dir_path,
                    tokenizer=self._model.tokenizer,
                    batch_size=self._params.batch_size,
                    sequence_length=self._model.params.block_size,
                )
            elif split == "val":
                return _DataLoader(
                    out_dir_path=self._params.validation_out_dir_path,
                    tokenizer=self._model.tokenizer,
                    batch_size=self._params.batch_size,
                    sequence_length=self._model.params.block_size,
                )
            else:
                raise ValueError(f"Invalid split: {split}")

        estimate_loss_data_loaders = {
            "train": _get_data_loader("train"),
            "val": _get_data_loader("val"),
        }

        # Estimate the loss over either split using many batches
        @torch.no_grad()
        def estimate_loss():
            out = {}
            self._model.eval()
            for split in ["train", "val"]:
                losses = torch.zeros(self._params.eval_iters)
                for i in range(self._params.eval_iters):
                    X, Y = estimate_loss_data_loaders[split].next_batch(device=self._device)
                    with ctx:
                        _, loss = self._model(X, Y)
                    losses[i] = loss.item()
                out[split] = losses.mean()
            self._model.train()
            return out

        def _get_lr(it):
            # 1) If it is before the warmup period, linearly increase the learning rate
            if it < self._params.learning_rate_warmup_iters:
                return (
                    self._params.max_learning_rate
                    * (it + 1)
                    / (self._params.learning_rate_warmup_iters + 1)
                )
            # 2) If it is after the learning rate decay period, return the minimum learning rate
            if it > self._params.learning_rate_decay_iters:
                return self._params.min_learning_rate
            # 3) In between, use cosine decay down to the minimum learning rate
            decay_ratio = (it - self._params.learning_rate_warmup_iters) / (
                self._params.learning_rate_decay_iters - self._params.learning_rate_warmup_iters
            )
            assert 0 <= decay_ratio <= 1
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            return self._params.min_learning_rate + coeff * (
                self._params.max_learning_rate - self._params.min_learning_rate
            )

        # Training loop
        training_data_loader = _get_data_loader("train")
        X, Y = training_data_loader.next_batch(device=self._device)
        t0 = time.time()
        running_flops_per_iter_per_sec = -1.0
        while True:
            # Get and update the learning rate for this iteration
            lr = _get_lr(self._iteration)
            for param_group in self._optimizer.param_groups:
                param_group["lr"] = lr

            # evaluate the loss on train/val sets and write checkpoints
            if self._iteration % self._params.eval_interval == 0 and master_process:
                losses = estimate_loss()
                self._logger.info(
                    f"=> Step {self._iteration}: train_loss={losses['train']:.4f}, val_loss={losses['val']:.4f}"
                )
                if (
                    enable_checkpointing
                    and self._iteration
                    % (self._params.eval_interval * self._params.checkpoint_eval_interval)
                    == 0
                    and losses["val"] < self._min_val_loss
                ):
                    self._min_val_loss = losses["val"]
                    if self._iteration > 0:
                        if self._params.checkpoint_backup_dir_path and os.path.exists(
                            self._params.checkpoint_dir_path
                        ):
                            if os.path.exists(self._params.checkpoint_backup_dir_path):
                                shutil.rmtree(self._params.checkpoint_backup_dir_path)
                            shutil.copytree(
                                self._params.checkpoint_dir_path,
                                self._params.checkpoint_backup_dir_path,
                            )
                        self.save_checkpoint(self._params.checkpoint_dir_path)

            for micro_step in range(gradient_accumulation_steps):
                if self._is_ddp:
                    # In DDP training we only need to sync gradients at the last micro step.
                    # The official way to do this is with model.no_sync() context manager, but that
                    # bloats the code and leads to code duplication.
                    # Looking at the source of that context manager, it just toggles this variable.
                    self._model.require_backward_grad_sync = (
                        micro_step == gradient_accumulation_steps - 1
                    )
                with ctx:
                    _, loss = self._model(X, Y)
                    # Scale the loss to account for gradient accumulation
                    loss = loss / gradient_accumulation_steps

                # Immediately async prefetch next batch while model is doing the forward pass
                X, Y = training_data_loader.next_batch(device=self._device)
                # Backward pass, with gradient scaling if training in fp16
                self._scaler.scale(loss).backward()

            # Clip the gradients
            if self._params.max_grad_norm != 0.0:
                self._scaler.unscale_(self._optimizer)
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._params.max_grad_norm)
            # Step the optimizer and scaler if training in fp16
            self._scaler.step(self._optimizer)
            self._scaler.update()
            # Flush the gradients as soon as we can, no need for this memory anymore
            self._optimizer.zero_grad(set_to_none=True)

            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            if self._iteration % self._params.log_interval == 0 and master_process:
                # get loss as float. note: this is a CPU-GPU sync point
                # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
                lossf = loss.item() * gradient_accumulation_steps
                if self._local_iterations >= 5:  # let the training loop settle a bit
                    flops_per_iter_per_sec = raw_model.estimate_flops_per_iter_per_sec(
                        time_per_iter=dt,
                        fwdbwd_per_iter=self._params.batch_size * gradient_accumulation_steps,
                    )
                    running_flops_per_iter_per_sec = (
                        flops_per_iter_per_sec
                        if running_flops_per_iter_per_sec == -1.0
                        else 0.9 * running_flops_per_iter_per_sec + 0.1 * flops_per_iter_per_sec
                    )
                self._logger.info(
                    f"=> Iteration {self._iteration}: loss={lossf:.4f}, time_per_iter={dt * 1000:.2f}ms, flops_per_iter_per_sec={running_flops_per_iter_per_sec * 100:.2f}%"
                )
            self._iteration += 1
            self._local_iterations += 1

            # Termination condition
            if self._iteration > self._params.max_iters:
                break

        if self._is_ddp:
            destroy_process_group()

    @classmethod
    def _load_new(
        cls,
        model: Model,
        params: TrainerParams,
        logger: logging.Logger | None = None,
    ) -> "Trainer":
        """
        Load a new trainer.

        Args:
            model: The model to use
            params: The params to use
            logger: The logger to use

        Returns:
            A trainer
        """
        logger = _get_logger(logger)

        logger.info("Loading untrained trainer")
        trainer = Trainer(
            create_key=_TRAINER_CREATE_KEY,
            model=model,
            params=params,
            logger=logger,
        )
        logger.info("Loaded untrained trainer")
        return trainer

    def save_checkpoint(self, file_path_prefix: str) -> None:
        """
        Save the trainer to file_path_prefix.trainer_params.pt and file_path_prefix.trainer_optimizer.pt.
        Save the model to file_path_prefix.model.pt.
        Save the params to file_path_prefix.model_params.pt.
        Save the tokenizer to file_path_prefix.tokenizer.model and human-readable file_path_prefix.tokenizer.vocab.

        Args:
            file_path_prefix: The prefix of the file path to save the model to. If the file
                already exists, it will be overwritten

        Args:
            file_path_prefix: The prefix of the file path to save the model to. If the file
                already exists, it will be overwritten
        """
        self._model.save(file_path_prefix)

        params_file_path = file_path_prefix + ".trainer_params.pt"
        self._logger.info(f"Saving trainer params to {params_file_path}")
        torch.save(
            {
                "params": self._params.model_dump(),
                "iteration": self._iteration,
                "min_val_loss": self._min_val_loss,
            },
            params_file_path,
        )
        self._logger.info(f"Saved trainer params to {params_file_path}")

        optimizer_file_path = file_path_prefix + ".trainer_optimizer.pt"
        self._logger.info(f"Saving trainer optimizer to {optimizer_file_path}")
        torch.save(self._optimizer.state_dict(), optimizer_file_path)
        self._logger.info(f"Saved trainer optimizer to {optimizer_file_path}")

    @classmethod
    async def _load_checkpointed(
        cls,
        file_path_prefix: str,
        device: str | None = None,
        logger: logging.Logger | None = None,
    ) -> "Trainer":
        """
        Load a checkpointed trainer that was saved to file_path_prefix.

        Args:
            file_path_prefix: The prefix of the file path to load the trainer from
            device: The device to use for the trainer
            logger: The logger to use for the trainer

        Returns:
            A trainer
        """
        logger = _get_logger(logger)

        model = await load_model(
            PretrainedModelConfig(file_path_prefix=file_path_prefix), device=device, logger=logger
        )

        params_file_path = file_path_prefix + ".trainer_params.pt"
        logger.info(f"Loading trainer params from {params_file_path}")
        params_dump = torch.load(params_file_path)
        params = TrainerParams(**params_dump["params"])
        iteration = params_dump["iteration"]
        min_val_loss = params_dump["min_val_loss"]
        logger.info(f"Loaded trainer params: {params}")

        optimizer_file_path = file_path_prefix + ".trainer_optimizer.pt"
        logger.info(f"Loading trainer optimizer from {optimizer_file_path}")
        optimizer = model.configure_optimizer(
            weight_decay=params.weight_decay,
            learning_rate=params.max_learning_rate,
            betas=params.betas,
        )
        trainer = Trainer(
            create_key=_TRAINER_CREATE_KEY,
            model=model,
            params=params,
            optimizer=optimizer,
            iteration=iteration,
            min_val_loss=min_val_loss,
            logger=logger,
        )
        optimizer.load_state_dict(
            torch.load(optimizer_file_path, map_location=device, weights_only=True)
        )
        logger.info(f"Loaded trainer from {optimizer_file_path}")
        return trainer
