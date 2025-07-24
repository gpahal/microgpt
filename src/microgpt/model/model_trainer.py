import logging
import math
import os
import random
import time
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import Any, cast

import torch
import torch.distributed as distributed
from pydantic import BaseModel
from torch.nn.parallel import DistributedDataParallel as DDP

from microgpt.common.ddp import _DDPParams, _get_ddp_params
from microgpt.common.device import _get_device, _get_device_type, _get_dtype, _get_torch_dtype
from microgpt.common.logger import _new_logger, _set_logging_level
from microgpt.common.trainer import TrainerCheckpointingConfig, _Trainer, _TrainerParams

from .data_loader import _DataLoader
from .hellaswag_utils import _get_most_likely_row, _render_example, _split_examples_iter
from .model import CustomTrainedModelConfig, Model, ModelConfig


class ModelTrainerConfig(BaseModel):
    model_konfig: ModelConfig
    output_dir_path: str | None = None
    checkpointing_config: TrainerCheckpointingConfig | None = None
    manual_seed: int | None = None
    should_compile: bool = True
    data_dir_path: str | None = None
    epochs: int = 1
    max_iterations_per_epoch: int
    batch_size: int
    gradient_accumulation_iterations: int = 1
    max_learning_rate: float | None = None
    min_learning_rate: float | None = None
    learning_rate_warmup_iterations: int | None = None
    learning_rate_decay_iterations: int | None = None
    weight_decay: float | None = None
    betas: tuple[float, float] = (0.9, 0.95)
    max_grad_norm: float = 1.0
    log_interval: int = 1
    eval_interval: int | None = None
    eval_iterations: int | None = None
    enable_hellaswag_eval: bool = False
    hellaswag_eval_interval: int | None = None
    generate_text_interval: int | None = None
    loss_output_file_path: str
    eval_output_file_path: str


class _ModelTrainerParams(BaseModel):
    manual_seed: int | None
    should_compile: bool
    data_dir_path: str | None
    epochs: int
    max_iterations_per_epoch: int
    batch_size: int
    gradient_accumulation_iterations: int
    max_learning_rate: float | None
    min_learning_rate: float | None
    learning_rate_warmup_iterations: int | None
    learning_rate_decay_iterations: int | None
    weight_decay: float | None
    betas: tuple[float, float]
    max_grad_norm: float
    log_interval: int
    eval_interval: int | None
    eval_iterations: int | None
    enable_hellaswag_eval: bool
    hellaswag_eval_interval: int | None
    generate_text_interval: int | None
    loss_output_file_path: str
    eval_output_file_path: str
    min_val_loss: float | None = None
    latest_val_loss: float | None = None


@dataclass
class _ModelTrainerRunContext:
    _model: torch.nn.Module
    _gradient_accumulation_iterations: int
    _eval_interval: int
    _eval_iterations: int
    _hellaswag_eval_interval: int
    _generate_text_interval: int
    _val_data_loader: _DataLoader
    _training_data_loader: _DataLoader
    _running_flops_per_iter_per_sec: float | None = None


_MODEL_TRAINER_NAME = "model"
_MODEL_TRAINER_CREATE_KEY = object()


class ModelTrainer(_Trainer[Model, _ModelTrainerRunContext]):
    _model: Model
    _is_compiled_model: bool
    _optimizer: torch.optim.Optimizer
    _params: _ModelTrainerParams
    _dtype: str
    _tdtype: torch.dtype
    _backend: str
    _grad_scaler: torch.GradScaler
    _is_ddp: bool
    _ddp_params: _DDPParams
    _is_master_process: bool

    def __init__(
        self,
        create_key: object,
        logger: logging.Logger,
        abstract_params: _TrainerParams,
        params: _ModelTrainerParams,
        model: Model,
        optimizer: torch.optim.Optimizer,
    ):
        """
        Initialize the model trainer. Do not call this constructor directly.
        Instead, use ModelTrainer.load.

        Args:
            create_key: A key to prevent instantiating the model trainer directly
            logger: The logger to use
            abstract_params: The abstract trainer params to use
            params: The model trainer params to use
            model: The model to train
            optimizer: The optimizer to use
        """
        if create_key != _MODEL_TRAINER_CREATE_KEY:
            raise ValueError("ModelTrainer cannot be instantiated directly. Use ModelTrainer.load")

        super().__init__(logger=logger, params=abstract_params)
        self._model = model
        self._is_compiled_model = False
        self._optimizer = optimizer
        self._params = params
        self._dtype = _get_dtype(model._device_type)
        self._tdtype = _get_torch_dtype(self._dtype)
        self._backend = "nccl" if model._device_type == "cuda" else "gloo"
        self._grad_scaler = torch.GradScaler(device=model._device_type, enabled=(self._dtype == "float16"))

        # DDP
        ddp_params = _get_ddp_params()
        self._is_ddp = ddp_params is not None
        self._ddp_params = ddp_params if ddp_params is not None else _DDPParams(_rank=0, _local_rank=0, _world_size=1)
        self._is_master_process = self._ddp_params._rank == 0

    def __str__(self) -> str:
        return (
            "ModelTrainer(\n"
            f"  name={self._name}\n"
            f"  iteration={self._iteration}\n"
            f"  latest_run_iteration={self._latest_run_iteration}\n"
            f"  output_dir_path={self._output_dir_path}\n"
            f"  checkpointing_config={self._checkpointing_config}\n"
            f"  params={self._params.model_dump()}\n"
            f"  model={self._model}\n"
            f"  optimizer={self._optimizer}\n"
            f"  dtype={self._dtype}\n"
            f"  tdtype={self._tdtype}\n"
            f"  backend={self._backend}\n"
            f"  grad_scaler={self._grad_scaler}\n"
            f"  is_grad_scaler_enabled={self._grad_scaler._enabled}\n"
            f"  is_ddp={self._is_ddp}\n"
            f"  ddp_params={self._ddp_params}\n"
            f"  is_master_process={self._is_master_process}\n"
            ")"
        )

    def __repr__(self) -> str:
        return str(self)

    @property
    def model(self) -> Model:
        return self._model

    def _get_params(self) -> BaseModel:
        return self._params

    async def _save_output_data(self, dir_path: str) -> None:
        await self._model.save(dir_path)

    async def _save_checkpoint_data(self, dir_path: str) -> None:
        await self._save_output_data(dir_path)

        weights_file_path = os.path.join(dir_path, "optimizer.pt")
        self._logger.info(f"Saving optimizer weights: file_path={weights_file_path}")
        torch.save(self._optimizer.state_dict(), weights_file_path)
        self._logger.info("Saved optimizer weights")

    async def _run_setup(self) -> tuple[int | None, bool, _ModelTrainerRunContext]:
        gradient_accumulation_iterations = self._params.gradient_accumulation_iterations
        if self._is_ddp:
            # Initialize the DDP process group
            distributed.init_process_group(backend=self._backend)

            # World size number of processes will be training simultaneously, so we can scale down the desired gradient
            # accumulation iterations per process proportionally
            assert gradient_accumulation_iterations % self._ddp_params._world_size == 0
            gradient_accumulation_iterations //= self._ddp_params._world_size

        self._logger.info(
            f"Training: "
            f"device={self._model._device} "
            f"rank={self._ddp_params._rank} "
            f"local_rank={self._ddp_params._local_rank} "
            f"world_size={self._ddp_params._world_size}"
        )

        if self._model._device_type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        tokens_per_iter = (
            self._params.batch_size
            * self._model._params.max_seq_len
            * gradient_accumulation_iterations
            * self._ddp_params._world_size
        )
        self._logger.info(f"Tokens per iteration: {tokens_per_iter}")

        # Compile the model
        if self._params.should_compile and not self._is_compiled_model:
            self._logger.info("Compiling model")
            self._model = cast(Model, torch.compile(self._model))
            self._logger.info("Compiled model")
            self._is_compiled_model = True

        model: torch.nn.Module = self._model
        # Wrap the model in a DDP container
        if self._is_ddp:
            model = DDP(self._model, device_ids=[self._ddp_params._local_rank])

        dirname = os.path.abspath(os.path.dirname(__file__))

        eval_interval = self._params.eval_interval
        if eval_interval is None:
            if self._params.max_iterations_per_epoch <= 10:
                eval_interval = 1
            elif self._params.max_iterations_per_epoch <= 100:
                eval_interval = 10
            elif self._params.max_iterations_per_epoch <= 1000:
                eval_interval = 100
            else:
                eval_interval = 250

        eval_iterations = self._params.eval_iterations
        if eval_iterations is None:
            if self._params.max_iterations_per_epoch <= 10:
                eval_iterations = min(3, self._params.max_iterations_per_epoch)
            elif self._params.max_iterations_per_epoch <= 100:
                eval_iterations = 10
            else:
                eval_iterations = 20

        hellaswag_eval_interval = self._params.hellaswag_eval_interval
        if hellaswag_eval_interval is None:
            hellaswag_eval_interval = eval_interval

        generate_text_interval = self._params.generate_text_interval
        if generate_text_interval is None:
            generate_text_interval = eval_interval

        if self._is_master_process:
            self._logger.info(
                f"Training eval: "
                f"eval_interval={eval_interval} "
                f"eval_iterations={eval_iterations} "
                f"hellaswag_eval_interval={hellaswag_eval_interval} "
                f"generate_text_interval={generate_text_interval}"
            )

        epoch = (self._iteration - 1) // self._params.max_iterations_per_epoch
        epoch_iteration = ((self._iteration - 1) % self._params.max_iterations_per_epoch) + 1
        shards_dir_path = (
            os.path.abspath(os.path.join(dirname, "..", "..", "..", "scripts/model/data/data_stage1/shards"))
            if self._params.data_dir_path is None
            else os.path.abspath(os.path.join(self._params.data_dir_path, "shards"))
        )
        val_data_loader = _DataLoader(
            shards_dir_path=shards_dir_path,
            split="val",
            tokenizer=self._model._tokenizer,
            batch_size=self._params.batch_size,
            sequence_length=self._model._params.max_seq_len,
            process_rank=self._ddp_params._rank,
            num_processes=self._ddp_params._world_size,
            seed=self._params.manual_seed,
        )
        training_data_loader = _DataLoader(
            shards_dir_path=shards_dir_path,
            split="train",
            tokenizer=self._model._tokenizer,
            batch_size=self._params.batch_size,
            sequence_length=self._model._params.max_seq_len,
            process_rank=self._ddp_params._rank,
            num_processes=self._ddp_params._world_size,
            seed=self._params.manual_seed,
            epoch=epoch,
            skip_batches=max(0, (epoch_iteration - 1) * gradient_accumulation_iterations),
        )
        if self._is_master_process:
            self._logger.info(
                f"Training data loader: "
                f"num_shards={len(training_data_loader._shards)} "
                f"current_shard={training_data_loader._curr_shard} "
                f"current_position={training_data_loader._curr_batch}"
            )

        return (
            self._params.epochs * self._params.max_iterations_per_epoch,
            self._is_master_process,
            _ModelTrainerRunContext(
                _model=model,
                _gradient_accumulation_iterations=gradient_accumulation_iterations,
                _eval_interval=eval_interval,
                _eval_iterations=eval_iterations,
                _hellaswag_eval_interval=hellaswag_eval_interval,
                _generate_text_interval=generate_text_interval,
                _val_data_loader=val_data_loader,
                _training_data_loader=training_data_loader,
            ),
        )

    def _get_model_forward_ctx(self) -> AbstractContextManager[Any]:
        return (
            nullcontext()
            if self._model._device_type == "cpu"
            else torch.autocast(device_type=self._model._device_type, dtype=self._tdtype)
        )

    async def _run_teardown(self, run_context: _ModelTrainerRunContext) -> None:
        if self._is_ddp:
            distributed.destroy_process_group()

    def _get_lr(self) -> float:
        max_learning_rate = self._params.max_learning_rate
        if max_learning_rate is None:
            num_params = self._model.get_num_params()
            if num_params <= 200_000_000:
                max_learning_rate = 1e-3
            elif num_params <= 1_000_000_000:
                max_learning_rate = 6e-4
            elif num_params <= 10_000_000_000:
                max_learning_rate = 4e-4
            else:
                max_learning_rate = 2e-4

        min_learning_rate = self._params.min_learning_rate
        if min_learning_rate is None:
            min_learning_rate = max_learning_rate * 0.1
        else:
            min_learning_rate = min(max_learning_rate, max(min_learning_rate, max_learning_rate * 0.05))

        learning_rate_warmup_iterations = min(
            1000,
            (
                self._params.learning_rate_warmup_iterations
                if self._params.learning_rate_warmup_iterations is not None
                else int(float(self._params.max_iterations_per_epoch) * 0.05)
            ),
        )
        learning_rate_decay_iterations = max(
            learning_rate_warmup_iterations + 1,
            self._params.learning_rate_decay_iterations
            if self._params.learning_rate_decay_iterations is not None
            else int(float(self._params.epochs * self._params.max_iterations_per_epoch) * 0.95),
        )
        if self._latest_run_iteration == 1 and self._is_master_process:
            self._logger.info(
                f"Learning rate params: "
                f"warmup_iterations: {learning_rate_warmup_iterations} "
                f"decay_iterations: {learning_rate_decay_iterations}"
            )

        # If it is before the warmup period, linearly increase the learning rate
        if self._iteration <= learning_rate_warmup_iterations:
            return (max_learning_rate * self._iteration) / learning_rate_warmup_iterations
        # If it is after the learning rate decay period, return the minimum learning rate
        if self._iteration > learning_rate_decay_iterations:
            return min_learning_rate
        # In between, use cosine decay down to the minimum learning rate
        decay_ratio = (self._iteration - learning_rate_warmup_iterations) / (
            learning_rate_decay_iterations - learning_rate_warmup_iterations
        )
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_learning_rate + coeff * (max_learning_rate - min_learning_rate)

    def _estimate_loss(self, run_context: _ModelTrainerRunContext) -> float:
        run_context._model.eval()
        run_context._val_data_loader._reset()
        loss_accumulated: torch.Tensor | None = None
        with torch.no_grad():
            for _ in range(run_context._eval_iterations):
                x, y = run_context._val_data_loader._next_batch(device=self._model._device)
                with self._get_model_forward_ctx():
                    _, loss = run_context._model(x, y)
                loss = loss / run_context._eval_iterations
                loss_detached = loss.detach()
                loss_accumulated = loss_accumulated + loss_detached if loss_accumulated is not None else loss_detached
        if self._is_ddp and loss_accumulated is not None:
            distributed.all_reduce(loss_accumulated, op=distributed.ReduceOp.AVG)
        return loss_accumulated.item() if loss_accumulated is not None else 0.0

    def _estimate_hellaswag_accuracy(self, run_context: _ModelTrainerRunContext) -> float:
        run_context._model.eval()
        n_total = 0
        n_correct = 0
        for i, example in enumerate(_split_examples_iter("val")):
            if i % self._ddp_params._world_size != self._ddp_params._rank:
                continue

            ids, mask, label = _render_example(
                tokenizer=self._model._tokenizer, example=example, device=self._model._device
            )
            with torch.no_grad():
                with self._get_model_forward_ctx():
                    logits, _ = run_context._model(ids, return_all_logits=True)
                pred_norm = _get_most_likely_row(ids, mask, logits)
            n_total += 1
            n_correct += int(pred_norm == label)
        if self._is_ddp:
            n_total_tensor = torch.tensor(n_total, dtype=torch.long, device=self._model._device)
            n_correct_tensor = torch.tensor(n_correct, dtype=torch.long, device=self._model._device)
            distributed.all_reduce(n_total_tensor, op=distributed.ReduceOp.SUM)
            distributed.all_reduce(n_correct_tensor, op=distributed.ReduceOp.SUM)
            n_total = cast(int, n_total_tensor.item())
            n_correct = cast(int, n_correct_tensor.item())
        accuracy = n_correct / n_total
        if self._is_master_process:
            self._logger.info(f"HellaSwag: accuracy={(accuracy * 100.0):.4f}%")
        return accuracy

    def _generate_text(self, run_context: _ModelTrainerRunContext) -> str:
        run_context._model.eval()
        text = self._model.generate_text(text="Hi, I'm a language model,", max_new_tokens=32)
        return text

    async def _run_iteration(self, run_context: _ModelTrainerRunContext) -> bool:
        t0 = time.time()

        is_first_current_run_iteration = self._latest_run_iteration == 1
        is_last_iteration = self._iteration == self._params.epochs * self._params.max_iterations_per_epoch
        is_last_epoch_iteration = self._iteration % self._params.max_iterations_per_epoch == 0

        run_context._model.train()
        self._optimizer.zero_grad()
        loss_accumulated = torch.tensor(0.0, device=self._model._device)
        for step in range(run_context._gradient_accumulation_iterations):
            x, y = run_context._training_data_loader._next_batch(device=self._model._device)
            if self._is_ddp:
                # In DDP training, we only need to sync gradients at the last micro step
                cast(DDP, run_context._model).require_backward_grad_sync = (
                    step == run_context._gradient_accumulation_iterations - 1
                )

            with self._get_model_forward_ctx():
                _, loss = run_context._model(x, y)

            # Scale the loss to account for gradient accumulation
            loss = loss / run_context._gradient_accumulation_iterations
            loss_accumulated += loss.detach()
            # Backward pass, with gradient scaling for fp16 training
            self._grad_scaler.scale(loss).backward()

        if self._is_ddp:
            distributed.all_reduce(loss_accumulated, op=distributed.ReduceOp.AVG)

        # Clip the gradients
        grad_norm = None
        if self._params.max_grad_norm > 0.0:
            self._grad_scaler.unscale_(self._optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(run_context._model.parameters(), self._params.max_grad_norm)

        # Get and update the learning rate for this iteration
        lr = self._get_lr()
        for param_group in self._optimizer.param_groups:
            param_group["lr"] = lr

        # Step the optimizer and scaler for fp16 training
        self._grad_scaler.step(self._optimizer)
        self._grad_scaler.update()
        # Wait for the GPU to finish work
        if self._model._device_type == "cuda":
            torch.cuda.synchronize()

        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if (self._iteration % self._params.log_interval == 0 or is_last_iteration) and self._is_master_process:
            tokens_processed = (
                self._params.batch_size
                * self._model._params.max_seq_len
                * run_context._gradient_accumulation_iterations
                * self._ddp_params._world_size
            )
            tokens_per_sec = tokens_processed / dt

            # let the training loop settle a bit
            if self._latest_run_iteration >= 3:
                flops_per_iter_per_sec = self._model.estimate_flops_per_iter_per_sec(
                    time_per_iter=dt,
                    fwdbwd_per_iter=self._params.batch_size * run_context._gradient_accumulation_iterations,
                )
                run_context._running_flops_per_iter_per_sec = (
                    flops_per_iter_per_sec
                    if run_context._running_flops_per_iter_per_sec is None
                    else 0.9 * run_context._running_flops_per_iter_per_sec + 0.1 * flops_per_iter_per_sec
                )
            self._logger.info(
                f"=> Iteration {self._iteration}: "
                f"loss={loss_accumulated.item():.6f} | "
                f"lr={lr:.4e} | "
                f"grad_norm={f'{grad_norm:.4f}' if grad_norm is not None else 'N/A'} | "
                f"dt={dt * 1000:.2f}ms | "
                f"toks/sec={tokens_per_sec:.2f} | "
                f"flops/sec={
                    f'{(run_context._running_flops_per_iter_per_sec / 1e12):.2f}TFLOPS'
                    if run_context._running_flops_per_iter_per_sec is not None
                    else 'N/A'
                }"
            )

        if self._is_master_process:
            with open(self._params.loss_output_file_path, "a") as f:
                f.write(f"{self._iteration},{loss_accumulated.item()}\n")

        # Evaluate the loss on train/val sets and write checkpoints
        if self._iteration % run_context._eval_interval == 0 or is_first_current_run_iteration or is_last_iteration:
            val_loss = self._estimate_loss(run_context)
            if self._is_master_process:
                self._params.latest_val_loss = val_loss
                if self._params.min_val_loss is None or val_loss < self._params.min_val_loss:
                    self._params.min_val_loss = val_loss
                with open(self._params.eval_output_file_path, "a") as f:
                    f.write(f"{self._iteration},val_loss,{val_loss},{loss_accumulated.item()}\n")
                self._logger.info(
                    f"=> Iteration {self._iteration}: "
                    f"val_loss={val_loss:.6f} | "
                    f"min_val_loss={self._params.min_val_loss:.6f}"
                )

        if self._params.enable_hellaswag_eval and (
            self._iteration % run_context._hellaswag_eval_interval == 0
            or is_first_current_run_iteration
            or is_last_iteration
        ):
            hellaswag_accuracy = self._estimate_hellaswag_accuracy(run_context)
            if self._is_master_process:
                with open(self._params.eval_output_file_path, "a") as f:
                    f.write(f"{self._iteration},hellaswag_accuracy,{hellaswag_accuracy}\n")
                self._logger.info(f"=> Iteration {self._iteration}: hellaswag_accuracy={hellaswag_accuracy:.4f}")

        if (
            self._iteration % run_context._generate_text_interval == 0
            or is_first_current_run_iteration
            or is_last_iteration
        ) and self._is_master_process:
            text = self._generate_text(run_context)
            self._logger.info(f"=> Iteration {self._iteration}: generated_text={text}")

        if is_last_epoch_iteration:
            if self._is_master_process:
                self._logger.info(f"Epoch completed: epoch={self._iteration // self._params.max_iterations_per_epoch}")
            if not is_last_iteration:
                run_context._training_data_loader._reset(epoch=self._iteration // self._params.max_iterations_per_epoch)

        return True

    def _get_return_value(self, run_context: _ModelTrainerRunContext) -> Model:
        return self._model

    @classmethod
    async def load(cls, config: ModelTrainerConfig) -> "ModelTrainer":
        """
        Load a model trainer.

        Args:
            config: The model trainer config to use
            logger: The logger to use

        Returns:
            A model trainer
        """
        logger = _new_logger(__name__)
        ddp_params = _get_ddp_params()
        device = _get_device()
        device_type = _get_device_type(device)
        if ddp_params is not None:
            device = f"cuda:{ddp_params._local_rank}"
            torch.cuda.set_device(device)
            logger.info(
                f"CUDA is available: "
                f"is_ddp={True} "
                f"rank={ddp_params._rank} "
                f"local_rank={ddp_params._local_rank} "
                f"world_size={ddp_params._world_size} "
                f"device={device}"
            )

            if ddp_params._local_rank > 0:
                logger.info("Not the master process: checkpointing disabled")
                config.checkpointing_config = None
                _set_logging_level(logging.WARNING)
            else:
                logger.info(
                    f"Master process: checkpointing {'disabled' if config.checkpointing_config is None else 'enabled'}"
                )
        else:
            if device_type == "cuda":
                logger.info(f"CUDA is available: is_ddp={False} rank={0} local_rank={0} world_size={1} device={device}")
            else:
                logger.info(f"CUDA is not available: device={device}")
            logger.info(
                f"Master process: checkpointing {'disabled' if config.checkpointing_config is None else 'enabled'}"
            )
            ddp_params = _DDPParams(
                _rank=0,
                _local_rank=0,
                _world_size=1,
            )

        logger.info(f"Loading model trainer: config={config}")

        abstract_params = _TrainerParams(
            name=_MODEL_TRAINER_NAME,
            output_dir_path=config.output_dir_path,
            checkpointing_config=config.checkpointing_config,
        )
        params = _ModelTrainerParams(
            manual_seed=config.manual_seed,
            should_compile=config.should_compile,
            data_dir_path=config.data_dir_path,
            epochs=config.epochs,
            max_iterations_per_epoch=config.max_iterations_per_epoch,
            batch_size=config.batch_size,
            gradient_accumulation_iterations=config.gradient_accumulation_iterations,
            max_learning_rate=config.max_learning_rate,
            min_learning_rate=config.min_learning_rate,
            learning_rate_warmup_iterations=config.learning_rate_warmup_iterations,
            learning_rate_decay_iterations=config.learning_rate_decay_iterations,
            weight_decay=config.weight_decay,
            betas=config.betas,
            max_grad_norm=config.max_grad_norm,
            log_interval=config.log_interval,
            eval_interval=config.eval_interval,
            eval_iterations=config.eval_iterations,
            enable_hellaswag_eval=config.enable_hellaswag_eval,
            hellaswag_eval_interval=config.hellaswag_eval_interval,
            generate_text_interval=config.generate_text_interval,
            loss_output_file_path=config.loss_output_file_path,
            eval_output_file_path=config.eval_output_file_path,
        )

        # Update torch settings
        manual_seed = (params.manual_seed if params.manual_seed is not None else 42) + ddp_params._rank
        random.seed(manual_seed)
        torch.manual_seed(manual_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(manual_seed)

        torch.set_float32_matmul_precision("high")

        async def _create_non_checkpointed_model_trainer() -> ModelTrainer:
            model = await Model.load(config=config.model_konfig, device=device)
            optimizer = model.configure_optimizer(
                weight_decay=params.weight_decay,
                learning_rate=params.max_learning_rate or 6e-4,
                betas=params.betas,
            )
            model_trainer = ModelTrainer(
                create_key=_MODEL_TRAINER_CREATE_KEY,
                logger=logger,
                abstract_params=abstract_params,
                params=params,
                model=model,
                optimizer=optimizer,
            )
            logger.info(f"Loaded non-checkpointed model trainer: model_trainer={model_trainer}")
            return model_trainer

        if config.checkpointing_config is None:
            return await _create_non_checkpointed_model_trainer()

        if config.output_dir_path is None:
            raise ValueError("Output directory path is required for checkpointed model trainer")

        async def _create_trainer(
            latest_checkpoint_dir_path: str,
            abstract_params: _TrainerParams,
            _params_json: Any,
        ) -> "ModelTrainer":
            model = await Model.load(
                config=CustomTrainedModelConfig(
                    dir_path=latest_checkpoint_dir_path,
                ),
                device=device,
            )
            try:
                await model._validate_matches_config(config.model_konfig)
            except ValueError as e:
                raise ValueError(
                    "Latest checkpointed model does not match config: "
                    f"checkpointed_model={model}, "
                    f"model_config={config.model_konfig}, "
                    f"error={e}"
                ) from e

            optimizer = model.configure_optimizer(
                weight_decay=params.weight_decay,
                learning_rate=params.max_learning_rate or 6e-4,
                betas=params.betas,
            )

            optimizer_weights_file_path = os.path.join(latest_checkpoint_dir_path, "optimizer.pt")
            logger.info(f"Loading optimizer weights: file_path={optimizer_weights_file_path}")
            optimizer.load_state_dict(
                torch.load(optimizer_weights_file_path, map_location=model._device, weights_only=True)
            )
            logger.info("Loaded optimizer weights")

            return ModelTrainer(
                create_key=_MODEL_TRAINER_CREATE_KEY,
                logger=logger,
                abstract_params=abstract_params,
                params=params,
                model=model,
                optimizer=optimizer,
            )

        model_trainer = await _Trainer._load_checkpointed_internal(
            abstract_params,
            _create_trainer,
            logger,
        )
        if model_trainer is None:
            return await _create_non_checkpointed_model_trainer()

        logger.info(f"Loaded model trainer: model_trainer={model_trainer}")
        return model_trainer
