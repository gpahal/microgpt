import asyncio
import logging
import random
from collections.abc import Coroutine
from contextlib import nullcontext
from typing import Annotated, Any, cast

import torch
import torch.distributed as distributed
import torch.nn.functional as F
import typer
from tqdm import tqdm

from microgpt import (
    CustomTrainedModelConfig,
    Model,
    ModelConfig,
    PretrainedGPT2ModelConfig,
    PretrainedGPT2ModelType,
    PretrainedModelConfig,
)
from microgpt.common.ddp import _DDPParams, _get_ddp_params
from microgpt.common.device import _get_device, _get_dtype, _get_torch_dtype
from microgpt.common.logger import _new_logger, _set_logging_level
from microgpt.model.hellaswag_utils import _NUM_VAL_EXAMPLES, _render_example, _split_examples_iter

logger = _new_logger(__name__)

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)


def _run_async[RT](coro: Coroutine[Any, Any, RT]) -> RT:
    return asyncio.get_event_loop().run_until_complete(coro)


def evaluate(model: Model, is_ddp: bool, ddp_params: _DDPParams) -> None:
    model.eval()
    device = model._device
    device_type = model._device_type
    dtype = _get_dtype(model._device_type)
    tdtype = _get_torch_dtype(dtype)
    is_master_process = ddp_params._local_rank == 0
    model_forward_ctx = nullcontext() if device_type == "cpu" else torch.autocast(device_type=device_type, dtype=tdtype)

    n_total = 0
    n_correct = 0
    if is_master_process:
        progress_bar = tqdm(total=_NUM_VAL_EXAMPLES, desc="Evaluating HellaSwag", unit="examples")
    for i, example in enumerate(_split_examples_iter("val")):
        if is_ddp and i % ddp_params._world_size != ddp_params._rank:
            if is_master_process:
                progress_bar.update(1)
            continue

        ids, mask, label = _render_example(tokenizer=model._tokenizer, example=example, device=device)
        with torch.no_grad():
            with model_forward_ctx:
                logits, _ = model(ids, return_all_logits=True)

            shift_logits = (logits[..., :-1, :]).contiguous()
            shift_tokens = (ids[..., 1:]).contiguous()
            flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            flat_shift_tokens = shift_tokens.view(-1)
            shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction="none")
            shift_losses = shift_losses.view(ids.size(0), -1)
            shift_mask = (mask[..., 1:]).contiguous()
            masked_shift_losses = shift_losses * shift_mask
            sum_loss = masked_shift_losses.sum(dim=1)
            avg_loss = sum_loss / shift_mask.sum(dim=1)
            pred_norm = avg_loss.argmin().item()

        n_total += 1
        n_correct += int(pred_norm == label)

        if is_master_process:
            progress_bar.update(1)

        if is_master_process:
            if n_total <= 3:
                print("---")
                print(f"Context:\n{example['ctx']}")
                print("Endings:")
                for i, end in enumerate(example["endings"]):
                    print(f"{i} (loss: {avg_loss[i].item():.4f}) {end}")
                print(f"Predicted: {pred_norm}, actual: {label}")

    if is_ddp:
        n_total_tensor = torch.tensor(n_total, dtype=torch.long, device=device)
        n_correct_tensor = torch.tensor(n_correct, dtype=torch.long, device=device)
        distributed.all_reduce(n_total_tensor, op=distributed.ReduceOp.SUM)
        distributed.all_reduce(n_correct_tensor, op=distributed.ReduceOp.SUM)
        n_total = cast(int, n_total_tensor.item())
        n_correct = cast(int, n_correct_tensor.item())

    if is_master_process:
        progress_bar.close()
        accuracy = (n_correct * 1.0) / n_total
        logger.info(f"HellaSwag: accuracy={(accuracy * 100.0):.4f}% | n_total={n_total} | n_correct={n_correct}")


async def _load_and_evaluate(config: ModelConfig) -> None:
    device = _get_device()

    ddp_params = _get_ddp_params()
    is_ddp = False
    if ddp_params is not None:
        is_ddp = True
        distributed.init_process_group(backend="nccl")
        device = f"cuda:{ddp_params._local_rank}"
        torch.cuda.set_device(device)
    else:
        ddp_params = _DDPParams(_rank=0, _local_rank=0, _world_size=1)

    if ddp_params._local_rank != 0:
        _set_logging_level(logging.WARNING)

    # Update torch settings
    manual_seed = 42 + ddp_params._rank
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(manual_seed)

    torch.set_float32_matmul_precision("high")

    model = await Model.load(config=config, device=device)
    evaluate(model, is_ddp, ddp_params)

    if is_ddp:
        distributed.destroy_process_group()


async def _custom_trained(dir_path: str | None = None) -> None:
    await _load_and_evaluate(CustomTrainedModelConfig(dir_path=dir_path))


async def _pretrained() -> None:
    await _load_and_evaluate(PretrainedModelConfig())


async def _pretrained_gpt2(model_type: PretrainedGPT2ModelType) -> None:
    await _load_and_evaluate(PretrainedGPT2ModelConfig(model_type=model_type))


@app.command(name="custom-trained")
def custom_trained(dir_path: Annotated[str | None, typer.Option("--dir-path", "-d")] = None) -> None:
    _run_async(_custom_trained(dir_path))


@app.command(name="pretrained")
def pretrained() -> None:
    _run_async(_pretrained())


@app.command(name="pretrained-gpt-2")
def pretrained_gpt_2(model_type: Annotated[PretrainedGPT2ModelType, typer.Option("--model-type", "-m")]) -> None:
    _run_async(_pretrained_gpt2(model_type))


if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    app()
