"""
Train a model.

To prepare the training data, see `scripts/model/data/prepare_data_stage1.py`.

Command to stage 1 train the pretrained model for 5 epochs:

```sh
uv run python -m scripts.model.train_model_stage1 \
    --epochs 5 \
    --max-iterations-per-epoch 18170 \
    --batch-size 64 \
    --gradient-accumulation-iterations 8 \
    --enable-hellaswag-eval \
    --max-seq-len 1024 \
    --d-model 768 \
    --n-layers 12 \
    --n-heads 12
```

For multi-GPU training, use the following command:

```sh
uv run torchrun --standalone --nproc_per_node=8 scripts/model/train_model_stage1.py \
    --epochs 5 \
    --max-iterations-per-epoch 18170 \
    --batch-size 64 \
    --gradient-accumulation-iterations 8 \
    --enable-hellaswag-eval \
    --max-seq-len 1024 \
    --d-model 768 \
    --n-layers 12 \
    --n-heads 12
```

To run in the background and detached from the terminal, wrap it like this: `nohup {COMMAND} &`
The command above will run indefinitely, even if the terminal is closed.
To check the logs, see `nohup.out`. You can live-tail the logs by running `tail -f nohup.out`.
"""

import asyncio
import os
from collections.abc import Coroutine
from typing import Annotated, Any

import typer

from microgpt import (
    ModelTrainer,
    ModelTrainerConfig,
    PretrainedTokenizerConfig,
    TrainerCheckpointingConfig,
    UntrainedModelConfig,
)
from microgpt.common.ddp import _get_ddp_params
from microgpt.common.logger import _new_logger

logger = _new_logger(__name__)

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)


def _run_async[RT](coro: Coroutine[Any, Any, RT]) -> RT:
    return asyncio.get_event_loop().run_until_complete(coro)


async def train(
    checkpointing_interval: int | None,
    epochs: int,
    max_iterations_per_epoch: int,
    batch_size: int,
    gradient_accumulation_iterations: int,
    enable_hellaswag_eval: bool,
    max_seq_len: int,
    d_model: int,
    n_layers: int,
    n_heads: int,
    use_rope: bool,
    init_std: float | None,
) -> None:
    dirname = os.path.dirname(os.path.abspath(__file__))
    trained_model_dir_path = os.path.join(dirname, "trained_model_stage1")
    ddp_params = _get_ddp_params()
    if ddp_params is None or ddp_params._local_rank == 0:
        if os.path.exists(os.path.join(trained_model_dir_path, "output/model.json")):
            logger.error("Trained model output already exists, skipping training")
            return

        os.makedirs(trained_model_dir_path, exist_ok=True)
        os.makedirs(os.path.join(trained_model_dir_path, "output"), exist_ok=True)

    if checkpointing_interval is None:
        checkpointing_interval = min(10000, max(10, max_iterations_per_epoch // 25))
        logger.info(f"Checkpointing iterations interval: {checkpointing_interval}")

    model_config = UntrainedModelConfig(
        tokenizer_config=PretrainedTokenizerConfig(),
        max_seq_len=max_seq_len,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        use_rope=use_rope,
        init_std=init_std,
    )
    model_trainer = await ModelTrainer.load(
        config=ModelTrainerConfig(
            model_konfig=model_config,
            output_dir_path=os.path.join(trained_model_dir_path, "output"),
            checkpointing_config=TrainerCheckpointingConfig(
                checkpointing_interval=checkpointing_interval,
                checkpoint_dir_path=os.path.join(trained_model_dir_path, "checkpoints"),
            ),
            manual_seed=42,
            epochs=epochs,
            max_iterations_per_epoch=max_iterations_per_epoch,
            batch_size=batch_size,
            gradient_accumulation_iterations=gradient_accumulation_iterations,
            enable_hellaswag_eval=enable_hellaswag_eval,
            loss_output_file_path=os.path.join(trained_model_dir_path, "output", "loss.txt"),
            eval_output_file_path=os.path.join(trained_model_dir_path, "output", "eval.txt"),
        ),
    )

    model = await model_trainer.run()
    text = model.generate_text("Hi, I'm a language model,", max_new_tokens=32)
    print(text)


@app.command()
def main(
    checkpointing_interval: Annotated[
        int | None,
        typer.Option(
            "--checkpointing-interval",
            help="The number of iterations between checkpoints. If not provided, it is set to "
            "min(10000, max(10, max_iterations // 100))",
            min=1,
        ),
    ] = None,
    epochs: Annotated[
        int,
        typer.Option(
            "--epochs",
            "-e",
            help="The number of epochs to train the model",
        ),
    ] = 1,
    max_iterations_per_epoch: Annotated[
        int,
        typer.Option(
            "--max-iterations-per-epoch",
            "-m",
            help="The maximum number of iterations to train the model",
        ),
    ] = 200,
    batch_size: Annotated[
        int,
        typer.Option(
            "--batch-size",
            "-b",
            help="The batch size of the model",
        ),
    ] = 8,
    gradient_accumulation_iterations: Annotated[
        int,
        typer.Option(
            "--gradient-accumulation-iterations",
            "-g",
            help="The number of gradient accumulation iterations. The model effectively trains on a batch size of "
            "batch_size * gradient_accumulation_iterations, with loss accumulated over the "
            "gradient_accumulation_iterations.",
        ),
    ] = 2,
    enable_hellaswag_eval: Annotated[
        bool,
        typer.Option(
            "--enable-hellaswag-eval",
            help="Whether to enable HellaSwag eval while training",
        ),
    ] = False,
    max_seq_len: Annotated[
        int,
        typer.Option(
            "--max-seq-len",
            "-l",
            help="The maximum sequence length of the model",
        ),
    ] = 256,
    d_model: Annotated[
        int,
        typer.Option(
            "--d-model",
            "-d",
            help="The number of embeddings of the model",
        ),
    ] = 192,
    n_layers: Annotated[
        int,
        typer.Option(
            "--n-layers",
            "-l",
            help="The number of layers of the model",
        ),
    ] = 3,
    n_heads: Annotated[
        int,
        typer.Option(
            "--n-heads",
            "-h",
            help="The number of heads of the model",
        ),
    ] = 3,
    use_rope: Annotated[
        bool,
        typer.Option(
            "--use-rope",
            help="Whether to use the RoPE positional encoding",
        ),
    ] = True,
    init_std: Annotated[
        float | None,
        typer.Option(
            "--init-std",
            help="The standard deviation of the initialization of the model",
        ),
    ] = 0.02,
) -> None:
    """
    Train a model. Save the model to the scripts/model/trained_model directory.
    """
    _run_async(
        train(
            checkpointing_interval=checkpointing_interval,
            epochs=epochs,
            max_iterations_per_epoch=max_iterations_per_epoch,
            batch_size=batch_size,
            gradient_accumulation_iterations=gradient_accumulation_iterations,
            enable_hellaswag_eval=enable_hellaswag_eval,
            max_seq_len=max_seq_len,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            use_rope=use_rope,
            init_std=init_std,
        )
    )


if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    app()
