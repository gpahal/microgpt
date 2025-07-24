"""
Train a tokenizer.

Config used for training the pretrained tokenizer:

DATA_SOURCES = [
    DatasetDataSource(
        name="fineweb-edu",
        params=DatasetParams(
            path="HuggingFaceFW/fineweb-edu",
            subset="sample-10BT",
            split="train",
        ),
        batch_size=1000,
        max_length=125000,  # ~588.79MB
    ),
]

Command used for training the pretrained tokenizer:

```sh
uv run python -m scripts.tokenizer.train_tokenizer --vocab-size 50257
```
"""

import asyncio
import os
from collections.abc import Coroutine
from typing import Annotated, Any

import typer

from microgpt import (
    DatasetDataSource,
    DatasetParams,
    TokenizerTrainer,
    TokenizerTrainerConfig,
    TrainerCheckpointingConfig,
    UntrainedTokenizerConfig,
)
from microgpt.common.logger import _new_logger

logger = _new_logger(__name__)

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)


def _run_async[RT](coro: Coroutine[Any, Any, RT]) -> RT:
    return asyncio.get_event_loop().run_until_complete(coro)


DATA_SOURCES = [
    DatasetDataSource(
        name="fineweb-edu",
        params=DatasetParams(
            path="HuggingFaceFW/fineweb-edu",
            subset="sample-10BT",
            split="train",
        ),
        batch_size=1000,
        max_length=5000,
    ),
]


async def train(
    checkpointing_interval: int | None,
    vocab_size: int,
    include_eot: bool,
    eot_id: int | None,
) -> None:
    dirname = os.path.dirname(os.path.abspath(__file__))
    trained_tokenizer_dir_path = os.path.join(dirname, "trained_tokenizer")
    if os.path.exists(os.path.join(trained_tokenizer_dir_path, "output/tokenizer.json")):
        logger.error("Trained tokenizer output already exists, skipping training")
        return

    if vocab_size == 257 and include_eot:
        logger.error("Vocabulary size is 257 and end-of-text token is included, skipping training")
        return
    if include_eot and eot_id is None:
        eot_id = vocab_size - 1
        logger.info(f"EOT id not provided, using (vocab size - 1) = {eot_id}")

    tokenizer_config = UntrainedTokenizerConfig(
        special_tokens={"<|endoftext|>": eot_id} if include_eot else {},
        eot_id=eot_id if include_eot else None,
    )

    if checkpointing_interval is None:
        checkpointing_interval = min(1000, max(10, vocab_size // 50))
        logger.info(f"Checkpointing iterations interval: {checkpointing_interval}")

    tokenizer_trainer = await TokenizerTrainer.load(
        config=TokenizerTrainerConfig(
            tokenizer_config=tokenizer_config,
            output_dir_path=os.path.join(trained_tokenizer_dir_path, "output"),
            checkpointing_config=TrainerCheckpointingConfig(
                checkpointing_interval=checkpointing_interval,
                checkpoint_dir_path=os.path.join(trained_tokenizer_dir_path, "checkpoints"),
            ),
            vocab_size=vocab_size,
            data_sources=DATA_SOURCES,
        ),
    )

    tokenizer = await tokenizer_trainer.run()
    ids = tokenizer.encode("Hello, world!")
    print(f"encode('Hello, world!') = {[tokenizer._vocab[id] for id in ids]}")
    decoded_text = tokenizer.decode(ids)
    print(f"decode(encode('Hello, world!')) = {decoded_text}")


@app.command()
def main(
    checkpointing_interval: Annotated[
        int | None,
        typer.Option(
            "--checkpointing-interval",
            help="The number of iterations between checkpoints. If not provided, it is set to "
            "min(1000, max(10, vocab_size // 50))",
            min=1,
        ),
    ] = None,
    vocab_size: Annotated[
        int,
        typer.Option(
            "--vocab-size",
            "-s",
            help="The size of the vocabulary. This includes 256 base tokens for each byte and optionally the "
            "end-of-text token",
            min=257,
        ),
    ] = 1257,
    include_eot: Annotated[
        bool,
        typer.Option(
            "--include-eot",
            help="Whether to include the end-of-text token",
        ),
    ] = True,
    eot_id: Annotated[
        int | None,
        typer.Option(
            "--eot-id",
            help="The id of the end-of-text token. If include_eot is False, this option is ignored. If include_eot is "
            "True and this option is not provided, the (vocab size - 1) is used as the end-of-text token id",
        ),
    ] = None,
) -> None:
    """
    Train a tokenizer. Save the tokenizer to the scripts/tokenizer/trained_tokenizer directory.
    """
    _run_async(
        train(
            checkpointing_interval=checkpointing_interval,
            vocab_size=vocab_size,
            include_eot=include_eot,
            eot_id=eot_id,
        )
    )


if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    app()
