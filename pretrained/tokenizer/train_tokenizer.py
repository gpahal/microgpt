import asyncio
import os
from collections.abc import Coroutine
from typing import Annotated, Any

import typer

from microgpt import DatasetParams, DatasetTextSource, UntrainedTokenizerConfig, load_tokenizer
from microgpt.logger import _new_logger

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)

logger = _new_logger(__name__)


def _run_async[RT](coro: Coroutine[Any, Any, RT]) -> RT:
    return asyncio.get_event_loop().run_until_complete(coro)


async def train(
    vocab_size: int,
    include_eot: bool = True,
    eot_id: int | None = None,
):
    if include_eot and eot_id is None:
        eot_id = vocab_size - 1
        logger.info(f"EOT id not provided, using (vocab size - 1) = {eot_id}")

    tokenizer = await load_tokenizer(
        config=UntrainedTokenizerConfig(
            special_tokens={"<|endoftext|>": eot_id} if include_eot else {},
            eot_id=eot_id if include_eot else None,
        ),
    )
    if vocab_size == 257 and include_eot:
        logger.info("Vocabulary size is 257 and end-of-text token is included, skipping training")
        return

    await tokenizer.train(
        text_sources=[
            DatasetTextSource(
                name="wikipedia",
                params=DatasetParams(
                    path="wikimedia/wikipedia",
                    subset="20231101.en",
                    split="train",
                ),
                batch_size=1000,
                max_length=75000,
            ),
            DatasetTextSource(
                name="fineweb",
                params=DatasetParams(
                    path="HuggingFaceFW/fineweb",
                    subset="sample-10BT",
                    split="train",
                ),
                batch_size=1000,
                max_length=125000,
            ),
            DatasetTextSource(
                name="yahoo_answers_topics",
                params=DatasetParams(
                    path="community-datasets/yahoo_answers_topics",
                    split="train",
                    field="best_answer",
                ),
                batch_size=5000,
                max_length=300000,
            ),
            DatasetTextSource(
                name="amazon_polarity",
                params=DatasetParams(
                    path="fancyzhx/amazon_polarity",
                    split="train",
                    field="content",
                ),
                batch_size=5000,
                max_length=300000,
            ),
            DatasetTextSource(
                name="imdb",
                params=DatasetParams(
                    path="stanfordnlp/imdb",
                    split="train",
                ),
                batch_size=5000,
                max_length=300000,
            ),
            DatasetTextSource(
                name="awesome-chatgpt-prompts",
                params=DatasetParams(
                    path="fka/awesome-chatgpt-prompts",
                    split="train",
                    field="prompt",
                ),
                batch_size=1000,
                max_length=1000,
            ),
            DatasetTextSource(
                name="python-codes-25k",
                params=DatasetParams(
                    path="flytech/python-codes-25k",
                    split="train",
                    field="output",
                ),
                batch_size=1000,
                max_length=50000,
            ),
        ],
        vocab_size=vocab_size,
    )
    ids = tokenizer.encode("Hello, world!")
    print(f"encode('Hello, world!') = {[tokenizer._vocab[idx] for idx in ids]}")
    decoded_text = tokenizer.decode(ids)
    print(f"decode(encode('Hello, world!')) = {decoded_text}")
    dirname = os.path.dirname(os.path.abspath(__file__))
    file_path_prefix = os.path.join(dirname, "data_new/pretrained")
    await tokenizer.save(file_path_prefix)


@app.command()
def main(
    vocab_size: Annotated[
        int,
        typer.Option(
            "--vocab-size",
            "-s",
            help="The size of the vocabulary. This includes 256 base tokens for each byte and optionally the end-of-text token",
            min=257,
        ),
    ],
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
            help="The id of the end-of-text token. If include_eot is False, this option is ignored. If include_eot is True and this option is not provided, the (vocab size - 1) is used as the end-of-text token id.",
        ),
    ] = None,
) -> None:
    """
    Train a tokenizer. Save the tokenizer to the pretrained/tokenizer/data_new directory.
    """
    _run_async(train(vocab_size=vocab_size, include_eot=include_eot, eot_id=eot_id))


if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    app()
