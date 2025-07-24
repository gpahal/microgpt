import os
from collections.abc import AsyncGenerator
from dataclasses import dataclass

import pytest
import pytest_asyncio
from tiktoken import get_encoding

from microgpt import (
    CustomTrainedTokenizerConfig,
    PretrainedGPTTokenizerConfig,
    PretrainedTokenizerConfig,
    TextDataSource,
    Tokenizer,
    TokenizerTrainer,
    TokenizerTrainerConfig,
    UntrainedTokenizerConfig,
)

FILE_TEXT_PREFIX = "FILE:"
TAYLOR_SWIFT_FILE_PATH = f"{FILE_TEXT_PREFIX}data/taylor_swift.txt"

SAMPLE_TEXTS = [
    "Hello, world!",
    "नमस्ते दुनिया!",
    "हिंदी में लिखा हुआ एक वाक्य",
    "こんにちは世界！",
    TAYLOR_SWIFT_FILE_PATH,
]

SAMPLE_SPECIAL_TOKENS_TEXTS = [
    "<|endoftext|>Hello, world!",
    "Hello, world!<|endoftext|>",
    "Hello<|endoftext|>, <|endoftext|>world!",
]


def unpack_text(text: str) -> str:
    if text.startswith(FILE_TEXT_PREFIX):
        file_path = text[len(FILE_TEXT_PREFIX) :]
        dirname = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(dirname, file_path)
        with open(file_path) as f:
            return f.read()
    else:
        return text


TRAINED_TOKENIZER_VOCAB_SIZE = 320


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def trained_tokenizer() -> AsyncGenerator[Tokenizer, None]:
    text = unpack_text(TAYLOR_SWIFT_FILE_PATH)
    tokenizer_trainer = await TokenizerTrainer.load(
        config=TokenizerTrainerConfig(
            tokenizer_config=UntrainedTokenizerConfig(),
            vocab_size=TRAINED_TOKENIZER_VOCAB_SIZE,
            data_sources=[TextDataSource(name="taylor_swift", text=text)],
        )
    )
    tokenizer = await tokenizer_trainer.run()
    yield tokenizer


@pytest.mark.asyncio(loop_scope="module")
async def test_vocab_size(trained_tokenizer: Tokenizer) -> None:
    assert trained_tokenizer.vocab_size == TRAINED_TOKENIZER_VOCAB_SIZE


@pytest.mark.asyncio(loop_scope="module")
async def test_save_and_load(trained_tokenizer: Tokenizer) -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp_dir_path:
        dir_path = os.path.join(tmp_dir_path, "test_tokenizer")
        await trained_tokenizer.save(dir_path)
        tokenizer = await Tokenizer.load(config=CustomTrainedTokenizerConfig(dir_path=dir_path))
        assert tokenizer.vocab_size == TRAINED_TOKENIZER_VOCAB_SIZE


@pytest.mark.asyncio(loop_scope="module")
@pytest.mark.parametrize("text", SAMPLE_TEXTS)
async def test_tokenizer(trained_tokenizer: Tokenizer, text: str) -> None:
    text = unpack_text(text)
    ids = trained_tokenizer.encode(text)
    decoded_text = trained_tokenizer.decode(ids)
    assert decoded_text == text


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def pretrained_tokenizer() -> AsyncGenerator[Tokenizer, None]:
    tokenizer = await Tokenizer.load(config=PretrainedTokenizerConfig())
    yield tokenizer


@pytest.mark.asyncio(loop_scope="module")
@pytest.mark.parametrize("text", SAMPLE_TEXTS)
async def test_pretrained_tokenizer(pretrained_tokenizer: Tokenizer, text: str) -> None:
    text = unpack_text(text)
    ids = pretrained_tokenizer.encode(text)
    decoded_text = pretrained_tokenizer.decode(ids)
    assert decoded_text == text


@pytest.mark.asyncio(loop_scope="module")
@pytest.mark.parametrize("text", SAMPLE_SPECIAL_TOKENS_TEXTS)
async def test_pretrained_tokenizer_special_tokens(pretrained_tokenizer: Tokenizer, text: str) -> None:
    text = unpack_text(text)
    ids = pretrained_tokenizer.encode(text, allowed_special_tokens="all")
    decoded_text = pretrained_tokenizer.decode(ids)
    assert decoded_text == text


@dataclass
class GPTEncodingMapping:
    tiktoken_encoding: str
    microgpt_gpt_encoding: str


GPT_ENCODING_MAPPINGS = [
    GPTEncodingMapping(tiktoken_encoding="gpt2", microgpt_gpt_encoding="gpt-2"),
    GPTEncodingMapping(tiktoken_encoding="r50k_base", microgpt_gpt_encoding="r50k_base"),
    GPTEncodingMapping(tiktoken_encoding="p50k_base", microgpt_gpt_encoding="p50k_base"),
    GPTEncodingMapping(tiktoken_encoding="p50k_edit", microgpt_gpt_encoding="p50k_edit"),
    GPTEncodingMapping(tiktoken_encoding="cl100k_base", microgpt_gpt_encoding="cl100k_base"),
    GPTEncodingMapping(tiktoken_encoding="o200k_base", microgpt_gpt_encoding="o200k_base"),
]


@pytest.mark.asyncio(loop_scope="module")
@pytest.mark.parametrize("mapping", GPT_ENCODING_MAPPINGS)
@pytest.mark.parametrize("text", SAMPLE_TEXTS)
async def test_gpt_tokenizer(mapping: GPTEncodingMapping, text: str) -> None:
    text = unpack_text(text)
    microgpt_gpt_tokenizer = await Tokenizer.load(
        config=PretrainedGPTTokenizerConfig(encoding_or_model_name=mapping.microgpt_gpt_encoding)
    )
    tiktoken_tokenizer = get_encoding(mapping.tiktoken_encoding)
    microgpt_ids = microgpt_gpt_tokenizer.encode(text)
    tiktoken_ids = tiktoken_tokenizer.encode(text)
    assert microgpt_ids == tiktoken_ids
    decoded_text = microgpt_gpt_tokenizer.decode(microgpt_ids)
    assert decoded_text == text


@pytest.mark.asyncio(loop_scope="module")
@pytest.mark.parametrize("mapping", GPT_ENCODING_MAPPINGS)
@pytest.mark.parametrize("text", SAMPLE_SPECIAL_TOKENS_TEXTS)
async def test_gpt_tokenizer_special_tokens(mapping: GPTEncodingMapping, text: str) -> None:
    text = unpack_text(text)
    microgpt_gpt_tokenizer = await Tokenizer.load(
        config=PretrainedGPTTokenizerConfig(encoding_or_model_name=mapping.microgpt_gpt_encoding)
    )
    tiktoken_tokenizer = get_encoding(mapping.tiktoken_encoding)
    microgpt_ids = microgpt_gpt_tokenizer.encode(text, allowed_special_tokens="all")
    tiktoken_ids = tiktoken_tokenizer.encode(text, allowed_special="all")
    assert microgpt_ids == tiktoken_ids
    decoded_text = microgpt_gpt_tokenizer.decode(microgpt_ids)
    assert decoded_text == text
