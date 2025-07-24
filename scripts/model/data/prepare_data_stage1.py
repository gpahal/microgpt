"""
Prepare the data for stage 1 model training - large amounts of mostly web based data.

Config used for preparing the data for stage 1 training a model:

DATASET_CONFIGS: list[DatasetConfig] = [
    DatasetConfig(
        name="fineweb-edu",
        params=DatasetParams(
            path="HuggingFaceFW/fineweb-edu",
            subset="sample-10BT",
            split="train",
        ),
        max_tokens_count=10_000_000_000,
    ),
]

Commands to prepare the data and upload to S3:

```sh
uv run python -m scripts.model.data.prepare_data_stage1
```
"""

import asyncio
import multiprocessing as mp
import os
import shutil
from dataclasses import dataclass
from typing import Any

import numpy as np
from datasets import Dataset, load_dataset  # type: ignore
from numpy.typing import NDArray
from tqdm import tqdm

from microgpt import DatasetParams, PretrainedTokenizerConfig, Tokenizer
from microgpt.common.logger import _new_logger
from microgpt.model.data_loader import _SHARD_FILE_PREFIX, _SHARD_FILE_SUFFIX, _SHARD_SIZE

logger = _new_logger(__name__)


@dataclass
class DatasetConfig:
    name: str
    params: DatasetParams
    max_tokens_count: int
    disallowed_substrings: list[str] | None = None


DATASET_CONFIGS: list[DatasetConfig] = [
    DatasetConfig(
        name="fineweb-edu",
        params=DatasetParams(
            path="HuggingFaceFW/fineweb-edu",
            subset="sample-10BT",
            split="train",
        ),
        max_tokens_count=10_000_000,
    ),
]


CPU_COUNT = os.cpu_count()
N_PROCS = max(1, CPU_COUNT // 2, CPU_COUNT - 4) if CPU_COUNT is not None else 1


def tokenize(tokenizer: Tokenizer, text: str) -> NDArray[np.uint16]:
    # Only supports 2**16 = 65536 vocab size. Pretrained tokenizer has 50257 - same as GPT-2.
    tokens = [tokenizer.eot_id] if tokenizer.eot_id is not None else []
    tokens.extend(tokenizer.encode_ordinary(text))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "Token dictionary too large for uint16"
    return tokens_np.astype(np.uint16)


class TokenizeWrapper:
    _tokenizer: Tokenizer
    _dataset_config: DatasetConfig

    def __init__(self, tokenizer: Tokenizer, dataset_config: DatasetConfig) -> None:
        self._tokenizer = tokenizer
        self._dataset_config = dataset_config

    def __call__(self, row: dict[str, Any]) -> NDArray[np.uint16] | None:
        text = row[self._dataset_config.params.field]
        if len(text) <= 16:
            return None
        if self._dataset_config.disallowed_substrings is not None:
            for substring in self._dataset_config.disallowed_substrings:
                if substring in text:
                    return None
        return tokenize(self._tokenizer, text)


async def save_tokenized_datasets_split(
    dataset_configs: list[DatasetConfig],
    split: str,
    ratio: float,
    shards_dir_path: str,
    tokenizer: Tokenizer,
    datasets: list[Dataset],
) -> None:
    """
    Save the tokenized datasets split in shards of SHARD_SIZE tokens.
    """
    logger.info(f"Saving tokenized dataset split: split={split} shards_dir_path={shards_dir_path}")

    with mp.Pool(N_PROCS) as pool:
        # Preallocate buffer for current shard
        all_tokens_np = np.empty((_SHARD_SIZE,), dtype=np.uint16)
        token_count = 0
        shard_index = 0

        for dataset_config, dataset in zip(dataset_configs, datasets, strict=False):
            logger.info(f"Processing dataset {dataset_config.name}")
            tokenize_wrapper = TokenizeWrapper(tokenizer, dataset_config)
            tokens_count_remaining = int(float(dataset_config.max_tokens_count) * ratio)
            data_source_progress_bar = tqdm(
                total=tokens_count_remaining, unit="tokens", desc=f"Dataset {dataset_config.name}"
            )

            for tokens in pool.imap(tokenize_wrapper, dataset, chunksize=32):
                if tokens is None or len(tokens) == 0:
                    continue

                if tokens_count_remaining <= 0:
                    data_source_progress_bar.close()
                    break

                if tokens_count_remaining < len(tokens):
                    tokens = tokens[:tokens_count_remaining]
                    tokens_count_remaining = 0
                else:
                    tokens_count_remaining -= len(tokens)

                data_source_progress_bar.update(len(tokens))

                # Check if we need to start a new shard
                while token_count + len(tokens) >= _SHARD_SIZE:
                    # Save current shard
                    shard_path = os.path.join(
                        shards_dir_path, f"{_SHARD_FILE_PREFIX}{shard_index:06d}{_SHARD_FILE_SUFFIX}"
                    )
                    remainder = _SHARD_SIZE - token_count
                    all_tokens_np[token_count : token_count + remainder] = tokens[:remainder]
                    np.save(shard_path, all_tokens_np)
                    logger.info(f"Saved shard: shard_index={shard_index} size={_SHARD_SIZE}")

                    tokens = tokens[remainder:]

                    all_tokens_np = np.empty((_SHARD_SIZE,), dtype=np.uint16)
                    token_count = 0
                    shard_index += 1

                all_tokens_np[token_count : token_count + len(tokens)] = tokens
                token_count += len(tokens)

            logger.info(
                f"Processed dataset {dataset_config.name}: "
                f"tokens_count={dataset_config.max_tokens_count - tokens_count_remaining}"
            )

        # Save final shard if it has any tokens
        if token_count > 0:
            shard_path = os.path.join(shards_dir_path, f"{_SHARD_FILE_PREFIX}{shard_index:06d}{_SHARD_FILE_SUFFIX}")
            remainder = _SHARD_SIZE - token_count
            np.save(shard_path, all_tokens_np[:token_count])
            logger.info(f"Saved shard: shard_index={shard_index} size={token_count}")

        logger.info("Saved tokenized data sources")


async def save_tokenized_datasets(dataset_configs: list[DatasetConfig], shards_dir_path: str) -> None:
    """
    Save the tokenized datasets in shards of SHARD_SIZE tokens.
    """
    logger.info(f"Saving tokenized datasets: shards_dir_path={shards_dir_path} cpu_count={CPU_COUNT} n_procs={N_PROCS}")

    logger.info("Loading datasets")
    train_datasets: list[Dataset] = []
    val_datasets: list[Dataset] = []
    for dataset_config in dataset_configs:
        logger.info(f"Loading dataset {dataset_config.name}")
        dataset: Dataset = load_dataset(
            dataset_config.params.path,
            dataset_config.params.subset,
            split=dataset_config.params.split,
            num_proc=min(N_PROCS, 2),
        )
        split_datasets = dataset.train_test_split(test_size=0.01, seed=42, shuffle=True)
        train_datasets.append(split_datasets["train"])
        val_datasets.append(split_datasets["test"])
        logger.info(f"Loaded dataset {dataset_config.name}")
    logger.info("Loaded datasets")

    if os.path.exists(shards_dir_path):
        shutil.rmtree(shards_dir_path)

    os.makedirs(shards_dir_path, exist_ok=True)
    train_dir_path = os.path.join(shards_dir_path, "train")
    val_dir_path = os.path.join(shards_dir_path, "val")
    os.makedirs(train_dir_path, exist_ok=True)
    os.makedirs(val_dir_path, exist_ok=True)

    logger.info("Loading tokenizer")
    tokenizer = await Tokenizer.load(config=PretrainedTokenizerConfig())
    logger.info(f"Loaded tokenizer: tokenizer={tokenizer}")

    await save_tokenized_datasets_split(dataset_configs, "train", 0.99, train_dir_path, tokenizer, train_datasets)
    await save_tokenized_datasets_split(dataset_configs, "val", 0.01, val_dir_path, tokenizer, val_datasets)
    logger.info("Saved tokenized datasets")


async def main() -> None:
    """
    Save the tokenized datasets in shards of SHARD_SIZE tokens.
    """
    dirname = os.path.abspath(os.path.dirname(__file__))
    shards_dir_path = os.path.abspath(os.path.join(dirname, "data_stage1/shards"))
    await save_tokenized_datasets(DATASET_CONFIGS, shards_dir_path)


if __name__ == "__main__":
    asyncio.run(main())
