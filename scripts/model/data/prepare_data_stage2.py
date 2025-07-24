"""
Prepare the data for stage 2 model training - smaller amounts of high quality data.

Config used for preparing the data for stage 2 training a model:

DATASET_CONFIGS: list[DatasetConfig] = [
    DatasetConfig(
        name="cosmopedia_wikihow",
        params=DatasetParams(
            path="HuggingFaceTB/cosmopedia",
            subset="wikihow",
            split="train",
        ),
        max_tokens_count=200_000_000,
    ),
    DatasetConfig(
        name="cosmopedia_openstax",
        params=DatasetParams(
            path="HuggingFaceTB/cosmopedia",
            subset="openstax",
            split="train",
        ),
        max_tokens_count=100_000_000,
    ),
    DatasetConfig(
        name="alpaca_cleaned",
        params=DatasetParams(
            path="yahma/alpaca-cleaned",
            split="train",
            field="output",
        ),
        max_tokens_count=10_000_000,
        disallowed_substrings=["```"],
    ),
]

Command to prepare the data and upload to S3:

```sh
uv run python -m scripts.model.data.prepare_data_stage2
```
"""

import asyncio
import os

from microgpt import DatasetParams
from microgpt.common.logger import _new_logger
from scripts.model.data.prepare_data_stage1 import DatasetConfig, save_tokenized_datasets

logger = _new_logger(__name__)

DATASET_CONFIGS: list[DatasetConfig] = [
    DatasetConfig(
        name="cosmopedia_wikihow",
        params=DatasetParams(
            path="HuggingFaceTB/cosmopedia",
            subset="wikihow",
            split="train",
        ),
        max_tokens_count=5_000_000,
    ),
    DatasetConfig(
        name="alpaca_cleaned",
        params=DatasetParams(
            path="yahma/alpaca-cleaned",
            split="train",
            field="output",
        ),
        max_tokens_count=1_000_000,
        disallowed_substrings=["```"],
    ),
]


async def main() -> None:
    """
    Save the tokenized datasets in shards of SHARD_SIZE tokens.
    """
    dirname = os.path.abspath(os.path.dirname(__file__))
    shards_dir_path = os.path.abspath(os.path.join(dirname, "data_stage2/shards"))
    await save_tokenized_datasets(DATASET_CONFIGS, shards_dir_path)


if __name__ == "__main__":
    asyncio.run(main())
