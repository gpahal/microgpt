import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import regex
from numpy.typing import NDArray
from pydantic import BaseModel

from microgpt.common.data_source import DataSource, _batch_read_data_source
from microgpt.common.logger import _new_logger
from microgpt.common.trainer import (
    TrainerCheckpointingConfig,
    _Trainer,
    _TrainerParams,
)

from .tokenizer import (
    CustomTrainedTokenizerConfig,
    GPTTokenizer,
    Tokenizer,
    TokenizerConfig,
)
from .tokenizer_utils import _get_counts_dict, _merge_ids_tensor


class TokenizerTrainerConfig(BaseModel):
    tokenizer_config: TokenizerConfig
    output_dir_path: str | None = None
    checkpointing_config: TrainerCheckpointingConfig | None = None
    vocab_size: int
    data_sources: DataSource | list[DataSource]


class _TokenizerTrainerParams(BaseModel):
    vocab_size: int
    data_sources: list[DataSource]


@dataclass
class _TokenizerTrainerRunContext:
    _ids: NDArray[np.int32]
    _counts_dict: dict[tuple[int, int], int]
    _starting_id: int


_TOKENIZER_TRAINER_NAME = "tokenizer"
_TOKENIZER_TRAINER_CREATE_KEY = object()


class TokenizerTrainer(_Trainer[Tokenizer, _TokenizerTrainerRunContext]):
    _tokenizer: Tokenizer
    _vocab_size: int
    _data_sources: list[DataSource]

    def __init__(
        self,
        create_key: object,
        logger: logging.Logger,
        abstract_params: _TrainerParams,
        params: _TokenizerTrainerParams,
        tokenizer: Tokenizer,
    ):
        """
        Initialize the tokenizer trainer. Do not call this constructor directly.
        Instead, use TokenizerTrainer.load.

        Args:
            create_key: A key to prevent instantiating the tokenizer trainer directly
            logger: The logger to use
            abstract_params: The abstract trainer params to use
            params: The tokenizer trainer params to use
            tokenizer: The tokenizer to train
        """
        if create_key != _TOKENIZER_TRAINER_CREATE_KEY:
            raise ValueError("TokenizerTrainer cannot be instantiated directly. Use TokenizerTrainer.load")
        if isinstance(tokenizer, GPTTokenizer):
            raise ValueError("GPTTokenizer cannot be trained")

        super().__init__(logger=logger, params=abstract_params)
        self._tokenizer = tokenizer
        self._init_vocab_size(params.vocab_size)
        self._data_sources = params.data_sources if isinstance(params.data_sources, list) else [params.data_sources]

    def _init_vocab_size(self, vocab_size: int) -> None:
        self._vocab_size = vocab_size
        if self._vocab_size < 256:
            raise ValueError(f"Vocab size {self._vocab_size} is less than the minimum vocab size of 256")
        if self._vocab_size < self._tokenizer.vocab_size:
            raise ValueError(
                f"Vocab size {self._vocab_size} is less than the tokenizer's vocab size {self._tokenizer.vocab_size}"
            )

    def __str__(self) -> str:
        return (
            "TokenizerTrainer(\n"
            f"  name={self._name}\n"
            f"  iteration={self._iteration}\n"
            f"  latest_run_iteration={self._latest_run_iteration}\n"
            f"  output_dir_path={self._output_dir_path}\n"
            f"  checkpointing_config={self._checkpointing_config}\n"
            f"  tokenizer={self._tokenizer}\n"
            f"  vocab_size={self._vocab_size}\n"
            f"  data_sources={self._data_sources}\n"
            ")"
        )

    @property
    def tokenizer(self) -> Tokenizer:
        return self._tokenizer

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def _get_params(self) -> BaseModel:
        return _TokenizerTrainerParams(
            vocab_size=self._vocab_size,
            data_sources=self._data_sources,
        )

    async def _save_output_data(self, dir_path: str) -> None:
        await self._tokenizer.save(dir_path)

    async def _run_setup(self) -> tuple[int | None, bool, _TokenizerTrainerRunContext]:
        # Calculate the number of merges needed to reach the desired vocab size
        num_merges = self._vocab_size - self._tokenizer.vocab_size
        max_iterations = num_merges + (self._iteration - 1)

        # Some merges already exist, so we need to merge the inputs before training again
        existing_merges_count = len(self._tokenizer._merges_dict)
        inputs_need_merging = existing_merges_count > 0

        self._logger.info(f"Training tokenizer: existing_merges_count={existing_merges_count} num_merges={num_merges}")

        self._logger.info(
            f"Processing data sources: inputs_need_merging={inputs_need_merging} data_sources={self._data_sources}"
        )
        ids: list[int] = []
        data_source_tokens_count_dict: dict[str, int] = {}
        for data_source in self._data_sources:
            data_source_tokens = 0
            async for text in _batch_read_data_source(data_source, shuffle_seed=42):
                # Split the text up into text chunks
                text_chunks: list[str] = regex.findall(self._tokenizer._compiled_split_pattern, text)
                # Input text preprocessing
                if inputs_need_merging:
                    for ch in text_chunks:
                        new_ids = self._tokenizer._encode_chunk(ch.encode("utf-8"))
                        ids.extend(new_ids)
                        ids.append(-1)
                        data_source_tokens += len(new_ids)
                else:
                    for ch in text_chunks:
                        new_ids = list(ch.encode("utf-8"))
                        ids.extend(new_ids)
                        ids.append(-1)
                        data_source_tokens += len(new_ids)

            data_source_tokens_count_dict[data_source.name] = data_source_tokens

        # Iteratively merge the most common pairs to create new tokens
        counts_dict = _get_counts_dict(ids)
        ids_array = np.array(ids, dtype=np.int32)
        starting_id = 256 + existing_merges_count
        self._logger.info(
            f"Processed data sources: "
            f"num_merges={num_merges} "
            f"max_iterations={max_iterations} "
            f"data_source_tokens={data_source_tokens_count_dict} "
            f"tokens={len(ids)} "
            f"counts_dict={len(counts_dict)} "
            f"starting_id={starting_id}"
        )
        return (
            max_iterations,
            True,
            _TokenizerTrainerRunContext(_ids=ids_array, _counts_dict=counts_dict, _starting_id=starting_id),
        )

    async def _run_teardown(self, run_context: _TokenizerTrainerRunContext) -> None:
        pass

    async def _run_iteration(self, run_context: _TokenizerTrainerRunContext) -> bool:
        if len(run_context._counts_dict) == 0:
            # Nothing else can be merged anymore
            return False

        # Find the pair with the highest count
        pair = max(run_context._counts_dict, key=lambda p: run_context._counts_dict[p])
        if run_context._counts_dict[pair] == 0:
            # Nothing else can be merged anymore
            return False

        # Mint a new token: assign it the next available id
        new_id = run_context._starting_id + (self._latest_run_iteration - 1)
        # Replace all occurrences of pair in ids with new_id
        run_context._ids = _merge_ids_tensor(
            ids=run_context._ids,
            pair=pair,
            new_id=new_id,
            counts_dict=run_context._counts_dict,
        )

        # Save the merge
        self._tokenizer._add_merge(pair, new_id)
        return True

    def _get_return_value(self, run_context: _TokenizerTrainerRunContext) -> Tokenizer:
        return self._tokenizer

    @classmethod
    async def load(cls, config: TokenizerTrainerConfig) -> "TokenizerTrainer":
        """
        Load a tokenizer trainer.

        Args:
            config: The tokenizer trainer config to use
            logger: The logger to use

        Returns:
            A tokenizer trainer
        """
        logger = _new_logger(__name__)
        logger.info(f"Loading tokenizer trainer: config={config}")

        abstract_params = _TrainerParams(
            name=_TOKENIZER_TRAINER_NAME,
            output_dir_path=config.output_dir_path,
            checkpointing_config=config.checkpointing_config,
        )
        params = _TokenizerTrainerParams(
            vocab_size=config.vocab_size,
            data_sources=config.data_sources,
        )

        async def _create_non_checkpointed_tokenizer_trainer() -> TokenizerTrainer:
            tokenizer = await Tokenizer.load(config=config.tokenizer_config)
            tokenizer_trainer = TokenizerTrainer(
                create_key=_TOKENIZER_TRAINER_CREATE_KEY,
                logger=logger,
                abstract_params=abstract_params,
                params=params,
                tokenizer=tokenizer,
            )
            logger.info(f"Loaded non-checkpointed tokenizer trainer: tokenizer_trainer={tokenizer_trainer}")
            return tokenizer_trainer

        if config.checkpointing_config is None:
            return await _create_non_checkpointed_tokenizer_trainer()

        if config.output_dir_path is None:
            raise ValueError("Output directory path is required for checkpointed tokenizer trainer")

        async def _create_trainer(
            latest_checkpoint_dir_path: str,
            abstract_params: _TrainerParams,
            params_json: Any,
        ) -> "TokenizerTrainer":
            tokenizer = await Tokenizer.load(
                config=CustomTrainedTokenizerConfig(
                    dir_path=latest_checkpoint_dir_path,
                ),
            )
            try:
                await tokenizer._validate_matches_config(config.tokenizer_config)
            except ValueError as e:
                raise ValueError(
                    "Latest checkpointed tokenizer does not match config: "
                    f"checkpointed_tokenizer={tokenizer}, "
                    f"tokenizer_config={config.tokenizer_config}, "
                    f"error={e}"
                ) from e

            loaded_params = _TokenizerTrainerParams(**params_json)
            if loaded_params.vocab_size > params.vocab_size:
                raise ValueError(
                    "Latest checkpointed tokenizer trainer vocab size is greater than expected vocab size: "
                    f"expected_vocab_size={params.vocab_size}, "
                    f"checkpointed_vocab_size={loaded_params.vocab_size}"
                )

            return TokenizerTrainer(
                create_key=_TOKENIZER_TRAINER_CREATE_KEY,
                logger=logger,
                abstract_params=abstract_params,
                params=params,
                tokenizer=tokenizer,
            )

        tokenizer_trainer = await _Trainer._load_checkpointed_internal(
            abstract_params,
            _create_trainer,
            logger,
        )
        if tokenizer_trainer is None:
            return await _create_non_checkpointed_tokenizer_trainer()

        logger.info(f"Loaded tokenizer trainer: tokenizer_trainer={tokenizer_trainer}")
        return tokenizer_trainer
