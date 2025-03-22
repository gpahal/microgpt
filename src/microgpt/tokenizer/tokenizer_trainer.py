import logging
from dataclasses import dataclass
from typing import Annotated, Any, Literal

import numpy as np
import regex
from pydantic import BaseModel, Field

from microgpt.common.data_source import DataSource, _batch_read_data_source
from microgpt.common.logger import _new_logger
from microgpt.common.trainer import (
    TrainerCheckpointingConfig,
    _AbstractTrainer,
    _AbstractTrainerParams,
)

from .tokenizer import GPTTokenizer, PretrainedTokenizerConfig, Tokenizer, TokenizerConfig
from .tokenizer_utils import _get_counts_dict, _merge_ids


class NonCheckpointedTokenizerTrainerConfig(BaseModel):
    type: Literal["non_checkpointed"] = "non_checkpointed"
    tokenizer_config: TokenizerConfig
    vocab_size: int
    data_sources: DataSource | list[DataSource]
    output_dir_path: str


class CheckpointedTokenizerTrainerConfig(BaseModel):
    type: Literal["checkpointed"] = "checkpointed"
    tokenizer_config: TokenizerConfig
    vocab_size: int
    data_sources: DataSource | list[DataSource]
    output_dir_path: str
    checkpointing_config: TrainerCheckpointingConfig


TokenizerTrainerConfig = Annotated[
    NonCheckpointedTokenizerTrainerConfig | CheckpointedTokenizerTrainerConfig,
    Field(discriminator="type"),
]


class _TokenizerTrainerParams(BaseModel):
    vocab_size: int
    data_sources: list[DataSource]


@dataclass
class TokenizerTrainerRunContext:
    ids: list[int]
    counts_dict: dict[tuple[int, int], int]
    starting_idx: int


_TOKENIZER_TRAINER_CREATE_KEY = object()


class TokenizerTrainer(_AbstractTrainer[Tokenizer, TokenizerTrainerRunContext]):
    _tokenizer: Tokenizer
    _vocab_size: int
    _data_sources: list[DataSource]

    def __init__(
        self,
        create_key: object,
        logger: logging.Logger,
        abstract_params: _AbstractTrainerParams,
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
            raise ValueError(
                "TokenizerTrainer cannot be instantiated directly. Use TokenizerTrainer.load"
            )
        if isinstance(tokenizer, GPTTokenizer):
            raise ValueError("GPTTokenizer cannot be trained")

        super().__init__(logger=logger, params=abstract_params)
        self._tokenizer = tokenizer
        self._init_vocab_size(params.vocab_size)
        self._data_sources = (
            params.data_sources if isinstance(params.data_sources, list) else [params.data_sources]
        )

    @property
    def tokenizer(self) -> Tokenizer:
        return self._tokenizer

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def _init_vocab_size(self, vocab_size: int) -> None:
        self._vocab_size = vocab_size
        if self._vocab_size < 256:
            raise ValueError(
                f"Vocab size {self._vocab_size} is less than the minimum vocab size of 256"
            )
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

    def _get_params(self) -> BaseModel:
        return _TokenizerTrainerParams(
            vocab_size=self._vocab_size,
            data_sources=self._data_sources,
        )

    async def _save_output_data(self, dir_path: str) -> None:
        await self._tokenizer.save(dir_path)

    async def _save_checkpoint_data(self, dir_path: str) -> None:
        await self._tokenizer.save(dir_path)

    async def _run_setup(self) -> tuple[int | None, TokenizerTrainerRunContext]:
        # Calculate the number of merges needed
        num_merges = self._vocab_size - self._tokenizer.vocab_size

        # Some merges already exist, so we need to merge the inputs before training again
        existing_merges_count = len(self._tokenizer._merges)
        inputs_need_merging = existing_merges_count > 0

        self._logger.info(
            f"Training tokenizer: existing_merges_count={existing_merges_count} num_merges={num_merges}"
        )

        self._logger.info(
            f"Processing data sources: inputs_need_merging={inputs_need_merging} data_sources={self._data_sources}"
        )
        ids: list[int] = []
        data_source_tokens_dict: dict[str, list[int]] = {}
        for data_source in self._data_sources:
            data_source_tokens = 0
            async for text in _batch_read_data_source(data_source):
                # Split the text up into text chunks
                text_chunks: list[str] = regex.findall(
                    self._tokenizer._compiled_split_pattern, text
                )
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

            data_source_tokens_dict[data_source.name] = data_source_tokens

        self._logger.info(
            f"Processed data sources: num_merges={num_merges} tokens={len(ids)} data_source_tokens={data_source_tokens_dict}"
        )

        # Iteratively merge the most common pairs to create new tokens
        counts_dict = _get_counts_dict(ids)
        ids = np.array(ids, dtype=np.int32)
        starting_idx = 256 + existing_merges_count
        return (
            min(num_merges, len(counts_dict)),
            TokenizerTrainerRunContext(ids=ids, counts_dict=counts_dict, starting_idx=starting_idx),
        )

    async def _run_teardown(self, run_context: TokenizerTrainerRunContext) -> None:
        pass

    async def _run_iteration(self, run_context: TokenizerTrainerRunContext) -> bool:
        if len(run_context.counts_dict) == 0:
            # Nothing else can be merged anymore
            return False

        # Find the pair with the highest count
        pair = max(run_context.counts_dict, key=lambda p: run_context.counts_dict.get(p, 0))
        if run_context.counts_dict.get(pair, 0) == 0:
            # Nothing else can be merged anymore
            return False

        # Mint a new token: assign it the next available id
        new_idx = run_context.starting_idx + self._latest_run_iteration
        # Replace all occurrences of pair in ids with idx
        run_context.ids = _merge_ids(
            ids=run_context.ids,
            pair=pair,
            new_id=new_idx,
            counts_dict=run_context.counts_dict,
        )

        # Save the merge
        self._tokenizer._merges[pair] = new_idx
        self._tokenizer._vocab[new_idx] = (
            self._tokenizer._vocab[pair[0]] + self._tokenizer._vocab[pair[1]]
        )
        self._tokenizer._vocab_size = len(self._tokenizer._vocab)
        return True

    async def _get_return_value(self, run_context: TokenizerTrainerRunContext) -> Tokenizer:
        """
        Get the return value of the trainer.
        """
        return self._tokenizer

    @classmethod
    async def _load_non_checkpointed(
        cls, config: NonCheckpointedTokenizerTrainerConfig, logger: logging.Logger
    ) -> "TokenizerTrainer":
        """
        Load a non-checkpointed tokenizer trainer.

        Args:
            config: The non-checkpointed tokenizer trainer config to use
            logger: The logger to use

        Returns:
            A tokenizer trainer
        """
        logger.info(f"Loading tokenizer trainer from non-checkpointed config: {config}")
        tokenizer = await Tokenizer.load(config=config.tokenizer_config)
        tokenizer_trainer = TokenizerTrainer(
            create_key=_TOKENIZER_TRAINER_CREATE_KEY,
            logger=logger,
            abstract_params=_AbstractTrainerParams(
                name="tokenizer",
                output_dir_path=config.output_dir_path,
            ),
            params=_TokenizerTrainerParams(
                vocab_size=config.vocab_size,
                data_sources=config.data_sources,
            ),
            tokenizer=tokenizer,
        )
        logger.info(
            f"Loaded non-checkpointed tokenizer trainer: tokenizer_trainer={tokenizer_trainer}"
        )
        return tokenizer_trainer

    @classmethod
    async def _load_checkpointed(
        cls, config: CheckpointedTokenizerTrainerConfig, logger: logging.Logger
    ) -> "TokenizerTrainer":
        """
        Load a checkpointed tokenizer trainer saved at checkpoint_dir_path.

        Args:
            config: The checkpointed tokenizer trainer config to use
            logger: The logger to use

        Returns:
            A tokenizer trainer
        """
        logger.info(f"Loading checkpointed tokenizer trainer: config={config}")

        async def _create_trainer(
            latest_checkpoint_dir_path: str,
            abstract_params: _AbstractTrainerParams,
            params_json: Any,
        ) -> "TokenizerTrainer":
            tokenizer = await Tokenizer._load_pretrained(
                config=PretrainedTokenizerConfig(
                    dir_path=latest_checkpoint_dir_path,
                ),
                logger=logger,
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

            params = _TokenizerTrainerParams(**params_json)
            return TokenizerTrainer(
                create_key=_TOKENIZER_TRAINER_CREATE_KEY,
                logger=logger,
                abstract_params=abstract_params,
                params=params,
                tokenizer=tokenizer,
            )

        abstract_params = _AbstractTrainerParams(
            name="tokenizer",
            output_dir_path=config.output_dir_path,
            checkpointing_config=config.checkpointing_config,
        )
        tokenizer_trainer = await _AbstractTrainer._load_checkpointed(
            abstract_params,
            _create_trainer,
            logger,
        )
        if tokenizer_trainer is None:
            tokenizer = await Tokenizer.load(config=config.tokenizer_config)
            tokenizer_trainer = TokenizerTrainer(
                create_key=_TOKENIZER_TRAINER_CREATE_KEY,
                logger=logger,
                abstract_params=abstract_params,
                params=_TokenizerTrainerParams(
                    vocab_size=config.vocab_size,
                    data_sources=config.data_sources,
                ),
                tokenizer=tokenizer,
            )

        logger.info(f"Loaded checkpointed tokenizer trainer: tokenizer_trainer={tokenizer_trainer}")
        return tokenizer_trainer

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
        if config.type == "non_checkpointed":
            return await TokenizerTrainer._load_non_checkpointed(config=config, logger=logger)
        elif config.type == "checkpointed":
            return await TokenizerTrainer._load_checkpointed(config=config, logger=logger)
        else:
            raise ValueError(f"Unknown tokenizer trainer config type: {config.type}")
