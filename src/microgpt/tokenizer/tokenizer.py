import logging
import os
from typing import Annotated, Literal

import numpy as np
import regex
from pydantic import BaseModel, Field

from microgpt.common.data_source import DataSource
from microgpt.common.logger import _new_logger

from .data_utils import _save_tokenized_data_sources
from .gpt_tokenizer_utils import (
    _GPT_TOKENIZER_ENCODING_VALUES_DICT,
    _MODEL_PREFIX_TO_ENCODING_DICT,
    _MODEL_TO_ENCODING_DICT,
    _GPTTokenizerConfig,
    _GPTTokenizerEncoding,
    _recover_merges,
)
from .tokenizer_utils import (
    _get_counts_dict,
    _merge_ids,
    _render_token,
)

# o200k_base encoding split pattern. Used by GPT-4o, o1 and other GPT models.
# See: https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py
_DEFAULT_SPLIT_PATTERN = "|".join(
    [
        r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
        r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
        r"""\p{N}{1,3}""",
        r""" ?[^\s\p{L}\p{N}]+[\r\n/]*""",
        r"""\s*[\r\n]+""",
        r"""\s+(?!\S)""",
        r"""\s+""",
    ]
)


class UntrainedTokenizerConfig(BaseModel):
    type: Literal["untrained"] = "untrained"
    split_pattern: str | None = None
    special_tokens: dict[str, int] | None = None
    eot_id: int | None = None


class PretrainedTokenizerConfig(BaseModel):
    type: Literal["pretrained"] = "pretrained"
    dir_path: str


class PretrainedGPTTokenizerConfig(BaseModel):
    type: Literal["pretrained_gpt"] = "pretrained_gpt"
    encoding_or_model_name: str


TokenizerConfig = Annotated[
    UntrainedTokenizerConfig | PretrainedTokenizerConfig | PretrainedGPTTokenizerConfig,
    Field(discriminator="type"),
]


class _TokenizerParams(BaseModel):
    split_pattern: str | None = None
    special_tokens: dict[str, int] | None = None
    eot_id: int | None = None
    merges: list[list[int]] | None = None


_TOKENIZER_CREATE_KEY = object()


class Tokenizer:
    """
    Tokenizer which uses a byte-level Byte Pair Encoding algorithm.

    See: https://github.com/karpathy/minbpe
    """

    _logger: logging.Logger
    _split_pattern: str
    _compiled_split_pattern: regex.Pattern
    _special_tokens: dict[str, int]
    _inverse_special_tokens: dict[int, str]
    _eot_id: int | None
    _merges: dict[tuple[int, int], int]
    _vocab: dict[int, bytes]
    _vocab_size: int

    def __init__(
        self,
        create_key: object,
        logger: logging.Logger,
        params: _TokenizerParams,
    ) -> None:
        """
        Initialize the tokenizer. Do not call this constructor directly.
        Instead, use Tokenizer.load.

        Args:
            create_key: A key to prevent instantiating the tokenizer directly
            logger: The logger to use
            params: The parameters to use
        """
        if create_key != _TOKENIZER_CREATE_KEY:
            raise ValueError("Tokenizer cannot be instantiated directly. Use Tokenizer.load")

        self._logger = logger
        self._init_split_pattern(params.split_pattern)
        self._init_special_tokens(params.special_tokens)
        self._eot_id = params.eot_id
        if self._eot_id is not None:
            assert self._special_tokens and self._eot_id in self._special_tokens.values()
        self._merges = (
            {tuple(merge[0:2]): merge[2] for merge in params.merges} if params.merges else {}
        )
        self._init_vocab()

    def _init_split_pattern(self, split_pattern: str | None = None) -> None:
        self._split_pattern = split_pattern if split_pattern is not None else _DEFAULT_SPLIT_PATTERN
        self._compiled_split_pattern = regex.compile(self._split_pattern)

    def _init_special_tokens(self, special_tokens: dict[str, int] | None = None) -> None:
        self._special_tokens = {} if special_tokens is None else special_tokens
        self._inverse_special_tokens = {v: k for k, v in self._special_tokens.items()}

    def _init_vocab(self) -> None:
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self._merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        for special, idx in self._special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        self._vocab = vocab
        self._vocab_size = len(vocab)

    def _get_params(self) -> _TokenizerParams:
        return _TokenizerParams(
            split_pattern=self._split_pattern,
            special_tokens=self._special_tokens,
            eot_id=self._eot_id,
            merges=[[pair[0], pair[1], id] for pair, id in self._merges.items()],
        )

    def __str__(self) -> str:
        return (
            "Tokenizer(\n"
            f"  split_pattern={self._split_pattern}\n"
            f"  special_tokens={self._special_tokens if self._special_tokens else 'None'}\n"
            f"  eot_id={self._eot_id if self._eot_id is not None else 'None'}\n"
            f"  merges_size={len(self._merges if hasattr(self, '_merges') and self._merges else {})}\n"
            f"  vocab_size={len(self._vocab if hasattr(self, '_vocab') and self._vocab else {})}\n"
            ")"
        )

    def __repr__(self) -> str:
        return str(self)

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def eot_id(self) -> int | None:
        return self._eot_id

    def add_special_tokens(self, special_tokens: dict[str, int]) -> None:
        """
        Add special tokens.

        Args:
            special_tokens: A dictionary of special tokens (str -> int)
        """
        special_tokens = {**self._special_tokens, **special_tokens}
        self._init_special_tokens(special_tokens)
        self._init_vocab()

    def _decode_special_token_id(self, id: int) -> bytes:
        return self._inverse_special_tokens[id].encode("utf-8")

    def _decode_vocab_id(self, id: int) -> bytes:
        return self._vocab[id]

    def decode(self, ids: list[int]) -> str:
        """
        Given a list of integers, decode them into a Python string.

        Args:
            ids: A list of integers

        Returns:
            A decoded Python string
        """
        part_bytes: list[bytes] = []
        for id in ids:
            if id in self._inverse_special_tokens:
                part_bytes.append(self._decode_special_token_id(id))
            elif id in self._vocab:
                part_bytes.append(self._decode_vocab_id(id))
            else:
                raise ValueError(f"Invalid token id: {id}")
        text_bytes = b"".join(part_bytes)
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    def _encode_bytes_to_ids(self, text_bytes: bytes) -> list[int]:
        return list(text_bytes)

    def _encode_chunk(self, text_bytes: bytes) -> list[int]:
        # First, convert all bytes to integers in range 0..255
        ids = self._encode_bytes_to_ids(text_bytes)
        if len(ids) < 2:
            return ids

        counts_dict = _get_counts_dict(ids)
        ids = np.array(ids, dtype=np.int32)
        while len(ids) >= 2:
            # Find the pair with the lowest merge index
            pair = min(counts_dict, key=lambda p: self._merges.get(p, float("inf")))
            # Otherwise, let's merge the best pair (lowest merge index)
            new_idx = self._merges.get(pair, None)
            if new_idx is None:
                # Nothing else can be merged anymore
                break
            ids = _merge_ids(ids=ids, pair=pair, new_id=new_idx, counts_dict=counts_dict)
        return ids.tolist()

    def encode_ordinary(self, text: str) -> list[int]:
        """
        Encode a text string. Ignores special tokens.

        Args:
            text: The text to encode

        Returns:
            A list of encoded integers
        """
        # Split text into chunks of text by categories defined in regex pattern
        text_chunks = regex.findall(self._compiled_split_pattern, text)
        # All chunks of text are encoded separately, then results are joined
        ids = []
        for chunk in text_chunks:
            chunk_bytes = chunk.encode("utf-8")  # raw bytes
            chunk_ids = self._encode_chunk(chunk_bytes)
            ids.extend(chunk_ids)
        return ids

    def encode(
        self,
        text: str,
        allowed_special_tokens: Literal["all", "none", "none_raise"] | set[str] = "none_raise",
    ) -> list[int]:
        """
        Encode a text string.
        Unlike encode_ordinary, this function handles special tokens.

        Args:
            text: The text to encode
            allowed_special_tokens: Can be "all" | "none" | "none_raise" or a set of special tokens.
                If none_raise, then an error is raised if any special token is encountered in text.
                This is the default tiktoken behavior right now as well.
                Any other behavior is either annoying, or a major footgun

        Returns:
            A list of encoded integers
        """
        special_tokens = None
        if allowed_special_tokens == "all":
            special_tokens = self._special_tokens
        elif allowed_special_tokens == "none":
            special_tokens = {}
        elif allowed_special_tokens == "none_raise":
            special_tokens = {}
            if any(token in text for token in self._special_tokens):
                raise ValueError("Special tokens found in text")
        elif isinstance(allowed_special_tokens, set):
            special_tokens = {
                k: v for k, v in self._special_tokens.items() if k in allowed_special_tokens
            }
        else:
            raise ValueError(f"Invalid allowed_special_tokens: {allowed_special_tokens}")

        if not special_tokens:
            # Shortcut: if no special tokens, just use the ordinary encoding
            return self.encode_ordinary(text)

        # Otherwise, we have to be careful with potential special tokens in text.
        # We handle special tokens by splitting the text based on the occurrence of any exact match
        # with any of the special tokens.
        # We can use regex.split for this. Note that surrounding the pattern with () makes it into a
        # capturing group, so the special tokens will be included in the output.
        special_pattern = "(" + "|".join(regex.escape(k) for k in special_tokens) + ")"
        special_chunks = regex.split(special_pattern, text)
        # Now all the special characters are separated from the rest of the text.
        # All chunks of text are encoded separately, then results are joined.
        ids = []
        for part in special_chunks:
            if part in special_tokens:
                # This is a special token, encode it separately as a special case
                ids.append(special_tokens[part])
            else:
                # This is an ordinary sequence, encode it normally
                ids.extend(self.encode_ordinary(part))
        return ids

    async def save_tokenized_data_sources(
        self, dir_path: str, data_sources: DataSource | list[DataSource]
    ) -> None:
        """
        Save the tokenized data sources to dir_path in shards.

        Args:
            dir_path: The directory to save the tokenized data sources to
            data_sources: The data sources to tokenize and save
        """
        await _save_tokenized_data_sources(self, dir_path, data_sources, self._logger)

    async def save(self, dir_path: str) -> None:
        """
        Save the tokenizer to two files: dir_path/tokenizer.model and dir_path/tokenizer.vocab.
        This is inspired (but not equivalent to) sentencepiece's model saving:
        - model file is the critical one, intended for load() later
        - vocab file is just a pretty printed version for human inspection only

        Args:
            dir_path: The directory to save the tokenizer to. If the directory already exists, files in it might be overwritten
        """
        import json

        import aiofiles

        os.makedirs(dir_path, exist_ok=True)

        # Write the model: to load the model later
        model_file_path = os.path.join(dir_path, "tokenizer.json")
        self._logger.info(f"Saving tokenizer model: file_path={model_file_path}")
        params = self._get_params()
        async with aiofiles.open(model_file_path, "w") as f:
            await f.write(params.model_dump_json(indent=2))
        self._logger.info("Saved tokenizer model")

        # Write the vocab: for human inspection only
        vocab_file_path = os.path.join(dir_path, "tokenizer_vocab.json")
        self._logger.info(f"Saving tokenizer vocab: file_path={vocab_file_path}")
        inverted_merges = {idx: pair for pair, idx in self._merges.items()}
        vocab_json = {}
        indices = list(self._vocab.keys())
        indices.sort()
        for idx in indices:
            token = self._vocab[idx]
            s = _render_token(token)
            # Find the children of this token, if any
            if idx in inverted_merges:
                # If this token has children, render it nicely as a merge
                idx0, idx1 = inverted_merges[idx]
                s0 = _render_token(self._vocab[idx0])
                s1 = _render_token(self._vocab[idx1])
                vocab_json[idx] = f"[{s0}][{s1}] -> [{s}]"
            else:
                # Otherwise this is a leaf token, just render it (this should just be the first 256
                # tokens, the bytes)
                vocab_json[idx] = s

        with open(vocab_file_path, "w", encoding="utf-8") as f:
            json.dump(vocab_json, f, indent=2)

        self._logger.info("Saved tokenizer vocab")

    async def _validate_matches_config(self, config: TokenizerConfig) -> None:
        if config.type == "pretrained_gpt":
            if not isinstance(self, GPTTokenizer):
                raise ValueError("Tokenizer is not a GPTTokenizer")
            elif GPTTokenizer.get_encoding(config.encoding_or_model_name) != self._encoding:
                raise ValueError(
                    "Tokenizer encoding does not match config: expected_encoding="
                    f"{GPTTokenizer.get_encoding(config.encoding_or_model_name).value}, "
                    f"actual_encoding={self._encoding.value}"
                )
        elif config.type == "untrained" or config.type == "pretrained":
            split_pattern = (
                config.split_pattern if config.split_pattern is not None else _DEFAULT_SPLIT_PATTERN
            )
            if self._split_pattern != split_pattern:
                raise ValueError(
                    "Tokenizer split pattern does not match config: "
                    f"expected_split_pattern={split_pattern}, "
                    f"actual_split_pattern={self._split_pattern}"
                )
            if self._special_tokens != config.special_tokens:
                raise ValueError(
                    "Tokenizer special tokens do not match config: "
                    f"expected_special_tokens={config.special_tokens}, "
                    f"actual_special_tokens={self._special_tokens}"
                )
            if self._eot_id != config.eot_id:
                raise ValueError(
                    "Tokenizer eot id does not match config: "
                    f"expected_eot_id={config.eot_id}, "
                    f"actual_eot_id={self._eot_id}"
                )

            if config.type == "pretrained":
                other_tokenizer = await Tokenizer._load_pretrained(
                    config=config, logger=self._logger
                )

                if isinstance(self, GPTTokenizer):
                    if not isinstance(other_tokenizer, GPTTokenizer):
                        raise ValueError(
                            "Tokenizer is a GPTTokenizer, but other tokenizer is not: "
                            f"expected_tokenizer_type={type(other_tokenizer)}, "
                            f"actual_tokenizer_type={type(self)}"
                        )

                    if self._encoding != other_tokenizer._encoding:
                        raise ValueError(
                            "Tokenizer encoding does not match config: "
                            f"expected_encoding={other_tokenizer._encoding.value}, "
                            f"actual_encoding={self._encoding.value}"
                        )
                    return

                if isinstance(other_tokenizer, GPTTokenizer):
                    raise ValueError(
                        "Tokenizer is not a GPTTokenizer: "
                        f"expected_tokenizer_type={type(other_tokenizer)}, "
                        f"actual_tokenizer_type={type(self)}"
                    )

                # Check if the vocabulary matches or is a subset
                if self._vocab_size < other_tokenizer._vocab_size:
                    raise ValueError(
                        "Tokenizer vocabulary size is less than expected: "
                        f"expected_vocab_size={other_tokenizer._vocab_size}, "
                        f"actual_vocab_size={self._vocab_size}"
                    )
                for idx in other_tokenizer._vocab:
                    if idx not in self._vocab:
                        raise ValueError(
                            "Tokenizer vocabulary is not a subset of the expected vocabulary"
                        )
        else:
            raise ValueError(f"Unknown tokenizer config type: {config.type}")

    @classmethod
    async def _load_untrained(
        cls,
        config: UntrainedTokenizerConfig,
        logger: logging.Logger,
    ) -> "Tokenizer":
        """
        Load an untrained tokenizer.

        Args:
            config: The untrained tokenizer config to use
            logger: The logger to use

        Returns:
            A tokenizer
        """
        logger.info(f"Loading untrained tokenizer: config={config}")
        tokenizer = Tokenizer(
            create_key=_TOKENIZER_CREATE_KEY,
            logger=logger,
            params=_TokenizerParams(
                split_pattern=config.split_pattern,
                special_tokens=config.special_tokens,
                eot_id=config.eot_id,
            ),
        )
        logger.info(f"Loaded untrained tokenizer: tokenizer={tokenizer}")
        return tokenizer

    @classmethod
    async def _load_pretrained(
        cls, config: PretrainedTokenizerConfig, logger: logging.Logger
    ) -> "Tokenizer":
        """
        Load a pretrained tokenizer that was saved to dir_path.

        Args:
            config: The pretrained tokenizer config to use
            logger: The logger to use

        Returns:
            A tokenizer
        """
        import json

        import aiofiles

        logger.info(f"Loading pretrained tokenizer: config={config}")
        file_path = os.path.join(config.dir_path, "tokenizer.json")

        async with aiofiles.open(file_path, encoding="utf-8") as f:
            params_json = json.loads(await f.read())

        if "encoding_name" in params_json:
            params = _GPTTokenizerParams(**params_json)
            tokenizer = GPTTokenizer(
                create_key=_TOKENIZER_CREATE_KEY,
                logger=logger,
                params=params,
            )
        else:
            params = _TokenizerParams(**params_json)
            tokenizer = Tokenizer(
                create_key=_TOKENIZER_CREATE_KEY,
                logger=logger,
                params=params,
            )

        logger.info(f"Loaded pretrained tokenizer: tokenizer={tokenizer}")
        return tokenizer

    @classmethod
    async def load(cls, config: TokenizerConfig) -> "Tokenizer":
        """
        Load a tokenizer.

        Args:
            config: The tokenizer config to use
            logger: The logger to use

        Returns:
            A tokenizer
        """
        logger = _new_logger(__name__)
        if config.type == "untrained":
            return await Tokenizer._load_untrained(
                config=config,
                logger=logger,
            )
        elif config.type == "pretrained":
            return await Tokenizer._load_pretrained(
                config=config,
                logger=logger,
            )
        elif config.type == "pretrained_gpt":
            return await GPTTokenizer._load_pretrained(
                config=config,
                logger=logger,
            )
        else:
            raise ValueError(f"Unknown tokenizer config type: {config.type}")


class _GPTTokenizerParams(BaseModel):
    encoding: _GPTTokenizerEncoding
    config: _GPTTokenizerConfig
    mergeable_ranks: dict[bytes, int]
    merges: dict[tuple[int, int], int]


class GPTTokenizer(Tokenizer):
    """
    GPT Tokenizer.

    See: https://github.com/karpathy/minbpe
    """

    _encoding: _GPTTokenizerEncoding
    _config: _GPTTokenizerConfig
    _mergeable_ranks: dict[bytes, int]
    _byte_shuffle: dict[int, int]
    _inverse_byte_shuffle: dict[int, int]

    def __init__(
        self,
        create_key: object,
        logger: logging.Logger,
        params: _GPTTokenizerParams,
    ) -> None:
        """
        Initialize the GPT tokenizer. Do not call this constructor directly.
        Instead, use Tokenizer.load.

        Args:
            create_key: A key to prevent instantiating the tokenizer directly
            logger: The logger to use
            params: The parameters to use
        """
        if create_key != _TOKENIZER_CREATE_KEY:
            raise ValueError("GPTTokenizer cannot be instantiated directly. Use Tokenizer.load")

        super().__init__(
            create_key=create_key,
            logger=logger,
            params=_TokenizerParams(
                split_pattern=params.config.split_pattern,
                special_tokens=params.config.special_tokens,
                eot_id=params.config.eot_id,
                merges=params.merges,
            ),
        )

        if self._config.vocab_size is not None:
            assert self._config.vocab_size == len(self._vocab), (
                f"Vocabulary size mismatch: {self._config.vocab_size} != {len(self._vocab)}"
            )

        self._encoding = params.encoding
        self._config = params.config
        self._mergeable_ranks = params.mergeable_ranks

        # The tokens corresponding to individual bytes are permuted in a different order.
        # The reason is no clear, but we have to deal with it here
        self._byte_shuffle = {i: self._mergeable_ranks[bytes([i])] for i in range(256)}
        self._inverse_byte_shuffle = {v: k for k, v in self._byte_shuffle.items()}

    def _get_params(self) -> _GPTTokenizerParams:
        return _GPTTokenizerParams(
            encoding_name=self._encoding.value,
        )

    def __str__(self) -> str:
        return (
            "GPTTokenizer(\n"
            f"  encoding={self._encoding.value if self._encoding else 'None'}\n"
            f"  special_tokens={self._special_tokens}\n"
            f"  eot_id={self._eot_id if self._eot_id is not None else 'None'}\n"
            f"  mergeable_ranks_size={len(self._mergeable_ranks) if hasattr(self, '_mergeable_ranks') and self._mergeable_ranks else 0}\n"
            f"  merges_size={len(self._merges) if hasattr(self, '_merges') and self._merges else 0}\n"
            f"  vocab_size={len(self._vocab) if hasattr(self, '_vocab') and self._vocab else 0}\n"
            ")"
        )

    def _decode_vocab_id(self, id: int) -> bytes:
        return bytes(self._inverse_byte_shuffle[b] for b in self._vocab[id])

    def _encode_bytes_to_ids(self, text_bytes: bytes) -> list[int]:
        return [self._byte_shuffle[b] for b in text_bytes]

    @classmethod
    def get_encoding(cls, encoding_or_model_name: str) -> _GPTTokenizerEncoding:
        assert isinstance(encoding_or_model_name, str), (
            f"Encoding or model name must be a string, got {type(encoding_or_model_name)}"
        )

        encoding: _GPTTokenizerEncoding
        if encoding_or_model_name in _GPT_TOKENIZER_ENCODING_VALUES_DICT:
            encoding = _GPT_TOKENIZER_ENCODING_VALUES_DICT[encoding_or_model_name]
        elif encoding_or_model_name in _MODEL_TO_ENCODING_DICT:
            encoding = _MODEL_TO_ENCODING_DICT[encoding_or_model_name]
        else:
            for model_prefix, model_encoding in _MODEL_PREFIX_TO_ENCODING_DICT.items():
                if encoding_or_model_name.startswith(model_prefix):
                    encoding = model_encoding
                    break
            else:
                raise ValueError(f"invalid encoding or model name: {encoding_or_model_name}")

        assert encoding.is_valid(), f"Invalid encoding or model name: {encoding_or_model_name}"
        return encoding

    @classmethod
    async def _load_pretrained(
        cls, config: PretrainedGPTTokenizerConfig, logger: logging.Logger
    ) -> "GPTTokenizer":
        """
        Load a pretrained GPT tokenizer.

        Args:
            config: The pretrained tokenizer config to use
            logger: The logger to use

        Returns:
            A tokenizer
        """
        logger.info(f"Loading pretrained gpt tokenizer: config={config}")
        assert isinstance(config.encoding_or_model_name, str)

        encoding = GPTTokenizer.get_encoding(config.encoding_or_model_name)
        logger.info(f"Pretrained gpt tokenizer encoding: encoding={encoding.value}")
        config = encoding.config()
        mergeable_ranks = await config.mergeable_ranks_fn()
        logger.info("Loaded mergeable ranks")
        merges = _recover_merges(mergeable_ranks)
        tokenizer = GPTTokenizer(
            create_key=_TOKENIZER_CREATE_KEY,
            logger=logger,
            params=_GPTTokenizerParams(
                encoding=encoding,
                config=config,
                mergeable_ranks=mergeable_ranks,
                merges=merges,
            ),
        )
        logger.info(f"Loaded pretrained gpt tokenizer: tokenizer={tokenizer}")
        return tokenizer
