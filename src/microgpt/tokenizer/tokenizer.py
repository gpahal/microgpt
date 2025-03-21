import logging
import os
from typing import Literal

import numpy as np
import regex

from microgpt.logger import _new_logger
from microgpt.types import TextSource
from microgpt.utils import _batch_read_text_source

from .data_utils import _save_tokenized_text_sources
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


def _get_logger(logger: logging.Logger | None = None) -> logging.Logger:
    if not logger:
        logger = _new_logger(__name__)
    return logger


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


_TOKENIZER_CREATE_KEY = object()


class Tokenizer:
    """
    Tokenizer which uses a byte-level Byte Pair Encoding algorithm.

    See: https://github.com/karpathy/minbpe
    """

    _logger: logging.Logger
    _merges: dict[tuple[int, int], int]
    _split_pattern: str
    _compiled_split_pattern: regex.Pattern
    _special_tokens: dict[str, int]
    _inverse_special_tokens: dict[int, str]
    _vocab: dict[int, bytes]
    _vocab_size: int
    _eot_id: int | None

    def __init__(
        self,
        create_key: object,
        pattern: str | None = None,
        special_tokens: dict[str, int] | None = None,
        eot_id: int | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        """
        Initialize the tokenizer. Do not call this constructor directly.
        Instead, use microgpt.tokenizer.load_tokenizer.

        Args:
            create_key: A key to prevent instantiating the tokenizer directly
            pattern: The pattern to use for splitting the text
            special_tokens: A dictionary of special tokens (str -> int)
            logger: The logger to use
        """
        if create_key != _TOKENIZER_CREATE_KEY:
            raise ValueError(
                "Tokenizer cannot be instantiated directly. Use microgpt.tokenizer.load_tokenizer"
            )

        self._logger = _get_logger(logger)
        self._merges = {}
        self._init_split_pattern(pattern)
        self._init_special_tokens(special_tokens)
        self._init_vocab()
        self._eot_id = eot_id
        if self._eot_id is not None:
            assert self._eot_id in self._vocab

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

    def __str__(self) -> str:
        return (
            "Tokenizer(\n"
            f"  pattern={self._split_pattern}\n"
            f"  special_tokens={self._special_tokens if self._special_tokens else 'None'}\n"
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

    async def save_tokenized_text_sources(
        self, out_dir_path: str, text_sources: TextSource | list[TextSource]
    ) -> None:
        """
        Save the tokenized text sources to out_dir_path in shards.

        Args:
            out_dir_path: The path to save the tokenized text sources to
            text_sources: The text sources to tokenize and save
        """
        await _save_tokenized_text_sources(self, out_dir_path, text_sources, self._logger)

    @classmethod
    async def _load_untrained(
        cls,
        split_pattern: str | None = None,
        special_tokens: dict[str, int] | None = None,
        eot_id: int | None = None,
        logger: logging.Logger | None = None,
    ) -> "Tokenizer":
        """
        Load an untrained tokenizer.

        Args:
            split_pattern: The pattern to use for splitting the text
            special_tokens: A dictionary of special tokens (str -> int)
            logger: The logger to use

        Returns:
            A tokenizer
        """
        logger = _get_logger(logger)
        inst = Tokenizer(
            create_key=_TOKENIZER_CREATE_KEY,
            pattern=split_pattern,
            special_tokens=special_tokens,
            eot_id=eot_id,
            logger=logger,
        )
        return inst

    async def _save_model_file(self, file_path_prefix: str) -> None:
        import aiofiles

        # Write the model: to load the model later
        model_file = file_path_prefix + ".tokenizer.model"
        self._logger.info(f"Saving tokenizer model to {model_file}")
        async with aiofiles.open(model_file, "w") as f:
            # Write the version, pattern and merges, that's all that's needed
            await f.write("microgpt.tokenizer v1\n")
            await f.write(f"{self._split_pattern}\n")
            # Write the special tokens, first the number of them, then each one
            await f.write(f"{len(self._special_tokens)}\n")
            for special, idx in self._special_tokens.items():
                await f.write(f"{special} {idx}\n")
            # Write the EOT id, if it exists, else -1
            await f.write(f"{self._eot_id if self._eot_id is not None else -1}\n")
            # Write the merges dict
            for idx1, idx2 in self._merges:
                await f.write(f"{idx1} {idx2}\n")

        self._logger.info(f"Saved tokenizer model to {model_file}")

    async def _save_vocab_file(self, file_path_prefix: str) -> None:
        import aiofiles

        # Write the vocab: for human inspection only
        vocab_file = file_path_prefix + ".tokenizer.vocab"
        self._logger.info(f"Saving tokenizer vocab to {vocab_file}")
        inverted_merges = {idx: pair for pair, idx in self._merges.items()}
        async with aiofiles.open(vocab_file, "w", encoding="utf-8") as f:
            for idx, token in self._vocab.items():
                # Note: many tokens may be partial utf-8 sequences and cannot be decoded into valid
                # strings. Here we're using errors='replace' to replace them with the replacement
                # char �.
                # This also means that we couldn't possibly use .vocab in load() because decoding
                # in this way is a lossy operation!
                s = _render_token(token)
                # Find the children of this token, if any
                if idx in inverted_merges:
                    # If this token has children, render it nicely as a merge
                    idx0, idx1 = inverted_merges[idx]
                    s0 = _render_token(self._vocab[idx0])
                    s1 = _render_token(self._vocab[idx1])
                    await f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
                else:
                    # Otherwise this is a leaf token, just print it (this should just be the first
                    # 256 tokens, the bytes)
                    await f.write(f"[{s}] {idx}\n")

        self._logger.info(f"Saved tokenizer vocab to {vocab_file}")

    async def save(self, file_path_prefix: str) -> None:
        """
        Save the tokenizer to two files: file_path_prefix.tokenizer.model and file_path_prefix.tokenizer.vocab.
        This is inspired (but not equivalent to) sentencepiece's model saving:
        - model file is the critical one, intended for load() later
        - vocab file is just a pretty printed version for human inspection only

        Args:
            file_path_prefix: The prefix of the file path to save the tokenizer to. If the file
                already exists, it will be overwritten
        """
        dirname = os.path.dirname(file_path_prefix)
        os.makedirs(dirname, exist_ok=True)
        await self._save_model_file(file_path_prefix)
        await self._save_vocab_file(file_path_prefix)

    @classmethod
    async def _load_pretrained(
        cls, file_path_prefix: str, logger: logging.Logger | None = None
    ) -> "Tokenizer":
        """
        Load a pretrained tokenizer that was saved to file_path_prefix.

        Args:
            file_path_prefix: The prefix of the file path to load the tokenizer from
            logger: The logger to use

        Returns:
            A tokenizer
        """
        assert not file_path_prefix.endswith(".tokenizer.model")
        file_path = file_path_prefix + ".tokenizer.model"

        import aiofiles

        logger = _get_logger(logger)
        logger.info(f"Loading tokenizer from {file_path}")

        # Read the model file
        merges = {}
        special_tokens = {}
        eot_id = None
        idx = 256
        inst: Tokenizer | None = None
        encoding_name: str | None = None
        async with aiofiles.open(file_path, encoding="utf-8") as f:
            # Read the version
            version = (await f.readline()).strip()
            if version.startswith("microgpt.tokenizer.gpt"):
                encoding_name = (await f.readline()).strip()
            else:
                assert version == "microgpt.tokenizer v1"

                inst = Tokenizer(create_key=_TOKENIZER_CREATE_KEY, logger=logger)
                # Read the pattern
                inst._split_pattern = (await f.readline()).strip()
                # Read the special tokens
                num_special = int((await f.readline()).strip())
                for _ in range(num_special):
                    special, special_idx = (await f.readline()).strip().split()
                    special_tokens[special] = int(special_idx)
                # Read the EOT id
                eot_id = int((await f.readline()).strip())
                if eot_id == -1:
                    eot_id = None
                # Read the merges
                async for line in f:
                    idx1, idx2 = map(int, line.split())
                    merges[(idx1, idx2)] = idx
                    idx += 1

        if inst is None:
            assert encoding_name is not None
            inst = await GPTTokenizer._load_pretrained(
                encoding_or_model_name=encoding_name,
                logger=logger,
            )
        else:
            inst._merges = merges
            inst._init_special_tokens(special_tokens)
            inst._init_vocab()
            inst._eot_id = eot_id
            if eot_id is not None:
                assert eot_id in inst._vocab

        logger.info(f"Tokenizer loaded from {file_path}")
        return inst

    async def train(self, text_sources: TextSource | list[TextSource], vocab_size: int) -> None:
        """
        Train the tokenizer on a given text.

        Args:
            text: The text to train the tokenizer on
            vocab_size: The size of the vocabulary to train the tokenizer on. Must be at least 256
        """
        from tqdm import tqdm

        assert vocab_size >= self._vocab_size
        num_merges = vocab_size - self._vocab_size

        # Some merges already exist, so we need to merge the inputs before training again
        existing_merges_count = len(self._merges)
        inputs_need_merging = existing_merges_count > 0

        self._logger.info(
            f"Training tokenizer: existing_merges_count={existing_merges_count} vocab_size={vocab_size}"
        )

        if not isinstance(text_sources, list):
            text_sources = [text_sources]

        ids: list[int] = []

        self._logger.info("Processing text sources")

        text_source_tokens_dict: dict[str, list[int]] = {}
        for text_source in text_sources:
            text_source_tokens = 0
            async for text in _batch_read_text_source(text_source):
                # Split the text up into text chunks
                text_chunks: list[str] = regex.findall(self._compiled_split_pattern, text)
                # Input text preprocessing
                if inputs_need_merging:
                    for ch in text_chunks:
                        new_ids = self._encode_chunk(ch.encode("utf-8"))
                        ids.extend(new_ids)
                        text_source_tokens += len(new_ids)
                else:
                    for ch in text_chunks:
                        new_ids = list(ch.encode("utf-8"))
                        ids.extend(new_ids)
                        text_source_tokens += len(new_ids)

            text_source_tokens_dict[text_source.name] = text_source_tokens

        self._logger.info(
            f"Processed text sources: num_merges={num_merges} tokens={len(ids)} text_source_tokens={text_source_tokens_dict}"
        )

        # Iteratively merge the most common pairs to create new tokens
        counts_dict = _get_counts_dict(ids)
        ids = np.array(ids, dtype=np.int32)
        starting_idx = 256 + existing_merges_count
        progress_bar = tqdm(
            total=num_merges,
            unit="merges",
            desc="Merging tokens",
        )

        for i in range(num_merges):
            if len(counts_dict) == 0:
                # Nothing else can be merged anymore
                break

            # Find the pair with the highest count
            pair = max(counts_dict, key=lambda p: counts_dict.get(p, 0))
            if counts_dict.get(pair, 0) == 0:
                # Nothing else can be merged anymore
                break

            # Mint a new token: assign it the next available id
            new_idx = starting_idx + i
            # Replace all occurrences of pair in ids with idx
            ids = _merge_ids(
                ids=ids,
                pair=pair,
                new_id=new_idx,
                counts_dict=counts_dict,
            )

            # Save the merge
            self._merges[pair] = new_idx
            self._vocab[new_idx] = self._vocab[pair[0]] + self._vocab[pair[1]]
            progress_bar.update(1)

        progress_bar.close()
        self._vocab_size = len(self._vocab)
        self._logger.info(f"Finished merging tokens: vocab size {self._vocab_size}")


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
        encoding: _GPTTokenizerEncoding,
        config: _GPTTokenizerConfig,
        mergeable_ranks: dict[bytes, int],
        eot_id: int | None,
        logger: logging.Logger | None = None,
    ) -> None:
        """
        Initialize the GPT tokenizer. Do not call this constructor directly.
        Instead, use microgpt.tokenizer.load_tokenizer.

        Args:
            create_key: A key to prevent instantiating the tokenizer directly
            encoding: The encoding to use
            config: The config to use
            mergeable_ranks: The mergeable ranks to use
            logger: The logger to use
        """
        if create_key != _TOKENIZER_CREATE_KEY:
            raise ValueError(
                "GPTTokenizer cannot be instantiated directly. Use microgpt.tokenizer.load_tokenizer"
            )

        super().__init__(create_key=create_key, logger=logger)

        self._encoding = encoding
        self._config = config
        self._mergeable_ranks = mergeable_ranks
        self._merges = _recover_merges(mergeable_ranks)
        self._init_split_pattern(self._config.split_pattern)
        self._init_special_tokens(self._config.special_tokens)
        self._init_vocab()

        if self._config.vocab_size is not None:
            assert self._config.vocab_size == len(self._vocab)

        self._eot_id = eot_id
        if self._eot_id is not None:
            assert self._eot_id in self._vocab

        # The tokens corresponding to individual bytes are permuted in a different order.
        # The reason is no clear, but we have to deal with it here
        self._byte_shuffle = {i: self._mergeable_ranks[bytes([i])] for i in range(256)}
        self._inverse_byte_shuffle = {v: k for k, v in self._byte_shuffle.items()}

    def __str__(self) -> str:
        return (
            "GPTTokenizer(\n"
            f"  encoding={self._encoding.value if self._encoding else 'None'}\n"
            f"  special_tokens={self._special_tokens}\n"
            f"  mergeable_ranks_size={len(self._mergeable_ranks) if hasattr(self, '_mergeable_ranks') and self._mergeable_ranks else 0}\n"
            f"  merges_size={len(self._merges) if hasattr(self, '_merges') and self._merges else 0}\n"
            f"  vocab_size={len(self._vocab) if hasattr(self, '_vocab') and self._vocab else 0}\n"
            ")"
        )

    def _decode_vocab_id(self, id: int) -> bytes:
        return bytes(self._inverse_byte_shuffle[b] for b in self._vocab[id])

    def _encode_bytes_to_ids(self, text_bytes: bytes) -> list[int]:
        return [self._byte_shuffle[b] for b in text_bytes]

    async def _save_model_file(self, file_path_prefix: str) -> None:
        import aiofiles

        # Write the model: to load the model later
        model_file = file_path_prefix + ".tokenizer.model"
        self._logger.info(f"Saving tokenizer model to {model_file}")
        async with aiofiles.open(model_file, "w") as f:
            # Write the version, pattern and merges, that's all that's needed
            await f.write("microgpt.tokenizer.gpt v1\n")
            await f.write(f"{self._encoding.value}\n")

        self._logger.info(f"Saved tokenizer model to {model_file}")

    @classmethod
    async def _load_pretrained(
        cls, encoding_or_model_name: str, logger: logging.Logger | None = None
    ) -> "GPTTokenizer":
        """
        Load a pretrained GPT tokenizer.

        Args:
            encoding_or_model_name: The encoding or model name to load
            logger: The logger to use

        Returns:
            A tokenizer
        """
        assert isinstance(encoding_or_model_name, str)

        logger = _get_logger(logger)

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

        assert encoding.is_valid()
        config = encoding.config()
        logger.info(
            f"Loading pretrained gpt tokenizer: encoding_or_model_name={encoding_or_model_name} encoding={encoding.value}"
        )
        mergeable_ranks = await config.mergeable_ranks_fn()
        logger.info("Loaded mergeable ranks")
        inst = GPTTokenizer(
            create_key=_TOKENIZER_CREATE_KEY,
            encoding=encoding,
            config=config,
            mergeable_ranks=mergeable_ranks,
            eot_id=config.eot_id,
            logger=logger,
        )
        logger.info(
            f"Loaded gpt tokenizer: encoding_or_model_name={encoding_or_model_name} encoding={encoding.value}"
        )
        return inst
