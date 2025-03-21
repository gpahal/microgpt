import logging
import os
from typing import TYPE_CHECKING

import numpy as np
import torch

from microgpt.logger import _new_logger
from microgpt.types import TextSource
from microgpt.utils import _batch_read_text_sources

if TYPE_CHECKING:
    from .tokenizer import Tokenizer


def _get_logger(logger: logging.Logger | None = None) -> logging.Logger:
    if logger is None:
        logger = _new_logger(__name__)
    return logger


SHARD_SIZE = int(1e8)  # 100M tokens per shard
SHARD_FILE_PREFIX = "shard_"
SHARD_FILE_SUFFIX = ".npy"


def _load_tokens_from_file(file_path: str) -> torch.Tensor:
    npt = np.load(file_path).astype(np.int32)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt


class _DataLoader:
    _out_dir_path: str
    _tokenizer: "Tokenizer"
    _batch_size: int
    _sequence_length: int
    _process_rank: int
    _num_processes: int
    _shards: list[str]

    def __init__(
        self,
        out_dir_path: str,
        tokenizer: "Tokenizer",
        batch_size: int,
        sequence_length: int,
        process_rank: int,
        num_processes: int,
    ) -> None:
        out_dir_path = os.path.abspath(out_dir_path)
        self._out_dir_path = out_dir_path
        self._tokenizer = tokenizer
        self._batch_size = batch_size
        self._sequence_length = sequence_length
        self._process_rank = process_rank
        self._num_processes = num_processes

        shards = os.listdir(out_dir_path)
        shards = [
            s for s in shards if s.startswith(SHARD_FILE_PREFIX) and s.endswith(SHARD_FILE_SUFFIX)
        ]
        shards.sort()
        shards = [os.path.join(out_dir_path, s) for s in shards]
        assert len(shards) > 0, f"No shards found for out_dir_path: {out_dir_path}"
        self._shards = shards
        self.reset()

    def reset(self) -> None:
        # state, init at shard zero
        self._current_shard = 0
        self._tokens = _load_tokens_from_file(self._shards[self._current_shard])
        self._current_position = self._batch_size * self._sequence_length * self._process_rank

    def next_batch(self, device: str | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length = self._batch_size, self._sequence_length
        buf = self._tokens[
            self._current_position : self._current_position + batch_size * sequence_length + 1
        ]
        x = (buf[:-1]).view(batch_size, sequence_length)
        y = (buf[1:]).view(batch_size, sequence_length)
        self._current_position += batch_size * sequence_length * self._num_processes
        # If loading the next batch would be out of bounds, advance to next shard
        if self._current_position + (batch_size * sequence_length * self._num_processes + 1) > len(
            self._tokens
        ):
            self._current_shard = (self._current_shard + 1) % len(self._shards)
            self._tokens = _load_tokens_from_file(self._shards[self._current_shard])
            self._current_position = batch_size * sequence_length * self._process_rank
        if device is not None:
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        return x, y


async def _save_tokenized_text_sources(
    tokenizer: "Tokenizer",
    out_dir_path: str,
    text_sources: TextSource | list[TextSource],
    logger: logging.Logger | None = None,
) -> None:
    """
    Save the tokenized text sources to out_dir_path in shards of SHARD_SIZE tokens.

    Args:
        tokenizer: The tokenizer to use to tokenize the text sources
        out_dir_path: The path to save the tokenized text sources to
        text_sources: The text sources to tokenize and save
        logger: The logger to use to log the progress of the tokenization
    """
    import multiprocessing as mp

    from tqdm import tqdm

    out_dir_path = os.path.abspath(out_dir_path)
    logger = _get_logger(logger)
    logger.info(f"Saving text sources to {out_dir_path}")

    os.makedirs(out_dir_path, exist_ok=True)

    def _tokenize(text: str) -> np.ndarray:
        tokens = [tokenizer.eot_id] if tokenizer.eot_id is not None else []
        tokens.extend(tokenizer.encode_ordinary(text))
        tokens_np = np.array(tokens)
        assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), (
            "Token dictionary too large for uint16"
        )
        return tokens_np.astype(np.uint16)

    pool_size = max(1, os.cpu_count() - 2)
    with mp.Pool(pool_size) as pool:
        # Preallocate buffer for current shard
        all_tokens_np = np.empty((SHARD_SIZE,), dtype=np.uint16)
        token_count = 0
        shard_index = 0
        progress_bar = tqdm(total=SHARD_SIZE, unit="tokens", desc=f"Shard {shard_index}")

        # Buffer to collect texts before batch processing
        text_buffer: list[str] = []
        buffer_size = pool_size * 20

        async for text in _batch_read_text_sources(text_sources):
            text_buffer.append(text)

            if len(text_buffer) >= buffer_size:
                # Process batch of texts in parallel
                tokens_list = pool.map(_tokenize, text_buffer)

                # Process all tokens from the batch
                for tokens in tokens_list:
                    # Check if we need to start a new shard
                    while token_count + len(tokens) >= SHARD_SIZE:
                        # Save current shard
                        shard_path = os.path.join(
                            out_dir_path, f"{SHARD_FILE_PREFIX}{shard_index:06d}{SHARD_FILE_SUFFIX}"
                        )
                        remainder = SHARD_SIZE - token_count
                        progress_bar.update(remainder)
                        all_tokens_np[token_count : token_count + remainder] = tokens[:remainder]
                        np.save(shard_path, all_tokens_np[:token_count])
                        progress_bar.close()

                        tokens = tokens[remainder:]

                        all_tokens_np = np.empty((SHARD_SIZE,), dtype=np.uint16)
                        token_count = 0
                        shard_index += 1
                        progress_bar = tqdm(
                            total=SHARD_SIZE, unit="tokens", desc=f"Shard {shard_index}"
                        )

                    # Add tokens to current shard
                    all_tokens_np[token_count : token_count + len(tokens)] = tokens
                    token_count += len(tokens)
                    progress_bar.update(len(tokens))

                # Clear buffer
                text_buffer = []

        # Process remaining texts in buffer
        if text_buffer:
            tokens_list = pool.map(_tokenize, text_buffer)
            for tokens in tokens_list:
                while token_count + len(tokens) >= SHARD_SIZE:
                    shard_path = os.path.join(
                        out_dir_path, f"{SHARD_FILE_PREFIX}{shard_index:06d}{SHARD_FILE_SUFFIX}"
                    )
                    remainder = SHARD_SIZE - token_count
                    progress_bar.update(remainder)
                    all_tokens_np[token_count : token_count + remainder] = tokens[:remainder]
                    np.save(shard_path, all_tokens_np[:token_count])
                    progress_bar.close()

                    tokens = tokens[remainder:]

                    all_tokens_np = np.empty((SHARD_SIZE,), dtype=np.uint16)
                    token_count = 0
                    shard_index += 1
                    progress_bar = tqdm(
                        total=SHARD_SIZE, unit="tokens", desc=f"Shard {shard_index}"
                    )

                all_tokens_np[token_count : token_count + len(tokens)] = tokens
                token_count += len(tokens)

        # Save final shard if it has any tokens
        if token_count > 0:
            shard_path = os.path.join(
                out_dir_path, f"{SHARD_FILE_PREFIX}{shard_index:06d}{SHARD_FILE_SUFFIX}"
            )
            remainder = SHARD_SIZE - token_count
            progress_bar.update(remainder)
            all_tokens_np[token_count : token_count + remainder] = tokens[:remainder]
            np.save(shard_path, all_tokens_np[:token_count])

        progress_bar.close()
