import os
import random
from typing import Literal

import numpy as np
import torch

from microgpt.tokenizer import Tokenizer

_SHARD_SIZE = int(1e8)  # 100M tokens per shard
_SHARD_FILE_PREFIX = "shard_"
_SHARD_FILE_SUFFIX = ".npy"


class _DataLoader:
    _shards_dir_path: str
    _split: Literal["train", "val"]
    _tokenizer: "Tokenizer"
    _batch_size: int
    _sequence_length: int
    _process_rank: int
    _num_processes: int
    _num_batches_per_shard: int
    _seed: int
    _epoch: int
    _random: random.Random
    _generator: torch.Generator
    _original_shards: list[str]
    _shards: list[str]
    _curr_shard: int
    _curr_batch: int
    _batches: torch.Tensor

    def __init__(
        self,
        shards_dir_path: str,
        split: Literal["train", "val"],
        tokenizer: "Tokenizer",
        batch_size: int,
        sequence_length: int,
        process_rank: int,
        num_processes: int,
        seed: int | None = None,
        epoch: int = 0,
        skip_batches: int = 0,
    ) -> None:
        shards_dir_path = os.path.abspath(shards_dir_path)
        self._shards_dir_path = shards_dir_path
        self._split = split
        self._tokenizer = tokenizer
        self._batch_size = batch_size
        self._sequence_length = sequence_length
        self._process_rank = process_rank
        self._num_processes = num_processes
        self._num_batches_per_shard = (_SHARD_SIZE - 1) // (batch_size * sequence_length * num_processes)
        self._seed = seed if seed is not None else 42
        self._epoch = epoch
        self._random = random.Random()
        self._generator = torch.Generator()

        split_shards_dir_path = os.path.abspath(os.path.join(shards_dir_path, split))
        shards = os.listdir(split_shards_dir_path)
        shards = [s for s in shards if s.startswith(_SHARD_FILE_PREFIX) and s.endswith(_SHARD_FILE_SUFFIX)]
        shards.sort()
        shards = [os.path.join(split_shards_dir_path, s) for s in shards]
        assert len(shards) > 0, f"No shards found: shards_dir_path={shards_dir_path} split={split}"
        self._original_shards = shards
        self._shards = shards
        self._reset(epoch=epoch, skip_batches=skip_batches)

    def _load_batches(self) -> torch.Tensor:
        file_path = self._shards[self._curr_shard]
        all_processes_batch_size = self._batch_size * self._sequence_length * self._num_processes
        npt = np.load(file_path).astype(np.uint16)
        tokens = torch.tensor(npt, dtype=torch.long)  # (n_tokens)
        if tokens.shape[0] == 0:
            return torch.zeros((0, all_processes_batch_size), dtype=torch.long)

        num_batches = (tokens.shape[0] - 1) // all_processes_batch_size
        if num_batches == 0:
            return torch.zeros((0, all_processes_batch_size), dtype=torch.long)

        max_start_idx = (tokens.shape[0] - 1) % all_processes_batch_size
        start_idx = 0
        if max_start_idx > 0:
            self._random.seed(self._seed + self._epoch)
            start_idx = self._random.randint(0, max_start_idx)

        batches = tokens[start_idx : start_idx + num_batches * all_processes_batch_size].view(
            (num_batches, all_processes_batch_size)
        )  # (n_batches, all_processes_batch_size)
        batches_extra = tokens[start_idx + all_processes_batch_size :: all_processes_batch_size][
            :num_batches, None
        ]  # (n_batches, 1)
        batches = torch.cat((batches, batches_extra), dim=-1)  # (n_batches, all_processes_batch_size + 1)
        # Shuffle the batches for each new epoch to ensure different training data order
        generator = self._generator.manual_seed(self._seed + self._epoch)
        batches = batches[torch.randperm(batches.size(0), generator=generator)]
        return batches

    def _reset(self, epoch: int = 0, skip_batches: int = 0) -> None:
        self._epoch = epoch
        # Shuffle the shards for each new epoch to ensure different training data order
        self._shards = self._original_shards.copy()
        self._random.seed(self._seed + epoch)
        # Don't shuffle the last shard as it might be smaller than the other shards
        shards_to_shuffle = self._shards[:-1]
        self._random.shuffle(shards_to_shuffle)
        self._shards = shards_to_shuffle + [self._shards[-1]]

        self._curr_shard = (skip_batches // self._num_batches_per_shard) % len(self._shards)
        self._batches = self._load_batches()
        self._curr_batch = skip_batches % self._num_batches_per_shard
        self._load_next_shard_if_needed()

    def _load_next_shard_if_needed(self) -> None:
        if self._curr_batch >= len(self._batches) - 1:
            self._curr_shard = (self._curr_shard + 1) % len(self._shards)
            self._batches = self._load_batches()
            self._curr_batch = 0

    def _next_batch(self, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length = self._batch_size, self._sequence_length
        buf = self._batches[self._curr_batch][
            batch_size * sequence_length * self._process_rank : (
                batch_size * sequence_length * (self._process_rank + 1)
            )
            + 1
        ]
        x = (buf[:-1]).view(batch_size, sequence_length)
        y = (buf[1:]).view(batch_size, sequence_length)
        self._curr_batch += 1
        self._load_next_shard_if_needed()
        x, y = x.to(device), y.to(device)
        return x, y
