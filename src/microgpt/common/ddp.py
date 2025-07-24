import os
from dataclasses import dataclass


@dataclass
class _DDPParams:
    _rank: int
    _local_rank: int
    _world_size: int


def _get_ddp_params() -> _DDPParams | None:
    rank = int(os.environ.get("RANK", -1))
    if rank == -1:
        return None
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))
    assert rank >= 0 and local_rank >= 0 and world_size >= 1, (
        f"Invalid DDP environment variables: RANK={rank}, LOCAL_RANK={local_rank}, WORLD_SIZE={world_size}"
    )
    return _DDPParams(_rank=rank, _local_rank=local_rank, _world_size=world_size)
