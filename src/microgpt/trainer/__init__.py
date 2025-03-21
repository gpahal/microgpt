import logging

from .trainer import Trainer
from .types import CheckpointedTrainerConfig, NewTrainerConfig, TrainerConfig


async def load_trainer(config: TrainerConfig, logger: logging.Logger | None = None) -> Trainer:
    if config.type == "new":
        return Trainer._load_new(
            model=config.model,
            params=config.params,
            logger=logger,
        )
    elif config.type == "checkpointed":
        return await Trainer._load_checkpointed(
            file_path_prefix=config.file_path_prefix, device=config.device, logger=logger
        )
    else:
        raise ValueError(f"Unknown trainer config type: {config.type}")


__all__ = [
    "Trainer",
    "TrainerConfig",
    "NewTrainerConfig",
    "CheckpointedTrainerConfig",
    "load_trainer",
]
