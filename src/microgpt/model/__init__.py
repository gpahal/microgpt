import logging

from .model import Model
from .types import (
    ModelConfig,
    PretrainedGPT2ModelConfig,
    PretrainedGPT2ModelType,
    PretrainedModelConfig,
    UntrainedModelConfig,
)


async def load_model(
    config: ModelConfig, device: str | None = None, logger: logging.Logger | None = None
) -> Model:
    if config.type == "untrained":
        return Model._load_untrained(
            tokenizer=config.tokenizer,
            params=config.params,
            device=device,
            logger=logger,
        )
    elif config.type == "pretrained":
        return Model._load_pretrained(
            dir_path=config.dir_path,
            device=device,
            logger=logger,
        )
    elif config.type == "pretrained_gpt2":
        return await Model._load_pretrained_gpt_2(
            model_type=config.model_type,
            dropout_p=config.dropout_p,
            bias=config.bias,
            device=device,
            logger=logger,
        )
    else:
        raise ValueError(f"Unknown model config type: {config.type}")


__all__ = [
    "Model",
    "ModelConfig",
    "UntrainedModelConfig",
    "PretrainedModelConfig",
    "PretrainedGPT2ModelType",
    "PretrainedGPT2ModelConfig",
    "load_model",
]
