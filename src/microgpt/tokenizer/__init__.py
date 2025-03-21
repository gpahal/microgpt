import logging

from .tokenizer import (
    GPTTokenizer,
    Tokenizer,
)
from .types import (
    PretrainedGPTTokenizerConfig,
    PretrainedTokenizerConfig,
    TokenizerConfig,
    UntrainedTokenizerConfig,
)


async def load_tokenizer(
    config: TokenizerConfig, logger: logging.Logger | None = None
) -> Tokenizer:
    if config.type == "untrained":
        return await Tokenizer._load_untrained(
            split_pattern=config.split_pattern,
            special_tokens=config.special_tokens,
            eot_id=config.eot_id,
            logger=logger,
        )
    elif config.type == "pretrained":
        return await Tokenizer._load_pretrained(
            file_path_prefix=config.file_path_prefix, logger=logger
        )
    elif config.type == "pretrained_gpt":
        return await GPTTokenizer._load_pretrained(
            encoding_or_model_name=config.encoding_or_model_name, logger=logger
        )
    else:
        raise ValueError(f"Unknown tokenizer config type: {config.type}")


__all__ = [
    "Tokenizer",
    "GPTTokenizer",
    "TokenizerConfig",
    "UntrainedTokenizerConfig",
    "PretrainedTokenizerConfig",
    "PretrainedGPTTokenizerConfig",
    "load_tokenizer",
]
