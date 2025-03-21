from typing import Annotated, Literal

from pydantic import BaseModel, Field


class UntrainedTokenizerConfig(BaseModel):
    type: Literal["untrained"] = "untrained"
    split_pattern: str | None = None
    special_tokens: dict[str, int] | None = None
    eot_id: int | None = None


class PretrainedTokenizerConfig(BaseModel):
    type: Literal["pretrained"] = "pretrained"
    file_path_prefix: str


class PretrainedGPTTokenizerConfig(BaseModel):
    type: Literal["pretrained_gpt"] = "pretrained_gpt"
    encoding_or_model_name: str


TokenizerConfig = Annotated[
    UntrainedTokenizerConfig | PretrainedTokenizerConfig | PretrainedGPTTokenizerConfig,
    Field(discriminator="type"),
]
