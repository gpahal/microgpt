import enum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from microgpt.tokenizer import PretrainedGPTTokenizerConfig, Tokenizer


class ModelParams(BaseModel):
    block_size: int = 512
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout_p: float = 0.0
    bias: bool = False


class UntrainedModelConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    type: Literal["untrained"] = "untrained"
    tokenizer: Tokenizer
    params: ModelParams


class PretrainedModelConfig(BaseModel):
    type: Literal["pretrained"] = "pretrained"
    dir_path: str


class PretrainedGPT2ModelType(enum.StrEnum):
    GPT_2 = "gpt-2"
    GPT_2_MEDIUM = "gpt-2-medium"
    GPT_2_LARGE = "gpt-2-large"
    GPT_2_XL = "gpt-2-xl"

    def is_valid(self) -> bool:
        return self in {self.GPT_2, self.GPT_2_MEDIUM, self.GPT_2_LARGE, self.GPT_2_XL}

    def huggingface_model_name(self) -> str:
        return {
            self.GPT_2: "gpt2",
            self.GPT_2_MEDIUM: "gpt2-medium",
            self.GPT_2_LARGE: "gpt2-large",
            self.GPT_2_XL: "gpt2-xl",
        }[self]

    def tokenizer_config(self) -> PretrainedGPTTokenizerConfig:
        return PretrainedGPTTokenizerConfig(
            encoding_or_model_name="gpt-2",
        )

    def params(self, dropout_p: float | None = None, bias: bool | None = None) -> ModelParams:
        params = {
            self.GPT_2: ModelParams(block_size=1024, n_layer=12, n_head=12, n_embd=768),
            self.GPT_2_MEDIUM: ModelParams(block_size=1024, n_layer=24, n_head=16, n_embd=1024),
            self.GPT_2_LARGE: ModelParams(block_size=1024, n_layer=36, n_head=20, n_embd=1280),
            self.GPT_2_XL: ModelParams(block_size=1024, n_layer=48, n_head=25, n_embd=1600),
        }[self]
        if dropout_p is not None:
            params.dropout_p = dropout_p
        params.bias = bias if bias is not None else True
        return params


class PretrainedGPT2ModelConfig(BaseModel):
    type: Literal["pretrained_gpt2"] = "pretrained_gpt2"
    model_type: PretrainedGPT2ModelType
    dropout_p: float | None = None
    bias: bool | None = None


ModelConfig = Annotated[
    UntrainedModelConfig | PretrainedModelConfig | PretrainedGPT2ModelConfig,
    Field(discriminator="type"),
]


class ModelTrainConfig(BaseModel):
    manual_seed: int | None = None
    should_compile: bool = True
    train_out_dir_path: str
    validation_out_dir_path: str
    max_iters: int
    log_interval: int = 1
    eval_interval: int = 1000
    eval_iters: int = 100
    checkpoint_dir_path: str | None = None
    checkpoint_interval: int | None = None
    batch_size: int
    gradient_accumulation_steps: int
    max_learning_rate: float = 6e-4
    min_learning_rate: float = 6e-5
    learning_rate_warmup_steps: int = 2000
    learning_rate_decay_iters: int | None = None
    weight_decay: float = 1e-1
    betas: tuple[float, float] = (0.9, 0.99)
    max_grad_norm: float = 1.0
