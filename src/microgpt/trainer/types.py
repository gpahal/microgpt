from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from microgpt.model import Model


class TrainerParams(BaseModel):
    manual_seed: int | None = None
    should_compile: bool = True
    train_out_dir_path: str
    validation_out_dir_path: str
    max_iters: int
    log_interval: int = 1
    eval_interval: int = 1000
    eval_iters: int = 100
    checkpoint_dir_path: str | None = None
    checkpoint_backup_dir_path: str | None = None
    checkpoint_eval_interval: int | None = None
    batch_size: int
    gradient_accumulation_steps: int
    max_learning_rate: float = 6e-4
    min_learning_rate: float = 6e-5
    learning_rate_warmup_iters: int = 2000
    learning_rate_decay_iters: int | None = None
    weight_decay: float = 1e-1
    betas: tuple[float, float] = (0.9, 0.99)
    max_grad_norm: float = 1.0


class NewTrainerConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    type: Literal["new"] = "new"
    model: Model
    params: TrainerParams


class CheckpointedTrainerConfig(BaseModel):
    type: Literal["checkpointed"] = "checkpointed"
    file_path_prefix: str
    device: str | None = None


TrainerConfig = Annotated[
    NewTrainerConfig | CheckpointedTrainerConfig,
    Field(discriminator="type"),
]
