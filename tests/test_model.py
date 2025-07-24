import os
from collections.abc import AsyncGenerator

import pytest
import pytest_asyncio

from microgpt import (
    CustomTrainedModelConfig,
    Model,
    ModelTrainer,
    ModelTrainerConfig,
    PretrainedModelConfig,
    PretrainedTokenizerConfig,
    UntrainedModelConfig,
)

GENERATE_TEXT_INPUT_TEXT = "Hi, I'm a language model,"


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def trained_model() -> AsyncGenerator[Model, None]:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp_dir_path:
        loss_output_file_path = os.path.join(tmp_dir_path, "loss.txt")
        eval_output_file_path = os.path.join(tmp_dir_path, "eval.txt")
        model_trainer = await ModelTrainer.load(
            config=ModelTrainerConfig(
                model_konfig=UntrainedModelConfig(
                    tokenizer_config=PretrainedTokenizerConfig(),
                    max_seq_len=64,
                    d_model=32,
                    n_layers=2,
                    n_heads=2,
                ),
                should_compile=False,
                max_iterations_per_epoch=2,
                batch_size=2,
                gradient_accumulation_iterations=1,
                enable_hellaswag_eval=False,
                loss_output_file_path=loss_output_file_path,
                eval_output_file_path=eval_output_file_path,
            )
        )
        model = await model_trainer.run()
        yield model


@pytest.mark.asyncio(loop_scope="module")
async def test_max_seq_len(trained_model: Model) -> None:
    assert trained_model._params.max_seq_len == 64


@pytest.mark.asyncio(loop_scope="module")
async def test_generate_text(trained_model: Model) -> None:
    generated_text = trained_model.generate_text(GENERATE_TEXT_INPUT_TEXT, max_new_tokens=10)
    assert len(generated_text) > len(GENERATE_TEXT_INPUT_TEXT)
    assert generated_text.startswith(GENERATE_TEXT_INPUT_TEXT)


@pytest.mark.asyncio(loop_scope="module")
async def test_save_and_load(trained_model: Model) -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp_dir_path:
        dir_path = os.path.join(tmp_dir_path, "test_model")
        await trained_model.save(dir_path)
        model = await Model.load(config=CustomTrainedModelConfig(dir_path=dir_path))
        assert model._params.max_seq_len == trained_model._params.max_seq_len


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def pretrained_model() -> AsyncGenerator[Model, None]:
    model = await Model.load(config=PretrainedModelConfig())
    yield model


@pytest.mark.asyncio(loop_scope="module")
async def test_pretrained_model(pretrained_model: Model) -> None:
    generated_text = pretrained_model.generate_text(GENERATE_TEXT_INPUT_TEXT, max_new_tokens=10)
    assert len(generated_text) > len(GENERATE_TEXT_INPUT_TEXT)
    assert generated_text.startswith(GENERATE_TEXT_INPUT_TEXT)
