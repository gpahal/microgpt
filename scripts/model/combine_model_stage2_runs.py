import asyncio
import os
import shutil

import torch

from microgpt import CustomTrainedModelConfig, Model
from microgpt.common.device import _get_device
from microgpt.common.logger import _new_logger

logger = _new_logger(__name__)


async def main() -> None:
    dirname = os.path.dirname(os.path.abspath(__file__))
    trained_model_dir_path = os.path.join(dirname, "trained_model_stage2")
    run_dirs = os.listdir(trained_model_dir_path)
    run_dirs = [s for s in run_dirs if s.startswith("run_") and os.path.isdir(os.path.join(trained_model_dir_path, s))]
    run_ints = [int(run[4:].strip()) for run in run_dirs]
    run_ints.sort()
    logger.info(f"Found {len(run_ints)} runs: runs={run_ints}")
    assert len(run_ints) > 0, "At least one run is required"

    output_dir_path = os.path.join(trained_model_dir_path, "output")
    os.makedirs(output_dir_path, exist_ok=True)

    model_weights: dict[str, torch.Tensor] = {}
    device = _get_device()
    total_runs = len(run_ints)
    for run_idx, run in enumerate(run_ints):
        run_output_dir_path = os.path.join(trained_model_dir_path, f"run_{run}/output")
        assert os.path.exists(run_output_dir_path), f"Trained model directory not found: {run_output_dir_path}"
        assert os.path.exists(os.path.join(run_output_dir_path, "model.json")), (
            f"Run {run} model output model file not found: {run_output_dir_path}"
        )
        assert os.path.exists(os.path.join(run_output_dir_path, "model.pt")), (
            f"Run {run} model output weights file not found: {run_output_dir_path}"
        )
        assert os.path.exists(os.path.join(run_output_dir_path, "tokenizer.json")), (
            f"Run {run} model output tokenizer file not found: {run_output_dir_path}"
        )
        assert os.path.exists(os.path.join(run_output_dir_path, "tokenizer_vocab.json")), (
            f"Run {run} model output tokenizer vocab file not found: {run_output_dir_path}"
        )
        assert os.path.exists(os.path.join(run_output_dir_path, "loss.txt")), (
            f"Run {run} model output loss file not found: {run_output_dir_path}"
        )
        assert os.path.exists(os.path.join(run_output_dir_path, "eval.txt")), (
            f"Run {run} model output eval file not found: {run_output_dir_path}"
        )

        os.makedirs(os.path.join(output_dir_path, f"run_{run}"), exist_ok=True)
        if run_idx == 0:
            file_names_to_new_file_names_map = {
                "model.json": "model.json",
                "tokenizer.json": "tokenizer.json",
                "tokenizer_vocab.json": "tokenizer_vocab.json",
                "loss.txt": f"run_{run}/loss.txt",
                "eval.txt": f"run_{run}/eval.txt",
            }
        else:
            file_names_to_new_file_names_map = {
                "loss.txt": f"run_{run}/loss.txt",
                "eval.txt": f"run_{run}/eval.txt",
            }

        for file_name, new_file_name in file_names_to_new_file_names_map.items():
            shutil.copy(os.path.join(run_output_dir_path, file_name), os.path.join(output_dir_path, new_file_name))

        run_model_weights: dict[str, torch.Tensor] = torch.load(
            os.path.join(run_output_dir_path, "model.pt"), map_location=device, weights_only=True
        )
        if run_idx == 0:
            for key, value in run_model_weights.items():
                model_weights[key] = value / (total_runs * 1.0)
        else:
            assert len(model_weights) == len(run_model_weights), "Model weights must have the same length"
            for key, value in run_model_weights.items():
                assert key in model_weights, f"Key {key} not found in model weights"
                assert model_weights[key].shape == value.shape, (
                    f"Shape mismatch for key {key}: {model_weights[key].shape} != {value.shape}"
                )
                model_weights[key] += value / (total_runs * 1.0)

    model_weights_file_path = os.path.join(output_dir_path, "model.pt")
    torch.save(model_weights, model_weights_file_path)
    model = await Model.load(config=CustomTrainedModelConfig(dir_path=output_dir_path), device=device)
    new_model_weights = model.state_dict()
    for key, value in model_weights.items():
        assert key in new_model_weights, f"Key {key} not found in new model weights"
        assert new_model_weights[key].shape == value.shape, (
            f"Shape mismatch for key {key}: {new_model_weights[key].shape} != {value.shape}"
        )
        assert new_model_weights[key].dtype == value.dtype, (
            f"Dtype mismatch for key {key}: {new_model_weights[key].dtype} != {value.dtype}"
        )
        assert torch.equal(new_model_weights[key], value), f"Values for key {key} are not equal"
    torch.save(new_model_weights, model_weights_file_path)


if __name__ == "__main__":
    asyncio.run(main())
