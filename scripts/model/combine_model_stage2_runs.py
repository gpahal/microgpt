import os
import shutil

import torch

from microgpt.common.logger import _new_logger

logger = _new_logger(__name__)


def main() -> None:
    dirname = os.path.dirname(os.path.abspath(__file__))
    trained_model_dir_path = os.path.join(dirname, "trained_model_stage2")
    run_dirs = os.listdir(trained_model_dir_path)
    run_dirs = [s for s in run_dirs if s.startswith("run_") and os.path.isdir(os.path.join(trained_model_dir_path, s))]
    logger.info(f"Found {len(run_dirs)} runs: run_dirs={run_dirs}")
    run_ints = [int(run[4:].strip()) for run in run_dirs]
    logger.info(f"Found {len(run_ints)} run ints: run_ints={run_ints}")
    assert len(run_ints) > 0, "At least one run is required"

    output_dir_path = os.path.join(trained_model_dir_path, "output")
    os.makedirs(output_dir_path, exist_ok=True)

    model_weights: dict[str, torch.Tensor]
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

        if run_idx == 0:
            file_names_to_new_file_names_map = {
                "model.json": "model.json",
                "tokenizer.json": "tokenizer.json",
                "tokenizer_vocab.json": "tokenizer_vocab.json",
                "loss.txt": f"loss_{run}.txt",
                "eval.txt": f"eval_{run}.txt",
            }
        else:
            file_names_to_new_file_names_map = {
                "loss.txt": f"loss_{run}.txt",
                "eval.txt": f"eval_{run}.txt",
            }

        for file_name, new_file_name in file_names_to_new_file_names_map.items():
            shutil.copy(os.path.join(run_output_dir_path, file_name), os.path.join(output_dir_path, new_file_name))

        run_model_weights = torch.load(os.path.join(run_output_dir_path, "model.pt"), weights_only=True)
        if run_idx == 0:
            model_weights = run_model_weights
        else:
            for key, value in run_model_weights.items():
                model_weights[key] += value

    for key, value in model_weights.items():
        model_weights[key] = value / len(run_ints)

    torch.save(model_weights, os.path.join(output_dir_path, "model.pt"))


if __name__ == "__main__":
    main()
