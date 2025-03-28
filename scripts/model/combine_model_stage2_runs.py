import os
import shutil
from typing import Annotated

import torch
import typer

from microgpt.common.logger import _new_logger

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)

logger = _new_logger(__name__)


@app.command()
def main(
    runs: Annotated[
        int,
        typer.Option("--run", help="The runs to combine. Comma-separated list of run numbers"),
    ],
) -> None:
    run_strs = runs.split(",")
    run_ints = [int(run.strip()) for run in run_strs]
    assert len(run_ints) > 0, "At least one run is required"

    dirname = os.path.dirname(os.path.abspath(__file__))
    output_dir_path = os.path.join(dirname, "trained_model_stage2/output")
    model_weights: dict[str, torch.Tensor]
    for run_idx, run in enumerate(run_ints):
        run_output_dir_path = os.path.join(dirname, f"trained_model_stage2/run_{run}/output")
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
    app()
