import abc
import logging
import os
import shutil
from collections.abc import Callable, Coroutine
from typing import Any

from pydantic import BaseModel, Field
from tqdm import tqdm

from .fs import _ensure_dir_path


class TrainerCheckpointingConfig(BaseModel):
    checkpointing_interval: int
    checkpoint_dir_path: str
    keep_last_n_checkpoints: int = Field(default=1, ge=1)


class _TrainerParams(BaseModel):
    name: str
    iteration: int = 1
    output_dir_path: str | None = None
    checkpointing_config: TrainerCheckpointingConfig | None = None


class _Trainer[TReturn, TRunContext](abc.ABC):
    _logger: logging.Logger
    _name: str
    _iteration: int
    _latest_run_iteration: int
    _output_dir_path: str | None
    _checkpointing_config: TrainerCheckpointingConfig | None

    def __init__(
        self,
        logger: logging.Logger,
        params: _TrainerParams,
    ):
        """
        Initialize the abstract trainer. Do not call this constructor directly.
        Instead, use microgpt.trainer.load_trainer.

        Args:
            logger: The logger to use
            params: The params to use
        """
        self._logger = logger
        self._name = params.name
        self._iteration = max(params.iteration, 1)
        self._latest_run_iteration = 1
        self._output_dir_path = params.output_dir_path
        self._checkpointing_config = params.checkpointing_config
        self._ensure_output_dir_path()
        self._ensure_checkpoint_dir_path()

    def __str__(self) -> str:
        return (
            "AbstractTrainer(\n"
            f"  name={self._name}\n"
            f"  iteration={self._iteration}\n"
            f"  latest_run_iteration={self._latest_run_iteration}\n"
            f"  output_dir_path={self._output_dir_path}\n"
            f"  checkpointing_config={self._checkpointing_config}\n"
            ")"
        )

    def __repr__(self) -> str:
        return self.__str__()

    @property
    def name(self) -> str:
        return self._name

    @property
    def iteration(self) -> int:
        return self._iteration

    @property
    def latest_run_iteration(self) -> int:
        return self._latest_run_iteration

    def is_checkpointing_enabled(self) -> bool:
        return self._checkpointing_config is not None

    @abc.abstractmethod
    def _get_params(self) -> BaseModel:
        raise NotImplementedError("Subclasses must implement this method")

    @abc.abstractmethod
    async def _save_output_data(self, dir_path: str) -> None:
        """
        Save the trainer output data to dir_path.

        Args:
            dir_path: The directory to save the trainer output data to
        """
        raise NotImplementedError("Subclasses must implement this method")

    async def _save_checkpoint_data(self, dir_path: str) -> None:
        """
        Save the trainer checkpoint data to dir_path.

        Args:
            dir_path: The directory to save the trainer checkpoint data to
        """
        await self._save_output_data(dir_path)

    @abc.abstractmethod
    async def _run_setup(self) -> tuple[int | None, bool, TRunContext]:
        """
        Run the trainer setup.

        Returns:
            A tuple containing the maximum number of iterations the trainer will run, whether to show a progress bar
            and the run context
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abc.abstractmethod
    async def _run_teardown(self, run_context: TRunContext) -> None:
        raise NotImplementedError("Subclasses must implement this method")

    @abc.abstractmethod
    async def _run_iteration(self, run_context: TRunContext) -> bool:
        """
        Run the trainer iteration.

        Returns:
            True if the iteration should continue, False if it should stop
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abc.abstractmethod
    def _get_return_value(self, run_context: TRunContext) -> TReturn:
        """
        Get the return value of the trainer.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def _ensure_output_dir_path(self) -> None:
        if self._output_dir_path is None:
            return

        _ensure_dir_path(
            self._output_dir_path,
            "output",
            self._logger,
        )

    def _ensure_checkpoint_dir_path(self) -> None:
        if self._checkpointing_config is None:
            return

        _ensure_dir_path(
            self._checkpointing_config.checkpoint_dir_path,
            "checkpoint",
            self._logger,
        )

    def _get_iteration_checkpoint_dir_path(self) -> str:
        if self._checkpointing_config is None:
            raise ValueError("Checkpointing is not enabled")

        return os.path.join(
            self._checkpointing_config.checkpoint_dir_path,
            f"iteration_{self._iteration:012d}",
        )

    async def _save_trainer_params(self, dir_path: str) -> None:
        import json

        import aiofiles

        params_file_path = os.path.join(dir_path, f"{self._name}_trainer.json")
        self._logger.info(f"Saving {self._name} trainer params: file_path={params_file_path}")
        params = _TrainerParams(
            name=self._name,
            iteration=self._iteration,
            latest_run_iteration=self._latest_run_iteration,
            output_dir_path=self._output_dir_path,
            checkpointing_config=self._checkpointing_config,
        )
        custom_params = self._get_params()
        async with aiofiles.open(params_file_path, "w") as f:
            await f.write(
                json.dumps(
                    {
                        "trainer": params.model_dump(),
                        f"{self._name}_trainer": custom_params.model_dump(),
                    },
                    indent=2,
                )
            )
        self._logger.info(
            f"Saved {self._name} trainer params: "
            f"trainer_params=[{params.model_dump()}] "
            f"{self._name}_trainer_params=[{custom_params.model_dump()}]"
        )

    async def _save_output(self) -> None:
        """
        Save the trainer output.
        """
        if self._output_dir_path is None:
            return

        self._logger.info(f"Saving {self._name} trainer output: dir_path={self._output_dir_path}")
        self._ensure_output_dir_path()
        await self._save_output_data(self._output_dir_path)

        self._logger.info(f"Saved {self._name} trainer output")

    async def _save_checkpoint(self) -> None:
        """
        Save the trainer checkpoint.
        """
        if self._checkpointing_config is None:
            return

        dir_path = self._get_iteration_checkpoint_dir_path()
        self._logger.info(
            f"Saving {self._name} trainer checkpoint: "
            f"dir_path={dir_path} "
            f"iteration={self._iteration} "
            f"latest_run_iteration={self._latest_run_iteration}"
        )
        self._ensure_checkpoint_dir_path()
        _ensure_dir_path(dir_path, "checkpoint", self._logger)
        await self._save_checkpoint_data(dir_path)

        await self._save_trainer_params(dir_path)

        self._logger.info(
            f"Saved {self._name} trainer checkpoint: "
            f"iteration={self._iteration} "
            f"latest_run_iteration={self._latest_run_iteration}"
        )

        keep_last_n_checkpoints = self._checkpointing_config.keep_last_n_checkpoints
        iteration_checkpoint_dir_paths = _Trainer._get_iteration_checkpoint_dir_paths(
            checkpoint_dir_path=self._checkpointing_config.checkpoint_dir_path,
            logger=self._logger,
        )
        if len(iteration_checkpoint_dir_paths) > keep_last_n_checkpoints:
            for iteration_checkpoint_dir_path in iteration_checkpoint_dir_paths[:-keep_last_n_checkpoints]:
                if os.path.exists(iteration_checkpoint_dir_path):
                    shutil.rmtree(iteration_checkpoint_dir_path)

    async def _run_loop(
        self, max_iterations: int | None, run_context: TRunContext, inc_progress_bar: Callable[[], None]
    ) -> None:
        while True:
            should_continue = await self._run_iteration(run_context)
            inc_progress_bar()
            if not should_continue:
                return

            if max_iterations is not None and self._iteration >= max_iterations:
                return

            if (
                self._checkpointing_config is not None
                and self._iteration % self._checkpointing_config.checkpointing_interval == 0
            ):
                await self._save_checkpoint()

            self._iteration += 1
            self._latest_run_iteration += 1

    async def run(self) -> TReturn:
        max_iterations, show_progress_bar, run_context = await self._run_setup()
        if show_progress_bar:
            progress_bar = tqdm(total=max_iterations, unit="iters", desc="Training", file=open(os.devnull, "w"))
        if show_progress_bar:
            self._logger.info(str(progress_bar))
            if self._iteration > 1:
                progress_bar.update(self._iteration - 1)

        def _inc_progress_bar() -> None:
            if show_progress_bar:
                progress_bar.update(1)
                self._logger.info(str(progress_bar))

        try:
            if max_iterations is not None and self._iteration > max_iterations:
                return self._get_return_value(run_context)

            await self._run_loop(max_iterations, run_context, _inc_progress_bar)
            await self._save_output()
            return self._get_return_value(run_context)
        finally:
            if show_progress_bar:
                progress_bar.close()
            await self._run_teardown(run_context)

    @classmethod
    def _get_iteration_checkpoint_dir_paths(
        cls,
        checkpoint_dir_path: str,
        logger: logging.Logger,
    ) -> list[str]:
        """
        Get the iteration checkpoint directory paths sorted by iteration number.

        Args:
            checkpoint_dir_path: The directory to look for checkpoints in
            logger: The logger to use

        Returns:
            The iteration checkpoint directory paths sorted by iteration number
        """
        import os

        logger.info(f"Finding iteration checkpoint directory paths: checkpoint_dir_path={checkpoint_dir_path}")

        # Get all directories that match the pattern iteration_*
        iterations: list[int] = []
        try:
            for entry in os.listdir(checkpoint_dir_path):
                full_path = os.path.join(checkpoint_dir_path, entry)
                if os.path.isdir(full_path) and entry.startswith("iteration_"):
                    iteration_str = entry[len("iteration_") :]
                    try:
                        iteration = int(iteration_str)
                    except ValueError:
                        logger.warning(f"Invalid iteration checkpoint: {entry}")
                        continue
                    iterations.append(iteration)
        except FileNotFoundError:
            return []

        iterations.sort()
        return [os.path.join(checkpoint_dir_path, f"iteration_{iteration:012d}") for iteration in iterations]

    @classmethod
    async def _load_checkpointed_internal[TrainerT: _Trainer[Any, Any]](
        cls,
        params: _TrainerParams,
        create_trainer_fn: Callable[[str, _TrainerParams, Any], Coroutine[Any, Any, TrainerT]],
        logger: logging.Logger,
    ) -> TrainerT | None:
        """
        Load a checkpointed trainer saved at checkpoint_dir_path.

        Args:
            params: The params to use
            create_trainer_fn: The function to create the trainer. This function should take the latest
                iteration checkpoint directory path, the abstract trainer params and trainer params json
                and returns a trainer instance.
            logger: The logger to use

        Returns:
            A trainer
        """
        import json

        import aiofiles

        if not params.checkpointing_config:
            return None

        iteration_checkpoint_dir_paths = _Trainer._get_iteration_checkpoint_dir_paths(
            checkpoint_dir_path=params.checkpointing_config.checkpoint_dir_path,
            logger=logger,
        )
        if not iteration_checkpoint_dir_paths:
            return None

        latest_iteration_checkpoint_dir_path = iteration_checkpoint_dir_paths[-1]
        params_file_path = os.path.join(latest_iteration_checkpoint_dir_path, f"{params.name}_trainer.json")
        logger.info(f"Loading checkpointed {params.name} trainer params: file_path={params_file_path}")
        async with aiofiles.open(params_file_path) as f:
            loaded_custom_params_json: dict[str, Any] = json.loads(await f.read())

        loaded_params_json = loaded_custom_params_json["trainer"]
        loaded_params = _TrainerParams(**loaded_params_json)
        if loaded_params.name != params.name:
            raise ValueError(
                f"Latest checkpointed {params.name} trainer name does not match expected name: "
                f"expected_name={params.name}, "
                f"checkpointed_name={loaded_params.name}"
            )
        if loaded_params.iteration + 1 < params.iteration:
            raise ValueError(
                f"Latest checkpointed {params.name} trainer iteration is less than expected iteration: "
                f"expected_iteration={params.iteration}, "
                f"checkpointed_iteration={loaded_params.iteration}"
            )

        # Override the output dir path and checkpointing config to match the current run
        params.iteration = loaded_params.iteration + 1

        loaded_custom_params_json = loaded_custom_params_json[f"{params.name}_trainer"]
        logger.info(
            f"Loaded checkpointed {params.name} trainer params: "
            f"trainer_params=[{params.model_dump()}] "
            f"{params.name}_trainer_params=[{loaded_custom_params_json}]"
        )

        trainer = await create_trainer_fn(latest_iteration_checkpoint_dir_path, params, loaded_custom_params_json)
        return trainer
