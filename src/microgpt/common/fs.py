import logging
import os


def _ensure_dir_path(
    dir_path: str, name: str | None = None, logger: logging.Logger | None = None
) -> None:
    name = f"{name} directory" if name is not None else "directory"
    # Check if dir_path exists
    if not os.path.exists(dir_path):
        if logger is not None:
            logger.info(f"Creating {name}: {dir_path}")
        os.makedirs(dir_path, exist_ok=True)

    # Check if the directory is a directory
    if not os.path.isdir(dir_path):
        raise ValueError(f"Cannot create {name}: {dir_path} exists but is not a directory")
