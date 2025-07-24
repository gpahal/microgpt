import logging
import os

from microgpt.common.logger import _new_logger

logger = _new_logger(__name__)


def _ensure_dir_path(dir_path: str, name: str | None = None, custom_logger: logging.Logger | None = None) -> None:
    custom_logger = custom_logger or logger
    name = f"{name} directory" if name is not None else "directory"
    # Check if dir_path exists
    if not os.path.exists(dir_path):
        custom_logger.info(f"Creating {name}: dir_path={dir_path}")
        os.makedirs(dir_path, exist_ok=True)

    # Check if the directory is a directory
    if not os.path.isdir(dir_path):
        raise ValueError(f"Cannot create {name}. Path already exists but is not a directory: dir_path={dir_path}")
