"""
Script to download pretrained model and tokenizer.
"""

import os

from dotenv import load_dotenv
from huggingface_hub import snapshot_download

from microgpt.common.logger import _new_logger

logger = _new_logger(__name__)
load_dotenv()


def main() -> None:
    pretrained_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "pretrained"))
    os.makedirs(pretrained_dir, exist_ok=True)

    logger.info(f"Downloading gpahal/microgpt from Hugging Face Hub into {pretrained_dir} ...")
    snapshot_download(
        repo_id="gpahal/microgpt",
        local_dir=pretrained_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    logger.info("Download complete.")


if __name__ == "__main__":
    main()
