import os
import shutil
import tempfile
import uuid
from typing import Annotated

import boto3
import typer
from botocore.exceptions import NoCredentialsError
from dotenv import load_dotenv

from microgpt.common.logger import _new_logger

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)

logger = _new_logger(__name__)
load_dotenv()

_DEFAULT_BUCKET_NAME = "microgpt"


def zip_dir(dir_path: str, zip_file_path: str) -> None:
    """
    Zip the directory into a compressed archive.

    Args:
        dir_path: Path to the directory to zip
        zip_file_path: Path to the directory zip file
    """
    if not os.path.exists(dir_path):
        logger.error(f"Directory not found: dir_path={dir_path}")
        raise FileNotFoundError(f"Directory not found: dir_path={dir_path}")

    logger.info(f"Creating zip archive of data folder: zip_file_path={zip_file_path}")
    shutil.make_archive(
        base_name=os.path.splitext(zip_file_path)[0],
        format="zip",
        root_dir=os.path.dirname(dir_path),
        base_dir=os.path.basename(dir_path),
    )
    logger.info(f"Created zip archive: zip_file_path={zip_file_path}")


def unzip_dir(zip_file_path: str, parent_dir_path: str, dir_path: str) -> None:
    """
    Unzip the directory from a compressed archive.

    Args:
        zip_file_path: Path to the directory zip file to unzip
        parent_dir_path: Path to the parent directory to unzip
    """
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)

    os.makedirs(parent_dir_path, exist_ok=True)

    logger.info(f"Unzipping data folder: zip_file_path={zip_file_path}")
    shutil.unpack_archive(zip_file_path, parent_dir_path)
    logger.info(f"Unzipped data folder: dir_path={dir_path}")


def _get_bucket_name() -> str:
    return os.environ.get("S3_BUCKET_NAME") or _DEFAULT_BUCKET_NAME


def download_from_s3(s3_key: str, file_path: str) -> None:
    """
    Download directory zip file from an S3 bucket.

    Args:
        s3_key: Key of the directory zip file to download
        file_path: Path to the downloaded directory zip file
    """
    bucket_name = _get_bucket_name()
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        region_name=os.environ.get("AWS_REGION"),
    )

    try:
        logger.info(f"Downloading file: bucket_name={bucket_name} s3_key={s3_key} file_path={file_path}")
        s3_client.download_file(bucket_name, s3_key, file_path)
        logger.info(f"Downloaded file: file_path={file_path}")
    except NoCredentialsError:
        logger.error("AWS credentials not found. Make sure AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY are set")
        raise
    except Exception as e:
        logger.error(f"Error downloading from S3: {str(e)}")
        raise


def upload_to_s3(file_path: str, s3_key: str) -> None:
    """
    Upload file to an S3 bucket.

    Args:
        file_path: Path to the file to upload
    """
    bucket_name = _get_bucket_name()
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        region_name=os.environ.get("AWS_REGION"),
    )

    try:
        logger.info(f"Uploading file: file_path={file_path} bucket_name={bucket_name} s3_key={s3_key}")
        s3_client.upload_file(file_path, bucket_name, s3_key)
        logger.info("Uploaded file")
    except FileNotFoundError:
        logger.error(f"File not found: file_path={file_path}")
        raise
    except NoCredentialsError:
        logger.error("AWS credentials not found. Make sure AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY are set")
        raise
    except Exception as e:
        logger.error(f"Error uploading to S3: {str(e)}")
        raise


def validate_credentials():
    aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
    aws_region = os.environ.get("AWS_REGION")
    if not aws_access_key_id or not aws_secret_access_key or not aws_region:
        logger.error(
            "AWS credentials not found. Make sure AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and AWS_REGION are set"
        )
        return


def _download_dir(s3_key: str, dir_path: str) -> None:
    validate_credentials()
    if not s3_key.endswith(".zip"):
        raise ValueError("s3_key must end with .zip")

    dir_path = os.path.abspath(dir_path)
    parent_dir_path = os.path.dirname(dir_path)
    with tempfile.TemporaryDirectory() as temp_dir:
        zip_file_path = os.path.join(temp_dir, f"temp_{uuid.uuid4()}.zip")
        download_from_s3(s3_key, zip_file_path)
        unzip_dir(zip_file_path, parent_dir_path, dir_path)

        try:
            os.remove(zip_file_path)
        except Exception as e:
            logger.warning(f"Error deleting directory zip file: {str(e)}")


def _upload_dir(dir_path: str, s3_key: str) -> None:
    validate_credentials()
    if not s3_key.endswith(".zip"):
        raise ValueError("s3_key must end with .zip")

    dir_path = os.path.abspath(dir_path)
    with tempfile.TemporaryDirectory() as temp_dir:
        zip_file_path = os.path.join(temp_dir, f"temp_{uuid.uuid4()}.zip")
        zip_dir(dir_path, zip_file_path)
        upload_to_s3(zip_file_path, s3_key)

        try:
            os.remove(zip_file_path)
        except Exception as e:
            logger.warning(f"Error deleting directory zip file: {str(e)}")


@app.command(name="download-dir")
def download_dir(
    s3_key: Annotated[str, typer.Option("--s3-key", "-k")],
    dir_path: Annotated[str, typer.Option("--dir-path", "-d")],
) -> None:
    _download_dir(s3_key, dir_path)


@app.command(name="upload-dir")
def upload_dir(
    dir_path: Annotated[str, typer.Option("--dir-path", "-d")],
    s3_key: Annotated[str | None, typer.Option("--s3-key", "-k")] = None,
) -> None:
    _upload_dir(dir_path, s3_key)


if __name__ == "__main__":
    app()
