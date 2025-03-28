from typing import Literal

import torch

DeviceType = Literal["cuda", "cpu"]


def _get_device_type(device: str) -> DeviceType:
    if "cuda" in device:
        return "cuda"
    else:
        return "cpu"


def _validate_device_type(device_type: str) -> None:
    if device_type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA is not available")
    elif device_type != "cpu":
        raise ValueError(f"Invalid device type: {device_type}")


def _get_device(device: str | None = None) -> str:
    if device is not None:
        _validate_device_type(_get_device_type(device))
        return device

    if torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def _get_dtype(device_type: DeviceType) -> str:
    _validate_device_type(device_type)
    return "bfloat16" if device_type == "cuda" and torch.cuda.is_bf16_supported() else "float16"


def _get_torch_dtype(dtype: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
