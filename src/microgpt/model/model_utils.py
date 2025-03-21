import torch


def _get_device_type(device: str) -> str:
    if "cuda" in device:
        return "cuda"
    elif "mps" in device:
        return "mps"
    else:
        return "cpu"


def _validate_device_type(device_type: str) -> None:
    if device_type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA is not available")
    elif device_type == "mps":
        if not torch.backends.mps.is_available():
            raise ValueError("MPS is not available")


def _get_device(device: str | None = None) -> str:
    if device is not None:
        _validate_device_type(_get_device_type(device))
        return device

    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def _get_dtype(device_type: str) -> str:
    _validate_device_type(device_type)
    return "bfloat16" if device_type == "cuda" and torch.cuda.is_bf16_supported() else "float16"
