"""
HellaSwag evaluation utilities.
See: https://github.com/rowanz/hellaswag

Example HellaSwag json item:

{
    "ind": 24,
    "activity_label": "Roof shingle removal",
    "ctx_a": "A man is sitting on a roof.",
    "ctx_b": "he",
    "ctx": "A man is sitting on a roof. he",
    "split": "val",
    "split_type": "indomain",
    "label": 3,
    "endings": [
        "is using wrap to wrap a pair of skis.",
        "is ripping level tiles off.",
        "is holding a rubik's cube.",
        "starts pulling up roofing on a roof."
    ],
    "source_id": "activitynet~v_-JhWjGDPHMY"
}

Fields:

ind: Dataset ID
activity_label: The ActivityNet or WikiHow label for this example
context: There are two formats. The full context is in ctx. When the context ends in an (incomplete) noun phrase,
    like for ActivityNet, this incomplete noun phrase is in ctx_b, and the context up until then is in ctx_a.
    This can be useful for models such as BERT that need the last sentence to be complete. However, it's never required.
    If ctx_b is nonempty, then ctx is the same thing as ctx_a, followed by a space, then ctx_b.
endings: a list of 4 endings. The correct index is given by label (0,1,2, or 3)
split: train, val, or test.
split_type: indomain if the activity label is seen during training, else zeroshot
source_id: Which video or WikiHow article this example came from

The validation set of HellaSwag has a total of 10,042 examples.

GPT-2 (124M):
- EleutherAI's LLM evaluation harness: acc=28.92% acc_norm=31.14% (multiple choice style)
- MicroGPT Pretrained GPT-2: accuracy=28.57% accuracy_normalized=29.56% (completion style)

Pretrained model:
- MicroGPT Pretrained: accuracy=28.57% accuracy_normalized=29.56% (completion style)
"""

import json
import os
from collections.abc import Generator
from typing import Any, cast

import requests
import torch
from torch.nn import functional as F
from tqdm import tqdm

from microgpt.common.logger import _new_logger
from microgpt.tokenizer import Tokenizer

logger = _new_logger(__name__)

_CACHE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".cache", "hellaswag"))
_NUM_VAL_EXAMPLES = 10_042

_SPLIT_URLS = {
    "train": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_train.jsonl",
    "val": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl",
    "test": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_test.jsonl",
}


def _download_file(url: str, file_path: str, chunk_size: int = 1024) -> None:
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    with (
        open(file_path, "wb") as file,
        tqdm(
            desc=file_path,
            total=total,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar,
    ):
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)


def _download_split(split: str) -> None:
    """Downloads HellaSwag DATA_CACHE_DIR"""
    os.makedirs(_CACHE_DIR, exist_ok=True)
    data_url = _SPLIT_URLS[split]
    file_path = os.path.join(_CACHE_DIR, f"hellaswag_{split}.jsonl")
    if not os.path.exists(file_path):
        print(f"Downloading split: split={split} url={data_url} file_path={file_path}...")
        _download_file(data_url, file_path)


def _split_examples_iter(split: str) -> Generator[dict[str, Any], None, None]:
    _download_split(split)
    with open(os.path.join(_CACHE_DIR, f"hellaswag_{split}.jsonl")) as f:
        for line in f:
            example = json.loads(line)
            yield example


def _render_example(
    tokenizer: Tokenizer, example: dict[str, Any], device: str | torch.device | None = None
) -> tuple[torch.Tensor, torch.Tensor, int]:
    label = example["label"]
    ctx = example["ctx"]
    ctx_ids = tokenizer.encode(ctx)
    rows = []
    mask_rows = []
    endings = example["endings"]
    for end in endings:
        end_ids = tokenizer.encode(" " + end)
        rows.append(ctx_ids + end_ids)
        mask_rows.append([0] * len(ctx_ids) + [1] * len(end_ids))

    max_len = max(len(row) for row in rows)
    ids = torch.zeros((4, max_len), dtype=torch.long, device=device)
    mask = torch.zeros((4, max_len), dtype=torch.long, device=device)
    for i, (row, mask_row) in enumerate(zip(rows, mask_rows, strict=False)):
        ids[i, : len(row)] = torch.tensor(row)
        mask[i, : len(mask_row)] = torch.tensor(mask_row)

    return ids, mask, label


def _get_most_likely_row(ids: torch.Tensor, mask: torch.Tensor, logits: torch.Tensor) -> int:
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (ids[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction="none")
    shift_losses = shift_losses.view(ids.size(0), -1)
    shift_mask = (mask[..., 1:]).contiguous()
    masked_shift_losses = shift_losses * shift_mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    pred_norm = avg_loss.argmin().item()
    return cast(int, pred_norm)
