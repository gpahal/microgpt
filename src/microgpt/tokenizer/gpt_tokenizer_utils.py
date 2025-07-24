import base64
import enum
import hashlib
import os
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import Any

_R50K_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s"""
_CL100K_SPLIT_PATTERN = "|".join(
    [
        r"""'(?i:[sdmt]|ll|ve|re)""",
        r"""[^\r\n\p{L}\p{N}]?+\p{L}++""",
        r"""\p{N}{1,3}+""",
        r""" ?[^\s\p{L}\p{N}]++[\r\n]*+""",
        r"""\s++$""",
        r"""\s*[\r\n]""",
        r"""\s+(?!\S)""",
        r"""\s""",
    ]
)
_O200K_SPLIT_PATTERN = "|".join(
    [
        r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
        r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
        r"""\p{N}{1,3}""",
        r""" ?[^\s\p{L}\p{N}]+[\r\n/]*""",
        r"""\s*[\r\n]+""",
        r"""\s+(?!\S)""",
        r"""\s+""",
    ]
)

_ENDOFTEXT_TOKEN = "<|endoftext|>"
_FIM_PREFIX_TOKEN = "<|fim_prefix|>"
_FIM_MIDDLE_TOKEN = "<|fim_middle|>"
_FIM_SUFFIX_TOKEN = "<|fim_suffix|>"
_ENDOFPROMPT_TOKEN = "<|endofprompt|>"


async def _read_file(blobpath: str) -> bytes:
    if not blobpath.startswith("http://") and not blobpath.startswith("https://"):
        import blobfile

        with blobfile.BlobFile(blobpath, "rb") as f:
            return f.read()

    import aiohttp

    async with aiohttp.ClientSession() as session:
        async with session.get(blobpath) as resp:
            resp.raise_for_status()
            return await resp.content.read()


def _check_hash(data: bytes, expected_hash: str) -> bool:
    actual_hash = hashlib.sha256(data).hexdigest()
    return actual_hash == expected_hash


_CACHE_DIR: str | None = None


async def _read_file_cached_internal(cache_dir: str, blobpath: str, expected_hash: str | None = None) -> bytes:
    if cache_dir == "":
        return await _read_file(blobpath)

    cache_key = hashlib.sha1(blobpath.encode()).hexdigest()
    cache_path = os.path.join(cache_dir, cache_key)
    if os.path.exists(cache_path):
        import aiofiles

        async with aiofiles.open(cache_path, "rb") as f:
            data = await f.read()
        if expected_hash is None or _check_hash(data, expected_hash):
            return data

        # The cached file does not match the hash, remove it and re-fetch
        try:
            os.remove(cache_path)
        except OSError:
            pass

    contents = await _read_file(blobpath)
    if expected_hash and not _check_hash(contents, expected_hash):
        raise ValueError(
            f"Hash mismatch for data downloaded from {blobpath} (expected {expected_hash}) -"
            f"this may indicate a corrupted download. Please try again"
        )

    import uuid

    try:
        os.makedirs(cache_dir, exist_ok=True)
        tmp_filename = cache_path + "." + str(uuid.uuid4()) + ".tmp"
        import aiofiles

        async with aiofiles.open(tmp_filename, "wb") as f:
            await f.write(contents)
        os.rename(tmp_filename, cache_path)
    except OSError:
        pass

    return contents


async def _read_file_cached(blobpath: str, expected_hash: str | None = None) -> bytes:
    global _CACHE_DIR

    if _CACHE_DIR is None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_dir:
            _CACHE_DIR = os.path.join(tmp_dir, "microgpt-gpt-tokenizer-data-cache")

    contents = await _read_file_cached_internal(_CACHE_DIR, blobpath, expected_hash)
    return contents


async def _load_mergeable_ranks_from_tiktoken_bpe(bpe_file_path: str, bpe_hash: str | None = None) -> dict[bytes, int]:
    contents = await _read_file_cached(bpe_file_path, bpe_hash)
    ret = {}
    for line in contents.splitlines():
        if not line:
            continue
        try:
            token, rank = line.split()
            ret[base64.b64decode(token)] = int(rank)
        except Exception as e:
            raise ValueError(f"Error parsing line {line!r} in {bpe_file_path}") from e
    return ret


async def _load_mergeable_ranks_from_bpe_and_encoder_json(
    bpe_file_path: str,
    encoder_json_file_path: str,
    bpe_hash: str | None = None,
    encoder_json_hash: str | None = None,
) -> dict[bytes, int]:
    rank_to_intbyte = [b for b in range(2**8) if chr(b).isprintable() and chr(b) != " "]

    data_gym_byte_to_byte = {chr(b): b for b in rank_to_intbyte}
    n = 0
    for b in range(2**8):
        if b not in rank_to_intbyte:
            rank_to_intbyte.append(b)
            data_gym_byte_to_byte[chr(2**8 + n)] = b
            n += 1
    assert len(rank_to_intbyte) == 2**8

    # bpe_file_path contains the merges along with associated ranks
    bpe_contents = (await _read_file_cached(bpe_file_path, bpe_hash)).decode(encoding="utf-8")
    bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_contents.split("\n")[1:-1]]

    def decode_data_gym(value: str) -> bytes:
        return bytes(data_gym_byte_to_byte[b] for b in value)

    # Add the single byte tokens
    bpe_ranks = {bytes([b]): i for i, b in enumerate(rank_to_intbyte)}
    # Add the merged tokens
    n = len(bpe_ranks)
    for first, second in bpe_merges:
        bpe_ranks[decode_data_gym(first) + decode_data_gym(second)] = n
        n += 1

    import json

    # Check that the encoder file matches the merges file.
    # This sanity check is important since tiktoken assumes that ranks are ordered the same as merge
    # priority
    encoder_json = json.loads(await _read_file_cached(encoder_json_file_path, encoder_json_hash))
    encoder_json_loaded = {decode_data_gym(k): v for k, v in encoder_json.items()}
    # Drop these two special tokens if present, since they're not mergeable bpe tokens
    encoder_json_loaded.pop(b"<|endoftext|>", None)
    encoder_json_loaded.pop(b"<|startoftext|>", None)
    assert bpe_ranks == encoder_json_loaded

    return bpe_ranks


@dataclass
class _GPTTokenizerConfig:
    vocab_size: int | None
    split_pattern: str
    mergeable_ranks_fn: Callable[[], Coroutine[Any, Any, dict[bytes, int]]]
    special_tokens: dict[str, int]
    eot_id: int | None = None


class _GPTTokenizerEncoding(enum.StrEnum):
    GPT_2 = "gpt-2"
    R50K_BASE = "r50k_base"
    P50K_BASE = "p50k_base"
    P50K_EDIT = "p50k_edit"
    CL100K_BASE = "cl100k_base"
    O200K_BASE = "o200k_base"

    def is_valid(self) -> bool:
        return self in {
            self.GPT_2,
            self.R50K_BASE,
            self.P50K_BASE,
            self.P50K_EDIT,
            self.CL100K_BASE,
            self.O200K_BASE,
        }

    def config(self) -> _GPTTokenizerConfig:
        if self == _GPTTokenizerEncoding.GPT_2:
            return _GPTTokenizerConfig(
                vocab_size=50257,
                split_pattern=_R50K_SPLIT_PATTERN,
                mergeable_ranks_fn=lambda: _load_mergeable_ranks_from_bpe_and_encoder_json(
                    bpe_file_path="https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/vocab.bpe",
                    encoder_json_file_path="https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/encoder.json",
                    bpe_hash="1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5",
                    encoder_json_hash="196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783",
                ),
                special_tokens={
                    _ENDOFTEXT_TOKEN: 50256,
                },
                eot_id=50256,
            )
        elif self == _GPTTokenizerEncoding.R50K_BASE:
            return _GPTTokenizerConfig(
                vocab_size=50257,
                split_pattern=_R50K_SPLIT_PATTERN,
                mergeable_ranks_fn=lambda: _load_mergeable_ranks_from_tiktoken_bpe(
                    bpe_file_path="https://openaipublic.blob.core.windows.net/encodings/r50k_base.tiktoken",
                    bpe_hash="306cd27f03c1a714eca7108e03d66b7dc042abe8c258b44c199a7ed9838dd930",
                ),
                special_tokens={
                    _ENDOFTEXT_TOKEN: 50256,
                },
                eot_id=50256,
            )
        elif self == _GPTTokenizerEncoding.P50K_BASE:
            return _GPTTokenizerConfig(
                vocab_size=50281,
                split_pattern=_R50K_SPLIT_PATTERN,
                mergeable_ranks_fn=lambda: _load_mergeable_ranks_from_tiktoken_bpe(
                    bpe_file_path="https://openaipublic.blob.core.windows.net/encodings/p50k_base.tiktoken",
                    bpe_hash="94b5ca7dff4d00767bc256fdd1b27e5b17361d7b8a5f968547f9f23eb70d2069",
                ),
                special_tokens={
                    _ENDOFTEXT_TOKEN: 50256,
                },
                eot_id=50256,
            )
        elif self == _GPTTokenizerEncoding.P50K_EDIT:
            return _GPTTokenizerConfig(
                vocab_size=None,
                split_pattern=_R50K_SPLIT_PATTERN,
                mergeable_ranks_fn=lambda: _load_mergeable_ranks_from_tiktoken_bpe(
                    bpe_file_path="https://openaipublic.blob.core.windows.net/encodings/p50k_base.tiktoken",
                    bpe_hash="94b5ca7dff4d00767bc256fdd1b27e5b17361d7b8a5f968547f9f23eb70d2069",
                ),
                special_tokens={
                    _ENDOFTEXT_TOKEN: 50256,
                    _FIM_PREFIX_TOKEN: 50281,
                    _FIM_MIDDLE_TOKEN: 50282,
                    _FIM_SUFFIX_TOKEN: 50283,
                },
                eot_id=50256,
            )
        elif self == _GPTTokenizerEncoding.CL100K_BASE:
            return _GPTTokenizerConfig(
                vocab_size=None,
                split_pattern=_CL100K_SPLIT_PATTERN,
                mergeable_ranks_fn=lambda: _load_mergeable_ranks_from_tiktoken_bpe(
                    bpe_file_path="https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken",
                    bpe_hash="223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7",
                ),
                special_tokens={
                    _ENDOFTEXT_TOKEN: 100257,
                    _FIM_PREFIX_TOKEN: 100258,
                    _FIM_MIDDLE_TOKEN: 100259,
                    _FIM_SUFFIX_TOKEN: 100260,
                    _ENDOFPROMPT_TOKEN: 100276,
                },
                eot_id=100257,
            )
        elif self == _GPTTokenizerEncoding.O200K_BASE:
            return _GPTTokenizerConfig(
                vocab_size=None,
                split_pattern=_O200K_SPLIT_PATTERN,
                mergeable_ranks_fn=lambda: _load_mergeable_ranks_from_tiktoken_bpe(
                    bpe_file_path="https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken",
                    bpe_hash="446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d",
                ),
                special_tokens={
                    _ENDOFTEXT_TOKEN: 199999,
                    _ENDOFPROMPT_TOKEN: 200018,
                },
                eot_id=199999,
            )
        else:
            raise ValueError(f"Unsupported encoding: {self}")


_GPT_TOKENIZER_ENCODING_VALUES_DICT = {e.value: e for e in _GPTTokenizerEncoding}


_MODEL_PREFIX_TO_ENCODING_DICT: dict[str, _GPTTokenizerEncoding] = {
    "o1-": _GPTTokenizerEncoding.O200K_BASE,
    "o3-": _GPTTokenizerEncoding.O200K_BASE,
    # Chat
    "chatgpt-4o-": _GPTTokenizerEncoding.O200K_BASE,
    "gpt-4o-": _GPTTokenizerEncoding.O200K_BASE,
    "gpt-4-": _GPTTokenizerEncoding.CL100K_BASE,
    "gpt-3.5-turbo-": _GPTTokenizerEncoding.CL100K_BASE,
    "gpt-35-turbo-": _GPTTokenizerEncoding.CL100K_BASE,
    "gpt-3.5-": _GPTTokenizerEncoding.CL100K_BASE,
    "gpt-35-": _GPTTokenizerEncoding.CL100K_BASE,
    "gpt-2-": _GPTTokenizerEncoding.GPT_2,
    # Fine-tuned
    "ft:gpt-4o": _GPTTokenizerEncoding.O200K_BASE,
    "ft:gpt-4": _GPTTokenizerEncoding.CL100K_BASE,
    "ft:gpt-3.5-turbo": _GPTTokenizerEncoding.CL100K_BASE,
    "ft:gpt-3.5": _GPTTokenizerEncoding.CL100K_BASE,
    "ft:gpt-35-turbo": _GPTTokenizerEncoding.CL100K_BASE,
    "ft:gpt-2": _GPTTokenizerEncoding.GPT_2,
    "ft:davinci-002": _GPTTokenizerEncoding.CL100K_BASE,
    "ft:babbage-002": _GPTTokenizerEncoding.CL100K_BASE,
}

_MODEL_TO_ENCODING_DICT: dict[str, _GPTTokenizerEncoding] = {
    # Reasoning
    "o1": _GPTTokenizerEncoding.O200K_BASE,
    "o3": _GPTTokenizerEncoding.O200K_BASE,
    # Chat
    "gpt-4o": _GPTTokenizerEncoding.O200K_BASE,
    "gpt-4": _GPTTokenizerEncoding.CL100K_BASE,
    "gpt-3.5-turbo": _GPTTokenizerEncoding.CL100K_BASE,
    "gpt-3.5": _GPTTokenizerEncoding.CL100K_BASE,
    "gpt-35-turbo": _GPTTokenizerEncoding.CL100K_BASE,
    "gpt-2": _GPTTokenizerEncoding.GPT_2,
    # Base
    "davinci-002": _GPTTokenizerEncoding.CL100K_BASE,
    "babbage-002": _GPTTokenizerEncoding.CL100K_BASE,
    # Embeddings
    "text-embedding-ada-002": _GPTTokenizerEncoding.CL100K_BASE,
    "text-embedding-3-small": _GPTTokenizerEncoding.CL100K_BASE,
    "text-embedding-3-large": _GPTTokenizerEncoding.CL100K_BASE,
}


def _bpe(mergeable_ranks: dict[bytes, int], token: bytes, max_rank: int | None = None) -> list[bytes]:
    """
    Reconstruct the merge forest from the mergeable ranks.

    Args:
        mergeable_ranks: The mergeable ranks
        token: The token to reconstruct
        max_rank: The maximum rank to use

    Returns:
        A reconstructed merge forest
    """
    parts = [bytes([b]) for b in token]
    while True:
        min_idx = None
        min_rank = None
        for i, pair in enumerate(zip(parts[:-1], parts[1:], strict=False)):
            rank = mergeable_ranks.get(pair[0] + pair[1])
            if rank is not None and (min_rank is None or rank < min_rank):
                min_idx = i
                min_rank = rank
        if min_rank is None or (max_rank is not None and min_rank >= max_rank):
            break
        assert min_idx is not None
        parts = parts[:min_idx] + [parts[min_idx] + parts[min_idx + 1]] + parts[min_idx + 2 :]
    return parts


def _recover_merges(mergeable_ranks: dict[bytes, int]) -> list[list[int]]:
    """
    Recover the merges from the mergeable ranks. The mergeable ranks contains byte sequences in
    their merged state and in order of their merge.

    See: https://github.com/openai/tiktoken/issues/60
    See: https://github.com/karpathy/minbpe/issues/11#issuecomment-1950805306

    Args:
        mergeable_ranks: The mergeable ranks

    Returns:
        A list of merges
    """
    merges: list[list[int]] = []
    for token, rank in mergeable_ranks.items():
        if len(token) == 1:
            continue
        pair_bytes = tuple(_bpe(mergeable_ranks, token, max_rank=rank))
        assert len(pair_bytes) == 2
        # Recover the integer ranks of the pair
        ix0 = mergeable_ranks[pair_bytes[0]]
        ix1 = mergeable_ranks[pair_bytes[1]]
        merges.append([ix0, ix1, rank])

    return merges
