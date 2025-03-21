from collections.abc import AsyncGenerator

from tqdm import tqdm

from .types import TextSource


async def _batch_read_text_source(text_source: TextSource) -> AsyncGenerator[str, None]:
    if text_source.type == "str":
        yield text_source.text
    elif text_source.type == "file":
        import aiofiles

        async with aiofiles.open(text_source.file_path, encoding="utf-8") as f:
            yield await f.read()
    elif text_source.type == "url":
        import aiohttp

        async with aiohttp.ClientSession() as session:
            async with session.get(text_source.url) as response:
                response.raise_for_status()
                yield await response.text()
    elif text_source.type == "dataset":
        import datasets

        params = text_source.params
        dataset = datasets.load_dataset(
            path=params.path,
            name=params.subset,
            split=params.split,
            streaming=True,
        )

        max_length: int | None = None
        dataset_info = dataset.info
        if (
            dataset_info
            and dataset_info.splits
            and params.split in dataset_info.splits
            and dataset_info.splits[params.split]
        ):
            max_length = dataset_info.splits[params.split].num_examples

        if text_source.max_length > 0 and (
            max_length is None or max_length > text_source.max_length
        ):
            max_length = text_source.max_length

        progress_bar = tqdm(
            total=max_length,
            unit="rows",
            desc=f"Processing dataset [{text_source.name}]",
        )
        batched_dataset = dataset.batch(text_source.batch_size)
        idx = 0
        for batch in batched_dataset:
            for text in batch[params.field]:
                yield text
                idx += 1
                progress_bar.update(1)
                if max_length is not None and idx >= max_length:
                    progress_bar.close()
                    return
        progress_bar.close()
    else:
        raise ValueError(f"Invalid text source type: {text_source.type}")


async def _batch_read_text_sources(
    text_sources: TextSource | list[TextSource],
) -> AsyncGenerator[str, None]:
    if not isinstance(text_sources, list):
        text_sources = [text_sources]
    for text_source in text_sources:
        async for text in _batch_read_text_source(text_source):
            yield text
