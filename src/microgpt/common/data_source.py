from collections.abc import AsyncGenerator
from typing import Annotated, Literal, cast

from pydantic import BaseModel, Field


class TextDataSource(BaseModel):
    type: Literal["str"] = "str"
    name: str
    text: str

    def __str__(self) -> str:
        return f"TextDataSource(name={self.name}, text_len={len(self.text)})"

    def __repr__(self) -> str:
        return self.__str__()


class FileDataSource(BaseModel):
    type: Literal["file"] = "file"
    name: str
    file_path: str

    def __str__(self) -> str:
        return f"FileDataSource(name={self.name}, file_path={self.file_path})"

    def __repr__(self) -> str:
        return self.__str__()


class UrlDataSource(BaseModel):
    type: Literal["url"] = "url"
    name: str
    url: str

    def __str__(self) -> str:
        return f"UrlDataSource(name={self.name}, url={self.url})"

    def __repr__(self) -> str:
        return self.__str__()


class DatasetParams(BaseModel):
    path: str
    subset: str | None = None
    split: str
    field: str = "text"

    def __str__(self) -> str:
        return f"DatasetParams(path={self.path}, subset={self.subset}, split={self.split}, field={self.field})"

    def __repr__(self) -> str:
        return self.__str__()


class DatasetDataSource(BaseModel):
    type: Literal["dataset"] = "dataset"
    name: str
    params: DatasetParams
    batch_size: int = 1000
    max_length: int = 0

    def __str__(self) -> str:
        return f"""DatasetDataSource(
            name={self.name},
            params={self.params},
            batch_size={self.batch_size},
            max_length={self.max_length}
        )"""

    def __repr__(self) -> str:
        return self.__str__()


DataSource = Annotated[
    TextDataSource | FileDataSource | UrlDataSource | DatasetDataSource,
    Field(discriminator="type"),
]


async def _batch_read_data_source(
    data_source: DataSource, shuffle_seed: int | None = None, show_progress: bool = True
) -> AsyncGenerator[str, None]:
    if data_source.type == "str":
        yield data_source.text
    elif data_source.type == "file":
        import aiofiles

        async with aiofiles.open(data_source.file_path, encoding="utf-8") as f:
            yield await f.read()
    elif data_source.type == "url":
        import aiohttp

        async with aiohttp.ClientSession() as session:
            async with session.get(data_source.url) as response:
                response.raise_for_status()
                yield await response.text()
    elif data_source.type == "dataset":
        import datasets  # type: ignore
        from tqdm import tqdm

        params = data_source.params
        dataset = datasets.load_dataset(
            path=params.path,
            name=params.subset,
            split=params.split,
            streaming=True,
        )
        if shuffle_seed is not None:
            dataset = dataset.shuffle(seed=shuffle_seed, buffer_size=data_source.batch_size * 10)

        max_length: int | None = None
        dataset_info = dataset.info
        if (
            dataset_info
            and dataset_info.splits
            and params.split in dataset_info.splits
            and dataset_info.splits[params.split]
        ):
            max_length = cast(int, dataset_info.splits[params.split].num_examples)

        if data_source.max_length > 0 and (max_length is None or max_length > data_source.max_length):
            max_length = data_source.max_length

        if show_progress:
            progress_bar = tqdm(
                total=max_length,
                unit="rows",
                desc=f"Processing dataset [{data_source.name}]",
            )
        batched_dataset = dataset.batch(data_source.batch_size)
        i = 0
        for batch in batched_dataset:
            for text in batch[params.field]:
                yield text
                i += 1
                if show_progress:
                    progress_bar.update(1)
                if max_length is not None and i >= max_length:
                    if show_progress:
                        progress_bar.close()
                    return
        if show_progress:
            progress_bar.close()
    else:
        raise ValueError(f"Invalid text source type: {data_source.type}")


async def _batch_read_data_sources(
    data_sources: DataSource | list[DataSource],
    shuffle_seed: int | None = None,
    show_progress: bool = True,
) -> AsyncGenerator[str, None]:
    if not isinstance(data_sources, list):
        data_sources = [data_sources]
    for data_source in data_sources:
        async for text in _batch_read_data_source(data_source, shuffle_seed, show_progress):
            yield text
