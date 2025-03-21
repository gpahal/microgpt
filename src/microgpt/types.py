from typing import Annotated, Literal

from pydantic import BaseModel, Field


class StrTextSource(BaseModel):
    type: Literal["str"] = "str"
    name: str
    text: str


class FileTextSource(BaseModel):
    type: Literal["file"] = "file"
    name: str
    file_path: str


class UrlTextSource(BaseModel):
    type: Literal["url"] = "url"
    name: str
    url: str


class DatasetParams(BaseModel):
    path: str
    subset: str | None = None
    split: str
    field: str = "text"


class DatasetTextSource(BaseModel):
    type: Literal["dataset"] = "dataset"
    name: str
    params: DatasetParams
    batch_size: int = 1000
    max_length: int = 0


TextSource = Annotated[
    StrTextSource | FileTextSource | UrlTextSource | DatasetTextSource,
    Field(discriminator="type"),
]
