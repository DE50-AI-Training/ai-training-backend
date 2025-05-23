from datetime import datetime
from enum import StrEnum

from fastapi_camelcase import CamelModel
from sqlmodel import SQLModel


class DatasetTypeEnum(StrEnum):
    csv = "csv"


class DatasetBase(SQLModel, CamelModel):
    name: str


class DatasetCreate(DatasetBase):
    pass


class DatasetRead(DatasetBase):
    id: int
    columns: list[str]
    row_count: int
    unique_values_per_column: list[int]
    created_at: datetime
    dataset_type: DatasetTypeEnum
    original_file_name: str
    is_draft: bool


class DatasetTransformation(SQLModel):
    pass
