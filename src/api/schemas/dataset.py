from datetime import datetime
from enum import StrEnum

from fastapi_camelcase import CamelModel
from sqlmodel import SQLModel


class DatasetTypeEnum(StrEnum):
    """Supported dataset file formats."""
    csv = "csv"


class DatasetColumnTypeEnum(StrEnum):
    """Column data types for automatic analysis."""
    numeric = "numeric"        # Numerical data (int, float)
    categorical = "categorical"  # Text/categorical data
    image = "image"           # Future: image data support


class DatasetColumn(SQLModel, CamelModel):
    name: str
    type: DatasetColumnTypeEnum
    unique_values: int = 0
    classes: list[str] | None = None
    null_count: int = 0
    dataset_id: int | None = None


class DatasetBase(SQLModel, CamelModel):
    name: str


class DatasetCreate(DatasetBase):
    pass


class DatasetRead(DatasetBase):
    id: int
    columns: list[DatasetColumn]
    row_count: int
    created_at: datetime
    dataset_type: DatasetTypeEnum
    original_file_name: str
    is_draft: bool


class DatasetTransformation(SQLModel):
    pass
