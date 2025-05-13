from fastapi_camelcase import CamelModel
from sqlmodel import Field, SQLModel


class DatasetBase(SQLModel, CamelModel):
    name: str


class DatasetCreate(DatasetBase):
    pass


class DatasetRead(DatasetBase):
    id: int
    columns: list[str]


class DatasetTransformation(SQLModel):
    pass
