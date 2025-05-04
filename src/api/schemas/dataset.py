from fastapi_camelcase import CamelModel
from sqlmodel import Field, SQLModel


class DatasetBase(SQLModel, CamelModel):
    name: str


class DatasetCreate(DatasetBase):
    pass


class DatasetRead(DatasetBase):
    id: int
    dataset_type: str = Field(alias="datasetType")


class DatasetTransformation(SQLModel):
    pass
