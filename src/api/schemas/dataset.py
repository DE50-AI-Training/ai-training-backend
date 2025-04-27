from sqlmodel import SQLModel


class DatasetBase(SQLModel):
    name: str
    dataset_type: str
