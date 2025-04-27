from typing import List, Optional, Union

from sqlalchemy import JSON
from sqlmodel import Column, Field, Relationship, SQLModel


class ArchitectureBase(SQLModel):
    input_size: int
    output_size: int


class DatasetBase(SQLModel):
    name: str


class MLPArchitecture(ArchitectureBase, table=True):
    __tablename__ = "mlparchitecture"

    id: Optional[int] = Field(default=None, primary_key=True)
    activation: str
    layers: List[int] = Field(default=[], sa_column=Column(JSON))

    model: Optional["Model"] = Relationship(back_populates="mlp_architecture")


class Model(SQLModel, table=True):
    __tablename__ = "model"

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    dataset_id: Optional[int] = Field(default=None, foreign_key="dataset.id")

    mlp_architecture_id: Optional[int] = Field(
        default=None, foreign_key="mlparchitecture.id"
    )

    mlp_architecture: Optional[MLPArchitecture] = Relationship(back_populates="model")

    # TODO: add validation to ensure that there can only be one architecture type per model


class Dataset(DatasetBase, table=True):
    __tablename__ = "dataset"
    id: int = Field(default=None, primary_key=True)
