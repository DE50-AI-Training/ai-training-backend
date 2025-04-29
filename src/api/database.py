from typing import Annotated, List, Optional, Union

from fastapi import Depends
from sqlalchemy import JSON
from sqlmodel import Column, Field, Relationship, Session, SQLModel, create_engine

from config import settings

# Create the database engine
connect_args = {"check_same_thread": False}
engine = create_engine(settings.database_url, connect_args=connect_args)


def get_session():
    with Session(engine) as session:
        yield session


SessionDep = Annotated[Session, Depends(get_session)]

# Define the database models


class ArchitectureBase(SQLModel):
    input_size: int
    output_size: int


class MLPArchitecture(ArchitectureBase, table=True):
    __tablename__ = "mlparchitecture"

    id: Optional[int] = Field(default=None, primary_key=True)
    activation: str
    layers: List[int] = Field(default=[], sa_column=Column(JSON))

    model: Optional["Model"] = Relationship(back_populates="mlp_architecture")


# Add here any other architecture types


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


class Dataset(SQLModel, table=True):
    __tablename__ = "dataset"
    id: int = Field(default=None, primary_key=True)
    name: str
