from datetime import datetime
from typing import Annotated, List, Optional

from fastapi import Depends
from sqlalchemy import JSON
from sqlmodel import (
    TIMESTAMP,
    Column,
    Enum,
    Field,
    Relationship,
    Session,
    SQLModel,
    create_engine,
    text,
)

from api.schemas.architecture import ActivationEnum
from api.schemas.dataset import DatasetColumnTypeEnum, DatasetTypeEnum
from api.schemas.model import ProblemTypeEnum
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
    activation: ActivationEnum = Field(
        default=None, sa_column=Column(Enum(ActivationEnum))
    )


class MLPArchitecture(ArchitectureBase, table=True):
    __tablename__ = "mlparchitecture"

    id: Optional[int] = Field(default=None, primary_key=True)
    activation: ActivationEnum = Field(default=None, nullable=False)
    layers: List[int] = Field(default=[], sa_column=Column(JSON))

    model: Optional["Model"] = Relationship(back_populates="mlp_architecture")


# Add here any other architecture types


class Model(SQLModel, table=True):
    __tablename__ = "model"

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(default=None, nullable=False)
    dataset_id: Optional[int] = Field(default=None, foreign_key="dataset.id")
    input_columns: List[int] = Field(default=None, sa_column=Column(JSON))
    output_columns: List[int] = Field(default=None, sa_column=Column(JSON))
    problem_type: ProblemTypeEnum = Field(default=None, nullable=False)
    created_at: Optional[datetime] = Field(
        sa_column=Column(
            TIMESTAMP(timezone=True),
            nullable=False,
            server_default=text("CURRENT_TIMESTAMP"),
        )
    )
    training_fraction: float = Field(default=None, nullable=False)
    last_batch_size: int = Field(default=None, nullable=False)
    last_max_epochs: int = Field(default=None, nullable=False)
    last_learning_rate: float = Field(default=None, nullable=False)
    training_time: int = Field(default=0, nullable=False)
    epochs_trained: int = Field(default=0, nullable=False)

    mlp_architecture_id: Optional[int] = Field(
        default=None, foreign_key="mlparchitecture.id"
    )

    mlp_architecture: Optional[MLPArchitecture] = Relationship(back_populates="model")
    dataset: Optional["Dataset"] = Relationship(back_populates="models")


class DatasetColumn(SQLModel, table=True):
    __tablename__ = "dataset_column"
    id: int = Field(default=None, primary_key=True)
    name: str = Field(default=None, nullable=False)
    type: DatasetColumnTypeEnum = Field(
        default=None, sa_column=Column(Enum(DatasetColumnTypeEnum))
    )
    unique_values: int = Field(default=0, nullable=False)
    classes: Optional[List[str]] = Field(
        default=None, sa_column=Column(JSON, nullable=True)
    )
    null_count: int = Field(default=0, nullable=False)
    dataset_id: int = Field(
        default=None, foreign_key="dataset.id"
    )  # Relation avec Dataset

    dataset: Optional["Dataset"] = Relationship(back_populates="columns")


class Dataset(SQLModel, table=True):
    __tablename__ = "dataset"
    id: int = Field(default=None, primary_key=True)
    name: str = Field(default=None, nullable=False)
    row_count: int = Field(default=0, nullable=False)
    created_at: Optional[datetime] = Field(
        sa_column=Column(
            TIMESTAMP(timezone=True),
            nullable=False,
            server_default=text("CURRENT_TIMESTAMP"),
        )
    )
    dataset_type: DatasetTypeEnum = Field(default=None, nullable=False)
    original_file_name: str = Field(default=None, nullable=False)
    delimiter: str = Field(default=",", nullable=False)
    is_draft: Optional[bool] = Field(default=True, nullable=False)

    columns: List[DatasetColumn] = Relationship(
        back_populates="dataset", cascade_delete=True
    )
    models: List[Model] = Relationship(back_populates="dataset", cascade_delete=True)
