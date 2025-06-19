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

# Database connection setup
connect_args = {"check_same_thread": False}
engine = create_engine(settings.database_url, connect_args=connect_args)


def get_session():
    """Database session dependency for FastAPI."""
    with Session(engine) as session:
        yield session


SessionDep = Annotated[Session, Depends(get_session)]

# Database Models


class ArchitectureBase(SQLModel):
    """Base class for neural network architectures."""
    activation: ActivationEnum = Field(
        default=None, sa_column=Column(Enum(ActivationEnum))
    )


class MLPArchitecture(ArchitectureBase, table=True):
    """Multi-Layer Perceptron architecture definition."""
    __tablename__ = "mlparchitecture"

    id: Optional[int] = Field(default=None, primary_key=True)
    activation: ActivationEnum = Field(default=None, nullable=False)
    layers: List[int] = Field(default=[], sa_column=Column(JSON))  # Layer sizes

    model: Optional["Model"] = Relationship(back_populates="mlp_architecture")


# Add here any other architecture types


class Model(SQLModel, table=True):
    """Machine learning model with training configuration and metrics."""
    __tablename__ = "model"

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(default=None, nullable=False)
    dataset_id: Optional[int] = Field(default=None, foreign_key="dataset.id")
    input_columns: List[int] = Field(default=None, sa_column=Column(JSON))  # Feature column indices
    output_columns: List[int] = Field(default=None, sa_column=Column(JSON))  # Target column indices
    problem_type: ProblemTypeEnum = Field(default=None, nullable=False)
    created_at: Optional[datetime] = Field(
        sa_column=Column(
            TIMESTAMP(timezone=True),
            nullable=False,
            server_default=text("CURRENT_TIMESTAMP"),
        )
    )
    training_fraction: float = Field(default=None, nullable=False)  # Data split ratio
    # Training hyperparameters (last used)
    last_batch_size: int = Field(default=None, nullable=False)
    last_max_epochs: int = Field(default=None, nullable=False)
    last_learning_rate: float = Field(default=None, nullable=False)
    # Training metrics
    training_time: int = Field(default=0, nullable=False)  # Cumulative seconds
    epochs_trained: int = Field(default=0, nullable=False)  # Total epochs completed

    mlp_architecture_id: Optional[int] = Field(
        default=None, foreign_key="mlparchitecture.id"
    )

    mlp_architecture: Optional[MLPArchitecture] = Relationship(back_populates="model")
    dataset: Optional["Dataset"] = Relationship(back_populates="models")


class DatasetColumn(SQLModel, table=True):
    """Column metadata and statistics for dataset analysis."""
    __tablename__ = "dataset_column"
    id: int = Field(default=None, primary_key=True)
    name: str = Field(default=None, nullable=False)
    type: DatasetColumnTypeEnum = Field(
        default=None, sa_column=Column(Enum(DatasetColumnTypeEnum))
    )
    unique_values: int = Field(default=0, nullable=False)
    classes: Optional[List[str]] = Field(  # Unique values for categorical columns
        default=None, sa_column=Column(JSON, nullable=True)
    )
    null_count: int = Field(default=0, nullable=False)
    dataset_id: int = Field(default=None, foreign_key="dataset.id")

    dataset: Optional["Dataset"] = Relationship(back_populates="columns")


class Dataset(SQLModel, table=True):
    """Dataset with file metadata and column analysis."""
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
    delimiter: str = Field(default=",", nullable=False)  # CSV delimiter
    is_draft: Optional[bool] = Field(default=True, nullable=False)  # Finalization status

    columns: List[DatasetColumn] = Relationship(
        back_populates="dataset", cascade_delete=True
    )
    models: List[Model] = Relationship(back_populates="dataset", cascade_delete=True)
