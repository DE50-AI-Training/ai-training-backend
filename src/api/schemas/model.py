from datetime import datetime
from enum import StrEnum
from typing import List, Optional, Union

from fastapi_camelcase import CamelModel
from sqlmodel import SQLModel

from api.schemas.architecture import MLPArchitectureCreate, MLPArchitectureRead
from api.schemas.training import TrainingRead


class ProblemTypeEnum(StrEnum):
    classification = "classification"
    regression = "regression"


class ModelBase(SQLModel, CamelModel):
    name: str
    input_columns: List[int]
    output_columns: List[int]
    problem_type: ProblemTypeEnum
    training_fraction: float


class ModelCreate(ModelBase):
    dataset_id: int
    mlp_architecture: Optional[MLPArchitectureCreate]

    # TODO: add validation to ensure that there can only be one architecture type per model


class ModelUpdate(SQLModel, CamelModel):
    pass


class ModelRead(ModelBase):
    id: int
    dataset_id: int
    created_at: datetime
    last_batch_size: int
    last_max_epochs: int
    last_learning_rate: float
    training_time: int
    epochs_trained: int

    class Config:
        from_attributes = True


class ModelWithArchitecture(ModelRead):
    mlp_architecture: Optional[MLPArchitectureRead] = None
