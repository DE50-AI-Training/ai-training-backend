from datetime import datetime
from enum import StrEnum
from typing import List, Optional, Union

from fastapi_camelcase import CamelModel
from sqlmodel import SQLModel


class TrainingStatusEnum(StrEnum):
    starting = "starting"
    training = "training"
    stopping = "stopping"
    stopped = "stopped"
    error = "error"


class TrainingBase(SQLModel, CamelModel):
    batch_size: int
    max_epochs: Optional[int] = None
    learning_rate: float


class TrainingStart(TrainingBase):
    pass


class TrainingRead(TrainingBase):
    epochs: int
    session_start: Optional[datetime]
    training_time_at_start: int
    status: TrainingStatusEnum
    model_id: int

    class Config:
        from_attributes = True
