from datetime import datetime
from typing import List, Optional, Union

from fastapi_camelcase import CamelModel
from sqlmodel import SQLModel


class TrainingBase(SQLModel, CamelModel):
    batch_size: int
    max_epochs: Optional[int] = None
    learning_rate: float


class TrainingStart(TrainingBase):
    pass


class TrainingRead(TrainingBase):
    total_epochs: int
    session_start: datetime
    training_time_at_start: int

    class Config:
        from_attributes = True
