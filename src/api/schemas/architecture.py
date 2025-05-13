from enum import StrEnum
from typing import List

from fastapi_camelcase import CamelModel
from sqlmodel import Enum, Field, SQLModel


class ActivationEnum(StrEnum):
    relu = "relu"
    sigmoid = "sigmoid"
    tanh = "tanh"


class ArchitectureBase(SQLModel, CamelModel):
    activation: ActivationEnum


class MLPArchitectureCreate(ArchitectureBase):
    layers: List[int]


class ArchitectureRead(ArchitectureBase):
    id: int

    class Config:
        from_attributes = True


class MLPArchitectureRead(ArchitectureRead):
    activation: str
    layers: List[int]
