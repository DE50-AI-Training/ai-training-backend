from typing import List

from sqlmodel import Field, SQLModel
from fastapi_camelcase import CamelModel


class ArchitectureBase(SQLModel, CamelModel):
    input_size: int
    output_size: int


class MLPArchitectureCreate(ArchitectureBase):
    activation: str
    layers: List[int]


class ArchitectureRead(ArchitectureBase):
    id: int

    class Config:
        from_attributes = True


class MLPArchitectureRead(ArchitectureRead):
    activation: str
    layers: List[int]
