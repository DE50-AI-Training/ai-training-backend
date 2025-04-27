from typing import List

from sqlmodel import SQLModel


class ArchitectureBase(SQLModel):
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
