from typing import List, Optional, Union

from fastapi_camelcase import CamelModel
from sqlmodel import SQLModel

from api.schemas.architecture import MLPArchitectureCreate, MLPArchitectureRead


class ModelBase(SQLModel, CamelModel):
    name: str
    input_columns: List[int]
    output_columns: List[int]


class ModelCreate(ModelBase):
    dataset_id: int
    mlp_architecture: Optional[MLPArchitectureCreate]

    # TODO: add validation to ensure that there can only be one architecture type per model


class ModelUpdate(ModelBase):
    pass


class ModelRead(ModelBase):
    id: int
    dataset_id: int

    class Config:
        from_attributes = True


class ModelWithArchitecture(ModelRead):
    mlp_architecture: Optional[MLPArchitectureRead]
