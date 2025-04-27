from typing import Optional, Union

from sqlmodel import SQLModel

from api.schemas.architecture import MLPArchitectureCreate, MLPArchitectureRead


class ModelCreate(SQLModel):
    name: str
    architecture: MLPArchitectureCreate


class ModelUpdate(SQLModel):
    name: Optional[str] = None


class ModelRead(SQLModel):
    id: int
    name: str

    class Config:
        from_attributes = True


class ModelWithArchitecture(ModelRead):
    architecture: Union[MLPArchitectureRead]
