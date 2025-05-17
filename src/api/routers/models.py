from fastapi import APIRouter
from sqlmodel import select

from api.database import MLPArchitecture, Model, SessionDep
from api.schemas.model import ModelCreate, ModelUpdate, ModelWithArchitecture

router = APIRouter(prefix="/models", tags=["models"])


@router.get("/")
async def get_models(session: SessionDep) -> list[ModelWithArchitecture]:
    statement = select(Model)
    models = session.exec(statement).all()
    return models


@router.post("/")
async def create_model(
    model: ModelCreate, session: SessionDep
) -> ModelWithArchitecture:
    mlp_architecture = None
    if model.mlp_architecture:
        mlp_architecture = MLPArchitecture(
            activation=model.mlp_architecture.activation,
            layers=model.mlp_architecture.layers,
        )
        session.add(mlp_architecture)
        session.flush()

    db_model = Model(
        name=model.name,
        dataset_id=model.dataset_id,
        problem_type=model.problem_type,
        mlp_architecture=mlp_architecture,
    )

    session.add(db_model)
    session.commit()
    session.refresh(db_model)
    return db_model


@router.put("/{model_id}")
async def update_model(model_id: int, model: ModelUpdate) -> ModelWithArchitecture:
    pass


@router.delete("/{model_id}")
async def delete_model(model_id: int) -> None:
    pass
