import json
import time
from datetime import datetime

from celery import Celery
from fastapi import APIRouter
from sqlmodel import select

from api.database import Dataset, MLPArchitecture, Model, SessionDep
from api.redis import (
    get_training,
    redis,
    set_training,
    trainer_stop,
    update_training_status,
)
from api.schemas.model import ModelCreate, ModelUpdate, ModelWithArchitecture
from api.schemas.training import TrainingRead, TrainingStart, TrainingStatusEnum
from config import settings

router = APIRouter(prefix="/models", tags=["models"])
celery_app = Celery("tasks", broker=settings.redis_url, backend=settings.redis_url)


@router.get("/")
async def get_models(session: SessionDep) -> list[ModelWithArchitecture]:
    statement = select(Model)
    models = session.exec(statement).all()

    return models


@router.get("/trainings")
async def get_trainings(session: SessionDep) -> list[TrainingRead]:
    trainings = []
    for key in redis.keys("training:*"):
        training = redis.get(key)
        if training:
            data = json.loads(training)
            data["model_id"] = int(key.split(":")[1])
            training_data = TrainingRead.model_validate(data)

            if training_data.status == TrainingStatusEnum.stopped:
                # TODO: Move this to another endpoint caled by trainer
                model = session.get(Model, training_data.model_id)
                if model:
                    model.last_batch_size = training_data.batch_size
                    model.last_max_epochs = training_data.max_epochs
                    model.last_learning_rate = training_data.learning_rate
                    model.training_time += int(
                        (datetime.now() - training_data.session_start).total_seconds()
                    )
                    model.epochs_trained += training_data.epochs
                    session.add(model)
                    session.commit()
                    session.refresh(model)

                redis.delete(key)
            else:
                trainings.append(training_data)
    return trainings


@router.get("/{model_id}")
async def get_model(model_id: int, session: SessionDep) -> ModelWithArchitecture:
    model = session.get(Model, model_id)
    if not model:
        raise ValueError(f"Model with id {model_id} not found.")
    return model


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
        input_columns=model.input_columns,
        output_columns=model.output_columns,
        mlp_architecture=mlp_architecture,
        last_batch_size=32,
        last_max_epochs=10,
        last_learning_rate=0.001,
    )

    session.add(db_model)
    session.commit()
    session.refresh(db_model)
    return db_model


@router.post("/{model_id}/train")
async def train_model(
    model_id: int, training_params: TrainingStart, session: SessionDep
) -> TrainingRead:
    if get_training(model_id):
        raise ValueError(f"Training with model_id {model_id} already exists.")
    model = session.get(Model, model_id)
    dataset = session.get(Dataset, model.dataset_id)

    training = TrainingRead(
        batch_size=training_params.batch_size,
        max_epochs=training_params.max_epochs,
        learning_rate=training_params.learning_rate,
        session_start=datetime.now(),
        training_time_at_start=model.training_time,
        epochs=0,
        status=TrainingStatusEnum.starting,
        model_id=model_id,
    )

    set_training(training)

    layers = model.mlp_architecture.layers
    config = {
        "csv_path": f"{settings.storage_path}/datasets/{model.dataset_id}.csv",
        "separator": dataset.delimiter,
        "target_columns": model.output_columns,
        "input_columns": model.input_columns,
        "model_arch": {
            "architecture": "MLP",
            "input_size": layers[0],
            "output_size": layers[-1],
            "layers": layers,
            "activation": model.mlp_architecture.activation,
        },
        "learning_rate": training.learning_rate,
        "epochs": training.max_epochs,
        "batch_size": training.batch_size,
        "fraction": 0.8,  # TODO: add fraction field to the model
        "cleaning": False,
        "seed": 42,
        "device": "cpu",
        "save_dir": f"{settings.storage_path}/models/{model_id}",
    }

    celery_app.send_task(
        "trainer.trainer.train_classification_model",
        kwargs={"model_id": model_id, "raw_config": config},
    )
    return training


@router.post("/{model_id}/stop")
async def stop_model(model_id: int) -> None:
    training = get_training(model_id)
    if training is None:
        raise ValueError(f"Training with model_id {model_id} not found. Cannot stop.")
    if (
        training.status == TrainingStatusEnum.stopped
        or training.status == TrainingStatusEnum.stopped
    ):
        return
    if training.status == TrainingStatusEnum.error:
        training.status = TrainingStatusEnum.stopped
        set_training(training)
        return

    trainer_stop(model_id)
    training.status = TrainingStatusEnum.stopping
    set_training(training)


@router.put("/{model_id}")
async def update_model(model_id: int, model: ModelUpdate) -> ModelWithArchitecture:
    pass


@router.delete("/{model_id}")
async def delete_model(model_id: int) -> None:
    pass
