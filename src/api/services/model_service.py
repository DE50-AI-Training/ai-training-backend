import json
import os
import shutil
from datetime import datetime
from typing import List

from celery import Celery
from fastapi import HTTPException
from fastapi.responses import FileResponse
from sqlmodel import Session, select

from api.database import Dataset, MLPArchitecture, Model
from api.redis import get_training, redis, set_training, trainer_stop
from api.schemas.architecture import MLPArchitectureExport
from api.schemas.model import (
    InferConfig,
    ModelCreate,
    ModelUpdate,
    ModelWithArchitecture,
    ProblemTypeEnum,
)
from api.schemas.training import TrainingRead, TrainingStart, TrainingStatusEnum
from config import settings
from trainer.infer import infer_on_dataset, infer_single_input


class ModelService:
    """Service layer for model management"""

    def __init__(self, celery_app: Celery):
        self.celery_app = celery_app

    def get_all_models(self, session: Session) -> List[ModelWithArchitecture]:
        """Get all models in the system"""
        statement = select(Model)
        models = session.exec(statement).all()
        return models

    def get_trainings(self, session: Session) -> List[TrainingRead]:
        """Get all active training sessions"""
        trainings = []
        for key in redis.keys("training:*"):
            training = redis.get(key)
            if training:
                data = json.loads(training)
                data["model_id"] = int(key.split(":")[1])
                training_data = TrainingRead.model_validate(data)

                if training_data.status == TrainingStatusEnum.stopped:
                    # TODO: Move this to another endpoint called by trainer
                    model = session.get(Model, training_data.model_id)
                    if model:
                        model.last_batch_size = training_data.batch_size
                        model.last_max_epochs = training_data.max_epochs
                        model.last_learning_rate = training_data.learning_rate
                        model.training_time += int(
                            (
                                datetime.now() - training_data.session_start
                            ).total_seconds()
                        )
                        model.epochs_trained += training_data.epochs
                        session.add(model)
                        session.commit()
                        session.refresh(model)

                    redis.delete(key)
                else:
                    trainings.append(training_data)
        return trainings

    def get_model(self, model_id: int, session: Session) -> ModelWithArchitecture:
        """Get a specific model by ID"""
        model = session.get(Model, model_id)
        if not model:
            raise HTTPException(
                status_code=404, detail=f"Model with id {model_id} not found."
            )
        return model

    def get_model_weights(self, model_id: int, session: Session) -> FileResponse:
        """Download model weights file"""
        model = session.get(Model, model_id)
        if not model:
            raise HTTPException(
                status_code=404, detail=f"Model with id {model_id} not found."
            )

        weights_path = f"{settings.storage_path}/models/{model_id}/model.pt"
        return FileResponse(
            path=weights_path,
            media_type="application/octet-stream",
            filename="model.pt",
        )

    def get_model_architecture(
        self, model_id: int, session: Session
    ) -> MLPArchitectureExport:
        """Get model architecture details"""
        model = session.get(Model, model_id)
        if not model:
            raise HTTPException(
                status_code=404, detail=f"Model with id {model_id} not found."
            )

        architecture = model.mlp_architecture
        if not architecture:
            raise HTTPException(
                status_code=404,
                detail=f"Model with id {model_id} has no architecture defined.",
            )

        # Filter dataset columns to only include those used in the model
        input_columns = [model.dataset.columns[i].name for i in model.input_columns]
        output_columns = [model.dataset.columns[i].name for i in model.output_columns]

        response = {
            "activation": architecture.activation,
            "layers": architecture.layers,
            "input_columns": input_columns,
            "output_columns": output_columns,
        }
        return response

    def create_model(
        self, model: ModelCreate, session: Session
    ) -> ModelWithArchitecture:
        """Create a new machine learning model"""
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
            training_fraction=model.training_fraction,
            last_batch_size=32,
            last_max_epochs=10,
            last_learning_rate=0.001,
        )

        session.add(db_model)
        session.commit()
        session.refresh(db_model)
        return db_model

    def train_model(
        self, model_id: int, training_params: TrainingStart, session: Session
    ) -> TrainingRead:
        """Start model training in the background"""
        if get_training(model_id):
            raise HTTPException(
                status_code=400,
                detail=f"Training with model_id {model_id} already exists.",
            )

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
            "csv_path": f"{settings.storage_path}/datasets/{model.dataset_id}/dataset.csv",
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
            "fraction": model.training_fraction,
            "cleaning": model.problem_type == ProblemTypeEnum.regression,
            "seed": 42,
            "device": "cpu",
            "save_dir": f"{settings.storage_path}/models/{model_id}",
        }

        if model.problem_type == ProblemTypeEnum.classification:
            self.celery_app.send_task(
                "trainer.trainer.train_classification_model",
                kwargs={"model_id": model_id, "raw_config": config},
            )
        elif model.problem_type == ProblemTypeEnum.regression:
            self.celery_app.send_task(
                "trainer.trainer.train_regression_model",
                kwargs={"model_id": model_id, "raw_config": config},
            )
        else:
            raise HTTPException(
                status_code=422,
                detail=f"Unsupported problem type: {model.problem_type}",
            )

        return training

    def stop_model(self, model_id: int) -> None:
        """Stop an ongoing training session"""
        training = get_training(model_id)
        if training is None:
            raise HTTPException(
                status_code=404,
                detail=f"Training with model_id {model_id} not found. Cannot stop.",
            )

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

    def infer_model(
        self, model_id: int, session: Session, params: InferConfig
    ) -> FileResponse:
        """Generate predictions on a complete dataset"""
        model = session.get(Model, model_id)
        dataset = session.get(Dataset, params.dataset_id)

        if not model:
            raise HTTPException(
                status_code=404, detail=f"Model with id {model_id} not found."
            )
        if not dataset:
            raise HTTPException(
                status_code=404, detail=f"Dataset with id {model.dataset_id} not found."
            )

        layers = model.mlp_architecture.layers
        classes = (
            dataset.columns[model.output_columns[0]].classes
            if model.problem_type == ProblemTypeEnum.classification
            else None
        )

        config = {
            "csv_path": f"{settings.storage_path}/datasets/{params.dataset_id}/dataset.csv",
            "separator": dataset.delimiter,
            "target_columns": model.output_columns,
            "input_columns": model.input_columns,
            "classification": model.problem_type == ProblemTypeEnum.classification,
            "classes": classes,
            "batch_size": params.batch_size,
            "model_arch": {
                "architecture": "MLP",
                "input_size": layers[0],
                "output_size": layers[-1],
                "layers": layers,
                "activation": model.mlp_architecture.activation,
            },
            "cleaning": False,
            "seed": 42,
            "device": "cpu",
            "save_dir": f"{settings.storage_path}/models/{model_id}",
        }

        infer_on_dataset(raw_config=config)

        response_file = (
            f"{settings.storage_path}/models/{model_id}/inference_results.csv"
        )
        if not os.path.exists(response_file):
            raise HTTPException(
                status_code=404,
                detail=f"Inference results for model {model_id} not found.",
            )

        return FileResponse(
            path=response_file, media_type="text/csv", filename="inference_results.csv"
        )

    def infer_single_model(
        self, model_id: int, input_data: List[float], session: Session
    ) -> dict:
        """Instant prediction on a single observation"""
        model = session.get(Model, model_id)
        if not model:
            raise HTTPException(
                status_code=404, detail=f"Model with id {model_id} not found."
            )

        layers = model.mlp_architecture.layers
        classes = (
            model.dataset.columns[model.output_columns[0]].classes
            if model.problem_type == ProblemTypeEnum.classification
            else None
        )

        config = {
            "csv_path": "",  # Not needed for single input
            "separator": "",  # Not needed for single input
            "target_columns": model.output_columns,
            "input_columns": model.input_columns,
            "classification": model.problem_type == ProblemTypeEnum.classification,
            "classes": classes,
            "batch_size": 1,
            "model_arch": {
                "architecture": "MLP",
                "input_size": layers[0],
                "output_size": layers[-1],
                "layers": layers,
                "activation": model.mlp_architecture.activation,
            },
            "cleaning": False,
            "seed": 42,
            "device": "cpu",
            "save_dir": f"{settings.storage_path}/models/{model_id}",
        }

        try:
            result = infer_single_input(raw_config=config, input_data=input_data)
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

    def update_model(self, model_id: int, model: ModelUpdate) -> ModelWithArchitecture:
        """Update an existing model (feature under development)"""
        # TODO: Implement model update logic
        pass

    def delete_model(self, model_id: int, session: Session) -> None:
        """Delete a model and its files"""
        model = session.get(Model, model_id)

        if not model:
            raise HTTPException(
                status_code=404, detail=f"Model with id {model_id} not found."
            )

        # Delete the model's weights file
        model_path = f"{settings.storage_path}/models/{model_id}"
        if os.path.exists(model_path):
            shutil.rmtree(model_path)

        # Delete the model from the database
        session.delete(model)
        session.commit()
        redis.delete(f"training:{model_id}")
        redis.delete(f"stop_signal:{model_id}")
