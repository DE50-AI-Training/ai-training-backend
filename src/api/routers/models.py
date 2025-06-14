import json
import os
import shutil
from datetime import datetime

from celery import Celery
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from sqlmodel import select

from api.database import Dataset, MLPArchitecture, Model, SessionDep
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
from api.services.model_service import ModelService
from config import settings
from trainer.infer import infer_on_dataset, infer_single_input

router = APIRouter(
    prefix="/models",
    tags=["Models & Training"],
)
celery_app = Celery("tasks", broker=settings.redis_url, backend=settings.redis_url)
model_service = ModelService(celery_app)


@router.get(
    "/",
    summary="Retrieve all models",
    description="Returns the complete list of all created models with their architectures and training metadata.",
    response_description="List of models with their MLP architectures",
)
async def get_models(session: SessionDep) -> list[ModelWithArchitecture]:
    """
    Retrieves all models in the system.

    **Included information:**
    - Model configuration (input/output columns, problem type)
    - MLP architecture (layers, activation)
    - Training history (time, epochs, hyperparameters)
    - Link to source dataset
    """
    return model_service.get_all_models(session)


@router.get(
    "/trainings",
    summary="Active training sessions",
    description="Retrieves all ongoing or recently completed training sessions, with their real-time metrics.",
    response_description="List of training sessions with statuses and metrics",
)
async def get_trainings(session: SessionDep) -> list[TrainingRead]:
    """
    Monitors ongoing training sessions.

    **Possible statuses:**
    - `starting`: Initialization in progress
    - `training`: Active training
    - `stopping`: Stop requested
    - `stopped`: Successfully completed
    - `error`: Error encountered

    **Available metrics:**
    - Number of completed epochs
    - Elapsed training time
    - Accuracy (classification) / MAE (regression)

    **Automatic cleanup:** Completed sessions are archived in the model
    """
    return model_service.get_trainings(session)


@router.get(
    "/{model_id}",
    summary="Model details",
    description="Retrieves complete information for a specific model, including its architecture and training statistics.",
    response_description="Complete model details with architecture",
)
async def get_model(model_id: int, session: SessionDep) -> ModelWithArchitecture:
    """
    Retrieves complete details for a model.

    **Detailed information:**
    - Complete configuration (dataset, columns, problem type)
    - MLP architecture (number of layers, sizes, activation function)
    - Complete training history
    - Latest hyperparameters used
    - Cumulative training time
    """
    return model_service.get_model(model_id, session)


@router.get(
    "/{model_id}/weights",
    summary="Download model weights",
    description="Downloads the trained PyTorch model weights file (.pt) for external use or backup.",
    response_description="PyTorch weights file (model.pt)",
    responses={
        200: {
            "description": "Weights file downloaded successfully",
            "content": {"application/octet-stream": {"example": "binary_pytorch_file"}},
        }
    },
)
async def get_model_weights(model_id: int, session: SessionDep) -> FileResponse:
    """
    Downloads the trained model weights in PyTorch format.

    **Format:** .pt file (PyTorch state_dict)
    **Usage:** Import into PyTorch, deployment, external backup

    **Prerequisites:** The model must have been trained at least once
    """
    return model_service.get_model_weights(model_id, session)


@router.get(
    "/{model_id}/architecture",
    summary="Model architecture",
    description="Retrieves the complete model architecture definition with input and output column names.",
    response_description="MLP architecture configuration with column names",
)
async def get_model_architecture(
    model_id: int, session: SessionDep
) -> MLPArchitectureExport:
    """
    Exports the model architecture in readable format.

    **Exported information:**
    - Layer structure (sizes)
    - Activation function used
    - Input column names (features)
    - Output column names (targets)

    **Format suitable** for model reconstruction or external integration
    """
    return model_service.get_model_architecture(model_id, session)


@router.post(
    "/",
    summary="Create a new model",
    description="Creates a new machine learning model with its associated MLP architecture, based on an existing dataset.",
    response_description="Created model with its architecture",
    status_code=201,
)
async def create_model(
    model: ModelCreate, session: SessionDep
) -> ModelWithArchitecture:
    """
    Creates a new machine learning model.

    **Required configuration:**
    - **Source dataset**: Must be finalized (not draft)
    - **Problem type**: Classification or regression
    - **Input columns**: Feature indices (explanatory variables)
    - **Output columns**: Target indices (variables to predict)

    **MLP Architecture:**
    - **Layers**: Array of sizes [input, hidden1, hidden2, ..., output]
    - **Activation**: relu, sigmoid, tanh, etc.

    **Default parameters:**
    - Training fraction: As specified
    - Batch size: 32
    - Max epochs: 10
    - Learning rate: 0.001
    """
    return model_service.create_model(model, session)


@router.post(
    "/{model_id}/train",
    summary="Start training",
    description="Starts an asynchronous training session for the model with specified hyperparameters.",
    response_description="Created and started training session",
    status_code=201,
)
async def train_model(
    model_id: int, training_params: TrainingStart, session: SessionDep
) -> TrainingRead:
    """
    Starts model training in the background.

    **Configurable hyperparameters:**
    - **Batch size**: Training batch size
    - **Max epochs**: Maximum number of epochs
    - **Learning rate**: Learning rate

    **Asynchronous process:**
    - Execution via Celery in background
    - Real-time monitoring via Redis
    - Automatically updated metrics

    **Training types:**
    - **Classification**: Crossentropy optimization + accuracy
    - **Regression**: MSE optimization + MAE

    **Prerequisites:** No ongoing training for this model
    """
    return model_service.train_model(model_id, training_params, session)


@router.post(
    "/{model_id}/stop",
    summary="Stop training",
    description="Requests graceful stop of an ongoing training session. The stop may take a few seconds.",
    response_description="Stop request sent",
)
async def stop_model(model_id: int) -> None:
    """
    Gracefully stops an ongoing training session.

    **Stop process:**
    1. Stop signal sent to Celery worker
    2. Save current weights
    3. Update model statistics
    4. Change status to 'stopping' then 'stopped'

    **Graceful stop:** Current epoch completes before stopping

    **Handled states:**
    - Active training → Stop requested
    - Error → Marked as stopped
    - Already stopped → No action
    """
    model_service.stop_model(model_id)


@router.post(
    "/{model_id}/infer",
    summary="Inference on a dataset",
    description="Performs predictions on a complete dataset and returns the results in CSV format.",
    response_description="CSV file with predictions",
    responses={
        200: {
            "description": "Predictions generated successfully",
            "content": {
                "text/csv": {"example": "input1,input2,prediction\n1.0,2.0,class_A\n"}
            },
        }
    },
)
async def infer_model(
    model_id: int, session: SessionDep, params: InferConfig
) -> FileResponse:
    """
    Generates predictions on a complete dataset.

    **Inference process:**
    - Loading trained weights
    - Dataset preparation (same format as training)
    - Batch predictions
    - CSV export with original columns + predictions

    **Output formats:**
    - **Classification**: Probabilities + predicted class
    - **Regression**: Predicted numerical values

    **Configuration:**
    - **Target dataset**: Can be different from training dataset
    - **Batch size**: Performance optimization

    **Prerequisites:** Trained model and compatible dataset
    """
    return model_service.infer_model(model_id, session, params)


@router.post(
    "/{model_id}/infer-single",
    summary="Inference on single input",
    description="Performs real-time prediction on a single observation provided as an array of numbers.",
    response_description="Prediction result with details",
)
async def infer_single_model(
    model_id: int, input_data: list[float], session: SessionDep
) -> dict:
    """
    Instant prediction on a single observation.

    **Input format:** Array of numbers corresponding to features
    ```json
    [1.5, 2.3, 0.8]  // Values for input columns
    ```

    **Output format:**
    - **Classification:**
    ```json
    {
      "prediction": [0.85, 0.15],    // Probabilities per class
      "predicted_class": "class_A"    // Class with max probability
    }
    ```

    - **Regression:**
    ```json
    {
      "prediction": [2.47]  // Predicted value
    }
    ```

    **Use cases:** Real-time API, quick tests, application integration
    """
    return model_service.infer_single_model(model_id, input_data, session)


@router.put(
    "/{model_id}",
    summary="Update a model",
    description="Modifies parameters of an existing model (feature under development).",
    response_description="Updated model",
)
async def update_model(model_id: int, model: ModelUpdate) -> ModelWithArchitecture:
    """
    Modifies parameters of an existing model.

    **🚧 Feature under development**

    **Planned modifications:**
    - Name change
    - Default hyperparameter modification
    - Training fraction adjustment
    - Configuration update

    **Limitations:** Architecture cannot be modified after creation
    """
    return model_service.update_model(model_id, model)


@router.delete(
    "/{model_id}",
    summary="Delete a model",
    description="Permanently deletes a model and all its associated files. This action is irreversible.",
    response_description="Deletion confirmed",
    status_code=204,
)
async def delete_model(model_id: int, session: SessionDep) -> None:
    """
    Permanently deletes a model and its data.

    **⚠️ WARNING: Irreversible action!**

    **Deleted elements:**
    - Weight files (.pt)
    - Inference results
    - Database metadata
    - Associated architecture
    - Complete storage folder

    **Automatic cleanup:**
    - Redis training sessions
    - Redis stop signals

    **Prerequisites:** Stop any ongoing training before deletion
    """
    model_service.delete_model(model_id, session)
