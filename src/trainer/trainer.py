import json
import os
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.optim as optim
from celery import Celery
from pydantic import Field, field_validator
from sqlmodel import SQLModel
from torch.utils.data import DataLoader

from api.redis import (
    get_training,
    redis,
    set_training,
    trainer_should_stop,
    update_training_status,
)
from api.schemas.training import TrainingRead, TrainingStatusEnum
from config import settings
from trainer.architectures import Architecture, MLPArchitecture
from trainer.datasets.classification_dataset import ClassificationDataset
from trainer.datasets.data_preparation import DataPreparation
from trainer.datasets.regression_dataset import RegressionDataset
from trainer.models import MLP, Model

app = Celery("tasks", broker=settings.redis_url, backend=settings.redis_url)


class Trainer:
    def __init__(
        self,
        training: TrainingRead,
        model: Model,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        self.training = training
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        should_stop = False

    def train_loop(self, dataloader: DataLoader, device: torch.device) -> None:
        self.model.to(device)
        self.model.train()
        for batch, (X, y) in enumerate(dataloader):
            # Compute prediction and loss
            X = X.to(device)
            y = y.to(device)

            pred = self.model(X)
            loss = self.loss_fn(pred, y)

            # Backpropagation
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

    def test_loop(self, dataloader: DataLoader, device: torch.device):
        self.model.to(device)
        self.model.eval()
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        test_loss, correct = 0, 0

        with torch.no_grad():
            for X, y in dataloader:
                X = X.to(device)
                y = y.to(device)

                pred = self.model(X)
                test_loss += self.loss_fn(pred, y).item()

                pred_class = torch.argmax(pred, dim=1)
                true_class = torch.argmax(y, dim=1)
                correct += (pred_class == true_class).type(torch.float).sum().item()

        test_loss /= num_batches
        correct /= size
        print(
            f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}\n"
        )
        self.training.score = correct

    def train(
        self,
        batch_size: int,
        train_dl: DataLoader,
        test_dl: DataLoader,
        epochs: int,
        device: torch.device,
    ) -> None:
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            self.train_loop(train_dl, device)
            self.test_loop(test_dl, device)

            self.training.epochs += 1
            set_training(self.training)

            if trainer_should_stop(self.training.model_id):
                print("Stopping training...")
                break

        print("Done!")

    def save(self, path: str) -> None:
        self.model.save(path)


class TrainerRegression(Trainer):
    def __init__(
        self, model: Model, loss_fn: nn.Module, optimizer: torch.optim.Optimizer
    ) -> None:
        super().__init__(model, loss_fn, optimizer)

    def train_loop(self, dataloader: DataLoader, device: torch.device) -> None:
        self.model.train()
        total_loss = 0

        for X, y in dataloader:
            # Move data to device
            X = X.to(device)
            y = y.to(device)

            # Forward pass
            pred = self.model(X)
            loss = self.loss_fn(pred, y)

            # Backpropagation
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            # Update statistics
            total_loss += loss.item() * len(X)

    def test_loop(self, dataloader: DataLoader, device: torch.device):
        self.model.eval()
        total_loss = 0
        num_batches = len(dataloader)
        size = len(dataloader.dataset)
        avg_abs_error = 0

        with torch.no_grad():
            for X, y in dataloader:
                X = X.to(device)
                y = y.to(device)

                pred = self.model(X)
                total_loss += self.loss_fn(pred, y).item()
                avg_abs_error += torch.abs(pred - y).sum().item()

        avg_loss = total_loss / num_batches
        avg_abs_error /= size
        # print(f"AvgLoss: {avg_loss:>8f}, AvgAbsError: {avg_abs_error:.3f}%")
        self.training.score = avg_abs_error


def create_model(arch_dict: dict) -> Model:
    if "architecture" not in arch_dict:
        raise ValueError("Architecture type is required")
    if arch_dict["architecture"] == "MLP":
        return MLP(MLPArchitecture(arch_dict))
    else:
        raise ValueError(
            f"Unsupported architecture: {arch_dict['architecture']}. Supported architectures are Currently MLP."
        )


class TrainClassificationConfig(SQLModel):
    csv_path: str
    target_columns: List[int]
    separator: str
    model_arch: Dict[str, Any]
    learning_rate: float
    epochs: Optional[int]
    batch_size: int
    fraction: float = Field(ge=0.0, le=1.0)
    cleaning: bool = False
    seed: int = 42
    device: str = "cpu"
    save_dir: str = "saved_models"
    image_column: Optional[str] = None


@app.task()
def train_classification_model(model_id: int, raw_config: dict):
    # --- read parameters ---
    try:
        config = TrainClassificationConfig(**raw_config)
        archi_info = raw_config["model_arch"]
        training = get_training(model_id)
        if training is None:
            raise ValueError(f"Training with model_id {model_id} not found.")

        # --- data prep ---
        data_prep = DataPreparation(
            config.csv_path,
            fraction=config.fraction,
            cleaning=config.cleaning,
            seed=config.seed,
        )
        data_prep.read_data(sep=config.separator)
        data_prep.extract_cols(config.target_columns)
        data_prep.split()
        train_set, test_set = data_prep.get_train_test()
        classes = data_prep.get_classes()

        train_ds = ClassificationDataset(train_set, classes)
        test_ds = ClassificationDataset(test_set, classes)

        train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=config.batch_size)

        # --- model handling---
        model = create_model(archi_info)
        device = torch.device(config.device)  # Could check if device is available
        model.to(device)

        # --- cost/opti ---
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

        # --- "real" training ---
        trainer = Trainer(training, model, criterion, optimizer)
        print("Ready to train")
        training.status = TrainingStatusEnum.training
        set_training(training)
        trainer.train(
            config.batch_size, train_loader, test_loader, config.epochs, device
        )
        training.status = TrainingStatusEnum.stopping
        set_training(training)
        # --- saving ---
        os.makedirs(config.save_dir, exist_ok=True)

        model_path = os.path.join(config.save_dir, "model.pt")
        torch.save(model.state_dict(), model_path)

        print(f"\nTraining complete. Model saved to: {model_path}")
        training.status = TrainingStatusEnum.stopped
        set_training(training)
    except Exception as e:
        print(f"Error during training: {e}")
        training.status = TrainingStatusEnum.error
        set_training(training)
        raise e


def train_regression_model(config: dict):
    csv_path = config["csv_path"]
    separator = config.get("separator", ",")
    target_columns = (
        config["target_column"]
        if isinstance(config["target_column"], list)
        else [config["target_column"]]
    )
    # model_class = config['model_class']
    archi_info = config["model_arch"]
    learning_rate = config.get("learning_rate", 0.001)
    epochs = config.get("epochs", 10)
    batch_size = config.get("batch_size", 32)
    test_fraction = config.get("fraction", 0.8)
    cleaning = config.get("cleaning", False)
    seed = config.get("seed", 42)
    device = config.get("device", "cpu")
    save_dir = config.get("save_dir", "saved_models")

    # --- data prep ---
    data_prep = DataPreparation(
        csv_path, fraction=test_fraction, cleaning=cleaning, seed=seed
    )
    data_prep.read_data(sep=separator)
    data_prep.split()
    train_set, test_set = data_prep.get_train_test()

    train_ds = RegressionDataset(train_set, target_cols=target_columns)
    test_ds = RegressionDataset(test_set, target_cols=target_columns)

    train_ds.normalize()
    test_ds.normalize()

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    # --- model handling---
    model = create_model(archi_info)
    device = torch.device(device)  # Could check if device is available
    model.to(device)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    trainer = TrainerRegression(model, loss_fn, optimizer)
    trainer.train(batch_size, train_loader, test_loader, epochs, device)
