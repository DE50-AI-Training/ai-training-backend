import json
import os
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.optim as optim
from celery import Celery
from pydantic import Field
from sqlmodel import SQLModel
from torch.utils.data import DataLoader

from api.redis import get_training, set_training, trainer_should_stop
from api.schemas.training import TrainingRead, TrainingStatusEnum
from config import settings
from trainer.architectures import Architecture, MLPArchitecture
from trainer.datasets.classification_dataset import ClassificationDataset
from trainer.datasets.data_preparation import DataPreparation
from trainer.datasets.regression_dataset import RegressionDataset
from trainer.models import MLP, Model

app = Celery("tasks", broker=settings.redis_url, backend=settings.redis_url)


class Trainer:
    """
    Base class for training neural network models.
    This class provides methods for training and testing the model, handling
    the training loop, and saving the model.
    Attributes:
        training (TrainingRead): Training session metadata.
        model (Model): The neural network model to be trained.
        loss_fn (nn.Module): Loss function used for training.
        optimizer (torch.optim.Optimizer): Optimizer used for updating model weights.
    """

    def __init__(
        self,
        training: TrainingRead,
        model: Model,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        """
        Initializes the Trainer with training metadata, model, loss function, and optimizer.
        :param training: Training session metadata.
        :param model: The neural network model to be trained.
        :param loss_fn: Loss function used for training.
        :param optimizer: Optimizer used for updating model weights.
        """

        self.training = training
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        should_stop = False

    def train_loop(self, dataloader: DataLoader, device: torch.device) -> None:
        """
        Runs a single training loop over the provided DataLoader.
        :param dataloader: DataLoader providing batches of training data.
        :param device: Device (CPU or GPU) on which to perform training.
        This method iterates over the DataLoader, computes predictions, calculates loss,
        performs backpropagation, and updates the model weights.
        """

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
        """
        Runs a single test loop over the provided DataLoader.
        :param dataloader: DataLoader providing batches of test data.
        :param device: Device (CPU or GPU) on which to perform testing.
        This method evaluates the model on the test dataset, computes the average loss,
        and calculates the accuracy for classification tasks.
        It prints the test results, including accuracy and average loss.
        """

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
        self.training.accuracy = correct

    def train(
        self,
        batch_size: int,
        train_dl: DataLoader,
        test_dl: DataLoader,
        epochs: int,
        device: torch.device,
    ) -> None:
        """
        Runs the training loop for a specified number of epochs.
        :param batch_size: Size of the batches to be used in training.
        :param train_dl: DataLoader providing training data.
        :param test_dl: DataLoader providing test data.
        :param epochs: Number of epochs to train the model.
        :param device: Device (CPU or GPU) on which to perform training.
        This method iterates over the specified number of epochs, calling the training
        and testing loops for each epoch. It updates the training metadata after each epoch
        and checks if the training should stop based on external conditions.
        """

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
        """
        Saves the model to the specified path.
        :param path: Path where the model will be saved.
        This method saves the model's state dictionary to the specified path.
        It is expected that the model class has a `save` method implemented.
        """

        self.model.save(path)


class TrainerRegression(Trainer):
    """
    Trainer class for regression tasks.
    Inherits from the base Trainer class and implements specific training and testing loops
    for regression models.
    Attributes:
        training (TrainingRead): Training session metadata.
        model (Model): The neural network model to be trained.
        loss_fn (nn.Module): Loss function used for training.
        optimizer (torch.optim.Optimizer): Optimizer used for updating model weights.
    """

    def __init__(
        self,
        training: TrainingRead,
        model: Model,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        """
        Initializes the TrainerRegression with training metadata, model, loss function, and optimizer.
        :param training: Training session metadata.
        :param model: The neural network model to be trained.
        :param loss_fn: Loss function used for training.
        :param optimizer: Optimizer used for updating model weights.
        """

        super().__init__(training, model, loss_fn, optimizer)

    def train_loop(self, dataloader: DataLoader, device: torch.device) -> None:
        """
        Runs a single training loop over the provided DataLoader for regression tasks.
        :param dataloader: DataLoader providing batches of training data.
        :param device: Device (CPU or GPU) on which to perform training.
        This method iterates over the DataLoader, computes predictions, calculates loss,
        performs backpropagation, and updates the model weights.
        It also handles moving data to the specified device (CPU or GPU).
        """

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
        """
        Runs a single test loop over the provided DataLoader for regression tasks.
        :param dataloader: DataLoader providing batches of test data.
        :param device: Device (CPU or GPU) on which to perform testing.
        This method evaluates the model on the test dataset, computes the average loss,
        and calculates the average absolute error.
        It prints the test results, including average loss and average absolute error.
        """

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
        print(f"AvgLoss: {avg_loss:>8f}, AvgAbsError: {avg_abs_error:.3f}")
        self.training.avg_abs_error = avg_abs_error


def create_model(arch_dict: dict) -> Model:
    """
    Creates a model instance based on the provided architecture dictionary.
    :param arch_dict: Dictionary containing architecture information, including:
                      - architecture: Type of architecture (e.g., "MLP").
                      - input_size: Size of the input features.
                      - output_size: Number of output classes or size.
    :return: An instance of the Model class corresponding to the specified architecture.
    :raises ValueError: If the architecture type is not supported or if required fields are missing.
    """

    if "architecture" not in arch_dict:
        raise ValueError("Architecture type is required")
    if arch_dict["architecture"] == "MLP":
        return MLP(MLPArchitecture(arch_dict))
    else:
        raise ValueError(
            f"Unsupported architecture: {arch_dict['architecture']}. Supported architectures are Currently MLP."
        )


def load_model_from_path(save_dir: str, model: Model) -> Model:
    """
    Loads a model from the specified path if it exists.
    :param save_dir: Directory where the model weights are saved.
    :param model: The model instance to load weights into.
    :return: The model instance with loaded weights.
    If the model weights file does not exist, it returns the model without loading any weights.
    """

    model_path = os.path.join(save_dir, "model.pt")
    if os.path.exists(model_path):
        print(f"Loaded weights from {model_path}")
        model.load(model_path)
    else:
        print("Starting training from scratch.")
    return model


class TrainConfig(SQLModel):
    """
    Configuration for training a model.
    This class defines the parameters required for training a model, including
    the path to the CSV file, target and input columns, model architecture,
    learning rate, number of epochs, batch size, and other settings.
    """

    csv_path: str
    target_columns: List[int]
    input_columns: List[int]
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
    """
    Task to train a classification model using the provided configuration.
    """

    # --- read parameters ---
    try:
        config = TrainConfig(**raw_config)
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
        data_prep.select_input_columns(config.input_columns, config.target_columns)
        data_prep.extract_cols(config.target_columns)
        data_prep.encode_categorical_inputs_as_dummies()

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

        model = load_model_from_path(config.save_dir, model)

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
        model.save(model_path)

        print(f"\nTraining complete. Model saved to: {model_path}")
        training.status = TrainingStatusEnum.stopped
        set_training(training)
    except Exception as e:
        print(f"Error during training: {e}")
        training.status = TrainingStatusEnum.error
        set_training(training)
        raise e


@app.task()
def train_regression_model(model_id: int, raw_config: dict):
    """
    Task to train a regression model using the provided configuration.
    """
    
    try:
        config = TrainConfig(**raw_config)
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
        data_prep.split()
        train_set, test_set = data_prep.get_train_test()

        train_ds = RegressionDataset(train_set, target_cols=config.target_columns)
        test_ds = RegressionDataset(test_set, target_cols=config.target_columns)

        train_ds.normalize()
        test_ds.normalize()

        train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=config.batch_size)

        # --- model handling---
        model = create_model(archi_info)
        device = torch.device(config.device)  # Could check if device is available
        model.to(device)

        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

        model = load_model_from_path(config.save_dir, model)

        # --- "real" training ---
        trainer = TrainerRegression(training, model, loss_fn, optimizer)
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
