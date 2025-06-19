import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from celery import Celery
from pydantic import Field
from sqlmodel import SQLModel
from torch.utils.data import DataLoader, Dataset

from config import settings
from trainer.datasets.data_preparation import DataPreparation
from trainer.trainer import create_model, load_model_from_path


class InferConfig(SQLModel):
    """
    Configuration for inference tasks.
    This class defines the parameters required for running inference on a dataset,
    including the path to the dataset, input and target columns, model architecture,
    batch size, and other settings.
    """

    csv_path: str
    input_columns: List[int]
    target_columns: List[int]
    classification: bool = False
    classes: Optional[List[str]] = None
    separator: str
    model_arch: Dict[str, Any]
    batch_size: int
    cleaning: bool = False
    seed: int = 42
    device: str = "cpu"
    save_dir: str = "saved_models"
    image_column: Optional[str] = None


class InferenceDataset(Dataset):
    """
    Dataset for inference tasks, compatible with PyTorch DataLoader.
    This dataset expects a NumPy array of features and normalizes them if necessary.
    """

    def __init__(self, data: np.ndarray, dtype: torch.dtype = torch.float32):
        """
        Initializes the InferenceDataset with a NumPy array of features.
        :param data: NumPy array containing the dataset features.
        :param dtype: Data type for the tensors, default is torch.float32.
        :raises
            ValueError: If the input data is empty or not a NumPy array.
        """

        self.data = data
        self.dtype = dtype

    def normalize(self) -> None:
        """
        Normalize the dataset features (x): only normalize columns that are not boolean.
        A column is considered boolean if it contains exactly 2 unique values.
        Same logic as RegressionDataset but with fixed axis handling.
        """
        # Check each column to see if it's boolean-like (has exactly 2 unique values)
        for col in range(self.data.shape[1]):
            unique_values = np.unique(self.data[:, col])            
            # If the column has more than 2 unique values, normalize it
            if len(unique_values) > 2:
                col_mean = np.mean(self.data[:, col])
                col_std = np.std(self.data[:, col])
                
                # Avoid division by zero
                if col_std > 0:
                    self.data[:, col] = (self.data[:, col] - col_mean) / col_std


    def __len__(self):
        """
        Returns the number of samples in the dataset, required by PyTorch DataLoader.
        :return: Number of samples in the dataset.
        """

        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns a single sample from the dataset at the specified index, required by PyTorch DataLoader.
        :param idx: Index of the sample to retrieve.
        :return: Sample at the specified index as a PyTorch tensor.
        """

        return torch.tensor(self.data[idx], dtype=self.dtype)


def infer_on_dataset(raw_config: dict):
    """
    Perform inference on a dataset using a pre-trained model.
    :param raw_config: Dictionary containing configuration parameters for inference.
    :raises ValueError: If the configuration is invalid or if the dataset cannot be loaded.
    """

    config = InferConfig(**raw_config)
    archi_info = raw_config["model_arch"]

    data_prep = DataPreparation(
        config.csv_path,
        fraction=1.0,
        cleaning=True,
        seed=config.seed,
    )

    data_prep.read_data(sep=config.separator)
    data_prep.split()
    dataset, _ = data_prep.get_train_test()
    original_data = dataset.copy()

    # Sélection des colonnes d'entrée
    dataset = dataset[[dataset.columns[idx] for idx in config.input_columns]]
    dataset = dataset.to_numpy()

    dataset = InferenceDataset(dataset)
    print(dataset[0])  # Print first item for debugging
    if not config.classification:
        # For regression, normalize the input
        dataset.normalize()
    print(dataset[0])  # Print first item for debugging


    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)

    model = create_model(archi_info)
    device = torch.device(config.device)  # Could check if device is available
    model.to(device)

    model = load_model_from_path(config.save_dir, model)
    print(f"Model loaded from {config.save_dir}/model.pt")

    model = create_model(archi_info)
    device = torch.device(config.device)  # Could check if device is available
    model.to(device)

    model = load_model_from_path(config.save_dir, model)

    results = []
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch.float()
            outputs = model(inputs)
            results.append(outputs.cpu().numpy())

    print(
        f"Inference completed. Saving results to {config.save_dir}/inference_results.csv"
    )

    all_results = np.vstack(results)

    results_df = original_data.copy()

    if config.classification:
        if all_results.shape[1] > 1:
            predicted_indices = np.argmax(all_results, axis=1)

            # Déduire les classes si non fournies
            class_names = config.classes

            predicted_classes = [class_names[idx] for idx in predicted_indices]
            results_df["predicted_class"] = predicted_classes

            # Comparaison avec la vérité terrain si disponible
            if config.target_columns:
                target_idx = config.target_columns[0]
                target_col = original_data.columns[target_idx]
                true_labels = results_df[target_col].astype(str).tolist()
                acc = np.mean([p == t for p, t in zip(predicted_classes, true_labels)])
                print(
                    f"Inference accuracy: {acc * 100:.2f}% (based on column '{target_col}')"
                )
        else:
            results_df["prediction"] = all_results.flatten()
    else:
        for i in range(all_results.shape[1]):
            results_df[f"prediction_{i}"] = all_results[:, i]

    os.makedirs(config.save_dir, exist_ok=True)

    results_df.to_csv(f"{config.save_dir}/inference_results.csv", index=False)


def infer_single_input(raw_config: dict, input_data: List[float]):
    """
    Perform inference on a single input using a pre-trained model.
    :param raw_config: Dictionary containing configuration parameters for inference.
    :param input_data: List of input features for the inference.
    :raises ValueError: If the configuration is invalid or if the input data is not compatible with the model.
    """
    
    config = InferConfig(**raw_config)
    archi_info = raw_config["model_arch"]

    # Convert input data to tensor
    input_tensor = torch.tensor([input_data], dtype=torch.float32)
    #Todo: apply a normalization if needed
    if not config.classification:
        # For regression, normalize the input
        input_tensor = (input_tensor - input_tensor.mean(dim=0)) / input_tensor.std(dim=0)

    # Load model
    model = create_model(archi_info)
    device = torch.device(config.device)
    model.to(device)
    model = load_model_from_path(config.save_dir, model)

    model.eval()

    # Perform inference
    with torch.no_grad():
        input_tensor = input_tensor.to(device)
        output = model(input_tensor)
        result = output.cpu().numpy()[0]  # Get first (and only) result

    # Process result based on problem type
    if config.classification:
        predicted_index = np.argmax(result)

        # Get class names
        class_names = config.classes
        predicted_class = class_names[predicted_index]

        return {
            "prediction": predicted_class,
            "confidence": float(result[predicted_index]),
        }
    else:
        return {"prediction": float(result[0])}
