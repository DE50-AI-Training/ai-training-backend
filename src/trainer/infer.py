import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import json
from celery import Celery

from typing import List, Dict, Any, Optional
from pydantic import Field
from sqlmodel import SQLModel
from config import settings

from trainer.trainer import create_model
from trainer.datasets.data_preparation import DataPreparation

app = Celery("tasks", broker=settings.redis_url, backend=settings.redis_url)


class TrainConfig(SQLModel):
    csv_path: str
    input_columns: List[int]
    column_name: List[str]
    classification: bool = False
    classes: Optional[List[str]] = None
    separator: str
    model_arch_path: str
    batch_size: int
    cleaning: bool = False
    seed: int = 42
    device: str = "cpu"
    save_dir: str = "saved_models"
    image_column: Optional[str] = None

class InferenceDataset(Dataset):
    def __init__(self, data: np.ndarray, dtype: torch.dtype = torch.float32):
        self.data = data
        self.dtype = dtype

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=self.dtype)
    

def load_model(model_arch_path: str):
    with open(model_arch_path, 'r') as f:
        arch = json.load(f)
    model = create_model(arch)
    if 'model_weights_path' in arch:
        model.load_state_dict(torch.load(arch['model_weights_path'], map_location=torch.device('cpu')))
    else:
        raise ValueError("Model architecture JSON does not contain a valid key for model state.")
    return model


@app.task()
def infer_on_dataset(raw_config: dict):
    # try:
        config = TrainConfig(**raw_config)

        data_prep = DataPreparation(
            config.csv_path,
            fraction=1.0,
            cleaning=config.cleaning,
            seed=config.seed,
        )

        data_prep.read_data(sep=config.separator)
        # Assuming the column_name is a list of column names and exists in the DataFrame and are empty
        # data_prep.select_input_columns(config.input_columns, config.column_name)
        data_prep.split()
        dataset, _ = data_prep.get_train_test()
        original_data = dataset.copy()
        dataset = dataset[[dataset.columns[idx] for idx in config.input_columns]]  # Select input columns and target column
        dataset = dataset.to_numpy()

        dataset = InferenceDataset(dataset)

        dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)
        model = load_model(config.model_arch_path)
        model.eval()
        print(f"Model loaded from {config.model_arch_path}")

        results = []
        with torch.no_grad():
            for batch in dataloader:
                inputs = batch.float()  # Adjust based on input columns
                outputs = model(inputs)                
                results.append(outputs.cpu().numpy())

        print(f"Inference completed. Saving results to {config.save_dir}/inference_results.csv")

        # Combine all results
        all_results = np.vstack(results)
        
        # Handle classification if specified
        if config.classification and config.classes:
            # Convert model outputs to class predictions
            if all_results.shape[1] == len(config.classes):
                # Multi-class classification - use argmax to get predicted class indices
                predicted_indices = np.argmax(all_results, axis=1)
                predicted_classes = [config.classes[idx] for idx in predicted_indices]
                
                # Create results DataFrame with original data and predictions
                results_df = original_data.copy()
                results_df['predicted_class'] = predicted_classes
            else:
                # Binary classification - just use the raw outputs
                results_df = original_data.copy()
                results_df['prediction'] = all_results.flatten()
        else:
            # Regression case - save original data with predictions
            results_df = original_data.copy()
            for i in range(all_results.shape[1]):
                results_df[f'prediction_{i}'] = all_results[:, i]
        
        results_df.to_csv(f"{config.save_dir}/inference_results.csv", index=False)

    # except Exception as e:
    #     print(f"Error during Inference: {e}")
