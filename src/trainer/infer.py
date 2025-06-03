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
    target_columns: List[int]
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
    config = TrainConfig(**raw_config)

    data_prep = DataPreparation(
        config.csv_path,
        fraction=1.0,
        cleaning=config.cleaning,
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

    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)
    model = load_model(config.model_arch_path)
    model.eval()
    print(f"Model loaded from {config.model_arch_path}")

    results = []
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch.float()
            outputs = model(inputs)
            results.append(outputs.cpu().numpy())

    print(f"Inference completed. Saving results to {config.save_dir}/inference_results.csv")

    all_results = np.vstack(results)

    results_df = original_data.copy()

    if config.classification:
        if all_results.shape[1] > 1:
            predicted_indices = np.argmax(all_results, axis=1)

            # Déduire les classes si non fournies
            if config.classes:
                class_names = config.classes
            else:
                target_col = original_data.columns[config.target_columns[0]]
                class_names = sorted(original_data[target_col].unique().astype(str).tolist())

            predicted_classes = [class_names[idx] for idx in predicted_indices]
            results_df["predicted_class"] = predicted_classes

            # Comparaison avec la vérité terrain si disponible
            if config.target_columns:
                target_idx = config.target_columns[0]
                target_col = original_data.columns[target_idx]
                true_labels = results_df[target_col].astype(str).tolist()
                acc = np.mean([p == t for p, t in zip(predicted_classes, true_labels)])
                print(f"Inference accuracy: {acc * 100:.2f}% (based on column '{target_col}')")
        else:
            results_df["prediction"] = all_results.flatten()
    else:
        for i in range(all_results.shape[1]):
            results_df[f"prediction_{i}"] = all_results[:, i]

    results_df.to_csv(f"{config.save_dir}/inference_results.csv", index=False)