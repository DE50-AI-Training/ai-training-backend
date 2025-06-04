import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from datetime import datetime

import pandas as pd

from api.redis import set_training
from api.schemas.training import TrainingRead, TrainingStatusEnum
from trainer.trainer import train_classification_model

# chargement du dataset + adaptation de la config
df = pd.read_csv("tests/iris.csv")
df = pd.get_dummies(df.iloc[:, [0, 1, 2]])
input_dim = df.shape[1]
output_dim = 3


training = TrainingRead(
    batch_size=16,
    max_epochs=10,
    learning_rate=0.001,
    session_start=datetime.now(),
    training_time_at_start=0,
    epochs=0,
    status=TrainingStatusEnum.starting,
    model_id=1,  # Dummy model_id for testing
)

# Needs redis to be running
set_training(training)

config = {
    "csv_path": "tests/iris.csv",
    "separator": ",",
    "target_columns": [4],
    "input_columns": [0, 1,2,3],
    # "image_column": None,
    # "model_class": MLP,
    "model_arch": {
        "architecture": "MLP",
        "input_size": 4,
        "output_size": 3,
        "layers": [4, 32, 3],
        "activation": "relu",
    },
    "learning_rate": training.learning_rate,
    "epochs": training.max_epochs,
    "batch_size": training.batch_size,
    "fraction": 0.8,
    "cleaning": False,
    "seed": 42,
    "device": "cpu",
    "save_dir": "saved_models/iris_run",
}

train_classification_model(training.model_id, config)
