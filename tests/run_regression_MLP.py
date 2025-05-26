import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from datetime import datetime

import pandas as pd

from api.redis import set_training
from api.schemas.training import TrainingRead, TrainingStatusEnum
from trainer.trainer import train_regression_model

# chargement du dataset + adaptation de la config
df = pd.read_csv("tests/babies.csv")
input_dim = df.drop(columns=["bwt"]).shape[1]


training = TrainingRead(
    batch_size=16,
    max_epochs=1000,
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
    "csv_path": "tests/babies.csv",
    "separator": ",",
    "target_columns": [0],
    "model_arch": {
        "architecture": "MLP",
        "input_size": input_dim,
        "output_size": 1,
        "layers": [input_dim, 32, 16, 1],
        "activation": "relu",
    },
    "learning_rate": training.learning_rate,
    "epochs": training.max_epochs,
    "batch_size": training.batch_size,
    "fraction": 0.8,
    "cleaning": True,
    "seed": 42,
    "device": "cpu",
    "save_dir": "saved_models/babies_run",
}

train_regression_model(training.model_id, config)
