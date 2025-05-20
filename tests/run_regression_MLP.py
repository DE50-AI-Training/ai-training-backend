import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from trainer.trainer import train_regression_model 

import pandas as pd

# chargement du dataset + adaptation de la config
df = pd.read_csv("tests/babies.csv")
input_dim = df.drop(columns=["bwt"]).shape[1]

config = {
    "csv_path": "tests/babies.csv",
    "separator": ",",
    "target_column": "bwt",
    "model_arch": {
        "architecture": "MLP",
        "input_size": input_dim,
        "output_size": 1,
        "layers": [input_dim, 32, 16, 1],
        "activation": "relu"
    },
    "learning_rate": 0.01,
    "epochs": 50,
    "batch_size": 512,
    "fraction": 0.8,
    "cleaning": True,
    "seed": 42,
    "device": "cpu",
    "save_dir": "saved_models/babies_run"
}

train_regression_model(config)

