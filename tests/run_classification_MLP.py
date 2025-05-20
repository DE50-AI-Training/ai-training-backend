import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from trainer.trainer import train_classification_model 

import pandas as pd

# chargement du dataset + adaptation de la config
df = pd.read_csv("tests/iris.csv")
input_dim = df.drop(columns=["variety"]).shape[1]
output_dim = df["variety"].nunique()

config = {
    "csv_path": "tests/iris.csv",
    "separator": ",",
    "target_column": "variety",
    # "image_column": None,
    # "model_class": MLP,
    "model_arch": {
        "architecture": "MLP",
        "input_size": input_dim,
        "output_size": output_dim,
        "layers": [input_dim, 32, output_dim],
        "activation": "relu"
    },
    "learning_rate": 0.001,
    "epochs": 10,
    "batch_size": 16,
    "fraction": 0.8,
    "cleaning": False,
    "seed": 42,
    "device": "cpu",
    "save_dir": "saved_models/iris_run"
}

train_classification_model(config)