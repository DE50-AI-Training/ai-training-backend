import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))


import pandas as pd

from trainer.infer import infer_on_dataset

# chargement du dataset + adaptation de la config
df = pd.read_csv("tests/iris.csv")

from trainer.infer import InferConfig

# Simulation de l'inférence d'un modèle de classification MLP
raw_config = {
    "csv_path": "tests/iris.csv",
    "input_columns": [0, 1, 2, 3],
    "target_columns": [4],
    "classification": True,
    "classes": ["Setosa", "Versicolor", "Virginica"],
    "separator": ",",
    "model_arch": {
        "architecture": "MLP",
        "input_size": 4,
        "output_size": 3,
        "layers": [4, 32, 3],
        "activation": "relu",
    },
    "batch_size": 16,
    "cleaning": False,
    "seed": 42,
    "device": "cpu",
    "save_dir": "saved_models/iris_run5",
    "image_column": None,
}

infer_on_dataset(raw_config)
# The inference results will be saved in the specified directory.
