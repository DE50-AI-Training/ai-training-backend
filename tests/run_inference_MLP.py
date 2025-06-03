import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))


import pandas as pd

from trainer.infer import infer_on_dataset  

# chargement du dataset + adaptation de la config
df = pd.read_csv("tests/iris.csv")

config = {
    "csv_path": "tests/iris.csv",
    "input_columns": [0, 1, 2, 3],
    "target_columns": [4],
    "classification": True,
    "classes": None,  # optionnel maintenant
    "separator": ",",
    "model_arch_path": "saved_models/iris_run5/model.json",
    "batch_size": 16,
    "cleaning": False,
    "seed": 42,
    "device": "cpu",
    "save_dir": "saved_models/iris_run5",
    "image_column": None
}

infer_on_dataset(config)
# The inference results will be saved in the specified directory.