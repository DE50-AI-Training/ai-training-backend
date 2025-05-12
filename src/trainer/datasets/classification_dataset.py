import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class ClassificationDataset(Dataset):
    def __init__(self, df: pd.DataFrame, class_names: dict[str, list[str]]) -> None:
        super().__init__()
        self.class_names = []
        for value_list in class_names.values():
            self.class_names.extend(value_list)

        self.x = df.drop(columns=self.class_names).to_numpy()
        self.y = df[self.class_names].to_numpy()

    def get_classnames(self) -> list[str]:
        return self.class_names

    def get_x_y(self) -> tuple[np.ndarray, np.ndarray]:
        return self.x, self.y
        
    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.tensor(self.x[idx], dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.float32)
        return x, y
