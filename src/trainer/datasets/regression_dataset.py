import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class RegressionDataset(Dataset):
    def __init__(
        self, df: pd.DataFrame, target_cols: list[int], dtype=torch.float32
    ) -> None:
        super().__init__()
        self.x = df.drop(df.columns[target_cols], axis=1).to_numpy()
        self.y = df.iloc[:, target_cols].to_numpy()
        if np.isnan(self.x).any():
            col_means = np.nanmean(self.x, axis=0)
            inds = np.where(np.isnan(self.x))
            self.x[inds] = np.take(col_means, inds[1])
        self.dtype = dtype

    def get_x_y(self) -> tuple[np.ndarray, np.ndarray]:
        return self.x, self.y

    def normalize(self) -> None:
        """
        Normalize the dataset features (x): only normalize columns that are not boolean.
        A column is considered boolean if it contains exactly 2 unique values.
        """
        # Check each column to see if it's boolean-like (has exactly 2 unique values)
        for col in range(self.x.shape[1]):
            unique_values = np.unique(self.x[:, col])

            # If the column has more than 2 unique values, normalize it
            if len(unique_values) > 2:
                self.x[:, col] = (
                    self.x[:, col] - np.mean(self.x[:, col], axis=0)
                ) / np.std(self.x[:, col], axis=0)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.tensor(self.x[idx], dtype=self.dtype)
        y = torch.tensor(self.y[idx], dtype=self.dtype)
        return x, y
