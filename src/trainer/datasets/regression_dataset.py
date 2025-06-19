import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class RegressionDataset(Dataset):
    """
    Dataset for regression tasks, compatible with PyTorch DataLoader.
    This dataset expects a DataFrame with features and target values.
    Attributes:
        x (np.ndarray): Features extracted from the DataFrame.
        y (np.ndarray): Target values extracted from the DataFrame.
        dtype (torch.dtype): Data type for the tensors, default is torch.float32.
    """

    def __init__(
        self, df: pd.DataFrame, target_cols: list[int], dtype=torch.float32
    ) -> None:
        """
        Initializes the RegressionDataset with a DataFrame and target columns.
        :param df: DataFrame containing the dataset with features and target values.
        :param target_cols: List of column indices in the DataFrame that represent the target values.
        :param
            dtype: Data type for the tensors, default is torch.float32.
        :raises ValueError: If more than one target column is specified.
        """

        super().__init__()

        self.x = df.drop(df.columns[target_cols], axis=1).to_numpy()
        self.y = df.iloc[:, target_cols].to_numpy()

        if np.isnan(self.x).any():
            col_means = np.nanmean(self.x, axis=0)
            inds = np.where(np.isnan(self.x))
            self.x[inds] = np.take(col_means, inds[1])
        
        self.dtype = dtype

    def get_x_y(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns the features and target values as numpy arrays.
        :return: Tuple containing features (x) and target values (y).
        """
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
        """
        Returns the number of samples in the dataset, required by PyTorch DataLoader.
        :return: Number of samples in the dataset.
        """

        return len(self.x)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns a single sample from the dataset at the specified index, required by PyTorch DataLoader.
        :param idx: Index of the sample to retrieve.
        :return: Tuple containing the features (x) and target value (y) as PyTorch tensors.
        """
        
        x = torch.tensor(self.x[idx], dtype=self.dtype)
        y = torch.tensor(self.y[idx], dtype=self.dtype)
        return x, y
