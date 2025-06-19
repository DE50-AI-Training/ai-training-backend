import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class ClassificationDataset(Dataset):
    """
    Dataset for classification tasks, compatible with PyTorch DataLoader.
    This dataset expects a DataFrame with features and target classes.
    Attributes:
        df (pd.DataFrame): DataFrame containing the dataset.
        class_names (list[str]): List of class names derived from the DataFrame.
        x (np.ndarray): Features extracted from the DataFrame.
        y (np.ndarray): Target classes extracted from the DataFrame.
    """

    def __init__(self, df: pd.DataFrame, class_names: dict[str, list[str]]) -> None:
        """
        Initializes the ClassificationDataset with a DataFrame and class names.
        :param df: DataFrame containing the dataset with features and target classes.
        :param class_names: Dictionary mapping class names to their corresponding columns in the DataFrame.
        """

        super().__init__()

        self.class_names = []
        for value_list in class_names.values():
            self.class_names.extend(value_list)

        self.x = df.drop(columns=self.class_names).to_numpy()
        self.y = df[self.class_names].to_numpy()

    def get_classnames(self) -> list[str]:
        """
        Returns the list of class names used in the dataset.
        :return: List of class names.
        """
        
        return self.class_names

    def get_x_y(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns the features and target classes as numpy arrays.
        :return: Tuple containing features (x) and target classes (y).
        """

        return self.x, self.y
        
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
        :return: Tuple containing the features (x) and target class (y) as PyTorch tensors.
        """

        x = torch.tensor(self.x[idx].astype(float), dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.float32)
        return x, y
