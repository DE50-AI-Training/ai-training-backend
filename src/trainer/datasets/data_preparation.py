from typing import Union
import pandas as pd


class DataPreparation:
    def __init__(
        self,
        filename: str,
        fraction: float = 0.8,
        cleaning: bool = True,
        seed: int = 42,
    ):
        self.filename = filename
        self.fraction = fraction
        self.cleaning = cleaning
        self.seed = seed

        self.df = None
        self.classes = {}

        self.train_df = None
        self.test_df = None

    def read_data(self, sep=",") -> pd.DataFrame:
        if self.filename.endswith(".csv"):
            self.df = pd.read_csv(self.filename, sep=sep)
        elif self.filename.endswith(".xlsx"):
            self.df = pd.read_excel(self.filename)
        else:
            raise ValueError("Unsupported file format. Please use .csv or .xlsx")

        if self.cleaning:
            self.df = self.df.dropna()
            self.df = self.df.drop_duplicates()
        return self.df
    
    def select_input_columns(self, input_cols: list[int], target_cols: list[int] = []) -> None:
        if len(target_cols) > 1:
            raise ValueError("Only one target column is allowed for now")
        
        total_columns = len(self.df.columns)
        all_indices = set(input_cols + target_cols)

        if not all(0 <= idx < total_columns for idx in all_indices):
            raise ValueError(
                f"Invalid column indices: {sorted(all_indices)}. Total columns: {total_columns}"
            )

        self.input_cols = input_cols
        self.target_cols_indices = target_cols

    def extract_cols(self, target_cols: list[int] = None) -> None:
        if target_cols is None:
            target_cols = self.target_cols_indices

        total_columns = len(self.df.columns)
        if not all(0 <= idx < total_columns for idx in target_cols):
            raise ValueError(
                f"Invalid target column indices: {target_cols}. Total columns: {total_columns}"
            )

        # convertis les indices en noms de colonnes avant de modifier self.df
        target_col_names = [self.df.columns[idx] for idx in target_cols]
        input_col_names = [self.df.columns[idx] for idx in self.input_cols]

        selected_columns = list(dict.fromkeys(input_col_names + target_col_names))  # garde l'ordre
        self.df = self.df[selected_columns]
        self.target_cols = target_col_names

        for col in self.target_cols:
            self.classes[col] = sorted(self.df[col].astype(str).unique().tolist())

        self.df = pd.get_dummies(self.df, columns=self.target_cols, prefix="", prefix_sep="")

    def encode_categorical_inputs_as_dummies(self, force_categorical_cols: list[int] = None) -> None:
        if self.df is None:
            raise ValueError("Data not loaded. Call read_data() first.")

        input_col_names = [self.df.columns[i] for i in self.input_cols]

        auto_categorical_cols = [
            col for col in input_col_names
            if self.df[col].dtype == object or self.df[col].dtype.name == "category"
        ]

        forced_cols = force_categorical_cols or []
        forced_col_names = [self.df.columns[self.input_cols[i]] for i in forced_cols if 0 <= i < len(self.input_cols)]

        all_categorical_cols = list(set(auto_categorical_cols + forced_col_names))

        self.df = pd.get_dummies(self.df, columns=all_categorical_cols, drop_first=False)

        new_input_col_names = [
            col for col in self.df.columns if any(col.startswith(base_col) for base_col in all_categorical_cols) or col in input_col_names
        ]
        self.input_cols = [self.df.columns.get_loc(col) for col in new_input_col_names]

    def split(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        if not (0 < self.fraction <= 1):
            raise ValueError("Fraction must be between 0 and 1")

        if self.df is None:
            raise ValueError("DataFrame is not initialized. Call read_data() first.")

        self.train_df = self.df.sample(frac=self.fraction, random_state=self.seed)
        self.test_df = self.df.drop(self.train_df.index)

        return self.train_df, self.test_df

    def get_train_test(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        if self.train_df is None or self.test_df is None:
            raise ValueError(
                "Train and test DataFrames are not initialized. Call split() first."
            )
        return self.train_df, self.test_df

    def get_classes(self) -> dict:
        if not self.classes:
            raise ValueError("Classes are not initialized. Call extract_col() first.")
        return self.classes
