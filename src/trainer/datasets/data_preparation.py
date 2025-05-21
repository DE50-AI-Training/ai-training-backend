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

    def extract_cols(self, target_cols: list[int]) -> None:
        if not all(0 <= idx < len(self.df.columns) for idx in target_cols):
            raise ValueError("One or more indices in target_cols are out of range.")

        # Convert indices to column names
        self.target_cols = [self.df.columns[idx] for idx in target_cols]

        for col in self.target_cols:
            self.classes[col] = [
                str(x) if not isinstance(x, str) else x
                for x in self.df[col].unique().tolist()
            ]
        self.df = pd.get_dummies(
            self.df, columns=self.target_cols, prefix="", prefix_sep=""
        )

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
        if self.classes is None:
            raise ValueError("Classes are not initialized. Call extract_col() first.")
        return self.classes
