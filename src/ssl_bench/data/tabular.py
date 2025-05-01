"""
Module pour données tabulaires, à partir de CSV ou DataFrame pandas
"""
import numpy as np
import pandas as pd
from typing import Optional, Tuple
from .base import DataModule

class TabularDataModule(DataModule):
    """
    DataModule générique pour jeux tabulaires.
    Lit un CSV et applique un pipeline pandas ou sklearn.
    """
    def __init__(self,
                 csv_path: Optional[str] = None,
                 df: Optional[pd.DataFrame] = None,
                 feature_columns: Optional[list] = None,
                 label_column: str = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.csv_path = csv_path
        self._df = df
        self.feature_columns = feature_columns
        self.label_column = label_column

    def load(self) -> Tuple[np.ndarray, np.ndarray]:
        if self._df is None and self.csv_path:
            self._df = pd.read_csv(self.csv_path)
        df = self._df.copy()
        X = df[self.feature_columns].values if self.feature_columns else df.drop(columns=[self.label_column]).values
        y = df[self.label_column].values
        return X, y