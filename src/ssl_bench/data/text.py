"""
Module pour données textuelles, vectorisation via sklearn
"""
import numpy as np
from typing import Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from .base import DataModule

class TextDataModule(DataModule):
    """
    DataModule générique pour jeux de textes en répertoires ou fichiers CSV.
    Utilise un vectorizer sklearn.
    """
    def __init__(self,
                 texts: Optional[list] = None,
                 labels: Optional[list] = None,
                 vectorizer: Optional[TfidfVectorizer] = None,
                 from_csv: Optional[str] = None,
                 text_column: str = 'text',
                 label_column: str = 'label',
                 **kwargs):
        super().__init__(**kwargs)
        self.texts = texts
        self.labels = labels
        self.vectorizer = vectorizer or TfidfVectorizer(max_features=10000)
        self.from_csv = from_csv
        self.text_column = text_column
        self.label_column = label_column

    def load(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.from_csv:
            import pandas as pd
            df = pd.read_csv(self.from_csv)
            self.texts = df[self.text_column].tolist()
            self.labels = df[self.label_column].tolist()
        X = self.vectorizer.fit_transform(self.texts).toarray()
        y = np.array(self.labels)
        return X, y