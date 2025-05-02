# src/ssl_bench/data/transformer.py

import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

class DataTransformer:
    """
    Pipeline de transformation :
      - normalisation (StandardScaler)
      - encodage des labels (OneHotEncoder)
      - split train/test
    """
    def __init__(
        self,
        scaler: StandardScaler = None,
        encoder: OneHotEncoder = None,
        test_size: float = 0.2,
        seed: int = None
    ):
        self.scaler = scaler or StandardScaler()
        # Utilise `sparse_output=False` pour sklearn >= 1.6
        self.encoder = encoder or OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.test_size = test_size
        self.seed = seed

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Fit scaler + encoder, puis transforme X et y.
        Retourne (X_scaled, y_encoded) :
          - X_scaled : array shape (n_samples, n_features)
          - y_encoded: array shape (n_samples, n_classes)
        """
        X_scaled = self.scaler.fit_transform(X)
        y_enc = self.encoder.fit_transform(y.reshape(-1, 1))
        return X_scaled, y_enc

    def transform(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Transform X et y avec un scaler/encoder déjà entraînés.
        """
        X_scaled = self.scaler.transform(X)
        y_enc = self.encoder.transform(y.reshape(-1, 1))
        return X_scaled, y_enc

    def split(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split X, y en (X_train, X_test, y_train, y_test) selon test_size et seed.
        """
        return train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.seed
        )