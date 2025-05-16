from abc import ABC, abstractmethod
import numpy as np

class BaseModel(ABC):
    """
    Base interface for all models.
    """

    @abstractmethod
    def train(
        self,
        X_l: np.ndarray,
        y_l: np.ndarray,
        sample_weight: np.ndarray = None
    ) -> None:
        """
        Train on (X_l, y_l), optionally weighting each example.
        If sample_weight is None, all weights = 1.
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        pass