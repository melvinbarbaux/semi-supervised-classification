from abc import ABC, abstractmethod
import numpy as np

class BaseModel(ABC):
    """
    Interface unifiée pour tous les classifieurs.

    Méthodes à implémenter :
      - train(X_labeled, y_labeled, X_unlabeled=None) : entraîne le modèle
      - predict(X) : retourne les prédictions de labels
      - predict_proba(X) : retourne les probabilités associées
    """

    @abstractmethod
    def train(
        self,
        X_l: np.ndarray,
        y_l: np.ndarray,
        X_u: np.ndarray = None
    ) -> None:
        """
        Entraîne le modèle sur les données étiquetées (X_l, y_l).
        Optionnellement, peut utiliser des données non étiquetées X_u.
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Retourne un array de prédictions de forme (n_samples,).
        """
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Retourne un array de probabilités de forme (n_samples, n_classes).
        """
        pass