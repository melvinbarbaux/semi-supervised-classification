# src/ssl_bench/datamodule/graph/base.py

from abc import ABC, abstractmethod
import numpy as np

class GraphBuilder(ABC):
    """
    Interface abstraite pour la construction de graphes à partir
    d'un ensemble de points X (numpy.ndarray, shape=(n_samples, n_features)).
    """

    @abstractmethod
    def fit(self, X: np.ndarray) -> None:
        """
        Construit la structure de graphe à partir de X.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Les données à partir desquelles construire le graphe.
        """
        pass

    @abstractmethod
    def adjacency_matrix(self) -> np.ndarray:
        """
        Retourne la matrice d’adjacence du graphe construit.

        Returns
        -------
        A : np.ndarray of shape (n_samples, n_samples)
            Matrice d’adjacence (pondérée ou binaire) décrivant les liens du graphe.
        """
        pass