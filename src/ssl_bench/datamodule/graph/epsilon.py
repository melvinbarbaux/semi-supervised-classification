# src/ssl_bench/datamodule/graph/epsilon.py

import numpy as np
from sklearn.metrics import pairwise_distances

from .base import GraphBuilder

class EpsilonGraph(GraphBuilder):
    """
    Construction d'un graphe d'ε-voisinage. 
    Chaque paire de points à distance ≤ eps est connectée.
    """
    def __init__(
        self,
        eps: float = 0.5,
        metric: str = "euclidean",
        mode: str = "connectivity"
    ):
        """
        Parameters
        ----------
        eps : float
            Seuil de distance pour créer une arête.
        metric : str
            Métrique de distance pour pairwise_distances.
        mode : str
            "connectivity" pour une matrice binaire (0/1),
            "distance" pour une matrice pondérée par (eps - d)/eps.
        """
        self.eps = eps
        self.metric = metric
        self.mode = mode
        self._adjacency = None

    def fit(self, X: np.ndarray) -> None:
        """
        Construit la matrice d'adjacence basée sur l'épsilon-voisinage.
        """
        # Calcul de toutes les distances paire-à-paire
        D = pairwise_distances(X, metric=self.metric)
        n_samples = D.shape[0]
        A = np.zeros((n_samples, n_samples), dtype=float)

        # Masque des paires à connecter
        mask = D <= self.eps
        if self.mode == "connectivity":
            A[mask] = 1.0
        elif self.mode == "distance":
            # Poids proportionnel à (eps - d) / eps
            W = np.clip((self.eps - D) / (self.eps + 1e-12), 0.0, 1.0)
            A = W * mask.astype(float)
        else:
            raise ValueError(f"Mode inconnu '{self.mode}', choisir 'connectivity' ou 'distance'.")

        # Pas de boucle sur soi-même
        np.fill_diagonal(A, 0.0)
        # Symétriser pour graphe non orienté
        self._adjacency = np.maximum(A, A.T)

    def adjacency_matrix(self) -> np.ndarray:
        """
        Retourne la matrice d’adjacence construite.

        Returns
        -------
        np.ndarray
            Matrice d'adjacence de forme (n_samples, n_samples).
        """
        if self._adjacency is None:
            raise ValueError("Appelez d'abord 'fit(X)' avant 'adjacency_matrix()'.")
        return self._adjacency
