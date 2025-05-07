# src/ssl_bench/datamodule/graph/knn.py

import numpy as np
from sklearn.neighbors import NearestNeighbors

from .base import GraphBuilder

class KNNGraph(GraphBuilder):
    """
    Construction d'un graphe k-plus proches voisins (k-NN). 
    Peut renvoyer une matrice binaire (connectivity) ou pondérée (distance).
    """
    def __init__(
        self,
        n_neighbors: int = 5,
        mode: str = "connectivity",
        metric: str = "euclidean"
    ):
        """
        Parameters
        ----------
        n_neighbors : int
            Nombre de voisins (k) à considérer par point.
        mode : str
            "connectivity" pour une matrice binaire,
            "distance" pour une matrice pondérée par similarité.
        metric : str
            Métrique utilisée pour calculer les distances.
        """
        self.n_neighbors = n_neighbors
        self.mode = mode
        self.metric = metric
        self._adjacency = None

    def fit(self, X: np.ndarray) -> None:
        """
        Fit du graphe k-NN sur les données X.
        """
        # On inclut le point lui-même pour faciliter l'exclusion ultérieure
        nn = NearestNeighbors(
            n_neighbors=self.n_neighbors + 1,
            metric=self.metric
        )
        nn.fit(X)
        distances, indices = nn.kneighbors(X)

        n_samples = X.shape[0]
        A = np.zeros((n_samples, n_samples), dtype=float)

        # Exclure le point lui-même (indice 0)
        neigh_idx = indices[:, 1:]
        neigh_dist = distances[:, 1:]

        if self.mode == "connectivity":
            # Matrice binaire
            for i in range(n_samples):
                A[i, neigh_idx[i]] = 1.0
        elif self.mode == "distance":
            # Poids = exp(-d^2 / (2 * sigma^2)) où sigma = moyenne des distances
            sigma = np.mean(neigh_dist)
            weights = np.exp(- (neigh_dist ** 2) / (2 * sigma ** 2 + 1e-12))
            for i in range(n_samples):
                A[i, neigh_idx[i]] = weights[i]
        else:
            raise ValueError(f"Mode inconnu '{self.mode}', choisir 'connectivity' ou 'distance'.")

        # Graphe non orienté (symétriser)
        self._adjacency = np.maximum(A, A.T)

    def adjacency_matrix(self) -> np.ndarray:
        """
        Retourne la matrice d'adjacence du graphe.
        """
        if self._adjacency is None:
            raise ValueError("Vous devez d'abord appeler fit(X) avant adjacency_matrix().")
        return self._adjacency
