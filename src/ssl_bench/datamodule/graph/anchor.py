# src/ssl_bench/datamodule/graph/anchor.py

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

from .base import GraphBuilder

class AnchorGraph(GraphBuilder):
    """
    Construction d'un Anchor Graph (graphe basé sur des points d'ancrage).
    On sélectionne un ensemble de points d'ancrage via KMeans, puis on mesure
    la similarité RBF entre chaque point X et ces ancres, et enfin on construit
    la matrice d’adjacence A = S @ S^T (graphe complet pondéré).
    """
    def __init__(
        self,
        n_anchors: int = 50,
        sigma: float = None,
        metric: str = "euclidean",
        random_state: int = None
    ):
        """
        Parameters
        ----------
        n_anchors : int
            Nombre de points d'ancrage (ancres) à utiliser.
        sigma : float or None
            Écart-type pour calcul RBF; si None, estimé comme moyenne des distances.
        metric : str
            Métrique pour calculer les distances (pairwise_distances).
        random_state : int
            Graine pour reproductibilité (KMeans).
        """
        self.n_anchors = n_anchors
        self.sigma = sigma
        self.metric = metric
        self.random_state = random_state
        self._adjacency = None
        self._anchors = None

    def fit(self, X: np.ndarray) -> None:
        """
        1) Sélectionne les ancres via KMeans sur X.
        2) Calcule la matrice de distances D(X, anchors).
        3) Estime sigma si nécessaire.
        4) Calcule la similarité S_{i,j} = exp(-d^2/(2*sigma^2)).
        5) Construit A = S @ S^T et met la diagonale à zéro.
        """
        # 1) KMeans pour extraire les ancres
        kmeans = KMeans(
            n_clusters=self.n_anchors,
            random_state=self.random_state
        )
        kmeans.fit(X)
        anchors = kmeans.cluster_centers_
        self._anchors = anchors

        # 2) Distances X -> anchors
        D = pairwise_distances(X, anchors, metric=self.metric)

        # 3) Sigma automatique
        if self.sigma is None:
            self.sigma = np.mean(D)

        # 4) Similarité RBF
        S = np.exp(- (D ** 2) / (2 * self.sigma ** 2 + 1e-12))  # shape (n_samples, n_anchors)

        # 5) Graphe complet pondéré
        A = S.dot(S.T)
        np.fill_diagonal(A, 0.0)

        self._adjacency = A

    def adjacency_matrix(self) -> np.ndarray:
        """
        Retourne la matrice d’adjacence construite (n_samples x n_samples).
        """
        if self._adjacency is None:
            raise ValueError("Veuillez appeler fit(X) avant adjacency_matrix().")
        return self._adjacency