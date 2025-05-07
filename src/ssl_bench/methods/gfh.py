# src/ssl_bench/methods/gfh.py
import numpy as np
import networkx as nx
import math
from typing import Tuple, List

from ssl_bench.methods.base import SemiSupervisedMethod
from ssl_bench.models.base import BaseModel
from ssl_bench.datamodule.graph.base import GraphBuilder

class GFHFModel(BaseModel):
    """
    Wrapper for Gaussian Fields & Harmonic Functions (GFHF) results.
    Stocke les prédictions (labels) pour l'ensemble L∪U et fournit predict et predict_proba.
    """
    def __init__(self, preds_all: np.ndarray, n_labeled: int, classes: np.ndarray):
        # preds_all: labels pour les n_l + n_u échantillons
        self.preds_all = preds_all
        self.n_labeled_ = n_labeled
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(classes)}

    def train(self, X, y, X_u=None):
        # Pas d'entraînement supplémentaire
        return self

    def predict(self, X) -> np.ndarray:
        # Retourne uniquement les prédictions pour la partie non-étiquetée
        return self.preds_all[self.n_labeled_:]

    def predict_proba(self, X) -> np.ndarray:
        # Encodage one-hot des prédictions
        preds_u = self.predict(X)
        n_u = len(preds_u)
        n_c = len(self.classes)
        probs = np.zeros((n_u, n_c))
        for i, lbl in enumerate(preds_u):
            idx = self.class_to_idx[lbl]
            probs[i, idx] = 1.0
        return probs

class GFHFMethod(SemiSupervisedMethod):
    """
    Gaussian Fields & Harmonic Functions (Zhu et al., 2003).
    Utilise un GraphBuilder pour construire le graphe W, calcule le Laplacien,
    résout la solution harmonique pour les données non-étiquetées.
    """
    def __init__(
        self,
        graph_builder: GraphBuilder,
        alpha: float = 0.99
    ):
        self.graph_builder = graph_builder
        self.alpha = alpha

    def run(
        self,
        X_l: np.ndarray,
        y_l: np.ndarray,
        X_u: np.ndarray
    ) -> Tuple[BaseModel, np.ndarray, np.ndarray]:
        # 1) Concaténer X_l et X_u
        X_all = np.vstack([X_l, X_u])
        n_l = X_l.shape[0]

        # 2) Construire graphe initial
        self.graph_builder.fit(X_all)
        W0 = self.graph_builder.adjacency_matrix()

        # 3) Convertir en networkx et calculer layout
        G = nx.from_numpy_array(W0)
        pos = nx.spring_layout(G, iterations=200)
        dims = len(next(iter(pos.values())))

        # 4) Calculer sigma par dimension
        sigma = [0.0] * dims
        for u, v in G.edges():
            for d in range(dims):
                d2 = (pos[u][d] - pos[v][d]) ** 2
                sigma[d] = max(sigma[d], d2)
        sigma = [s if s > 0 else 1e-6 for s in sigma]

        # 5) Recalculer matrice W avec RBF sur layout
        n = X_all.shape[0]
        W = np.zeros((n, n))
        for u, v in G.edges():
            s = sum((pos[u][d] - pos[v][d]) ** 2 / sigma[d] for d in range(dims))
            w = math.exp(-s)
            W[u, v] = W[v, u] = w

        # 6) Construire Laplacien L = D - W
        D = np.diag(W.sum(axis=1))
        L = D - W

        # 7) Partitionner L en blocs
        L_ul = L[n_l:, :n_l]
        L_uu = L[n_l:, n_l:]

        # 8) One-hot encoder y_l
        classes = np.unique(y_l)
        n_c = classes.shape[0]
        idx_map = {c: i for i, c in enumerate(classes)}
        Y_l = np.zeros((n_l, n_c))
        for i, c in enumerate(y_l):
            Y_l[i, idx_map[c]] = 1

        # 9) Résoudre f_u = -L_uu^{-1} (L_ul Y_l)
        B = -L_ul.dot(Y_l)
        try:
            f_u = np.linalg.solve(L_uu, B)
        except np.linalg.LinAlgError:
            jitter = 1e-6 * np.eye(L_uu.shape[0])
            f_u = np.linalg.solve(L_uu + jitter, B)

        # 10) argmax pour prédictions
        idxs = np.argmax(f_u, axis=1)
        preds_u = classes[idxs]

        # 11) assembler y_l et preds_u
        preds_all = np.concatenate([y_l, preds_u])

        # 12) retourner modèle et données labellisées
        model = GFHFModel(preds_all=preds_all, n_labeled=n_l, classes=classes)
        return model, X_l, y_l
