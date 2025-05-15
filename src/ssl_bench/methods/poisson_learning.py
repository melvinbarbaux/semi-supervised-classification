# src/ssl_bench/methods/poisson_learning.py
import numpy as np
import logging
from typing import Tuple

from ssl_bench.methods.base import SemiSupervisedMethod
from ssl_bench.models.base import BaseModel
from ssl_bench.datamodule.graph.base import GraphBuilder

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    ch.setFormatter(fmt)
    logger.addHandler(ch)


class PoissonModel(BaseModel):
    """
    Poisson Learning model (Calder et al., 2020).
    Stores learned potentials for unlabeled nodes and provides predict / predict_proba.
    """
    def __init__(self, f_u: np.ndarray, classes: np.ndarray):
        # f_u: array of shape (n_u, n_classes)
        self.f_u = f_u
        self.classes = classes

    def train(self, X, y, X_u=None):
        # No further training needed
        return self

    def predict(self, X) -> np.ndarray:
        # Return class with highest potential
        idxs = np.argmax(self.f_u, axis=1)
        return self.classes[idxs]

    def predict_proba(self, X) -> np.ndarray:
        # Normalize potentials to probabilities
        f = self.f_u.copy()
        # Shift to non-negative
        f -= f.min(axis=1, keepdims=True)
        sums = f.sum(axis=1, keepdims=True)
        # Avoid zero division
        zero_mask = (sums == 0)
        sums[zero_mask] = 1.0
        probs = f / sums
        # If all zeros, assign uniform
        probs[zero_mask.flatten(), :] = 1.0 / probs.shape[1]
        return probs


class PoissonLearningMethod(SemiSupervisedMethod):
    """
    Poisson Learning (Calder et al., 2020) for graph-based SSL at very low label rates.
    Implements Algorithm 1: solves U = sum_{t=1}^T D^{-1}(B - L U) and reweights.
    """
    def __init__(
        self,
        graph_builder: GraphBuilder,
        T: int = 200,
        eps_mix: float = None,
        verbose: bool = False
    ):
        """
        :param graph_builder: builds adjacency matrix W
        :param T: number of Poisson iterations (hand-tuned, default ~200)
        :param eps_mix: optional mixing threshold to stop early (∞-norm criterion)
        """
        self.graph_builder = graph_builder
        self.T = T
        self.eps_mix = eps_mix
        self.verbose = verbose

    def run(
        self,
        X_l: np.ndarray,
        y_l: np.ndarray,
        X_u: np.ndarray
    ) -> Tuple[BaseModel, np.ndarray, np.ndarray]:
        # 1) Combine data
        X_all = np.vstack([X_l, X_u])
        n_l = X_l.shape[0]
        n = X_all.shape[0]

        # 2) Build graph
        self.graph_builder.fit(X_all)
        W = self.graph_builder.adjacency_matrix()

        # 3) Degree and Laplacian
        d = W.sum(axis=1)
        D_inv = np.diag(1.0 / np.where(d > 0, d, 1.0))
        L = np.diag(d) - W

        # 4) Build label-source matrix B (n × k)
        classes = np.unique(y_l)
        k = classes.shape[0]
        # F: m × k one-hot
        F = np.zeros((n_l, k))
        class_to_idx = {c: i for i, c in enumerate(classes)}
        for i, c in enumerate(y_l):
            F[i, class_to_idx[c]] = 1.0
        # ȳ: average label vector
        y_bar = F.mean(axis=0, keepdims=True)  # shape (1,k)
        # B: (F - ȳ) for labeled, zeros for unlabeled
        B = np.vstack([F - y_bar, np.zeros((n - n_l, k))])

        # 5) Initialize potentials U = 0 (n × k)
        U = np.zeros((n, k))

        # 6) Iteratively accumulate Poisson updates
        for t in range(self.T):
            U += D_inv.dot(B - L.dot(U))
            if self.eps_mix is not None:
                # optional early stopping when change is small
                delta = np.linalg.norm((B - L.dot(U)), ord=np.inf)
                if delta < self.eps_mix:
                    if self.verbose:
                        logger.info(f"Poisson: early stop at iter {t+1}, Δ={delta:.2e}")
                    break

        # 7) Extract unlabeled potentials f_u = U[n_l:, :]
        f_u = U[n_l:, :]

        # 8) Reweight to compensate class imbalance (b / ȳ)
        # Here b = uniform 1/k if prior unknown
        b = np.ones((1, k)) / k
        reweight = (b / (y_bar + 1e-12)).flatten()
        f_u *= reweight[None, :]

        # 9) Wrap and return
        model = PoissonModel(f_u=f_u, classes=classes)
        return model, X_l, y_l