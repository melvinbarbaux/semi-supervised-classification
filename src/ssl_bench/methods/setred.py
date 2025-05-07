# src/ssl_bench/methods/setred.py
"""
Implementation of SETRED: Self-Training with Editing
Ming Li and Zhi-Hua Zhou, 2005
Pseudo-code steps are included as comments corresponding to the original algorithm's description.
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors
from copy import deepcopy
from typing import Tuple, Optional

from ssl_bench.methods.base import SemiSupervisedMethod
from ssl_bench.models.base import BaseModel

class SetredMethod(SemiSupervisedMethod):
    """
    SETRED: Self-Training with Editing
    - Filters mislabeled examples via local cut edge statistics on a neighborhood graph.
    """
    def __init__(
        self,
        base_model: BaseModel,
        theta: float = 0.1,
        max_iter: int = 10,
        pool_size: Optional[int] = None,
        n_neighbors: int = 10,
        random_state: Optional[int] = None
    ):
        """
        :param base_model: underlying classifier implementing train/predict/predict_proba
        :param theta: left rejection threshold for cut-edge proportion
        :param max_iter: maximum self-training iterations M
        :param pool_size: size of U' pool (None -> use all unlabeled)
        :param n_neighbors: neighborhood size for editing
        :param random_state: random seed
        """
        self.model = deepcopy(base_model)
        self.theta = theta
        self.max_iter = max_iter
        self.pool_size = pool_size
        self.n_neighbors = n_neighbors
        self.random_state = random_state

    def run(
        self,
        X_l: np.ndarray,
        y_l: np.ndarray,
        X_u: np.ndarray
    ) -> Tuple[BaseModel, np.ndarray, np.ndarray]:
        # 1) Create U' by sampling from X_u
        rng = np.random.RandomState(self.random_state)
        if self.pool_size is None or self.pool_size >= len(X_u):
            U_prime = X_u.copy()
        else:
            idx = rng.choice(len(X_u), size=self.pool_size, replace=False)
            U_prime = X_u[idx]

        # 2) Initial learning on L
        #    h <- Learn(L)
        h = deepcopy(self.model)
        h.train(X_l, y_l)

        L_x = X_l.copy()
        L_y = y_l.copy()

        # 3) Repeat for max_iter iterations
        for it in range(self.max_iter):
            # L' <- empty set
            Lp_x, Lp_y = [], []

            # for each possible label yj:
            classes, counts = np.unique(L_y, return_counts=True)
            total = len(L_y)
            # compute selection sizes proportional to class distribution
            n_select = min(len(U_prime), int(0.1 * len(U_prime))) or len(classes)
            proba = h.predict_proba(U_prime)
            for j, cls in enumerate(classes):
                kj = max(1, int(counts[j] / total * n_select))
                confidences = proba[:, cls]
                top_idx = np.argsort(confidences)[-kj:]
                Lp_x.append(U_prime[top_idx])
                Lp_y.append(np.full(kj, cls, dtype=int))

            if not Lp_x:
                break
            # merge lists
            Lp_x = np.vstack(Lp_x)
            Lp_y = np.concatenate(Lp_y)

            # 4) Build neighborhood graph G on L ∪ L'
            X_all = np.vstack([L_x, Lp_x])
            y_all = np.concatenate([L_y, Lp_y])
            nn = NearestNeighbors(n_neighbors=self.n_neighbors + 1)
            nn.fit(X_all)
            Lp_indices = np.arange(len(L_x), len(L_x) + len(Lp_x))
            to_keep = []
            for idx in Lp_indices:
                neigh = nn.kneighbors(X_all[idx:idx+1], return_distance=False)[0]
                neigh = neigh[neigh != idx][:self.n_neighbors]
                cut = (y_all[neigh] != y_all[idx]).astype(int)
                Ji = cut.sum()
                if Ji / self.n_neighbors <= self.theta:
                    to_keep.append(idx)

            if not to_keep:
                break
            Lp_x = X_all[to_keep]
            Lp_y = y_all[to_keep]

            # 5) h <- Learn(L ∪ L')
            L_x = np.vstack([L_x, Lp_x])
            L_y = np.concatenate([L_y, Lp_y])
            h = deepcopy(self.model)
            h.train(L_x, L_y)

            # 6) Replenish U'
            if self.pool_size is not None and self.pool_size < len(X_u):
                idx = rng.choice(len(X_u), size=self.pool_size, replace=False)
                U_prime = X_u[idx]
            else:
                U_prime = X_u.copy()

        # End of Repeat
        return h, L_x, L_y
