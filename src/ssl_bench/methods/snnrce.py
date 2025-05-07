# src/ssl_bench/methods/snnrce.py
"""
Implementation of SNNRCE: Self-training Nearest Neighbor Rule using Cut Edges
Wang et al., Knowledge-Based Systems 23 (2010)
Pseudo-code steps are included as comments corresponding to the original algorithm description.
"""
import numpy as np
from copy import deepcopy
from typing import Tuple, Optional
from scipy.stats import norm

from sklearn.neighbors import NearestNeighbors
from ssl_bench.methods.base import SemiSupervisedMethod
from ssl_bench.models.base import BaseModel

class SnnrceMethod(SemiSupervisedMethod):
    """
    SNNRCE: Self-training Nearest Neighbor Rule using Cut Edges
    Combines NN classification, self-training, and cut-edge based editing.
    """
    def __init__(
        self,
        base_model: BaseModel,
        n_neighbors: int = 10,
        alpha: float = 0.05,
        random_state: Optional[int] = None
    ):
        """
        :param base_model: classifier implementing train/predict/predict_proba
        :param n_neighbors: neighborhood size for cut-edge calculations
        :param alpha: significance level for label modification test
        :param random_state: random seed
        """
        self.base_model = deepcopy(base_model)
        self.n_neighbors = n_neighbors
        self.alpha = alpha
        self.random_state = random_state

    def run(
        self,
        X_l: np.ndarray,
        y_l: np.ndarray,
        X_u: np.ndarray
    ) -> Tuple[BaseModel, np.ndarray, np.ndarray]:
        # Step 1: compute class‐proportions and N_max
        classes, counts = np.unique(y_l, return_counts=True)
        total_L = len(y_l)
        ratio = {c: counts[i] / total_L for i, c in enumerate(classes)}
        N_max = {c: int(ratio[c] * len(X_u)) for c in classes}

        # Initialize L and U
        L_x, L_y = X_l.copy(), y_l.copy()
        U_x = X_u.copy()

        rng = np.random.RandomState(self.random_state)

        # Step 2: initial NN training on L
        model = deepcopy(self.base_model)
        model.train(L_x, L_y)

        # Step 3: initial cut-edge based labeling
        nn = NearestNeighbors(n_neighbors=self.n_neighbors + 1)
        nn.fit(L_x)
        # find neighbors for U
        nbrs_U = nn.kneighbors(U_x, return_distance=False)[:,1:]
        labeled_mask = np.zeros(len(U_x), dtype=bool)
        new_labels = np.full(len(U_x), -1, dtype=int)

        for i, nbrs in enumerate(nbrs_U):
            # distances & weights
            dists = np.linalg.norm(U_x[i] - L_x[nbrs], axis=1)
            weights = 1.0 / (1.0 + dists)
            # cut‐edge ratio R_i
            pred_i = model.predict(U_x[i:i+1])[0]
            mismatch = L_y[nbrs] != pred_i
            R_i = weights[mismatch].sum() / (weights.sum() + 1e-12)
            if R_i == 0.0:
                # label by NN
                labeled_mask[i] = True
                new_labels[i] = pred_i

        if labeled_mask.any():
            L_x = np.vstack([L_x, U_x[labeled_mask]])
            L_y = np.concatenate([L_y, new_labels[labeled_mask]])
            U_x = U_x[~labeled_mask]

        # Step 4: self‐training with dynamic confidence until N_max reached
        while True:
            model = deepcopy(self.base_model)
            model.train(L_x, L_y)
            if len(U_x) == 0:
                break

            # compute confidence = exp(−distance to nearest neighbor)
            dists, idxs = nn.kneighbors(U_x, n_neighbors=1)
            confidences = np.exp(-dists.flatten())

            added_any = False
            # sort indices descending by confidence
            order = np.argsort(confidences)[::-1]
            for c in classes:
                need = N_max[c] - np.sum(L_y == c)
                if need <= 0:
                    continue
                # find top‐confidence points predicted as c
                preds = model.predict(U_x)
                sel = [i for i in order if preds[i] == c]
                to_take = sel[:need]
                if not to_take:
                    continue
                L_x = np.vstack([L_x, U_x[to_take]])
                L_y = np.concatenate([L_y, np.full(len(to_take), c)])
                mask = np.ones(len(U_x), dtype=bool)
                mask[to_take] = False
                U_x = U_x[mask]
                added_any = True
            if not added_any:
                break

        # Step 5: label modification via cut‐edge distribution on final L
        nn_L = NearestNeighbors(n_neighbors=self.n_neighbors + 1)
        nn_L.fit(L_x)
        nbrs_L = nn_L.kneighbors(L_x, return_distance=False)[:,1:]
        R_vals = []
        for i, nbrs in enumerate(nbrs_L):
            dists = np.linalg.norm(L_x[i] - L_x[nbrs], axis=1)
            weights = 1.0 / (1.0 + dists)
            mismatch = L_y[nbrs] != L_y[i]
            R_vals.append(weights[mismatch].sum() / (weights.sum() + 1e-12))
        R_vals = np.array(R_vals)
        mu, sigma = R_vals.mean(), R_vals.std()
        crit = mu + norm.ppf(1 - self.alpha/2) * sigma
        for i, R_i in enumerate(R_vals):
            if R_i > crit:
                # flip label to the other class
                alt = [cls for cls in classes if cls != L_y[i]][0]
                L_y[i] = alt

        # Step 6: final classification of remaining U
        if len(U_x) > 0:
            final = deepcopy(self.base_model)
            final.train(L_x, L_y)
            preds_U = final.predict(U_x)
            L_x = np.vstack([L_x, U_x])
            L_y = np.concatenate([L_y, preds_U])
        else:
            final = model

        return final, L_x, L_y