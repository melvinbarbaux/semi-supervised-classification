"""
TTADEC: Tri-Training with Adaptive Density Editing and Cross-Entropy Evaluation.
Jia Zhao, Yuhang Luo, Renbin Xiao, Runxiu Wu & Tanghuai Fan
"""

import numpy as np
from copy import deepcopy
from typing import List, Tuple, Any
from sklearn.neighbors import NearestNeighbors
import logging

from ssl_bench.methods.base import SemiSupervisedMethod
from ssl_bench.models.base import BaseModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(fmt)
    logger.addHandler(ch)

class TTADECEnsemble(BaseModel):
    def __init__(self, learners: List[BaseModel], alphas: List[float]):
        self.learners = learners
        self.alphas = alphas

    def train(self, X, y, X_u=None):
        # Already trained
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        votes = np.stack([clf.predict(X) for clf in self.learners], axis=1)
        # weighted vote by alpha
        final = []
        for row in votes:
            scores = {}
            for i, label in enumerate(row):
                scores[label] = scores.get(label, 0.0) + self.alphas[i]
            final.append(max(scores, key=scores.get))
        return np.array(final)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probas = [clf.predict_proba(X) * alpha for clf, alpha in zip(self.learners, self.alphas)]
        stacked = np.stack(probas, axis=2)
        return np.sum(stacked, axis=2) / sum(self.alphas)

class TTADECMethod(SemiSupervisedMethod):
    def __init__(
        self,
        learners: List[BaseModel],
        k: int = 10,
        random_state: int = None,
        verbose: bool = False
    ):
        assert len(learners) == 3, "Require three base learners"
        super().__init__(learners[0])
        # create three bootstrap copies
        self.learners = [deepcopy(l) for l in learners]
        self.k = k
        self.rng = np.random.RandomState(random_state)
        self.verbose = verbose

    def run(
        self,
        X_l: np.ndarray,
        y_l: np.ndarray,
        X_u: np.ndarray
    ) -> Tuple[BaseModel, np.ndarray, np.ndarray]:
        # Phase 1: Bootstrap initialization
        L_X, L_y = X_l.copy(), y_l.copy()
        U = X_u.copy()
        # flatten U for nearest-neighbors operations
        U_flat = U.reshape(U.shape[0], -1)
        n_l = len(L_y)

        # Train three initial learners on bootstrap samples
        for i in range(3):
            idxs = self.rng.choice(n_l, size=n_l, replace=True)
            self.learners[i].train(L_X[idxs], L_y[idxs])

        # iterative refinement
        updated = True
        while updated and U.shape[0] > 0:
            updated = False
            # predictions on L and U
            preds_L = [h.predict(L_X) for h in self.learners]
            preds_U = [h.predict(U)   for h in self.learners]
            # for each learner, propose new labels where other two agree
            for i in range(3):
                j, k_idx = (i+1)%3, (i+2)%3
                agree_L = (preds_L[j] == preds_L[k_idx])
                # error on L
                if agree_L.sum() == 0:
                    continue
                e_i = np.mean(preds_L[j][agree_L] != L_y[agree_L])
                # candidate pool where U predictions agree
                agree_U = (preds_U[j] == preds_U[k_idx])
                X_cand = U[agree_U]
                y_cand = preds_U[j][agree_U]
                if len(y_cand) == 0:
                    continue
                # density editing: keep only those with at least k/2 same-label neighbors in U
                # build neighbor index on flattened features
                nbrs = NearestNeighbors(n_neighbors=min(self.k, U_flat.shape[0]), algorithm='auto') \
                           .fit(U_flat)
                # flatten candidate points
                X_cand_flat = X_cand.reshape(X_cand.shape[0], -1)
                distances, neighbors = nbrs.kneighbors(X_cand_flat)
                keep_mask = []
                for idx_row, nbr_ids in enumerate(neighbors):
                    labels_nbr = preds_U[j][nbr_ids]
                    count_same = np.sum(labels_nbr == y_cand[idx_row])
                    keep_mask.append(count_same >= (self.k // 2))
                keep_mask = np.array(keep_mask)
                X_new = X_cand[keep_mask]
                y_new = y_cand[keep_mask]
                if len(y_new) == 0:
                    continue
                # add to L, remove from U
                L_X = np.vstack([L_X, X_new])
                L_y = np.concatenate([L_y, y_new])
                remove_idx = np.where(agree_U)[0][keep_mask]
                mask_remain = np.ones(len(U), dtype=bool)
                mask_remain[remove_idx] = False
                U = U[mask_remain]
                # also update flattened representation
                U_flat = U_flat[mask_remain]
                # retrain learner i on expanded L
                self.learners[i].train(L_X, L_y)
                updated = True

        # final ensemble by equal weights
        alphas = [1.0/3]*3
        final_model = TTADECEnsemble(self.learners, alphas)
        return final_model, L_X, L_y