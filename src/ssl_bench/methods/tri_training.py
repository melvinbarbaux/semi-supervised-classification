"""
Tri-Training 
Zhou & Li, 2005.
"""

import numpy as np
import logging
from copy import deepcopy
from typing import List, Tuple, Any

from ssl_bench.methods.base import SemiSupervisedMethod
from ssl_bench.models.base import BaseModel

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    ch.setFormatter(fmt)
    logger.addHandler(ch)


class TriTrainingEnsemble(BaseModel):
    def __init__(
        self,
        classifiers: List[BaseModel],
        model_indices: List[np.ndarray],
        instances_index: np.ndarray,
        model_index_map: List[np.ndarray],
        verbose: bool = False
    ):
        super().__init__()
        self.classifiers = classifiers
        self.model_indices = model_indices
        self.instances_index = instances_index
        self.model_index_map = model_index_map
        self.verbose = verbose

    def train(self, X, y, X_u=None):
        # Already trained during run
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        votes = np.stack([clf.predict(X) for clf in self.classifiers], axis=1)
        return np.array([np.bincount(row).argmax() for row in votes])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probas = [clf.predict_proba(X) for clf in self.classifiers]
        stacked = np.stack(probas, axis=2)
        return np.mean(stacked, axis=2)


class TriTrainingMethod(SemiSupervisedMethod):
    def __init__(
        self,
        base_model: Any,
        *,
        random_state: Any = None,
        verbose: bool = False,
        **kwargs,
    ):
        super().__init__(base_model, **kwargs)
        self.base_model = deepcopy(base_model)
        self.random_state = random_state
        self.verbose = verbose

    def run(
        self,
        X_labeled: np.ndarray,
        y_labeled: np.ndarray,
        X_unlabeled: np.ndarray
    ) -> Tuple[BaseModel, np.ndarray, np.ndarray]:
        # 1) Initialize datasets and indices
        L_X = X_labeled.copy()
        L_y = y_labeled.copy()
        U_X = X_unlabeled.copy()

        labeled_idx = np.arange(len(L_y))
        unlabeled_idx = np.arange(len(U_X))

        # 2) Bootstrap three classifiers
        rng = np.random.RandomState(self.random_state)
        h: List[BaseModel] = []
        model_indices: List[np.ndarray] = []
        e_prev = [0.5] * 3
        l_prev = [0] * 3
        for i in range(3):
            idx = rng.choice(len(L_X), size=len(L_X), replace=True)
            clf = deepcopy(self.base_model)
            clf.train(L_X[idx], L_y[idx])
            h.append(clf)
            model_indices.append(labeled_idx[idx].copy())
        if self.verbose:
            logger.info(f"TriTraining start: |L|={len(L_y)}, |U|={len(U_X)}")

        # 3) Iterative labeling
        updated = True
        while updated and U_X.shape[0] > 0:
            updated = False
            for i in range(3):
                j, k = (i+1)%3, (i+2)%3
                # 3a) joint error on L
                pj = h[j].predict(L_X)
                pk = h[k].predict(L_X)
                agree_L = pj == pk
                n_agree_L = agree_L.sum()
                if n_agree_L > 0:
                    e_i = (pj[agree_L] != L_y[agree_L]).sum() / n_agree_L
                else:
                    e_i = 1.0

                # 3b) select U where hj==hk
                pj_u = h[j].predict(U_X)
                pk_u = h[k].predict(U_X)
                agree_U = pj_u == pk_u
                U_idx = np.where(agree_U)[0]

                sel = None
                if e_i < e_prev[i] and U_idx.size > 0:
                    # compute l' if first time
                    if l_prev[i] == 0:
                        l_i = max(1, int(np.floor(e_i/(e_prev[i]-e_i) + 1)))
                    else:
                        l_i = l_prev[i]
                    # cond1: add all
                    if U_idx.size > l_i and e_i*U_idx.size < e_prev[i]*l_i:
                        sel = U_idx
                    # cond2: sample subset
                    elif e_i > 0 and (e_prev[i]-e_i) > 0 and l_i > e_i/(e_prev[i]-e_i):
                        s = int(np.ceil(e_prev[i]*l_i/e_i - 1))
                        if 0 < s < U_idx.size:
                            sel = rng.choice(U_idx, size=s, replace=False)
                # 3c) apply update
                if sel is not None and sel.size > 0:
                    new_X = U_X[sel]
                    new_y = pj_u[sel]
                    # update L and indices
                    L_X = np.vstack([L_X, new_X])
                    L_y = np.concatenate([L_y, new_y])
                    new_idx = unlabeled_idx[sel]
                    labeled_idx = np.concatenate([labeled_idx, new_idx])
                    mask = np.ones(len(U_X), bool)
                    mask[sel] = False
                    U_X = U_X[mask]
                    unlabeled_idx = unlabeled_idx[mask]
                    # retrain hi
                    hi_new = deepcopy(self.base_model)
                    hi_new.train(L_X, L_y)
                    h[i] = hi_new
                    # update stats
                    e_prev[i] = e_i
                    l_prev[i] = sel.size
                    model_indices[i] = np.concatenate([model_indices[i], new_idx])
                    updated = True
            if self.verbose:
                logger.info(f"TriTraining iteration: updated={updated}, remaining U={U_X.shape[0]}")

        # 4) Build indices mapping
        instances_index = np.unique(np.concatenate(model_indices))
        model_index_map: List[np.ndarray] = []
        for inds in model_indices:
            pos = np.array([np.where(instances_index==idx)[0][0] for idx in inds])
            model_index_map.append(pos)

        # 5) Final ensemble
        final_model = TriTrainingEnsemble(h, model_indices, instances_index, model_index_map)
        if self.verbose:
            logger.info(f"TriTraining completed: final |L|={len(L_y)}, classifiers=3")
        return final_model, L_X, L_y