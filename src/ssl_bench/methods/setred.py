"""
SETRED: Self-Training with Editing
Ming Li & Zhi-Hua Zhou, 2005.
"""

import numpy as np
import logging
from copy import deepcopy
from typing import Tuple, Optional

from sklearn.neighbors import kneighbors_graph
from ssl_bench.methods.base import SemiSupervisedMethod
from ssl_bench.models.base import BaseModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    logger.addHandler(ch)


class SetredMethod(SemiSupervisedMethod):
    def __init__(
        self,
        base_model: BaseModel,
        theta: float = 0.1,
        max_iter: int = 10,
        pool_size: Optional[int] = None,
        n_neighbors: int = 10,
        perc_full: float = 0.9,
        random_state: Optional[int] = None
    ):
        """
        :param base_model: classifier providing train / predict / predict_proba
        :param theta: cut-edge ratio threshold
        :param max_iter: max self-training iterations
        :param pool_size: U' size each round (None = all of U)
        :param n_neighbors: neighbors for editing graph
        :param perc_full: stop when |L| ≥ perc_full·(|L0|+|U0|)
        :param random_state: RNG seed
        """
        self.base_model = deepcopy(base_model)
        self.theta = theta
        self.max_iter = max_iter
        self.pool_size = pool_size
        self.n_neighbors = n_neighbors
        self.perc_full = perc_full
        self.random_state = random_state

    def run(
        self,
        X_l: np.ndarray,
        y_l: np.ndarray,
        X_u: np.ndarray
    ) -> Tuple[BaseModel, np.ndarray, np.ndarray]:
        rng = np.random.RandomState(self.random_state)

        # 1) Flatten for graph if images (ndim>2), else use directly as features
        U_raw = X_u.copy()
        U_feat = (U_raw.reshape(len(U_raw), -1)
                  if U_raw.ndim > 2 else U_raw.copy())
        L_raw = X_l.copy()
        L_feat = (L_raw.reshape(len(L_raw), -1)
                  if L_raw.ndim > 2 else L_raw.copy())
        L_y = y_l.copy()
        total0 = len(L_feat) + len(U_feat)

        # 2) Initial training on L
        h = deepcopy(self.base_model)
        h.train(L_raw, L_y)

        logger.info(f"SETRED start: |L|={len(L_y)}, |U|={len(U_feat)}, θ={self.theta}")

        # 3) Self-training + editing loop
        for it in range(1, self.max_iter + 1):
            # stop if labeled fraction reached
            if len(L_feat) >= self.perc_full * total0:
                logger.info(f"Stop iter {it}: reached perc_full ({len(L_feat)}/{total0})")
                break

            # 3a) build U′ by sampling (or all)
            if self.pool_size is None or self.pool_size >= len(U_feat):
                pool_inds = np.arange(len(U_feat))
            else:
                pool_inds = rng.choice(len(U_feat), size=self.pool_size, replace=False)
            pool_raw = U_raw[pool_inds]
            pool_feat = U_feat[pool_inds]

            # 3b) pseudo-label top-confidence candidates proportional to class distribution
            proba = h.predict_proba(pool_raw)
            classes, counts = np.unique(L_y, return_counts=True)
            total_L = len(L_y)
            n_select = max(1, int(0.1 * len(pool_feat)))

            cand_raws, cand_feats, cand_labels = [], [], []
            for cls, cnt in zip(classes, counts):
                k = max(1, int(cnt / total_L * n_select))
                idxs = np.argsort(proba[:, cls])[-k:]
                cand_raws.append(pool_raw[idxs])
                cand_feats.append(pool_feat[idxs])
                cand_labels.append(np.full(k, cls, dtype=int))

            if not cand_feats:
                logger.info(f"Iter {it}: no candidates")
                break

            cand_raws  = np.vstack(cand_raws)
            cand_feats = np.vstack(cand_feats)
            cand_labels= np.concatenate(cand_labels)

            # 3c) build k-NN graph on L_feat ∪ cand_feats
            X_all = np.vstack([L_feat, cand_feats])
            y_all = np.concatenate([L_y, cand_labels])
            A = kneighbors_graph(X_all,
                                 n_neighbors=self.n_neighbors+1,
                                 include_self=False).tolil()

            # compute cut-edge ratio for each candidate
            start = len(L_feat)
            keep = []
            for j in range(start, start + len(cand_feats)):
                neighs = [n for n in A.rows[j] if n != j][: self.n_neighbors]
                cut = np.sum(y_all[neighs] != y_all[j])
                if cut / self.n_neighbors <= self.theta:
                    keep.append(j - start)

            if not keep:
                logger.info(f"Iter {it}: none survive editing")
                break

            logger.info(f"Iter {it}: {len(keep)}/{len(cand_feats)} kept")

            # 3d) add the kept candidates to L
            kept_raw  = cand_raws[keep]
            kept_feat = cand_feats[keep]
            kept_y    = cand_labels[keep]
            L_raw  = np.vstack([L_raw,  kept_raw])
            L_feat = np.vstack([L_feat, kept_feat])
            L_y    = np.concatenate([L_y,   kept_y])

            # remove accepted from U
            actual = pool_inds[keep]
            mask = np.ones(len(U_raw), dtype=bool)
            mask[actual] = False
            U_raw  = U_raw[mask]
            U_feat = U_feat[mask]

            # 3e) retrain h on raw L
            h = deepcopy(self.base_model)
            h.train(L_raw, L_y)

        logger.info(f"SETRED end: |L|={len(L_y)}, |U|={len(U_feat)} (added {len(L_y) - len(X_l)}), it={it}")

        # 4) return final model and the enlarged labeled set (raw)
        return h, L_raw, L_y