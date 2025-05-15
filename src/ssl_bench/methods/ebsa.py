import numpy as np
import logging
from copy import deepcopy
from typing import Tuple

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


class EBSAEnsemble(BaseModel):
    def __init__(self, model: BaseModel):
        super().__init__()
        self.model = model

    def train(self, X, y, X_u=None):
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)


class EBSAMethod(SemiSupervisedMethod):
    def __init__(self, base_model: BaseModel, random_state: int = None, verbose: bool = False):
        super().__init__(base_model)
        self.base_model = deepcopy(base_model)
        self.rng = np.random.RandomState(random_state)
        self.verbose = verbose

    def run(
        self,
        X_l: np.ndarray,
        y_l: np.ndarray,
        X_u: np.ndarray
    ) -> Tuple[BaseModel, np.ndarray, np.ndarray]:
        # --- Algorithm 4 Step 1: initialize sets L and U; high-confidence S ← ∅ ---
        L_X = X_l.copy()        # L ← labeled data
        L_y = y_l.copy()
        U_X = X_u.copy()        # U ← unlabeled data
        # --- Algorithm 4 Step 2: train classifier H on L ---
        H = deepcopy(self.base_model)
        H.train(L_X, L_y)
        if self.verbose:
            logger.info(f"EBSA: start |L|={len(L_y)}, |U|={len(U_X)}")

        # --- Algorithm 4 Step 8: while not converged ---
        while True:
            # stop if no unlabeled samples remain
            if U_X.shape[0] == 0:
                break

            # --- Algorithm 4 Step 3–6: assign pseudo-labels to U (form clusters Cₚ) ---
            pseudo = H.predict(U_X)            # Step 3–4: y_i = H(x_i)
            clusters = {}                      # Step 5: C_{y_i} ← C_{y_i} ∪ x_i
            for idx, lbl in enumerate(pseudo):
                clusters.setdefault(lbl, []).append(idx)

            # --- Algorithm 2 & 3: ball-cluster partition (EC) and high-confidence selection (SG) ---
            new_indices = []  # will collect S = ⋃ₚ (stable region minus disputed)  [oai_citation:0‡12-2023-Fast semisupervised selftraining algorithm based on data editing.pdf](file-service://file-1D1gxQ6V9LUDavd5F48Si1)
            for lbl, idxs in clusters.items():
                Xp = U_X[idxs]
                if Xp.shape[0] == 0:
                    continue

                # Algorithm 2 Step 3–4: compute center and radius of cluster Cₚ
                center = Xp.mean(axis=0)
                dists = np.linalg.norm(Xp - center, axis=1)
                radius = dists.max()

                # Algorithm 2 Step 7–11: find neighbor clusters Nₚq
                neighbor_centers = []
                for other_lbl, other_idxs in clusters.items():
                    if other_lbl == lbl:
                        continue
                    oc = U_X[other_idxs].mean(axis=0)
                    d_cent = np.linalg.norm(center - oc)   # distance dₚq
                    if d_cent / 2 < radius:               # if ½·dₚq < rₚ
                        neighbor_centers.append((oc, d_cent))

                # Algorithm 2 Step 15: compute stable radius bₚᵣ = ½ · minₙ dₚq
                if neighbor_centers:
                    min_dist = min(d for _, d in neighbor_centers)
                    b_r = min_dist / 2.0
                else:
                    b_r = radius

                # Algorithm 2 Step 16–18: compute disputed region indices
                disputed = set()
                for oc, d_cent in neighbor_centers:
                    e_r = d_cent / 2.0   # er = ½·dₚq
                    d_other = np.linalg.norm(Xp - oc, axis=1)
                    disputed.update(np.where(d_other <= e_r)[0].tolist())

                # Algorithm 2 Step 18–22: edit stable points (optional label correction)
                stable_idx = np.where(dists <= b_r)[0]
                for j in stable_idx:
                    pseudo[idxs[j]] = lbl  # optionally reinforce label

                # Algorithm 3 Step 3–5: select high-confidence S₀ = Xp \ disputed
                for j in range(len(idxs)):
                    if j not in disputed:
                        new_indices.append(idxs[j])

            new_indices = sorted(set(new_indices))
            # if no new high-confidence points, convergence
            if not new_indices:
                break

            # --- Algorithm 4 Step 11: update L ← L ∪ S ---
            X_new = U_X[new_indices]
            y_new = pseudo[new_indices]
            L_X = np.vstack([L_X, X_new])
            L_y = np.concatenate([L_y, y_new])

            # remove newly labeled from U
            mask = np.ones(len(U_X), dtype=bool)
            mask[new_indices] = False
            U_X = U_X[mask]

            # --- Algorithm 4 Step 12: retrain H on updated L ---
            H = deepcopy(self.base_model)
            H.train(L_X, L_y)
            if self.verbose:
                logger.info(f"EBSA iter: added={len(new_indices)}, remaining U={len(U_X)}")

        if self.verbose:
            logger.info(f"EBSA completed: |L|={len(L_y)}")

        # --- Algorithm 4 Step 14: return final classifier H and expanded L ---
        final_model = EBSAEnsemble(H)
        return final_model, L_X, L_y