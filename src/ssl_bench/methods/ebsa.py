# src/ssl_bench/methods/ebsa.py

"""
EBSA: Fast semi-supervised self-training based on data editing
Bing Lia, Jikui Wang, Zhengguo Yang, Jihai Yi & Feiping Nie
"""

import numpy as np
import logging
from copy import deepcopy
from typing import Tuple, List

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
        # Entraînement déjà effectué dans run()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)


class EBSAMethod(SemiSupervisedMethod):
    def __init__(self, base_model: BaseModel, random_state: int = None):
        super().__init__(base_model)
        self.base_model = deepcopy(base_model)
        self.rng = np.random.RandomState(random_state)

    def run(
        self,
        X_l: np.ndarray,
        y_l: np.ndarray,
        X_u: np.ndarray
    ) -> Tuple[BaseModel, np.ndarray, np.ndarray]:
        # 1) Initialisation
        L_X = X_l.copy()
        L_y = y_l.copy()
        U_X = X_u.copy()
        H = deepcopy(self.base_model)
        H.train(L_X, L_y)
        logger.info(f"EBSA: start |L|={len(L_y)}, |U|={len(U_X)}")

        # 2) Boucle principale
        while True:
            # Si plus rien à annoter, on sort
            if U_X.shape[0] == 0:
                break

            # 3) Pseudo-étiquette
            pseudo = H.predict(U_X)

            # 4) Regrouper par étiquette
            clusters = {}
            for idx, lbl in enumerate(pseudo):
                clusters.setdefault(lbl, []).append(idx)

            new_indices = []

            # 5) Pour chaque cluster Cp
            for lbl, idxs in clusters.items():
                Xp = U_X[idxs]
                if Xp.shape[0] == 0:
                    continue

                # 6) Centre et distances
                center = Xp.mean(axis=0)
                dists = np.linalg.norm(Xp - center, axis=1)
                radius = dists.max()

                # 7) Voisins de centres
                neighbor_centers = []
                for other_lbl, other_idxs in clusters.items():
                    if other_lbl == lbl:
                        continue
                    oc = U_X[other_idxs].mean(axis=0)
                    d_cent = np.linalg.norm(center - oc)
                    if d_cent/2 < radius:
                        neighbor_centers.append((oc, d_cent))

                # 8) Rayon stable br
                if neighbor_centers:
                    min_dist = min(d_cent for _, d_cent in neighbor_centers)
                    br = min_dist / 2.0
                else:
                    br = radius

                # 9) Région disputée
                disputed = set()
                for oc, d_cent in neighbor_centers:
                    er = d_cent / 2.0
                    d_other = np.linalg.norm(Xp - oc, axis=1)
                    disputed.update(np.where(d_other <= er)[0].tolist())

                # 10) Points stables
                stable_idx = np.where(dists <= br)[0]
                for j in stable_idx:
                    # pas besoin de réétiqueter, mais conservé pour cohérence avec article
                    pseudo[idxs[j]] = lbl

                # 11) Ajouter au set à étiqueter
                for j in range(len(idxs)):
                    if j not in disputed:
                        new_indices.append(idxs[j])

            new_indices = sorted(set(new_indices))
            if not new_indices:
                break

            # 12) MàJ L et U
            X_new = U_X[new_indices]
            y_new = pseudo[new_indices]
            L_X = np.vstack([L_X, X_new])
            L_y = np.concatenate([L_y, y_new])
            mask = np.ones(len(U_X), dtype=bool)
            mask[new_indices] = False
            U_X = U_X[mask]

            # 13) Réentraînement
            H = deepcopy(self.base_model)
            H.train(L_X, L_y)
            logger.info(f"EBSA iter: added={len(new_indices)}, remaining U={len(U_X)}")

        logger.info(f"EBSA completed: |L|={len(L_y)}")
        final_model = EBSAEnsemble(H)
        return final_model, L_X, L_y