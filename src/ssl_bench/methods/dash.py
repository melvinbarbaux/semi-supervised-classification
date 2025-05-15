""""
Dash: Semi-Supervised Learning with Dynamic Thresholding 
Xu et al, 2021
"""

import numpy as np
import logging
from copy import deepcopy
from typing import Tuple, Optional

from ssl_bench.methods.base import SemiSupervisedMethod
from ssl_bench.models.base import BaseModel

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(fmt)
    logger.addHandler(ch)

class DashMethod(SemiSupervisedMethod):
    def __init__(
        self,
        base_model: BaseModel,
        C: float = 1.0001,
        gamma: float = 1.1,
        rho_min: float = 0.0,
        max_iter: int = 10,
        verbose: Optional[bool] = False
    ):
        """
        :param base_model: modèle fournissant train / predict / predict_proba
        :param C: constante C > 1 pour seuil initial (Eq.15)
        :param gamma: facteur gamma > 1 pour décroissance du seuil (Eq.15)
        :param rho_min: seuil minimal autorisé pour ρ_t
        :param max_iter: nombre maximal d'itérations
        """
        self.base_model = base_model
        self.C = C
        self.gamma = gamma
        self.rho_min = rho_min
        self.max_iter = max_iter
        self.verbose = verbose

    def run(
        self,
        X_labeled: np.ndarray,
        y_labeled: np.ndarray,
        X_unlabeled: np.ndarray
    ) -> Tuple[BaseModel, np.ndarray, np.ndarray]:
        """
        Run DASH algorithm.

        :param X_labeled: labeled feature matrix
        :param y_labeled: labeled targets
        :param X_unlabeled: unlabeled feature matrix
        :return: final model, augmented labeled X, augmented labeled y
        """
        # Step 1: Warm-up - initial training on labeled data L
        model = deepcopy(self.base_model)
        L_X = X_labeled.copy()
        L_y = y_labeled.copy()
        model.train(L_X, L_y)

        # Step 2: Estimate rho_hat = mean over L of loss f(x) = -log(max p)
        probs_L = model.predict_proba(L_X)
        max_p_L = np.max(probs_L, axis=1)
        losses_L = -np.log(np.clip(max_p_L, 1e-12, 1.0))
        rho_hat = float(np.mean(losses_L))  # Eq.17

        # U = pool of unlabeled samples
        U = X_unlabeled.copy()
        if self.verbose:
            logger.info(f"DASH start: |L|={len(L_y)}, |U|={len(U)}, C={self.C}, gamma={self.gamma}, rho_min={self.rho_min}")

        # Step 3: Selection iterations t = 1..T
        for t in range(1, self.max_iter + 1):
            # 3.1 Compute dynamic threshold rho_t = C * gamma^{-(t-1)} * rho_hat, floored by rho_min (Eq.15)
            rho_t = self.C * (self.gamma ** (-(t - 1))) * rho_hat
            rho_t = max(self.rho_min, rho_t)

            # 3.2 Compute losses on U: f(x) = -log(max p)
            probs_U = model.predict_proba(U)
            max_p_U = np.max(probs_U, axis=1)
            losses_U = -np.log(np.clip(max_p_U, 1e-12, 1.0))

            # 3.3 Select indices where loss <= rho_t
            idxs = np.where(losses_U <= rho_t)[0]
            if self.verbose:
                logger.info(f"DASH iter {t}: threshold={rho_t:.4f}, selected={len(idxs)}")
            if idxs.size == 0:
                break

            # 3.4 Generate pseudo-labels y_sel = argmax p(x)
            preds_U = np.argmax(probs_U, axis=1)
            X_sel = U[idxs]
            y_sel = preds_U[idxs]

            # 3.5 Augment L with selected pseudo-labeled samples
            L_X = np.vstack([L_X, X_sel])
            L_y = np.concatenate([L_y, y_sel])

            # Remove selected from U to avoid duplicates in subsequent iterations
            mask_remain = np.ones(U.shape[0], dtype=bool)
            mask_remain[idxs] = False
            U = U[mask_remain]

            # 3.6 Retrain model on augmented L
            model = deepcopy(self.base_model)
            model.train(L_X, L_y)

            if self.verbose:
                logger.info(f"DASH completed: final |L|={len(L_y)}, added={len(L_y)-len(y_labeled)}, iters={t}")
        # Return final model and labeled pool
        return model, L_X, L_y
