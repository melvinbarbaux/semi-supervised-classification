"""
SELF-TRAINING: Unsupervised Word Sense Disambiguation Rivaling Supervised Methods
Yarowsky, 1995.
"""

import numpy as np
import logging
from copy import deepcopy
from typing import Tuple
from .base import SemiSupervisedMethod
from ssl_bench.models.base import BaseModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    logger.addHandler(ch)


class SelfTrainingMethod(SemiSupervisedMethod):
    def __init__(
        self,
        base_model: BaseModel,
        threshold: float = 0.8,
        max_iter: int = 5,
        verbose: bool = False
    ):
        """
        :param base_model: classifier providing train / predict / predict_proba
        :param threshold: confidence threshold for pseudo-labels
        :param max_iter: maximum number of self-training rounds
        """
        self.base_model = deepcopy(base_model)
        self.threshold = threshold
        self.max_iter = max_iter
        self.verbose = verbose

    def run(
        self,
        X_l: np.ndarray,
        y_l: np.ndarray,
        X_u: np.ndarray
    ) -> Tuple[BaseModel, np.ndarray, np.ndarray]:
        """
        :returns: (trained_model, X_l_enlarged, y_l_enlarged)
        """
        assert X_u is not None, "SelfTraining requires an unlabeled set X_u"

        # 0) Initialize
        L_X = X_l.copy()
        L_y = y_l.copy()
        U_X = X_u.copy()
        h = deepcopy(self.base_model)

        if self.verbose:
            logger.info(f"SelfTraining start: |L|={len(L_y)}, |U|={len(U_X)}, thr={self.threshold}")

        # 1..max_iter:
        for it in range(1, self.max_iter + 1):
            if U_X.shape[0] == 0:
                logger.info("SelfTraining: no remaining unlabeled samples, stopping")
                break

            # 1) Train on L
            h.train(L_X, L_y)

            # 2) Predict on U
            probs = h.predict_proba(U_X)
            confidences = probs.max(axis=1)

            # 3) Select indices above threshold
            sel = np.where(confidences >= self.threshold)[0]
            if sel.size == 0:
                if self.verbose:
                    logger.info(f"SelfTraining: iteration {it}, no examples >= threshold, stopping")
                break

            # 4) Add pseudo-labels to L, remove from U
            new_X = U_X[sel]
            new_y = probs[sel].argmax(axis=1)
            L_X = np.vstack([L_X, new_X])
            L_y = np.concatenate([L_y, new_y])
            U_X = np.delete(U_X, sel, axis=0)

            if self.verbose:
                logger.info(f"Iter {it}: added {len(sel)} pseudo-labels → |L|={len(L_y)}, |U|={len(U_X)}")

        if self.verbose:
            logger.info(f"SelfTraining end: |L|={len(L_y)}, |U|={len(X_l)} (added {len(L_y) - len(X_l)}), it={it}")

        return h, L_X, L_y