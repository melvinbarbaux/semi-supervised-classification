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

class AdaptiveThresholdingMethod(SemiSupervisedMethod):
    def __init__(
        self,
        base_model: BaseModel,
        tau1: float = 0.8,
        max_iter: int = 10,
        lambda_u: float = 1.0,
        random_state: int = None,
        verbose: bool = False,
        update_s_freq: int = 1,
    ):
        super().__init__(base_model)
        self.base_model     = base_model
        self.tau1           = tau1
        self.max_iter       = max_iter
        self.lambda_u       = lambda_u
        self.rng            = np.random.RandomState(random_state)
        self.verbose        = verbose
        self.update_s_freq  = update_s_freq
        self.s              = None

    def run(
        self,
        X_l: np.ndarray,
        y_l: np.ndarray,
        X_u: np.ndarray
    ) -> Tuple[BaseModel, np.ndarray, np.ndarray]:
        if self.verbose:
            logger.info(
                "Starting AdaptiveThresholding: tau1=%.2f, max_iter=%d, λ_u=%.2f, update_s_freq=%d",
                self.tau1, self.max_iter, self.lambda_u, self.update_s_freq
            )

        # record original labeled count for weighting
        n_initial = len(y_l)

        # 1) Warm start: supervised only
        L_X, L_y = X_l.copy(), y_l.copy()
        U = X_u.copy()
        model = deepcopy(self.base_model)
        model.train(L_X, L_y)

        for it in range(self.max_iter):
            n_u = U.shape[0]
            if self.verbose:
                logger.info(
                    "Iteration %d/%d: unlabeled remaining=%d, labeled so far=%d",
                    it + 1, self.max_iter, n_u, len(L_y)
                )
            if n_u == 0:
                if self.verbose:
                    logger.info("No unlabeled instances left, stopping.")
                break

            # 2) Use all unlabeled
            X_pool = U

            # 3) Pseudo-labels & confidences
            P = model.predict_proba(X_pool)
            if P.shape[0] == 0:
                if self.verbose:
                    logger.info("predict_proba returned empty, stopping.")
                break

            preds = np.argmax(P, axis=1)
            if preds.size == 0:
                if self.verbose:
                    logger.info("No predictions, stopping.")
                break

            # 4) Group confidences by pseudo-class
            classes_pool, counts = np.unique(preds, return_counts=True)
            if counts.size == 0:
                if self.verbose:
                    logger.info("No classes detected, stopping.")
                break
            confidences = {
                c: np.sort(P[preds == c, c])[::-1]
                for c in classes_pool
            }

            # 5) Reference class & rho (dynamic as before)
            ref_class = classes_pool[np.argmax(counts)]
            C1 = confidences.get(ref_class, np.array([]))
            if C1.size == 0:
                if self.verbose:
                    logger.info("Empty confidences for ref_class %d, stopping.", ref_class)
                break
            sel1 = np.sum(C1 >= self.tau1)
            rho  = sel1 / float(len(C1))
            if self.verbose:
                logger.info("ref_class=%d, sel1=%d, rho=%.4f", ref_class, sel1, rho)

            # 6) Compute τ_k and update s-vector only every update_s_freq iterations
            if it % self.update_s_freq == 0:
                n_cls = P.shape[1]
                tau   = {}
                for k in range(n_cls):
                    conf_k = np.sort(P[preds == k, k])[::-1]
                    if conf_k.size == 0:
                        tau[k] = 1.0
                    else:
                        idx = int(np.ceil(rho * len(conf_k))) - 1
                        idx = max(0, min(idx, len(conf_k) - 1))
                        tau[k] = conf_k[idx]
                tau_arr = np.array([tau[k] for k in range(n_cls)])
                self.s = -np.log(tau_arr)
                if self.verbose:
                    logger.info("Updated s-vector at it %d: %s", it, self.s.tolist())
            else:
                # reuse previous s to get tau thresholds
                tau_arr = np.exp(-self.s)

            # 7) Select pseudo-labeled instances
            keep = [i for i, k in enumerate(preds) if P[i, k] >= tau_arr[k]]
            if not keep:
                if self.verbose:
                    logger.info("No pseudo-labels above thresholds, stopping.")
                break
            if self.verbose:
                logger.info("Selected %d pseudo-labels.", len(keep))

            X_new = U[keep]
            y_new = preds[keep]

            # 8) Remove from U, add to L
            mask = np.ones(n_u, bool)
            mask[keep] = False
            U = U[mask]
            L_X = np.vstack([L_X, X_new])
            L_y = np.concatenate([L_y, y_new])

            # 9) Retrain with unified loss via sample_weight
            model = deepcopy(self.base_model)
            n_pseudo_total = len(L_y) - n_initial
            weights = np.concatenate([
                np.ones(n_initial, dtype=float),
                np.full(n_pseudo_total, self.lambda_u, dtype=float)
            ])
            model.train(L_X, L_y, sample_weight=weights)

        if self.verbose:
            logger.info(
                "AdaptiveThresholding finished. Final labeled size=%d", len(L_y)
            )

        return model, L_X, L_y