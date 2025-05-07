import numpy as np
from copy import deepcopy
from typing import Tuple, List
from sklearn.metrics.pairwise import rbf_kernel
import logging

from ssl_bench.methods.base import SemiSupervisedMethod
from ssl_bench.models.base import BaseModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)


class EnsembleMSSBoost(BaseModel):
    """
    Modèle final MSSBoost : vote pondéré des learners avec leurs alphas.
    Défini au niveau module pour permettre le pickling.
    """
    def __init__(self, learners: List[BaseModel], alphas: List[float]):
        self.learners = learners
        self.alphas   = alphas

    def train(self, X, y, X_u=None):
        # Entraînement déjà effectué individuellement
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba_sum = sum(
            α * clf.predict_proba(X)
            for clf, α in zip(self.learners, self.alphas)
        )
        return proba_sum.argmax(axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        stacked = np.stack([
            clf.predict_proba(X) for clf in self.learners
        ], axis=2)
        return np.mean(stacked, axis=2)


class MSSBoostMethod(SemiSupervisedMethod):
    """
    MSSBoost (Tanha et al. 2018): semi-supervised multiclass boosting.
    Pseudo-code fidèle aux Eqs. (14)–(17) de l'article.
    """
    def __init__(
        self,
        base_model: BaseModel,
        n_estimators: int = 20,
        lambda_u: float = 0.1,
        gamma: float = 0.5,           # pour le noyau RBF
    ):
        self.base_model   = base_model
        self.n_estimators = n_estimators
        self.lambda_u     = lambda_u
        self.gamma        = gamma

    def run(
        self,
        X_l: np.ndarray,
        y_l: np.ndarray,
        X_u: np.ndarray
    ) -> Tuple[BaseModel, np.ndarray, np.ndarray]:
        # 1) Copies des données
        L_X_img = X_l.copy()
        L_y     = y_l.copy()
        U_img   = X_u.copy()

        nL, nU = len(L_y), len(U_img)
        n_classes = int(np.unique(L_y).size)

        logger.info(f"MSSBoost: démarrage T={self.n_estimators}, λ_u={self.lambda_u:.3f}, γ={self.gamma:.3f}")
        logger.info(f"  • Labeled images shape = {L_X_img.shape}, Unlabeled images shape = {U_img.shape}")

        # 2) Aplatir pour kernel RBF et wrappers sklearn
        L_flat = L_X_img.reshape(nL, -1)
        U_flat = U_img.reshape(nU, -1)
        S_LU = rbf_kernel(L_flat, U_flat, gamma=self.gamma)
        logger.info(f"  • Kernel RBF calculé : S_LU.shape = {S_LU.shape}")

        learners: List[BaseModel] = []
        alphas:   List[float]     = []

        for t in range(self.n_estimators):
            logger.info(f"MSSBoost: itération {t+1}/{self.n_estimators}")

            # 3) Poids w_i (Eq.14)
            f_L = self._ensemble_score(L_X_img, learners, alphas, n_classes)
            w_i = np.zeros(nL)
            for i in range(nL):
                c = L_y[i]
                margin = f_L[i, c] - np.max(np.delete(f_L[i], c))
                w_i[i] = 0.5 * np.exp(-0.5 * margin)

            # 4) Poids v_j (Eq.15)
            f_U = self._ensemble_score(U_img, learners, alphas, n_classes)
            v_j = np.zeros(nU)
            for j in range(nU):
                margins = np.array([
                    f_U[j, L_y[i]] - np.max(np.delete(f_U[j], L_y[i]))
                    for i in range(nL)
                ])
                v_j[j] = 0.5 * np.dot(S_LU[:, j], np.exp(-0.5 * margins))

            # 5) Pseudo-labels
            if learners:
                pseudo = f_U.argmax(axis=1)
            else:
                maj = np.argmax(np.bincount(L_y))
                pseudo = np.full(nU, maj, dtype=int)

            # 6) Training set pondéré
            X_train_img = np.vstack([L_X_img, U_img])
            y_train     = np.concatenate([L_y, pseudo])
            sample_w    = np.concatenate([
                w_i,
                self.lambda_u * v_j / (v_j.sum() + 1e-12)
            ])
            logger.info(f"  • Training set size = {X_train_img.shape[0]}, sample_weights sum = {sample_w.sum():.3f}")

            # 7) Entraînement de g_t
            clf = deepcopy(self.base_model)
            if hasattr(clf, "clf"):
                # wrapper sklearn
                X_train_flat = X_train_img.reshape(len(X_train_img), -1)
                clf.clf.fit(X_train_flat, y_train, sample_weight=sample_w)
            else:
                # wrapper torch
                clf.train(X_train_img, y_train)

            # 8) α_t (Eq.16)
            preds_L = clf.predict(L_X_img)
            err_t   = np.sum(w_i * (preds_L != L_y)) / (w_i.sum() + 1e-12)
            alpha_t = 0.5 * np.log((1 - err_t) / (err_t + 1e-12))
            logger.info(f"  • err_t={err_t:.3f} → α_t={alpha_t:.3f}")

            learners.append(clf)
            alphas.append(alpha_t)

        # 9) Modèle final
        final_model = EnsembleMSSBoost(learners, alphas)
        return final_model, L_X_img, L_y

    def _ensemble_score(
        self,
        X: np.ndarray,
        learners: List[BaseModel],
        alphas: List[float],
        n_classes: int
    ) -> np.ndarray:
        if not learners:
            return np.zeros((len(X), n_classes))
        return sum(
            α * clf.predict_proba(X)
            for clf, α in zip(learners, alphas)
        )
