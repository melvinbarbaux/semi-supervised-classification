"""
MSSBoost: multi-class semi-supervised boosting with metric learning 
Tanha & al., 2018.
"""
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
    def __init__(self, learners: List[BaseModel], alphas: List[float]):
        self.learners = learners
        self.alphas   = alphas

    def train(self, X, y, X_u=None):
        # Training already performed individually
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba_sum = sum(
            alpha * clf.predict_proba(X)
            for clf, alpha in zip(self.learners, self.alphas)
        )
        return proba_sum.argmax(axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        stacked = np.stack([
            clf.predict_proba(X) for clf in self.learners
        ], axis=2)
        return np.mean(stacked, axis=2)


class MSSBoostMethod(SemiSupervisedMethod):
    def __init__(
        self,
        base_model: BaseModel,
        n_estimators: int = 20,
        lambda_u: float = 0.1,
        gamma: float = 0.5,
        verbose: bool = False
    ):
        super().__init__(base_model)
        self.base_model   = base_model
        self.n_estimators = n_estimators
        self.lambda_u     = lambda_u
        self.gamma        = gamma
        self.verbose       = verbose

    def run(
        self,
        X_l: np.ndarray,
        y_l: np.ndarray,
        X_u: np.ndarray
    ) -> Tuple[BaseModel, np.ndarray, np.ndarray]:
        # 1) Copy data
        L_X_img = X_l.copy()
        L_y     = y_l.copy()
        U_img   = X_u.copy()

        nL, nU = len(L_y), len(U_img)
        if self.verbose:
            logger.info(f"MSSBoost start: |L|={nL}, |U|={nU}, T={self.n_estimators}, lambda_u={self.lambda_u:.3f}, gamma={self.gamma:.3f}")

        n_classes = int(np.unique(L_y).size)
        
        if self.verbose:
            logger.info(f"  • Labeled X shape = {L_X_img.shape}, Unlabeled X shape = {U_img.shape}")

        # 2) Flatten for RBF kernel and sklearn wrappers
        L_flat = L_X_img.reshape(nL, -1)
        U_flat = U_img.reshape(nU, -1)
        # Initialize diagonal metric A (shape d×d)
        d = L_flat.shape[1]
        A = np.eye(d)
        def compute_similarity(X1, X2, A_mat):
            # RBF with metric A: exp(- (x - x')^T A (x - x'))
            diffs = X1[:, None, :] - X2[None, :, :]
            weighted = np.einsum('ijk,kl,ijl->ij', diffs, A_mat, diffs)
            return np.exp(-weighted)
        S_LU = compute_similarity(L_flat, U_flat, A)
        if self.verbose:
            logger.info(f"  • Initial similarity computed with A=I: S_LU shape = {S_LU.shape}")

        learners: List[BaseModel] = []
        alphas:   List[float]     = []

        for t in range(self.n_estimators):
            if self.verbose:
                logger.info(f"MSSBoost iter {t+1}/{self.n_estimators} - computing weights and metric")

            # 3) Compute weights w_i (Eq.14)
            f_L = self._ensemble_score(L_X_img, learners, alphas, n_classes)
            w_i = np.zeros(nL)
            for i in range(nL):
                c = L_y[i]
                margin = f_L[i, c] - np.max(np.delete(f_L[i], c))
                w_i[i] = 0.5 * np.exp(-0.5 * margin)

            # 4) Compute weights v_j (Eq.15)
            f_U = self._ensemble_score(U_img, learners, alphas, n_classes)
            v_j = np.zeros(nU)
            for j in range(nU):
                margins = np.array([
                    f_U[j, L_y[i]] - np.max(np.delete(f_U[j], L_y[i]))
                    for i in range(nL)
                ])
                v_j[j] = 0.5 * np.dot(S_LU[:, j], np.exp(-0.5 * margins))

            # --- 2.1) Metric learning: update A ---
            # Compute weights W_{ij} and P_{ij} for all (i,j)
            # W_ij = w_i * (self.lambda_u * v_j / (v_j.sum() + 1e-12))
            W = np.outer(w_i, self.lambda_u * v_j / (v_j.sum() + 1e-12))
            # P_{ij} = 1 if pseudo[i] == L_y[i], else 0
            # but as in the paper, use indicator of agreement between f_U[j] and true f_L
            P = np.zeros_like(W)
            for i in range(nL):
                for j in range(nU):
                    P[i, j] = 1 if f_U[j, L_y[i]] > np.max(np.delete(f_U[j], L_y[i])) else 0
            # Compute gradient G = sum_{i,j} P_ij * W_ij * (x_i - x'_j)(x_i - x'_j)^T
            G = np.zeros((d, d))
            for i in range(nL):
                for j in range(nU):
                    diff = L_flat[i] - U_flat[j]
                    G += P[i, j] * W[i, j] * np.outer(diff, diff)
            # Normalize gradient to diagonal form (only keep diagonal)
            g_diag = np.diag(G)
            # Find best A_star as sign of negative gradient on each diag (clipped)
            A_star = np.sign(-g_diag)
            # Line search for optimal step size beta via simple backtracking:
            beta = 1.0
            curr_risk = np.sum(W * P)  # proxy for risk derivative
            for _ in range(10):
                A_trial = A + beta * np.diag(A_star)
                S_LU_trial = compute_similarity(L_flat, U_flat, A_trial)
                # proxy risk: negative sum of weighted similarity
                risk = -np.sum(W * S_LU_trial)
                if risk < curr_risk:
                    break
                beta *= 0.5
            # Update metric matrix and clamp diagonal to non-negative, replace NaNs with zero
            A += beta * np.diag(A_star)
            diagA = np.diag(A)
            # replace NaNs with 0, then clip negatives
            diagA = np.nan_to_num(diagA, nan=0.0, posinf=None, neginf=None)
            diagA_clipped = np.clip(diagA, 0.0, None)
            np.fill_diagonal(A, diagA_clipped)
            S_LU = compute_similarity(L_flat, U_flat, A)
            if self.verbose:
                logger.info(f"  • Metric updated (beta={beta:.4f}), updated A diagonal first entries: {np.diag(A)[:5]}")
                logger.info(f"  • After metric update: first diag(A) entries: {np.diag(A)[:5]}")

            # 5) Pseudo-labels
            if learners:
                pseudo = f_U.argmax(axis=1)
            else:
                maj = np.argmax(np.bincount(L_y))
                pseudo = np.full(nU, maj, dtype=int)

            # 6) Weighted training set
            X_train_img = np.vstack([L_X_img, U_img])
            y_train     = np.concatenate([L_y, pseudo])
            sample_w    = np.concatenate([
                w_i,
                self.lambda_u * v_j / (v_j.sum() + 1e-12)
            ])
            # Replace any NaN or infinite sample weights with zero
            sample_w = np.nan_to_num(sample_w, nan=0.0, posinf=0.0, neginf=0.0)
            if self.verbose:
                logger.info(f"  • Training set size = {X_train_img.shape[0]}, sample_weights sum = {sample_w.sum():.3f}")

            # 7) Train g_t
            clf = deepcopy(self.base_model)
            if hasattr(clf, "clf"):
                # sklearn wrapper
                X_train_flat = X_train_img.reshape(len(X_train_img), -1)
                clf.clf.fit(X_train_flat, y_train, sample_weight=sample_w)
            else:
                # torch wrapper
                clf.train(X_train_img, y_train)

            # 8) alpha_t (Eq.16)
            preds_L = clf.predict(L_X_img)
            err_t   = np.sum(w_i * (preds_L != L_y)) / (w_i.sum() + 1e-12)
            alpha_t = 0.5 * np.log((1 - err_t) / (err_t + 1e-12))
            if self.verbose:
                logger.info(f"  • err_t={err_t:.3f} → alpha_t={alpha_t:.3f}")
                logger.info(f"  • Appending classifier {t+1}, alpha={alpha_t:.3f}")

            learners.append(clf)
            alphas.append(alpha_t)

        if self.verbose:
            logger.info(f"MSSBoost completed: total learners={len(learners)}, original labeled used={nL}")
        # 9) Final model
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
            alpha * clf.predict_proba(X)
            for clf, alpha in zip(learners, alphas)
        )
