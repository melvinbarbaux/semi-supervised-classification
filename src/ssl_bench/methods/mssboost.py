# src/ssl_bench/methods/mssboost.py

import numpy as np
from copy import deepcopy
from typing import Tuple, List
from sklearn.metrics.pairwise import rbf_kernel

from ssl_bench.methods.base import SemiSupervisedMethod
from ssl_bench.models.base import BaseModel

class MSSBoostMethod(SemiSupervisedMethod):
    """
    MSSBoost (Tanha et al. 2018) : semi-supervised multiclass boosting.
    Implémentation fidèle des Eqs. (14)–(17) de l'article.
    """
    def __init__(
        self,
        base_model: BaseModel,
        n_estimators: int = 20,
        lambda_u: float = 0.1,
        gamma: float = 0.5,           # pour le noyau RBF de similarité
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
        # --- Initialisation (Alg.1) ---
        L_X, L_y = X_l.copy(), y_l.copy()  # données labellisées
        U        = X_u.copy()              # données non-labellisées
        nL, nU   = len(L_y), len(U)
        n_classes = int(np.unique(L_y).size)

        # ensembles de base learners et leurs poids alpha
        learners: List[BaseModel] = []  # {g1...gT}
        alphas:   List[float]      = []  # {α1...αT}

        # Pré-calcul similarité RBF S(x_i, x_j) entre L et U (Eq.15)
        S_LU = rbf_kernel(L_X, U, gamma=self.gamma)

        for t in range(self.n_estimators):
            # --- Étape 1 (Eq.14): w_i pour chaque xi∈L ---
            f_L = self._ensemble_score(L_X, learners, alphas, n_classes)
            w_i = np.zeros(nL)
            for i in range(nL):
                c = L_y[i]
                true_score = f_L[i, c]
                # score max sur classes ≠ c
                other_score = np.max(np.delete(f_L[i], c))
                margin = true_score - other_score
                w_i[i] = 0.5 * np.exp(-0.5 * margin)

            # --- Étape 2 (Eq.15): v_j pour chaque x̃j∈U ---
            f_U = self._ensemble_score(U, learners, alphas, n_classes)
            v_j = np.zeros(nU)
            for j in range(nU):
                # calcul margin par échantillon de L
                margins = np.zeros(nL)
                for i in range(nL):
                    c = L_y[i]
                    margins[i] = f_U[j, c] - np.max(np.delete(f_U[j], c))
                v_j[j] = 0.5 * np.sum(S_LU[:, j] * np.exp(-0.5 * margins))

            # --- Étape 3: pseudo-étiquettes via vote de f_{t-1} ---
            if learners:
                pseudo_labels = np.argmax(f_U, axis=1)
            else:
                majority = np.argmax(np.bincount(L_y))
                pseudo_labels = np.full(nU, majority, dtype=int)

            # former l'ensemble d'entraînement pondéré
            X_train = np.vstack([L_X, U])
            y_train = np.concatenate([L_y, pseudo_labels])
            sample_weights = np.concatenate([
                w_i,
                self.lambda_u * v_j / (np.sum(v_j) + 1e-12)
            ])

            # --- Étape 4: entraîner g_t ---
            clf = deepcopy(self.base_model)
            if hasattr(clf, 'clf'):
                clf.clf.fit(X_train, y_train, sample_weight=sample_weights)
            else:
                clf.train(X_train, y_train)

            # --- Étape 5 (Eq.16): calcul α_t via erreur pondérée sur L ---
            preds_L = clf.predict(L_X)
            err_t = np.sum(w_i * (preds_L != L_y)) / (np.sum(w_i) + 1e-12)
            alpha_t = 0.5 * np.log((1 - err_t) / (err_t + 1e-12))

            learners.append(clf)
            alphas.append(alpha_t)

        # --- Modèle final (Eq.17) : vote pondéré ---
        class EnsembleMSSBoost(BaseModel):
            def __init__(self, learners: List[BaseModel], alphas: List[float]):
                self.learners = learners
                self.alphas   = alphas

            def train(self, X, y, X_u=None): pass

            def predict(self, X: np.ndarray) -> np.ndarray:
                proba_sum = sum(
                    alpha * learner.predict_proba(X)
                    for learner, alpha in zip(self.learners, self.alphas)
                )
                return proba_sum.argmax(axis=1)

            def predict_proba(self, X: np.ndarray) -> np.ndarray:
                stacked = np.stack([
                    learner.predict_proba(X) for learner in self.learners
                ], axis=2)
                return np.mean(stacked, axis=2)

        return EnsembleMSSBoost(learners, alphas), L_X, L_y

    def _ensemble_score(
        self,
        X: np.ndarray,
        learners,
        alphas,
        n_classes: int
    ) -> np.ndarray:
        """
        Calcule f_t(x) = somme_{s < t} α_s h_s.predict_proba(x)
        Retourne tableau (n_samples, n_classes).
        """
        if not learners:
            return np.zeros((X.shape[0], n_classes))
        proba_list = [
            alpha * clf.predict_proba(X)
            for clf, alpha in zip(learners, alphas)
        ]
        return np.sum(proba_list, axis=0)
