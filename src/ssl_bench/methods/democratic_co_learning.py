import numpy as np
from copy import deepcopy
from typing import List, Tuple
import logging

from ssl_bench.methods.base import SemiSupervisedMethod
from ssl_bench.models.base import BaseModel
from scipy.stats import norm

logger = logging.getLogger(__name__)

class DemocraticEnsemble(BaseModel):
    """
    Vote pondéré final pour Democratic Co-Learning.
    """
    def __init__(self, learners: List[BaseModel], weights: List[float]):
        self.learners = learners
        self.weights = weights

    def train(self, X, y, X_u=None):
        # déjà entraînés
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        hat = np.vstack([clf.predict(X) for clf in self.learners]).T
        preds = []
        for row in hat:
            score = {}
            for i, c in enumerate(row):
                score[c] = score.get(c, 0.0) + self.weights[i]
            preds.append(max(score, key=score.get))
        return np.array(preds)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probas = np.stack(
            [clf.predict_proba(X) * w for clf, w in zip(self.learners, self.weights)],
            axis=2
        )
        return np.sum(probas, axis=2) / sum(self.weights)


class DemocraticCoLearningMethod(SemiSupervisedMethod):
    """
    Implémentation fidèle de Democratic Co-Learning (Zhou & Goldman, 2004).
    """
    def __init__(self, learners: List[BaseModel], alpha: float = 0.05, random_state: int = None):
        self.learners = [deepcopy(l) for l in learners]
        self.alpha = alpha
        self.rng = np.random.RandomState(random_state)

    def run(
        self,
        X_l: np.ndarray,
        y_l: np.ndarray,
        X_u: np.ndarray
    ) -> Tuple[BaseModel, np.ndarray, np.ndarray]:
        # --- Étape 1: apprentissage initial ---
        M = len(self.learners)
        L, Y = X_l.copy(), y_l.copy()
        Li_X = [L.copy() for _ in range(M)]
        Li_y = [Y.copy() for _ in range(M)]
        ei = [0]*M

        for i, h in enumerate(self.learners):
            h.train(Li_X[i], Li_y[i])
            ei[i] = int((h.predict(Li_X[i]) != Li_y[i]).sum())

        # --- Calcul des poids wi ---
        nL = len(L)
        wi = []
        for i, h in enumerate(self.learners):
            pi = np.mean(h.predict(L) == Y)
            delta = 1.96*np.sqrt(pi*(1-pi)/nL)
            li, hi = max(0, pi-delta), min(1, pi+delta)
            wi.append((li+hi)/2)

        U = X_u.copy()
        # --- Boucle principale de pseudo-étiquetage & fusion ---
        while True:
            changed = False
            probs = [h.predict_proba(U) for h in self.learners]
            hat = np.array([p.argmax(axis=1) for p in probs]).T

            Li_pX = [[] for _ in range(M)]
            Li_pY = [[] for _ in range(M)]

            for j in range(U.shape[0]):
                lbls = hat[j]
                classes = np.unique(lbls)
                score = {c: sum(wi[i] for i in np.where(lbls==c)[0]) for c in classes}
                c_star = max(score, key=score.get)
                if any(score[c] >= score[c_star] for c in classes if c!=c_star):
                    continue
                for i in range(M):
                    if lbls[i] != c_star:
                        Li_pX[i].append(U[j])
                        Li_pY[i].append(c_star)

            for i in range(M):
                if not Li_pX[i]:
                    continue
                Li_size = len(Li_X[i])
                qi = Li_size*(1-2*(ei[i]/max(1,Li_size)))**2
                pi_err = 1 - ei[i]/max(1,Li_size)
                delta = 1.96*np.sqrt(pi_err*(1-pi_err)/nL)
                li_p = max(0, pi_err-delta)
                Li1_size = len(Li_pX[i])
                ei_p = Li1_size*(1-li_p)
                union = Li_size + Li1_size
                qi_p = union*(1-2*((ei[i]+ei_p)/max(1,union)))**2
                if qi_p < qi:
                    Li_X[i] = np.vstack([Li_X[i], np.vstack(Li_pX[i])])
                    Li_y[i] = np.concatenate([Li_y[i], np.array(Li_pY[i],dtype=int)])
                    ei[i] += ei_p
                    self.learners[i].train(Li_X[i], Li_y[i])
                    changed = True

            if not changed:
                break

        # --- Étape 6: vote final pondéré ---
        final_model = DemocraticEnsemble(self.learners, wi)
        all_L = np.vstack(Li_X)
        unique_L = np.unique(all_L, axis=0)
        dummy_y = np.zeros(len(unique_L), dtype=int)
        return final_model, unique_L, dummy_y