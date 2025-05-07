# src/ssl_bench/methods/tri_training.py

import numpy as np
from copy import deepcopy
from typing import List, Tuple

from ssl_bench.methods.base import SemiSupervisedMethod
from ssl_bench.models.base import BaseModel

class TriTrainingEnsemble(BaseModel):
    """Vote majoritaire pour Tri-Training — défini au niveau module."""
    def __init__(self, classifiers: List[BaseModel]):
        self.classifiers = classifiers

    def train(self, X, y, X_u=None):
        # déjà entraînés
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        votes = np.stack([clf.predict(X) for clf in self.classifiers], axis=1)
        return np.array([np.bincount(row).argmax() for row in votes])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probas = [clf.predict_proba(X) for clf in self.classifiers]
        stacked = np.stack(probas, axis=2)
        return np.mean(stacked, axis=2)


class TriTrainingMethod(SemiSupervisedMethod):
    """
    Méthode Tri-Training (Zhou & Li, 2005), version complète.
    """
    def __init__(self, base_model: BaseModel, random_state: int = None):
        self.base_model = deepcopy(base_model)
        self.random_state = random_state

    def run(
        self,
        X_labeled: np.ndarray,
        y_labeled: np.ndarray,
        X_unlabeled: np.ndarray
    ) -> Tuple[BaseModel, np.ndarray, np.ndarray]:
        rng = np.random.RandomState(self.random_state)
        L_X = X_labeled.copy()
        L_y = y_labeled.copy()
        U = X_unlabeled.copy()

        # --- Phase 1 : bootstrap initial
        h = []
        e_prev = [0.5] * 3
        l_prev = [0] * 3
        for i in range(3):
            idx = rng.choice(len(L_X), size=len(L_X), replace=True)
            clf = deepcopy(self.base_model)
            clf.train(L_X[idx], L_y[idx])
            h.append(clf)

        # --- Phase 2 : itération jusqu’à convergence
        while True:
            if U.shape[0] == 0:
                break
            updated = False
            for i in range(3):
                j, k = (i+1) % 3, (i+2) % 3

                # 1) erreur conjointe sur L_X
                pj = h[j].predict(L_X)
                pk = h[k].predict(L_X)
                agree = pj == pk
                agree_count = agree.sum()
                if agree_count == 0:
                    e_i = 0.0
                else:
                    e_i = ((pj[agree] != L_y[agree]).sum()) / agree_count

                # 2) sélectionner U où hj==hk
                pj_u = h[j].predict(U)
                pk_u = h[k].predict(U)
                mask = pj_u == pk_u
                L_i_X = U[mask]
                L_i_y = pj_u[mask]

                # 3) conditions d’ajout (Eq.9–10)
                do_update = False
                if e_i < e_prev[i] and L_i_X.shape[0] > 0:
                    if l_prev[i] == 0:
                        l_prev[i] = max(1, int(np.floor(e_i / (e_prev[i] - e_i)))) if e_i < e_prev[i] else 1
                        do_update = True
                    else:
                        cond1 = (L_i_X.shape[0] > l_prev[i] and
                                 e_i * L_i_X.shape[0] < e_prev[i] * l_prev[i])
                        cond2 = False
                        if e_i > 0 and (e_prev[i] - e_i) > 0 and l_prev[i] > e_i / (e_prev[i] - e_i):
                            s = int(np.ceil(e_prev[i] * l_prev[i] / e_i - 1))
                            if 0 < s < L_i_X.shape[0]:
                                sel = rng.choice(L_i_X.shape[0], size=s, replace=False)
                                L_i_X = L_i_X[sel]
                                L_i_y = L_i_y[sel]
                                cond2 = True
                        do_update = cond1 or cond2

                if do_update:
                    # 4) mettre à jour L_X, L_y et U
                    idxs = np.where(mask)[0]
                    L_X = np.vstack([L_X, L_i_X])
                    L_y = np.concatenate([L_y, L_i_y])
                    U = np.delete(U, idxs, axis=0)

                    # 5) réentraîner hi
                    new_clf = deepcopy(self.base_model)
                    new_clf.train(L_X, L_y)
                    h[i] = new_clf
                    e_prev[i] = e_i
                    l_prev[i] = L_i_X.shape[0]
                    updated = True

            if not updated:
                break

        # --- Phase 3 : vote final
        final_model = TriTrainingEnsemble(h)
        return final_model, L_X, L_y