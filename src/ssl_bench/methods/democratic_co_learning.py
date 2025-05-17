"""
Democratic Co-Learning 
Zhou & Goldman, 2004.
"""

import numpy as np
import logging
from copy import deepcopy
from typing import List, Tuple, Any
from scipy.stats import norm

from ssl_bench.methods.base import SemiSupervisedMethod
from ssl_bench.models.base import BaseModel

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    ch.setFormatter(fmt)
    logger.addHandler(ch)


class DemocraticEnsemble(BaseModel):
    def __init__(
        self,
        learners: List[BaseModel],
        weights: List[float],
        model_indices: List[np.ndarray],
        instances_index: np.ndarray,
        model_index_map: List[np.ndarray],
        verbose: bool = False
    ):
        super().__init__()
        self.learners = learners
        self.weights = weights
        self.model_indices = model_indices
        self.instances_index = instances_index
        self.model_index_map = model_index_map
        self.verbose = verbose

    def train(self, X, y, X_u=None):
        # Already trained during run
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        # 1) Stack the predictions from each classifier
        votes = np.stack([clf.predict(X) for clf in self.learners], axis=1)

        # 2) Select classifiers with weight > 0.5
        selected = [i for i, w in enumerate(self.weights) if w > 0.5]
        if not selected:
            raise ValueError("No classifier selected: all weights ≤ 0.5")

        preds = []
        for row in votes:
            # Count votes and weight sums per class
            vote_counts = {}  # number of votes per class
            weight_sums = {}  # sum of weights per class
            for i in selected:
                c = row[i]
                vote_counts[c] = vote_counts.get(c, 0) + 1
                weight_sums[c] = weight_sums.get(c, 0.0) + self.weights[i]

            # Compute Laplace-corrected score for each class
            scores = {}
            for c, count in vote_counts.items():
                # (count + 0.5) / (count + 1) * (weight_sums[c] / count)
                scores[c] = (count + 0.5) / (count + 1) * (weight_sums[c] / count)

            # Choose the class with the max score
            preds.append(max(scores, key=scores.get))

        return np.array(preds)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # For consistency, only sum probabilities of selected classifiers
        selected = [i for i, w in enumerate(self.weights) if w > 0.5]
        if not selected:
            raise ValueError("No classifier selected: all weights ≤ 0.5")

        probas = [
            clf.predict_proba(X) * self.weights[i]
            for i, clf in enumerate(self.learners)
            if i in selected
        ]
        stacked = np.stack(probas, axis=2)
        total_w = sum(self.weights[i] for i in selected)
        return np.sum(stacked, axis=2) / total_w


class DemocraticCoLearningMethod(SemiSupervisedMethod):
    def __init__(
        self,
        learners: List[BaseModel],
        *,
        alpha: float = 0.05,
        random_state: Any = None,
        verbose: bool = False,
        **kwargs
    ):
        super().__init__(learners[0], **kwargs)
        self.learners = [deepcopy(l) for l in learners]
        self.alpha = alpha
        self.rng = np.random.RandomState(random_state)
        self.verbose = verbose

    def run(
        self,
        X_l: np.ndarray,
        y_l: np.ndarray,
        X_u: np.ndarray
    ) -> Tuple[BaseModel, np.ndarray, np.ndarray]:
        # 1) Build combined dataset for index tracking
        L0 = len(X_l)
        U0 = len(X_u)
        full_X = np.vstack([X_l, X_u])
        full_y = np.concatenate([y_l, np.array([None]*U0)])
        labeled_idx = np.arange(L0)
        unlabeled_idx = np.arange(L0, L0+U0)

        # 2) Initial training and error count
        M = len(self.learners)
        Li_idx = [labeled_idx.copy() for _ in range(M)]
        errors = [0.0] * M
        for i, clf in enumerate(self.learners):
            Xi = full_X[Li_idx[i]]
            yi = full_y[Li_idx[i]].astype(int)
            clf.train(Xi, yi)
            pred = clf.predict(Xi)
            errors[i] = (pred != yi).sum()
        nL = len(labeled_idx)

        # New log: summarize start of algorithm
        if self.verbose:
            logger.info(f"Starting Democratic Co-Learning: M={M}, alpha={self.alpha}")
            logger.info(f"Iteration 1 start: remaining unlabeled={len(unlabeled_idx)}")

        # 3) Compute 95% CI–based confidence weights
        weights = []
        for i, clf in enumerate(self.learners):
            pred = clf.predict(full_X[labeled_idx])
            pi = np.mean(pred == y_l)
            delta = norm.ppf(1 - self.alpha/2) * np.sqrt(pi*(1-pi)/nL)
            lo, hi = max(0, pi-delta), min(1, pi+delta)
            weights.append((lo + hi) / 2)
        if self.verbose:
            logger.info(f"Democratic weights: {[float(w) for w in weights]}")

        # 4) Iterative pseudo-labeling & retraining
        changed = True
        iteration = 1
        while changed and len(unlabeled_idx) > 0:
            if self.verbose:
                logger.info(f"Iteration {iteration} loop: remaining unlabeled={len(unlabeled_idx)}")
            changed = False

            U_X = full_X[unlabeled_idx]
            probas = [clf.predict_proba(U_X) for clf in self.learners]
            hat = np.array([p.argmax(axis=1) for p in probas]).T

            prop_idx = [[] for _ in range(M)]
            prop_label = [[] for _ in range(M)]
            if self.verbose:
                logger.info(f"Proposal slots init: {[len(p) for p in prop_idx]}")
            for j, idx in enumerate(unlabeled_idx):
                lbls = hat[j]
                classes = np.unique(lbls)
                score = {c: sum(weights[i] for i in np.where(lbls==c)[0]) for c in classes}
                c_star = max(score, key=score.get)
                if any(score[c] >= score[c_star] for c in classes if c != c_star):
                    continue
                for i in range(M):
                    if lbls[i] != c_star:
                        prop_idx[i].append(idx)
                        prop_label[i].append(c_star)
            if self.verbose:
                logger.info(f"Proposals per classifier before accept/reject: {[len(p) for p in prop_idx]}")

            # Evaluate Q vs Q' and accept valid proposals
            accepted = []
            for i in range(M):
                if not prop_idx[i]:
                    continue
                old_idxs = Li_idx[i]
                new_idxs = np.array(prop_idx[i], dtype=int)
                Li_size = len(old_idxs)
                new_size = len(new_idxs)

                q = Li_size * (1 - 2*(errors[i]/max(1,Li_size)))**2
                p_err = 1 - errors[i]/max(1,Li_size)
                delta = norm.ppf(1 - self.alpha/2) * np.sqrt(p_err*(1-p_err)/nL)
                l_bound = max(0, p_err - delta)
                e_new = new_size * (1 - l_bound)
                union = Li_size + new_size
                q_new = union * (1 - 2*((errors[i] + e_new)/max(1,union)))**2

                if q_new > q:
                    Li_idx[i] = np.concatenate([old_idxs, new_idxs])
                    full_y[new_idxs] = prop_label[i]
                    errors[i] += e_new
                    self.learners[i].train(full_X[Li_idx[i]], full_y[Li_idx[i]].astype(int))
                    changed = True
                    accepted.extend(new_idxs.tolist())

            # Remove only truly accepted examples
            to_remove = np.unique(accepted) if accepted else np.array([], dtype=int)
            unlabeled_idx = np.setdiff1d(unlabeled_idx, to_remove)
            if self.verbose:
                logger.info(
                    f"Iteration {iteration} end: changed={changed}, "
                    f"accepted={len(to_remove)} samples, remaining U={len(unlabeled_idx)}"
                )

            iteration += 1

        # 5) Build final index mappings
        instances_index = np.unique(np.concatenate(Li_idx))
        model_index_map = [
            np.array([np.where(instances_index == idx)[0][0] for idx in inds], dtype=int)
            for inds in Li_idx
        ]

        # 6) Final weighted ensemble
        final_model = DemocraticEnsemble(
            self.learners,
            weights,
            Li_idx,
            instances_index,
            model_index_map
        )

        final_X = full_X[instances_index]
        final_y = full_y[instances_index].astype(int)

        if self.verbose:
            logger.info(f"Democratic completed: labeled={len(instances_index)}, classifiers={M}")
        return final_model, final_X, final_y