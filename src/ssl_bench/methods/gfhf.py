import numpy as np
import logging
from typing import Tuple

from ssl_bench.methods.base import SemiSupervisedMethod
from ssl_bench.models.base import BaseModel
from ssl_bench.datamodule.graph.base import GraphBuilder

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(fmt)
    logger.addHandler(ch)

class GFHFModel(BaseModel):
    """
    Gaussian Fields & Harmonic Functions (Zhu et al., 2003) result wrapper.
    Stores labels for the full dataset (labeled + unlabeled) and provides
    predict / predict_proba for the unlabeled portion.
    """
    def __init__(self, preds_all: np.ndarray, n_labeled: int, classes: np.ndarray):
        self.preds_all = preds_all
        self.n_labeled = n_labeled
        self.classes = classes
        self.class_to_index = {c: i for i, c in enumerate(classes)}

    def train(self, X, y, X_u=None):
        # No additional training required
        return self

    def predict(self, X) -> np.ndarray:
        # Return predictions for unlabeled data
        return self.preds_all[self.n_labeled:]

    def predict_proba(self, X) -> np.ndarray:
        # One-hot encoding of hard predictions
        preds_u = self.predict(X)
        n_u = len(preds_u)
        n_c = len(self.classes)
        probs = np.zeros((n_u, n_c))
        for i, lbl in enumerate(preds_u):
            idx = self.class_to_index[lbl]
            probs[i, idx] = 1.0
        return probs

class GFHFMethod(SemiSupervisedMethod):
    """
    Implementation of Gaussian Fields & Harmonic Functions (Zhu et al., 2003).
    Builds a graph, computes the combinatorial Laplacian, solves the harmonic
    solution for unlabeled nodes via closed-form linear system.
    """
    def __init__(self, graph_builder: GraphBuilder, verbose: bool = False):
        super().__init__(None)
        self.graph_builder = graph_builder
        self.verbose = verbose

    def run(
        self,
        X_l: np.ndarray,
        y_l: np.ndarray,
        X_u: np.ndarray
    ) -> Tuple[BaseModel, np.ndarray, np.ndarray]:
        # 1) Concatenate labeled and unlabeled data
        X_all = np.vstack([X_l, X_u])
        n_l = X_l.shape[0]
        n_u = X_u.shape[0]

        if self.verbose:
            logger.info(f"GFHF start: |L|={n_l}, |U|={n_u}")

        # 2) Fit graph builder and get adjacency W
        self.graph_builder.fit(X_all)
        W = self.graph_builder.adjacency_matrix()

        # 3) Compute Laplacian L = D - W
        D = np.diag(W.sum(axis=1))
        L = D - W

        # 4) Partition L into blocks
        L_ul = L[n_l:, :n_l]
        L_uu = L[n_l:, n_l:]

        # 5) One-hot encode y_l
        classes = np.unique(y_l)
        idx_map = {c: i for i, c in enumerate(classes)}
        Y_l = np.zeros((n_l, len(classes)))
        for i, c in enumerate(y_l):
            Y_l[i, idx_map[c]] = 1

        # 6) Solve harmonic function: f_u = - L_uu^{-1} L_ul Y_l
        B = - L_ul.dot(Y_l)
        try:
            f_u = np.linalg.solve(L_uu, B)
        except np.linalg.LinAlgError:
            jitter = 1e-6 * np.eye(L_uu.shape[0])
            f_u = np.linalg.solve(L_uu + jitter, B)

        # 7) Assign labels by argmax and assemble full label vector
        preds_u = classes[np.argmax(f_u, axis=1)]
        preds_all = np.concatenate([y_l, preds_u])

        model = GFHFModel(preds_all, n_labeled=n_l, classes=classes)

        if self.verbose:
            logger.info(f"GFHF completed: labeled used = {n_l}")
        return model, X_l, y_l