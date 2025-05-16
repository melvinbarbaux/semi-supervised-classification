import numpy as np
from sklearn.base import BaseEstimator
from .base import BaseModel

class SklearnModel(BaseModel):
    """
    Generic wrapper for any scikit-learn classifier. 
    Accepts optional sample_weight to weight pseudo-labels.
    """
    def __init__(self, clf: BaseEstimator):
        self.clf = clf

    def train(
        self,
        X_l: np.ndarray,
        y_l: np.ndarray,
        sample_weight: np.ndarray = None
    ) -> None:
        """
        If sample_weight is None → pure supervised.
        Else → fit on the combined dataset with those weights.
        """
        if sample_weight is None:
            self.clf.fit(X_l, y_l)
        else:
            # X_l and y_l already _include_ pseudo-labels appended, 
            # and sample_weight aligns to that concatenation.
            self.clf.fit(X_l, y_l, sample_weight=sample_weight)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if hasattr(self.clf, "predict_proba"):
            return self.clf.predict_proba(X)
        # fallback on decision_function + softmax
        scores = self.clf.decision_function(X)
        scores = scores - scores.max(axis=1, keepdims=True)
        exp   = np.exp(scores)
        return exp / exp.sum(axis=1, keepdims=True)