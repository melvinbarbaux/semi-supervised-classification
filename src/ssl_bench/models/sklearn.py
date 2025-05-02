import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from .base import BaseModel

class SklearnModel(BaseModel):
    """
    Wrapper générique pour classifieurs scikit-learn.
    Implémente train, predict et predict_proba via le classifieur sous-jacent.
    """
    def __init__(self, clf):
        self.clf = clf

    def train(self, X_l, y_l, X_u=None):
        # X_u (non étiquetées) est ignoré par les classifieurs purement supervisés
        self.clf.fit(X_l, y_l)

    def predict(self, X):
        return self.clf.predict(X)

    def predict_proba(self, X):
        # si predict_proba existe, on l'utilise
        if hasattr(self.clf, 'predict_proba'):
            return self.clf.predict_proba(X)
        # sinon on fait un softmax sur decision_function
        scores = self.clf.decision_function(X)
        # stabilisation numérique
        scores = scores - scores.max(axis=1, keepdims=True)
        exp_scores = np.exp(scores)
        return exp_scores / exp_scores.sum(axis=1, keepdims=True)

class RandomForestWrapper(SklearnModel):
    """Classifieur Forêt aléatoire."""
    def __init__(self, **kwargs):
        super().__init__(RandomForestClassifier(**kwargs))

class SVMWrapper(SklearnModel):
    """Classifieur SVM avec probabilités."""
    def __init__(self, probability=True, **kwargs):
        super().__init__(SVC(probability=probability, **kwargs))