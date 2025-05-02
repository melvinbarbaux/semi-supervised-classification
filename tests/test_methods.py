# tests/test_methods.py

import numpy as np
import pytest

from ssl_bench.models.sklearn import RandomForestWrapper
from ssl_bench.methods.self_training import SelfTrainingMethod
from ssl_bench.methods.supervised import SupervisedMethod

def test_self_training_method():
    # 20 points en 2 classes
    X = np.vstack([np.random.randn(10, 2) - 2, np.random.randn(10, 2) + 2])
    y = np.array([0]*10 + [1]*10)

    # on n'étiquette que 5 points
    X_l, y_l = X[:5], y[:5]
    X_u = X[5:]

    model = RandomForestWrapper(n_estimators=5)
    method = SelfTrainingMethod(model, threshold=0.6, max_iter=3)
    trained_model, Xl2, yl2 = method.run(X_l, y_l, X_u)

    # Au moins les 5 originaux doivent être présents
    assert Xl2.shape[0] >= 5
    assert yl2.shape[0] == Xl2.shape[0]

    # On peut prédire sur tout le dataset
    preds = trained_model.predict(X)
    assert preds.shape == (20,)

def test_supervised_method_baseline():
    # 20 points en 2 classes
    X = np.vstack([np.zeros((10, 3)), np.ones((10, 3))])
    y = np.array([0]*10 + [1]*10)
    X_u = np.random.randn(5, 3)  # non utilisé

    model = RandomForestWrapper(n_estimators=5)
    method = SupervisedMethod(model)
    trained_model, Xf, yf = method.run(X, y, X_u)

    # Jeu étiqueté final identique au jeu d'entrée
    assert Xf.shape == X.shape
    assert np.array_equal(yf, y)

    # Performance baseline raisonnable (>80% sur le train)
    preds = trained_model.predict(X)
    acc = (preds == y).mean()
    assert acc > 0.8