import numpy as np
from ssl_bench.models.sklearn import RandomForestWrapper
from ssl_bench.methods.supervised import SupervisedMethod

def test_supervised_baseline():
    # jeu simple
    X = np.vstack([np.zeros((5,2)), np.ones((5,2))])
    y = np.array([0]*5 + [1]*5)
    X_l, y_l = X, y
    X_u = np.random.randn(3,2)  # ignoré

    model = RandomForestWrapper(n_estimators=3)
    method = SupervisedMethod(model)
    trained_model, Xf, yf = method.run(X_l, y_l, X_u)

    # Vérifie que le modèle a été entraîné sur X_l / y_l
    preds = trained_model.predict(X_l)
    # au moins >80% d'exactitude sur le même set
    acc = (preds == y_l).mean()
    assert acc > 0.8

    # Xf et yf sont identiques à X_l et y_l
    assert Xf.shape == X_l.shape
    assert np.array_equal(yf, y_l)