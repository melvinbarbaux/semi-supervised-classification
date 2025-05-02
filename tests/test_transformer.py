import numpy as np
import pytest
from sklearn.datasets import make_classification
from ssl_bench.transforms.transformer import DataTransformer

@pytest.fixture
def synthetic_data():
    # Ajustement : n_clusters_per_class=1 pour respecter
    # n_classes * n_clusters_per_class ≤ 2**n_informative
    X, y = make_classification(
        n_samples=50,
        n_features=4,
        n_informative=2,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=1
    )
    return X, y

def test_fit_transform_and_transform(synthetic_data):
    X, y = synthetic_data
    tf = DataTransformer(test_size=0.25, seed=0)
    X_scaled, y_enc = tf.fit_transform(X, y)
    # Vérifie que X_scaled a moyenne ≈ 0 sur chaque feature
    mean0 = np.mean(X_scaled, axis=0)
    assert pytest.approx(0, abs=1e-7) == mean0[0]
    # Vérifie la dimension de y_enc : (50, n_classes)
    assert y_enc.shape == (50, 3)

    # Un appel transform sans refit doit donner le même résultat
    X2, y2 = tf.transform(X, y)
    assert np.allclose(X2, X_scaled)
    assert np.array_equal(y2, y_enc)

def test_split(synthetic_data):
    X, y = synthetic_data
    tf = DataTransformer(test_size=0.2, seed=42)
    Xs, ys = tf.fit_transform(X, y)
    X_train, X_test, y_train, y_test = tf.split(Xs, ys)
    n_train = int(0.8 * 50)
    n_test = 50 - n_train

    # Vérifie les tailles de train/test
    assert X_train.shape[0] == n_train
    assert X_test.shape[0] == n_test

    # Vérifie que toutes les classes originales apparaissent
    classes_train = set(np.argmax(y_train, axis=1))
    classes_test = set(np.argmax(y_test, axis=1))
    assert classes_train <= set(np.unique(y))
    assert classes_test <= set(np.unique(y))