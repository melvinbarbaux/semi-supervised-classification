import numpy as np
from ssl_bench.data.dataset_loader import DatasetLoader

class DummyLoader(DatasetLoader):
    def load(self):
        X = np.ones((5, 3))
        y = np.array([0, 1, 0, 1, 0])
        return X, y


def test_dummy_loader():
    loader = DummyLoader()
    X, y = loader.load()
    assert X.shape == (5, 3)
    assert y.dtype == int
    assert set(y.tolist()) == {0,1}