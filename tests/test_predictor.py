# tests/test_predictor.py

import numpy as np
import pytest

from ssl_bench.predictor import Predictor
from ssl_bench.experiment import ExperimentConfig, ExperimentResult

class DummyModel:
    def predict(self, X):
        return np.ones(len(X), dtype=int)

@pytest.fixture
def fake_cache(tmp_path):
    # Prépare un config et un cache avec un DummyModel
    config = ExperimentConfig(
        method='dummy', model='dummy', dataset='D', labeled_fraction=0.1, seed=0
    )
    config.model_hyperparams = {}
    config.method_hyperparams = {}

    # Crée le cache et y stocke un ExperimentResult factice
    from ssl_bench.cache import CacheManager
    cache = CacheManager(cache_dir=str(tmp_path / 'cache'))
    dummy_model = DummyModel()
    result = ExperimentResult(
        config=config,
        model=dummy_model,
        metrics={'accuracy': 0.5},
        duration=0.0
    )
    cache.save(config.__dict__, result)
    return config, str(tmp_path / 'cache')

def test_predictor(fake_cache):
    config, cache_dir = fake_cache
    pred = Predictor(cache_dir=cache_dir)

    # load()
    result = pred.load(config)
    assert hasattr(result, 'model')
    assert isinstance(result.model, DummyModel)

    # predict()
    X_new = np.zeros((4,3))
    y = pred.predict(config, X_new)
    assert np.array_equal(y, np.ones(4, dtype=int))