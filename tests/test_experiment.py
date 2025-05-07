# tests/test_experiment.py

import pytest
import time
from ssl_bench.cache import CacheManager
from ssl_bench.experiment import ExperimentRunner, ExperimentConfig, ExperimentResult

@pytest.fixture
def cache(tmp_path):
    return CacheManager(cache_dir=str(tmp_path / 'cache'))

@pytest.fixture
def runner(cache):
    return ExperimentRunner(cache)

def dummy_train_eval(config):
    # Simule un calcul de 10ms
    time.sleep(0.01)
    # Retourne un modèle factice et une métrique basée sur labeled_fraction
    dummy_model = object()
    metrics = {'accuracy': round(config.labeled_fraction, 2)}
    return dummy_model, metrics

def test_experiment_caching_behavior(runner):
    cfg = ExperimentConfig(
        method='TestMethod',
        model='TestModel',
        dataset='TestDataset',
        labeled_fraction=0.42,
        seed=7
    )
    # 1ère exécution : pas de cache
    res1 = runner.run(cfg, dummy_train_eval)
    assert isinstance(res1, ExperimentResult)
    assert 'accuracy' in res1.metrics
    assert res1.duration > 0
    assert res1.model is not None

    # 2ᵉ exécution : doit venir du cache (rapide)
    start = time.time()
    res2 = runner.run(cfg, dummy_train_eval)
    elapsed = time.time() - start

    # Les métriques et la config doivent être identiques
    assert res2.metrics == res1.metrics
    assert res2.config == res1.config
    # Le modèle rechargé a bien le même type
    assert type(res2.model) is type(res1.model)
    # La récupération depuis cache doit être plus rapide
    assert elapsed < res1.duration
