import numpy as np
from typing import Any

from .experiment import ExperimentConfig, ExperimentResult

class Predictor:
    def __init__(self, cache_dir: str = "data/processed/experiment_cache"):
        self.cache = CacheManager(cache_dir)

    def load(self, config: ExperimentConfig) -> ExperimentResult:
        cfg_dict = config.__dict__.copy()
        cfg_dict['model_hyperparams'] = config.model_hyperparams or {}
        cfg_dict['method_hyperparams'] = config.method_hyperparams or {}
        return self.cache.load(cfg_dict)

    def predict(self, config: ExperimentConfig, X: np.ndarray) -> np.ndarray:
        result = self.load(config)
        model = result.model
        return model.predict(X)