# src/ssl_bench/experiment.py

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Tuple

@dataclass
class ExperimentConfig:
    method: str
    model: str
    dataset: str
    labeled_fraction: float
    seed: int
    model_hyperparams: Dict = None
    method_hyperparams: Dict = None

@dataclass
class ExperimentResult:
    config: ExperimentConfig
    model: Any
    metrics: Dict
    duration: float

class ExperimentRunner:
    """
    Exécute directement la fonction train+eval et renvoie le résultat.
    """
    def __init__(self):
        pass

    def run(
        self,
        config: ExperimentConfig,
        func_train_eval: Callable[[ExperimentConfig], Tuple[Any, Dict]]
    ) -> ExperimentResult:
        start = time.time()
        trained_model, metrics = func_train_eval(config)
        duration = time.time() - start
        return ExperimentResult(config, trained_model, metrics, duration)