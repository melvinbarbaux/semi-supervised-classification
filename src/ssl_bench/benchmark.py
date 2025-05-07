# src/ssl_bench/benchmark.py

import pandas as pd
import logging
from typing import List, Dict, Any

from ssl_bench.data.dataset_loader import DatasetLoader
from ssl_bench.transforms import DataTransformer
from ssl_bench.experiment import ExperimentRunner, ExperimentConfig, ExperimentResult
from ssl_bench.cache import CacheManager
from ssl_bench.registry import ModelRegistry
from ssl_bench.models.base import BaseModel
from ssl_bench.methods.base import SemiSupervisedMethod

class BenchmarkSuite:
    """
    Lance un benchmark systématique pour un ensemble de modèles et méthodes
    sur un dataset donné.
    """
    def __init__(
        self,
        loader: DatasetLoader,
        models: List[BaseModel],
        methods: List[SemiSupervisedMethod],
        dataset_name: str,
        label_fractions: List[float],
        seeds: List[int],
        cache_dir: str = "data/processed/experiment_cache",
        registry_dir: str = "data/processed/registry"
    ):
        # Configuration du logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        if not self.logger.handlers:
            self.logger.addHandler(handler)

        self.loader = loader
        self.models = models
        self.methods = methods
        self.dataset_name = dataset_name
        self.label_fractions = label_fractions
        self.seeds = seeds

        self.cache = CacheManager(cache_dir=cache_dir)
        self.registry = ModelRegistry(registry_dir=registry_dir)
        self.runner = ExperimentRunner(self.cache)

    def run(self) -> pd.DataFrame:
        # Chargement du dataset
        self.logger.info(f"Loading dataset '{self.dataset_name}' using {self.loader.__class__.__name__}")
        X, y = self.loader.load()
        self.logger.info(f"Dataset loaded: X.shape={X.shape}, y.shape={y.shape}")

        records: List[Dict[str, Any]] = []
        for frac in self.label_fractions:
            for seed in self.seeds:
                self.logger.info(f"Starting experiments for fraction={frac}, seed={seed}")
                # Transformation et split
                transformer = DataTransformer(test_size=1 - frac, seed=seed)
                Xs, ys_enc = transformer.fit_transform(X, y)
                X_train, X_test, y_train_enc, y_test_enc = transformer.split(Xs, ys_enc)
                y_train = y_train_enc.argmax(axis=1)
                y_test = y_test_enc.argmax(axis=1)
                self.logger.info(
                    f"Split complete: train={X_train.shape}, test={X_test.shape}"
                )

                for model in self.models:
                    model_name = model.__class__.__name__
                    # Clone du modèle
                    params = getattr(model, 'params', {}) if hasattr(model, 'params') else {}
                    model_copy = model.__class__(**params) if params else model

                    for method in self.methods:
                        method_name = method.__class__.__name__
                        # Clone de la méthode
                        method_params: Dict[str, Any] = {}
                        for attr in ('threshold', 'max_iter'):
                            if hasattr(method, attr):
                                method_params[attr] = getattr(method, attr)
                        method_copy = method.__class__(model_copy, **method_params)

                        self.logger.info(
                            f"Running {method_name} on model {model_name} "
                            f"(params={params}, method_params={method_params})"
                        )

                        # Prépare la configuration d'expérience
                        config = ExperimentConfig(
                            method=method_name,
                            model=model_name,
                            dataset=self.dataset_name,
                            labeled_fraction=frac,
                            seed=seed,
                            model_hyperparams=params,
                            method_hyperparams=method_params
                        )

                        # Fonction train/eval spécifique
                        def train_eval(cfg):  # noqa: F811
                            tm, X_labelled, y_labelled = method_copy.run(
                                X_train, y_train, X_test
                            )
                            preds = tm.predict(X_test)
                            acc = (preds == y_test).mean()
                            return tm, {
                                'accuracy': acc,
                                'n_labeled': X_labelled.shape[0]
                            }

                        # Exécution de l'expérience
                        result: ExperimentResult = self.runner.run(config, train_eval)

                        self.logger.info(
                            f"Result for {method_name}-{model_name}-frac{frac}-seed{seed}: "
                            f"accuracy={result.metrics['accuracy']:.4f}, "
                            f"n_labeled={result.metrics['n_labeled']}"
                        )

                        # Enregistrement dans le registry
                        self.registry.register_run(
                            dataset=self.dataset_name,
                            model_name=model_name,
                            method=method_name,
                            trained_model=result.model,
                            metrics=result.metrics
                        )

                        # Collecte des résultats
                        rec: Dict[str, Any] = {
                            'dataset': self.dataset_name,
                            'model': model_name,
                            'method': method_name,
                            'fraction': frac,
                            'seed': seed,
                            **result.metrics
                        }
                        records.append(rec)

        df = pd.DataFrame.from_records(records)
        self.logger.info(f"Benchmark completed: {len(records)} runs recorded")
        return df
