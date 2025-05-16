# examples/experiment_loop.py

import numpy as np
import pandas as pd
from copy import deepcopy
from typing import Any

from ssl_bench.cache import CacheManager
from ssl_bench.registry import ModelRegistry
from ssl_bench.experiment import ExperimentRunner, ExperimentConfig
from ssl_bench.data.dataset_loader import DatasetLoader
from ssl_bench.transforms import DataTransformer
from ssl_bench.models.sklearn import RandomForestWrapper
from ssl_bench.methods.supervised import SupervisedMethod
from ssl_bench.methods.self_training import SelfTrainingMethod
from ssl_bench.methods.setred import SetredMethod
from ssl_bench.methods.tri_training import TriTrainingMethod
from ssl_bench.methods.democratic_co_learning import DemocraticCoLearningMethod
from ssl_bench.methods.adsh import AdaptiveThresholdingMethod

class SyntheticLoader(DatasetLoader):
    def load(self):
        from sklearn.datasets import make_classification
        X, y = make_classification(
            n_samples=200, n_features=20, n_informative=10,
            n_redundant=5, n_classes=2, random_state=0
        )
        return X, y

def train_and_eval(config: ExperimentConfig) -> tuple[Any, dict]:
    # 1) Chargement et split
    loader = SyntheticLoader()
    X, y = loader.load()
    transformer = DataTransformer(test_size=1 - config.labeled_fraction, seed=config.seed)
    Xs, ys_enc = transformer.fit_transform(X, y)
    X_train, X_test, y_train_enc, y_test_enc = transformer.split(Xs, ys_enc)
    y_train = y_train_enc.argmax(axis=1)
    y_test  = y_test_enc.argmax(axis=1)

    # 2) Base model
    rf = RandomForestWrapper(n_estimators=config.model_hyperparams.get("n_estimators", 10))

    # 3) Choix de la méthode
    m = config.method
    if m == "supervised":
        method = SupervisedMethod(rf)
    elif m == "self_training":
        method = SelfTrainingMethod(
            rf,
            threshold=config.method_hyperparams.get("threshold", 0.8),
            max_iter=config.method_hyperparams.get("max_iter", 5)
        )
    elif m == "setred":
        method = SetredMethod(
            rf,
            theta=config.method_hyperparams.get("theta", 0.1),
            max_iter=config.method_hyperparams.get("max_iter", 10),
            pool_size=config.method_hyperparams.get("pool_size", None),
            n_neighbors=config.method_hyperparams.get("n_neighbors", 10),
            random_state=config.seed
        )
    elif m == "tri_training":
        method = TriTrainingMethod(deepcopy(rf))
    elif m == "democratic_co_learning":
        learners = [deepcopy(rf) for _ in range(3)]
        method = DemocraticCoLearningMethod(
            learners=learners,
            alpha=config.method_hyperparams.get("alpha", 0.05),
            random_state=config.seed
        )
    elif m == "adsh":
        method = AdaptiveThresholdingMethod(
            deepcopy(rf),
            C=config.method_hyperparams.get("C", 1.0001),
            gamma=config.method_hyperparams.get("gamma", 1.1),
            rho_min=config.method_hyperparams.get("rho_min", 0.0),
            max_iter=config.method_hyperparams.get("max_iter", 10)
        )
    else:
        raise ValueError(f"Unknown method {config.method!r}")

    # 4) Run & evaluate
    trained, metrics = method.run(X_train, y_train, X_test)
    preds = trained.predict(X_test)
    metrics["accuracy"] = (preds == y_test).mean()
    metrics["n_labeled"] = len(getattr(trained, "n_labeled_", y_train))

    return trained, metrics

def main():
    cache = CacheManager(cache_dir="data/processed/experiment_cache")
    runner = ExperimentRunner(cache)
    registry = ModelRegistry(registry_dir="data/processed/registry")

    methods = [
        "supervised",
        "self_training",
        "setred",
        "snnrce",
        "tri_training",
        "democratic_co_learning",
        "adsh"
    ]
    fractions = [0.1, 0.3, 0.5]
    seeds = [0, 1]

    # Hyperparam grid
    rf_params       = [{"n_estimators":10}, {"n_estimators":50}]
    st_params       = [{"threshold":0.7,"max_iter":3}, {"threshold":0.9,"max_iter":5}]
    setred_params   = [{"theta":0.1,"max_iter":5,"pool_size":None,"n_neighbors":10}]
    tri_params      = [{}]
    dcl_params      = [{"alpha":0.05}]
    adsh_params     = [{"C":1.0001,"gamma":1.1,"rho_min":0.0,"max_iter":10}]

    records = []
    for m in methods:
        for frac in fractions:
            for seed in seeds:
                if m == "supervised":
                    mp_list, mhp_list = rf_params, [{}]
                elif m == "self_training":
                    mp_list, mhp_list = rf_params, st_params
                elif m == "setred":
                    mp_list, mhp_list = rf_params, setred_params
                elif m == "tri_training":
                    mp_list, mhp_list = rf_params, tri_params
                elif m == "democratic_co_learning":
                    mp_list, mhp_list = rf_params, dcl_params
                elif m == "adsh":
                    mp_list, mhp_list = rf_params, adsh_params
                else:
                    continue

                for mp in mp_list:
                    for mhp in mhp_list:
                        cfg = ExperimentConfig(
                            method=m,
                            model="RandomForest",
                            dataset="Synthetic",
                            labeled_fraction=frac,
                            seed=seed,
                            model_hyperparams=mp,
                            method_hyperparams=mhp
                        )
                        res = runner.run(cfg, train_and_eval)
                        registry.register_run(
                            dataset=cfg.dataset,
                            model_name=cfg.model,
                            method=cfg.method,
                            trained_model=res.model,
                            metrics=res.metrics
                        )
                        records.append({
                            "method":m,
                            "labeled_fraction":frac,
                            "seed":seed,
                            **mp, **mhp,
                            "accuracy":res.metrics["accuracy"],
                            "n_labeled":res.metrics["n_labeled"]
                        })

    df = pd.DataFrame(records)
    df = df.sort_values(["method","labeled_fraction","n_estimators"])
    print(df)
    df.to_csv("experiments_summary.csv", index=False)

if __name__ == "__main__":
    main()