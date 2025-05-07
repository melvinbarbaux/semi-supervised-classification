"""
Run full benchmark on (a small fraction of) CIFAR-10 pour toutes les méthodes classiques
et graph-based (GFHF), sans gestion de cache.
"""
import os
import pickle
import numpy as np
import pandas as pd

from ssl_bench.experiment import ExperimentRunner, ExperimentConfig
from ssl_bench.registry import ModelRegistry

from ssl_bench.data.loaders.cifar10_raw import CIFAR10RawLoader
from ssl_bench.models.sklearn import RandomForestWrapper, SVMWrapper

from ssl_bench.methods.supervised import SupervisedMethod
from ssl_bench.methods.self_training import SelfTrainingMethod
from ssl_bench.methods.setred import SetredMethod
from ssl_bench.methods.snnrce import SnnrceMethod
from ssl_bench.methods.tri_training import TriTrainingMethod
from ssl_bench.methods.democratic_co_learning import DemocraticCoLearningMethod

from ssl_bench.datamodule.graph.knn import KNNGraph
from ssl_bench.datamodule.graph.epsilon import EpsilonGraph
from ssl_bench.datamodule.graph.anchor import AnchorGraph
from ssl_bench.methods.gfh import GFHFMethod


def main():
    # 1) Loader + sous-échantillon 1 % de CIFAR-10 (~500 échantillons)
    loader = CIFAR10RawLoader(batch_dir="data/raw/cifar-10-batches-py")
    X_full, y_full = loader.load()
    N = len(X_full)
    idx_small = np.random.default_rng(0).choice(N, size=max(1, N // 100), replace=False)
    X_small, y_small = X_full[idx_small], y_full[idx_small]
    print(f"→ On travaille sur {len(X_small)} échantillons (1 % de {N})\n")

    # 2) Modèles de base
    models = {
        "RF":  RandomForestWrapper(n_estimators=20),
        "SVM": SVMWrapper(gamma="scale"),
    }

    # 3) Méthodes « classiques »
    classical = {
        "supervised":    lambda m: SupervisedMethod(m),
        "self_training": lambda m: SelfTrainingMethod(m, threshold=0.8, max_iter=5),
        "setred":        lambda m: SetredMethod(m, theta=0.1, max_iter=10, n_neighbors=15, random_state=0),
        "tri_training":  lambda m: TriTrainingMethod(m),
        "democratic":    lambda m: DemocraticCoLearningMethod([m, m, m], alpha=0.05, random_state=0),
    }

    # 4) Variantes GFHF (sur RF uniquement ici)
    gf_builders = {
        "GFHF_kNN":    KNNGraph(n_neighbors=3,  mode="connectivity"),
        "GFHF_eps":    EpsilonGraph(eps=1.5, mode="connectivity"),
        "GFHF_anchor": AnchorGraph(n_anchors=10, sigma=None, random_state=0),
    }

    # 5) Construit la liste de toutes les méthodes à tester
    all_methods = {}
    for name, base in models.items():
        for mname, factory in classical.items():
            all_methods[f"{mname}_{name}"] = factory(base)
    for name, builder in gf_builders.items():
        all_methods[f"{name}_RF"] = GFHFMethod(graph_builder=builder)

    # 6) Prépare runner + registry (plus de cache)
    runner   = ExperimentRunner()
    registry = ModelRegistry(registry_dir="data/processed/registry")

    # 7) Boucle rapide sur 10 % d’étiquettes, seed=0
    frac = 0.1
    seed = 0
    results = []

    for run_name, method in all_methods.items():
        cfg = ExperimentConfig(
            method=run_name,
            model=run_name.split("_")[-1],
            dataset="CIFAR10_1pct",
            labeled_fraction=frac,
            seed=seed,
            model_hyperparams=getattr(method, "params", {}),
            method_hyperparams={}
        )

        def train_eval(cfg):
            # split semi-supervisé : on sépare X_small en labeled / unlabeled
            from ssl_bench.transforms import DataTransformer
            transformer = DataTransformer(test_size=1 - frac, seed=seed)
            Xs, ys_enc = transformer.fit_transform(X_small, y_small)
            X_lab, X_unl, y_lab_enc, y_unl_enc = transformer.split(Xs, ys_enc)
            y_lab = y_lab_enc.argmax(axis=1)
            y_unl = y_unl_enc.argmax(axis=1)

            trained, Xl_final, _ = method.run(X_lab, y_lab, X_unl)
            preds = trained.predict(X_unl)
            acc = (preds == y_unl).mean()
            return trained, {"accuracy": acc, "n_labeled": Xl_final.shape[0]}

        # 8) Exécution
        result = runner.run(cfg, train_eval)

        # 9) Enregistrement dans le registry + dump du modèle
        print(f"Enregistrement du run {cfg.method} dans le registry...")    
        registry.register_run(
            dataset=cfg.dataset,
            model_name=cfg.model,
            method=cfg.method,
            trained_model=result.model,
            metrics=result.metrics
        )
        outdir = os.path.join("data", "processed", "saved_models", cfg.dataset, cfg.method)
        os.makedirs(outdir, exist_ok=True)
        with open(os.path.join(outdir, "model.pkl"), "wb") as f:
            pickle.dump(result.model, f)

        # 10) Collecte pour le résumé
        results.append({
            "method": cfg.method,
            **result.metrics
        })
        print(f"✅ {cfg.method}: acc={result.metrics['accuracy']:.3f}, labeled={result.metrics['n_labeled']}")

    # 11) Affiche la synthèse
    df = pd.DataFrame(results)
    print("\n--- Synthèse 1 % CIFAR-10 ---")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()