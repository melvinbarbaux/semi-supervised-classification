#!/usr/bin/env python3
"""
Run full benchmark on (a small fraction of) CIFAR-10 pour toutes les méthodes classiques
et graph-based (GFHF), sans cache.
"""
import os
import pickle
import numpy as np
import pandas as pd
from copy import deepcopy
from sklearn.model_selection import train_test_split
import torch
from torchvision.models import resnet18

from ssl_bench.experiment import ExperimentRunner, ExperimentConfig
from ssl_bench.registry import ModelRegistry
from ssl_bench.data.loaders.cifar10_raw import CIFAR10RawLoader

from ssl_bench.methods.supervised              import SupervisedMethod
from ssl_bench.methods.self_training           import SelfTrainingMethod
from ssl_bench.methods.setred                  import SetredMethod
from ssl_bench.methods.tri_training            import TriTrainingMethod
from ssl_bench.methods.democratic_co_learning  import DemocraticCoLearningMethod
from ssl_bench.methods.mssboost                import MSSBoostMethod
from ssl_bench.methods.dash                    import DashMethod

from ssl_bench.models.torch_model                    import TorchModel

# pour GFHF
from ssl_bench.datamodule.graph.knn     import KNNGraph
from ssl_bench.datamodule.graph.epsilon import EpsilonGraph
from ssl_bench.datamodule.graph.anchor  import AnchorGraph
from ssl_bench.methods.gfhf               import GFHFMethod


def main():
    import logging
    logging.basicConfig(
    level=logging.INFO,  # Changez en DEBUG si vous voulez plus de détails
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
    # 1) Charger 1% de CIFAR-10
    loader = CIFAR10RawLoader(batch_dir="data/raw/cifar-10-batches-py")
    X_full, y_full = loader.load()        # shape (50000, 3072)
    N = len(X_full)
    idx_small = np.random.default_rng(0).choice(N, size=max(1, N // 100), replace=False)
    X_small = X_full[idx_small].astype(np.float32).reshape(-1, 3, 32, 32) / 255.0
    y_small = y_full[idx_small]
    print(f"→ On travaille sur {len(X_small)} échantillons (1% de {N})\n")

    # 2) Split initial en labeled / unlabeled
    frac = 0.1
    seed = 0
    X_lab, X_unl, y_lab, y_unl = train_test_split(
        X_small, y_small,
        test_size=1 - frac,
        random_state=seed,
        stratify=y_small
    )

    # 3) Préparer le modèle Torch : ResNet18
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cnn = resnet18(weights=None, num_classes=10).to(device)
    base_model = TorchModel(
        net=cnn,
        lr=1e-3,
        epochs=10,
        batch_size=32
    )

    # 4) Usines de méthodes classiques
    classical = {
        "supervised":    lambda m: SupervisedMethod(m),
        "self_training": lambda m: SelfTrainingMethod(m, threshold=0.8, max_iter=5),
        "setred":        lambda m: SetredMethod(m, theta=0.1, max_iter=10, n_neighbors=15, random_state=seed),
        "tri_training":  lambda m: TriTrainingMethod(m, random_state=seed),
        "democratic":    lambda m: DemocraticCoLearningMethod([m, m, m], alpha=0.05, random_state=seed),
        "mssboost":      lambda m: MSSBoostMethod(m, n_estimators=20, lambda_u=0.1),
        "dash":          lambda m: DashMethod(m, C=1.0001, gamma=1.1, rho_min=0.05, max_iter=5),
    }

    # 5) Construire et lancer les méthodes classiques
    runner   = ExperimentRunner()
    registry = ModelRegistry(registry_dir="data/processed/registry")
    results  = []

    for name, factory in classical.items():
        method_name = f"{name}_ResNet"
        method = factory(deepcopy(base_model))
        cfg = ExperimentConfig(
            method=method_name,
            model="ResNet18",
            dataset="CIFAR10_1pct_CNN",
            labeled_fraction=frac,
            seed=seed,
            model_hyperparams={},
            method_hyperparams={}
        )

        def train_eval(cfg):
            # entraîner la méthode
            trained, Xl_final, _ = method.run(X_lab, y_lab, X_unl)
            # évaluer
            preds = trained.predict(X_unl)
            acc = (preds == y_unl).mean()
            return trained, {"accuracy": acc, "n_labeled": Xl_final.shape[0]}

        result = runner.run(cfg, train_eval)
        registry.register_run(
            dataset=cfg.dataset,
            model_name=cfg.model,
            method=cfg.method,
            trained_model=result.model,
            metrics=result.metrics
        )
        print(f"✅ {cfg.method}: acc={result.metrics['accuracy']:.3f}, labeled={result.metrics['n_labeled']}")
        results.append({"method": cfg.method, **result.metrics})

    # 6) Variantes GFHF
    #    → on travaille sur la version plate pour construire le graphe
    X_flat = X_small.reshape(len(X_small), -1)  # (500, 3072)
    Xl_flat, Xu_flat, yl_flat, yu_flat = train_test_split(
        X_flat, y_small,
        test_size=1 - frac,
        random_state=seed,
        stratify=y_small
    )

    gf_builders = {
        "GFHF_kNN":    KNNGraph(n_neighbors=3,  mode="connectivity"),
        "GFHF_eps":    EpsilonGraph(eps=1.5, mode="connectivity"),
        "GFHF_anchor": AnchorGraph(n_anchors=10, sigma=None, random_state=seed),
    }

    for name, builder in gf_builders.items():
        method_name = f"{name}_ResNet"
        # On n’injecte plus base_model ici
        method = GFHFMethod(graph_builder=builder)

        cfg = ExperimentConfig(
            method=method_name,
            model="ResNet18",
            dataset="CIFAR10_1pct_CNN",
            labeled_fraction=frac,
            seed=seed,
            model_hyperparams={},
            method_hyperparams={}
        )

        def train_eval_gfh(cfg):
            # 1) Propagation GFHF sur les vecteurs plats
            trained_gfh, Xl_final_flat, yl_final = method.run(
                Xl_flat, yl_flat, Xu_flat
            )

            # 2) On reshape les samples labellisés en (n,3,32,32)
            Xl_final_4d = Xl_final_flat.reshape(-1, 3, 32, 32)

            # 3) Fine-tuning rapide du ResNet sur ces Xl_final_4d
            cnn_ft = deepcopy(cnn)
            ft_model = TorchModel(
                net=cnn_ft,
                lr=1e-3,
                epochs=5,
                batch_size=32
            )
            ft_model.train(Xl_final_4d, yl_final)

            # 4) Évaluation sur le pool non-labellisé d’origine
            X_unl_4d = Xu_flat.reshape(-1, 3, 32, 32)
            preds = ft_model.predict(X_unl_4d)
            acc = (preds == yu_flat).mean()

            return ft_model, {"accuracy": acc, "n_labeled": Xl_final_4d.shape[0]}

        result = runner.run(cfg, train_eval_gfh)
        registry.register_run(
            dataset=cfg.dataset,
            model_name=cfg.model,
            method=cfg.method,
            trained_model=result.model,
            metrics=result.metrics
        )
        print(f"✅ {cfg.method}: acc={result.metrics['accuracy']:.3f}, labeled={result.metrics['n_labeled']}")
        results.append({"method": cfg.method, **result.metrics})

if __name__ == "__main__":
    main()