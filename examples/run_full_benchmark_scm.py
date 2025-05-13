#!/usr/bin/env python3
"""
Run full benchmark on (a small fraction of) CIFAR-10 for all methods,
including classical semi-supervised methods and graph-based label propagation.
"""
import os
import numpy as np
from copy import deepcopy
from sklearn.model_selection import train_test_split
import torch
import logging

from ssl_bench.experiment import ExperimentRunner, ExperimentConfig
from ssl_bench.registry import ModelRegistry
from ssl_bench.data.loaders.cifar10_raw import CIFAR10RawLoader

# Supervised & semi-supervised methods
from ssl_bench.methods.supervised              import SupervisedMethod
from ssl_bench.methods.self_training           import SelfTrainingMethod
from ssl_bench.methods.setred                  import SetredMethod
from ssl_bench.methods.tri_training            import TriTrainingMethod
from ssl_bench.methods.democratic_co_learning  import DemocraticCoLearningMethod
from ssl_bench.methods.mssboost                import MSSBoostMethod
from ssl_bench.methods.dash                    import DashMethod
from ssl_bench.methods.ebsa                    import EBSAMethod
from ssl_bench.methods.ttadec                  import TTADECMethod

# Graph-based propagation methods
from ssl_bench.methods.gfhf                    import GFHFMethod
from ssl_bench.methods.poisson_learning        import PoissonLearningMethod
from ssl_bench.datamodule.graph.knn            import KNNGraph
from ssl_bench.datamodule.graph.epsilon        import EpsilonGraph
from ssl_bench.datamodule.graph.anchor         import AnchorGraph

# Torch model wrapper
from ssl_bench.models.torch_model import TorchModel
from torchvision.models import resnet18


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    # 1) Load 1% of CIFAR-10
    loader = CIFAR10RawLoader(batch_dir="data/raw/cifar-10-batches-py")
    X_full, y_full = loader.load()  # shape (50000, 3072)
    N = len(X_full)
    
    # Utiliser toutes les données
    X_small = X_full.astype(np.float32).reshape(-1, 3, 32, 32) / 255.0
    y_small = y_full
    print(f"→ Working on {len(X_small)} samples (100% of {N})\n")

    # 2) Initial labeled vs unlabeled split
    frac = 0.1
    seed = 0
    X_lab, X_unl, y_lab, y_unl = train_test_split(
        X_small, y_small,
        test_size=1 - frac,
        random_state=seed,
        stratify=y_small
    )

    # 3) Base CNN: ResNet18
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cnn = resnet18(weights=None, num_classes=10).to(device)
    base_model = TorchModel(net=cnn, lr=1e-3, epochs=10, batch_size=32)

    # 4) Classical SSL methods factory
    classical = {
        "supervised":    lambda m: SupervisedMethod(m),
        "self_training": lambda m: SelfTrainingMethod(m, threshold=0.8, max_iter=5),
        "setred":        lambda m: SetredMethod(m, theta=0.1, max_iter=10, n_neighbors=15, random_state=seed),
        "tri_training":  lambda m: TriTrainingMethod(m, random_state=seed),
        "democratic":    lambda m: DemocraticCoLearningMethod([m, m, m], alpha=0.05, random_state=seed),
        "dash":          lambda m: DashMethod(m, C=1.0001, gamma=1.1, rho_min=0.05, max_iter=5),
        "mssboost":      lambda m: MSSBoostMethod(m, n_estimators=10, lambda_u=0.1),
        "ebsa":          lambda m: EBSAMethod(m, random_state=seed),
#        "ttadec":        lambda m: TTADECMethod([m, m, m])
    }

    # 5) Run classical methods
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
            trained, Xl_final, _ = method.run(X_lab, y_lab, X_unl)
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

    # 6) Graph-based propagation variants
    # Flatten for graph building
    X_flat = X_small.reshape(len(X_small), -1)
    Xl_flat, Xu_flat, yl_flat, yu_flat = train_test_split(
        X_flat, y_small,
        test_size=1 - frac,
        random_state=seed,
        stratify=y_small
    )

    graph_builders = {
        "kNN":      KNNGraph(n_neighbors=3, mode="connectivity"),
        "eps":      EpsilonGraph(eps=1.5, mode="connectivity"),
        "anchor":   AnchorGraph(n_anchors=10, sigma=None, random_state=seed)
    }

    # 6a) GFHF
    for tag, builder in graph_builders.items():
        method_name = f"GFHF_{tag}_ResNet"
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
            trained_gfh, Xl_f, yl_f = method.run(Xl_flat, yl_flat, Xu_flat)
            # reshape for fine-tuning
            Xl_4d = Xl_f.reshape(-1, 3, 32, 32)
            cnn_ft = deepcopy(cnn)
            ft = TorchModel(net=cnn_ft, lr=1e-3, epochs=5, batch_size=32)
            ft.train(Xl_4d, yl_f)
            X_unl_4d = Xu_flat.reshape(-1, 3, 32, 32)
            preds = ft.predict(X_unl_4d)
            acc = (preds == yu_flat).mean()
            return ft, {"accuracy": acc, "n_labeled": Xl_4d.shape[0]}

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

    # 6b) Poisson Learning
    for tag, builder in graph_builders.items():
        method_name = f"Poisson_{tag}_ResNet"
        method = PoissonLearningMethod(graph_builder=builder)
        cfg = ExperimentConfig(
            method=method_name,
            model="ResNet18",
            dataset="CIFAR10_1pct_CNN",
            labeled_fraction=frac,
            seed=seed,
            model_hyperparams={},
            method_hyperparams={}
        )

        def train_eval_poisson(cfg):
            trained_p, Xl_f, yl_f = method.run(Xl_flat, yl_flat, Xu_flat)
            # fine-tune
            Xl_4d = Xl_f.reshape(-1, 3, 32, 32)
            cnn_ft = deepcopy(cnn)
            ft = TorchModel(net=cnn_ft, lr=1e-3, epochs=5, batch_size=32)
            ft.train(Xl_4d, yl_f)
            X_unl_4d = Xu_flat.reshape(-1, 3, 32, 32)
            preds = ft.predict(X_unl_4d)
            acc = (preds == yu_flat).mean()
            return ft, {"accuracy": acc, "n_labeled": Xl_4d.shape[0]}

        result = runner.run(cfg, train_eval_poisson)
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
