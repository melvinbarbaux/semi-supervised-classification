#!/usr/bin/env python3
"""
Run full benchmark on CIFAR-10 for all methods,
incluant supervised, semi-supervised et propagation graph methods.
"""
import os
import numpy as np
import pandas as pd
from copy import deepcopy
from sklearn.model_selection import train_test_split
import torch
import logging
import time
import resource

from ssl_bench.experiment import ExperimentRunner, ExperimentConfig
from ssl_bench.registry import ModelRegistry
from ssl_bench.data.loaders.cifar10_raw import CIFAR10RawLoader

# Methods
from ssl_bench.methods.supervised              import SupervisedMethod
from ssl_bench.methods.self_training           import SelfTrainingMethod
from ssl_bench.methods.setred                  import SetredMethod
from ssl_bench.methods.tri_training            import TriTrainingMethod
from ssl_bench.methods.democratic_co_learning  import DemocraticCoLearningMethod
from ssl_bench.methods.mssboost                import MSSBoostMethod
from ssl_bench.methods.adsh                    import AdaptiveThresholdingMethod
from ssl_bench.methods.ebsa                    import EBSAMethod

# Graph-based
from ssl_bench.methods.gfhf                    import GFHFMethod
from ssl_bench.methods.poisson_learning        import PoissonLearningMethod
from ssl_bench.datamodule.graph.knn            import KNNGraph
from ssl_bench.datamodule.graph.epsilon        import EpsilonGraph
from ssl_bench.datamodule.graph.anchor         import AnchorGraph

# Torch wrapper and model
from ssl_bench.models.torch_model import TorchModel
from torchvision.models import resnet18

# Configurable parameters
LOAD_PERCENT = 1  # pourcentage de données CIFAR-10 à charger (1-100)

def split_dataset(X, y, fraction, seed, flatten=False):
    """Split data into labeled and unlabeled subsets."""
    if flatten:
        X = X.reshape(len(X), -1)
    return train_test_split(
        X, y,
        test_size=1 - fraction,
        random_state=seed,
        stratify=y
    )

def train_and_register(runner, registry, cfg, train_fn):
    """
    Run training, register results, handle errors
    """
    # Démarrage des mesures
    start_time = time.time()
    start_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    gpu_available = torch.cuda.is_available()
    if gpu_available:
        torch.cuda.reset_peak_memory_stats()

    try:
        # Lancement de l'expérience
        result = runner.run(cfg, train_fn)

        # Calcul des métriques de temps et mémoire
        duration = time.time() - start_time
        end_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        mem_usage = end_mem - start_mem

        # Enrichir les metrics retournées
        metrics = result.metrics.copy()
        metrics['duration_s']     = duration
        metrics['mem_usage_kb']   = mem_usage
        if gpu_available:
            metrics['gpu_max_mem_bytes'] = torch.cuda.max_memory_allocated()

        # Enregistrement dans le registry
        registry.register_run(
            dataset=cfg.dataset,
            model_name=cfg.model,
            method=cfg.method,
            trained_model=result.model,
            metrics=metrics
        )

        # Log enrichi avec nombre de pseudo-labels
        logging.getLogger(__name__).info(
            "✅ %s: acc=%.3f, labeled=%d (%d pseudo), time=%.2fs, mem=%dkB%s",
            cfg.method,
            metrics['accuracy'],
            metrics['n_labeled'],
            metrics.get('n_pseudo', 0),
            metrics['duration_s'],
            metrics['mem_usage_kb'],
            f", gpu_peak={metrics.get('gpu_max_mem_bytes', 0)} bytes" if gpu_available else ""
        )
        return {"method": cfg.method, **metrics}

    except Exception:
        logging.getLogger(__name__).exception("Erreur lors de %s", cfg.method)
        return None

    finally:
        # Cleanup mémoire GPU
        try:
            del result
        except NameError:
            pass
        if gpu_available:
            torch.cuda.empty_cache()

def main():
    # Logging configuration
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("benchmark.log")
        ]
    )
    logger = logging.getLogger(__name__)

    # Seeds for reproducibility
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info("Seed set to %d", seed)

    # 1) Load dataset
    loader = CIFAR10RawLoader(batch_dir="data/raw/cifar-10-batches-py")
    X_full, y_full = loader.load()
    X_full = X_full.astype(np.float32).reshape(-1, 3, 32, 32) / 255.0

    # Subsample based on LOAD_PERCENT
    total_full = len(X_full)
    n_load = int(total_full * LOAD_PERCENT / 100)
    rng = np.random.RandomState(seed)
    indices = rng.choice(total_full, n_load, replace=False)
    X_all = X_full[indices]
    y_all = y_full[indices]
    total = len(X_all)
    logger.info("Working on %d samples (%d%% of %d)", total, LOAD_PERCENT, total_full)

    # 2) Initial split
    frac = 0.1
    X_lab, X_unl, y_lab, y_unl = split_dataset(X_all, y_all, frac, seed)
    initial_n_lab = len(y_lab)  # Pour mesurer les pseudo-labels ajoutés

    # 3) Base CNN setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info("Using device %s", device)
    cnn_base = resnet18(weights=None, num_classes=10).to(device)
    base_model = TorchModel(net=cnn_base, lr=1e-3, epochs=10, batch_size=32)

    # 4) Classical SSL methods factory
    classical = {
#        "supervised":    lambda m: SupervisedMethod(m),
#        "self_training": lambda m: SelfTrainingMethod(m, threshold=0.8, max_iter=5),
        "adsh":          lambda m: AdaptiveThresholdingMethod(m, mu=1.0, tau1=0.8, max_iter=10),
#        "setred":        lambda m: SetredMethod(m, theta=0.1, max_iter=10, n_neighbors=15, random_state=seed),
#        "tri_training":  lambda m: TriTrainingMethod(m, random_state=seed),
#        "democratic":    lambda m: DemocraticCoLearningMethod([m, m, m], alpha=0.05, random_state=seed),
#        "mssboost":      lambda m: MSSBoostMethod(m, n_estimators=10, lambda_u=0.1),
#        "ebsa":          lambda m: EBSAMethod(m, random_state=seed),
#        "ttadec":        lambda m: TTADECMethod([m, m, m])
    }

    runner   = ExperimentRunner()
    registry = ModelRegistry(registry_dir="data/processed/registry")
    results  = []

    # 5) Run classical methods
    for name, factory in classical.items():
        method_label = f"{name}_ResNet"
        method = factory(deepcopy(base_model))
        cfg = ExperimentConfig(
            method=method_label,
            model="ResNet18",
            dataset=f"CIFAR10_{int(frac*100)}pct_CNN",
            labeled_fraction=frac,
            seed=seed,
            model_hyperparams={'lr': 1e-3, 'epochs': 10, 'batch_size': 32},
            method_hyperparams={}
        )

        def eval_fn(cfg):
            trained, Xl_final, _ = method.run(X_lab, y_lab, X_unl)
            final_n = Xl_final.shape[0]
            n_pseudo = final_n - initial_n_lab
            preds = trained.predict(X_unl)
            acc   = (preds == y_unl).mean()
            return trained, {
                "accuracy":  acc,
                "n_labeled": final_n,
                "n_pseudo":  n_pseudo
            }

        res = train_and_register(runner, registry, cfg, eval_fn)
        if res:
            results.append(res)

    # 6) Graph-based
    Xl_flat, Xu_flat, yl_flat, yu_flat = split_dataset(X_all, y_all, frac, seed, flatten=True)
    initial_n_graph = len(yl_flat)

    graph_builders = {
        "kNN":    KNNGraph(n_neighbors=3, mode="connectivity"),
    #    "eps":    EpsilonGraph(eps=1.5, mode="connectivity"),
    #    "anchor": AnchorGraph(n_anchors=10, sigma=None, random_state=seed)
    }

    for cls, method_cls in [(GFHFMethod, 'GFHF'), (PoissonLearningMethod, 'Poisson')]:
        for tag, builder in graph_builders.items():
            method_label = f"{method_cls}_{tag}_ResNet"
            method = cls(graph_builder=builder)
            cfg = ExperimentConfig(
                method=method_label,
                model="ResNet18",
                dataset=f"CIFAR10_{int(frac*100)}pct_CNN",
                labeled_fraction=frac,
                seed=seed,
                model_hyperparams={},
                method_hyperparams={}
            )

            def eval_graph(cfg, method=method, builder=builder):
                trained_p, Xl_prop, yl_prop = method.run(Xl_flat, yl_flat, Xu_flat)
                final_n = Xl_prop.shape[0]
                n_pseudo = final_n - initial_n_graph
                # fine-tune
                Xl_4d = Xl_prop.reshape(-1, 3, 32, 32)
                cnn_ft = deepcopy(cnn_base)
                ft = TorchModel(net=cnn_ft, lr=1e-3, epochs=5, batch_size=32)
                ft.train(Xl_4d, yl_prop)
                X_unl_4d = Xu_flat.reshape(-1, 3, 32, 32)
                preds = ft.predict(X_unl_4d)
                acc = (preds == yu_flat).mean()
                return ft, {
                    "accuracy":  acc,
                    "n_labeled": final_n,
                    "n_pseudo":  n_pseudo
                }

            res = train_and_register(runner, registry, cfg, eval_graph)
            if res:
                results.append(res)

    # 7) Export results
    timestamp = time.strftime("%Y%m%d-%H%M")
    results_dir = os.path.join("results", timestamp)
    os.makedirs(results_dir, exist_ok=True)

    registry_path = os.path.join(results_dir, "results.csv")
    df = pd.DataFrame(results)
    df.to_csv(registry_path, index=False)
    logger.info(f"Results saved to {registry_path}")


if __name__ == "__main__":
    main()