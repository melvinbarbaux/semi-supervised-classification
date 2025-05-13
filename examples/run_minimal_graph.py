import warnings
import numpy as np
from sklearn.preprocessing import StandardScaler

from ssl_bench.data.loaders.cifar10_raw import CIFAR10RawLoader
from ssl_bench.datamodule.graph.knn import KNNGraph
from ssl_bench.datamodule.graph.epsilon import EpsilonGraph
from ssl_bench.datamodule.graph.anchor import AnchorGraph
from ssl_bench.methods.gfhf import GFHFMethod
from ssl_bench.methods.poisson_learning import PoissonLearningMethod

warnings.filterwarnings("ignore", category=RuntimeWarning)


def sample_per_class(
    X: np.ndarray,
    y: np.ndarray,
    samples_per_class: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sélectionne un sous-ensemble équilibré : `samples_per_class` par classe.
    """
    classes = np.unique(y)
    idxs = []
    for c in classes:
        class_idxs = np.where(y == c)[0]
        if len(class_idxs) <= samples_per_class:
            chosen = class_idxs
        else:
            chosen = np.random.choice(class_idxs, samples_per_class, replace=False)
        idxs.extend(chosen.tolist())
    return X[idxs], y[idxs]


def main():
    print("🚀 Lancement de run_minimal_graph.py …")

    # 1) Charger un petit sous-ensemble de CIFAR-10
    loader = CIFAR10RawLoader(batch_dir="data/raw/cifar-10-batches-py")
    X, y = loader.load()
    print(f"Chargé CIFAR-10 : {X.shape[0]} exemples total, classes = {np.unique(y)}")

    # 2) Sous-échantillonnage : 5 images par classe
    X_sub, y_sub = sample_per_class(X, y, samples_per_class=5)
    print(f"Sous-échantillon équilibré : {X_sub.shape[0]} images (5 par classe)")

    # 3) Flatten + standardisation
    N = X_sub.shape[0]
    X_flat = X_sub.reshape(N, -1).astype(np.float32)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_flat)

    # 4) Partition label/non-label
    classes = np.unique(y_sub)
    n_lab_per_class = 2
    idxs_labeled = []
    idxs_unlabeled = []
    for c in classes:
        inds = np.where(y_sub == c)[0]
        idxs_labeled.extend(inds[:n_lab_per_class].tolist())
        idxs_unlabeled.extend(inds[n_lab_per_class:].tolist())
    X_l, y_l = Xs[idxs_labeled], y_sub[idxs_labeled]
    X_u, y_u = Xs[idxs_unlabeled], y_sub[idxs_unlabeled]
    print(f"Partition : {X_l.shape[0]} labellisées, {X_u.shape[0]} non-labellisées\n")

    # 5) Boucle sur différents builders de graphe
    builders = {
        "k-NN (k=3)":      KNNGraph(n_neighbors=3, mode="connectivity"),
        "ε-Graph (ε=1.5)": EpsilonGraph(eps=1.5, mode="connectivity"),
        "Anchor (m=10)":   AnchorGraph(n_anchors=10, sigma=None, random_state=0),
    }

    for name, builder in builders.items():
        print(f"--- GFHF via {name:<15s} ---")
        gf_method = GFHFMethod(graph_builder=builder)
        model_gf, _, _ = gf_method.run(X_l, y_l, X_u)
        preds_gf = model_gf.predict(X_u)
        acc_gf = (preds_gf == y_u).mean()
        print(f"Accuracy GFHF on U = {acc_gf:.2f}\n")

        # Poisson Learning
        print(f"--- Poisson Learning via {name:<15s} ---")
        pl_method = PoissonLearningMethod(graph_builder=builder)
        model_pl, _, _ = pl_method.run(X_l, y_l, X_u)
        preds_pl = model_pl.predict(X_u)
        acc_pl = (preds_pl == y_u).mean()
        print(f"Accuracy Poisson on U = {acc_pl:.2f}\n")


if __name__ == "__main__":
    main()
