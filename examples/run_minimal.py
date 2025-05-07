# examples/run_minimal_all_methods.py

import numpy as np
from copy import deepcopy
from sklearn.datasets import make_classification

from ssl_bench.data.dataset_loader import DatasetLoader
from ssl_bench.transforms import DataTransformer

# SKLearn wrappers & methods
from ssl_bench.models.sklearn import RandomForestWrapper
from ssl_bench.methods.supervised import SupervisedMethod
from ssl_bench.methods.self_training import SelfTrainingMethod
from ssl_bench.methods.setred import SetredMethod
from ssl_bench.methods.snnrce import SnnrceMethod
from ssl_bench.methods.tri_training import TriTrainingMethod
from ssl_bench.methods.democratic_co_learning import DemocraticCoLearningMethod
from ssl_bench.methods.mssboost import MSSBoostMethod

# Dash method
from ssl_bench.methods.dash import DashMethod

# Torch wrapper & network
import torch
import torch.nn as nn
import torch.nn.functional as F
from ssl_bench.models.torch import TorchModel


class SyntheticLoader(DatasetLoader):
    def load(self):
        X, y = make_classification(
            n_samples=200,
            n_features=20,
            n_informative=10,
            n_redundant=5,
            n_classes=2,
            random_state=0
        )
        return X.astype(np.float32), y.astype(np.int64)


class SimpleMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_classes: int = 2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def run_all(name, method, X_train, y_train, X_test, y_test):
    trained, Xl, yl = method.run(X_train, y_train, X_test)
    preds = trained.predict(X_test)
    acc = (preds == y_test).mean()
    print(f"{name:25s} accuracy: {acc:.2f}, labeled used: {Xl.shape[0]}")


def main():
    # 1) Chargement et split du dataset synthétique
    loader = SyntheticLoader()
    X, y = loader.load()
    transformer = DataTransformer(test_size=0.3, seed=42)
    Xs, y_enc = transformer.fit_transform(X, y)
    X_train, X_test, y_train_enc, y_test_enc = transformer.split(Xs, y_enc)
    y_train = y_train_enc.argmax(axis=1)
    y_test  = y_test_enc.argmax(axis=1)

    print("=== Dataset synthétique ===")
    print("Train shapes:", X_train.shape, y_train.shape)
    print("Test shapes :", X_test.shape, y_test.shape)
    print()

    # 2) Préparer wrappers
    rf_base = RandomForestWrapper(n_estimators=10)
    input_dim = X_train.shape[1]
    mlp_module = SimpleMLP(input_dim=input_dim, hidden_dim=64, num_classes=y_enc.shape[1])
    torch_base = TorchModel(mlp_module)

    # 3) Définir méthodes pour chaque wrapper
    wrappers = [
        ("rf", rf_base),
        ("torch", torch_base)
    ]
    methods_specs = [
        ("supervised",    SupervisedMethod),
        ("self_training", SelfTrainingMethod),
        ("setred",        SetredMethod),
        ("snnrce",        SnnrceMethod),
        ("tri_training",  TriTrainingMethod),
        ("democratic",    DemocraticCoLearningMethod),
        ("dash",          DashMethod),
        ("mssboost",      MSSBoostMethod),
    ]

    # 4) Exécuter tests
    for tag, base in wrappers:
        print(f"--- Testing wrapper: {tag} ---")
        for name, MethodCls in methods_specs:
            full_name = f"{name}_{tag}"
            # créer instance de méthode selon le type
            if name == "self_training":
                method = MethodCls(deepcopy(base), threshold=0.8, max_iter=5)
            elif name == "setred":
                method = MethodCls(
                    deepcopy(base),
                    theta=0.1, max_iter=5,
                    pool_size=None, n_neighbors=10, random_state=42
                )
            elif name == "snnrce":
                method = MethodCls(
                    deepcopy(base),
                    n_neighbors=10, alpha=0.05, random_state=42
                )
            elif name == "tri_training":
                method = MethodCls(deepcopy(base))
            elif name == "democratic":
                learners = [deepcopy(base) for _ in range(3)]
                method = MethodCls(learners, alpha=0.05, random_state=42)
            elif name == "dash":
                method = MethodCls(
                    deepcopy(base),
                    C=1.0001, gamma=1.1, rho_min=0.05, max_iter=5
                )
            elif name == "mssboost":
                method = MethodCls(
                    deepcopy(base),
                    n_estimators=20, lambda_u=0.1
                )
            else:
                # supervised and any others without hyperparams
                method = MethodCls(deepcopy(base))

            run_all(full_name, method, X_train, y_train, X_test, y_test)
        print()


if __name__ == '__main__':
    main()