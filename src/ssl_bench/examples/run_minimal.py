"""
Script minimal de test :
  - création d'un dataset synthétique
  - transformation et split
  - baseline full-supervisé
  - méthode semi-supervisée Self-Training
"""

import numpy as np
from sklearn.datasets import make_classification

from ssl_bench.data.dataset_loader import DatasetLoader
from ssl_bench.transforms import DataTransformer


class SyntheticLoader(DatasetLoader):
    def load(self):
        # 100 échantillons, 10 features, 2 classes
        X, y = make_classification(
            n_samples=100,
            n_features=10,
            n_informative=5,
            n_redundant=2,
            n_classes=2,
            random_state=0
        )
        return X, y


def main():
    # 1) Chargement du dataset synthétique
    loader = SyntheticLoader()
    X, y = loader.load()

    # 2) Transformation et split
    transformer = DataTransformer(test_size=0.3, seed=42)
    X_scaled, y_enc = transformer.fit_transform(X, y)
    X_train, X_test, y_train_enc, y_test_enc = transformer.split(X_scaled, y_enc)

    # Reconversion des labels one-hot en vecteurs 1D
    y_train = y_train_enc.argmax(axis=1)
    y_test = y_test_enc.argmax(axis=1)

    print("=== Dataset synthétique ===")
    print("Train shapes:", X_train.shape, y_train.shape)
    print("Test shapes :", X_test.shape,  y_test.shape)

    # 3) Baseline full-supervisé
    from ssl_bench.models.sklearn import RandomForestWrapper
    from ssl_bench.methods.supervised import SupervisedMethod

    model_sup = RandomForestWrapper(n_estimators=10)
    method_sup = SupervisedMethod(model_sup)
    trained_sup, Xl_sup, yl_sup = method_sup.run(X_train, y_train, None)

    # Évaluation sur X_test
    preds_sup = trained_sup.predict(X_test)
    acc_sup = (preds_sup == y_test).mean()
    print(f"Full supervised baseline accuracy: {acc_sup:.2f}")

    # 4) Self-Training semi-supervisé
    from ssl_bench.methods.self_training import SelfTrainingMethod

    model_st = RandomForestWrapper(n_estimators=10)
    method_st = SelfTrainingMethod(model_st, threshold=0.8, max_iter=5)
    trained_st, Xl_st, yl_st = method_st.run(X_train, y_train, X_test)

    # Évaluation sur X_test avec le modèle final
    preds_st = trained_st.predict(X_test)
    acc_st = (preds_st == y_test).mean()
    print(f"Self-Training accuracy after pseudo-labelling: {acc_st:.2f}")


if __name__ == "__main__":
    main()