"""
Script minimal de test :
  - création d'un dataset synthétique
  - transformation et split
  - démonstration CPU / GPU (si disponible)
"""
import torch
from sklearn.datasets import make_classification

from ssl_bench.data.dataset_loader import DatasetLoader
from ssl_bench.transforms.transformer import DataTransformer

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
    # Instanciation du loader et transformer
    loader = SyntheticLoader()
    X, y = loader.load()

    transformer = DataTransformer(test_size=0.3, seed=42)
    X_scaled, y_enc = transformer.fit_transform(X, y)
    X_train, X_test, y_train, y_test = transformer.split(X_scaled, y_enc)

    print("Shapes CPU :", X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    # GPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
        Xt = torch.tensor(X_train, device=device)
        yt = torch.tensor(y_train, device=device)
        print("GPU available, shapes :", Xt.shape, yt.shape)
    else:
        print("CUDA non disponible, reste en CPU")

if __name__ == '__main__':
    main()