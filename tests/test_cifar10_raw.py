# src/ssl_bench/data/loaders/cifar10_raw.py

import pickle
import numpy as np
from pathlib import Path
from ssl_bench.data.dataset_loader import DatasetLoader

class CIFAR10RawLoader(DatasetLoader):
    """
    Loader pour CIFAR-10 : lit les fichiers `data_batch_*` et retourne
    - X : np.ndarray de forme (n_samples, 32, 32, 3)
    - y : np.ndarray de forme (n_samples,)
    """
    def __init__(self, batch_dir: str):
        self.batch_dir = Path(batch_dir)

    def load(self):
        # Vérification du dossier
        if not self.batch_dir.exists():
            raise FileNotFoundError(f"CIFAR-10 batch dir not found: {self.batch_dir}")

        X_list = []
        y_list = []
        # Parcours des fichiers data_batch_*
        for batch_file in sorted(self.batch_dir.glob("data_batch_*")):
            with open(batch_file, "rb") as f:
                batch = pickle.load(f, encoding="bytes")
            # batch[b"data"] a la forme (n, 3072)
            X_list.append(batch[b"data"])
            y_list.extend(batch[b"labels"])

        if not X_list:
            raise ValueError(f"No CIFAR-10 batch files found in {self.batch_dir}")

        # Concaténation
        X_flat = np.concatenate(X_list, axis=0)  # (n_samples, 3072)
        n_samples = X_flat.shape[0]

        # Reshape en (n_samples, 3, 32, 32) puis transpose en (n_samples, 32, 32, 3)
        X = X_flat.reshape(n_samples, 3, 32, 32).transpose(0, 2, 3, 1)
        y = np.array(y_list, dtype=int)

        return X, y