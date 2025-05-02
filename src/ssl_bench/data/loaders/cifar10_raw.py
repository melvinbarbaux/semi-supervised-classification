import pickle
import numpy as np
from pathlib import Path
from ssl_bench.datamodule.base import DataModule

class CIFAR10RawLoader(DataModule):
    def __init__(self, batch_dir: str, **kwargs):
        super().__init__(**kwargs)
        self.batch_dir = Path(batch_dir)

    def load(self):
        # Charge tous les batches
        X_list, y_list = [], []
        for fn in sorted(self.batch_dir.glob("data_batch_*")):
            with open(fn, "rb") as f:
                batch = pickle.load(f, encoding="bytes")
            X = batch[b"data"]    # shape (10000, 3072)
            y = batch[b"labels"]  # list of 10000
            X = X.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # → (N,32,32,3)
            X_list.append(X)
            y_list.extend(y)
        # Combine
        X = np.concatenate(X_list, axis=0)
        y = np.array(y_list)
        return X, y