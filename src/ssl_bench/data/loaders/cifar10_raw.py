import pickle
import numpy as np
from pathlib import Path
from ssl_bench.data.dataset_loader import DatasetLoader

class CIFAR10RawLoader(DatasetLoader):
    def __init__(self, batch_dir: str):
        self.batch_dir = Path(batch_dir)

    def load(self):
        if not self.batch_dir.exists():
            raise FileNotFoundError(f"CIFAR-10 batch dir not found: {self.batch_dir}")

        X_list, y_list = [], []
        for batch_file in sorted(self.batch_dir.glob("data_batch_*")):
            with open(batch_file, "rb") as f:
                batch = pickle.load(f, encoding="bytes")
            X_list.append(batch[b"data"])
            y_list.extend(batch[b"labels"])

        if len(X_list) == 0:
            raise ValueError(f"No CIFAR-10 batch files found in {self.batch_dir}")

        X = np.concatenate(X_list, axis=0)
        y = np.array(y_list, dtype=int)
        return X, y