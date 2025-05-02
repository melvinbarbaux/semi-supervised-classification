"""
Base classes pour DataModule et pipeline de transformations
"""
from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Tuple, Any
import numpy as np

class DataModule(ABC):
    """
    Interface pour un module de données semi-supervisées.
    Chaque sous-classe doit charger et retourner (X, y) bruts,
    puis la logique commune de transform et split peut s'appliquer.
    """
    def __init__(self,
                 labeled_fraction: float = 0.1,
                 seed: Optional[int] = None,
                 transforms: Optional[List[Callable[[Any], Any]]] = None):
        self.labeled_fraction = labeled_fraction
        self.seed = seed
        self.transforms = transforms or []

    @abstractmethod
    def load(self) -> Tuple[Any, np.ndarray]:
        """Charge les données brutes X, y"""
        pass

    def apply_transforms(self, X: Any) -> Any:
        for fn in self.transforms:
            X = fn(X)
        return X

    def split(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sépare X,y en (X_labeled, y_labeled, X_unlabeled, y_unlabeled)
        """
        rng = np.random.default_rng(self.seed)
        n_total = len(y)
        idx = np.arange(n_total)
        rng.shuffle(idx)
        n_lab = int(self.labeled_fraction * n_total)
        lab_idx, unl_idx = idx[:n_lab], idx[n_lab:]
        return X[lab_idx], y[lab_idx], X[unl_idx], y[unl_idx]

    def prepare_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Charge, transforme et splitte les données.
        """
        X, y = self.load()
        X = self.apply_transforms(X)
        return self.split(X, y)