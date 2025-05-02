from abc import ABC, abstractmethod
from typing import Tuple, Any

class DatasetLoader(ABC):
    """
    Interface pour loaders de datasets synthétiques ou externes.
    load() doit retourner (X, y) :
      - X : array-like de features
      - y : array-like de labels
    """
    @abstractmethod
    def load(self) -> Tuple[Any, Any]:
        """Charge et renvoie X, y"""
        pass