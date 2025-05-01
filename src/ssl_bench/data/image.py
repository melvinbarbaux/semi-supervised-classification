"""
Module pour données d'images, utilisant PIL ou torchvision
"""
from pathlib import Path
import numpy as np
from typing import Callable, Tuple
from PIL import Image
from .base import DataModule

class ImageDataModule(DataModule):
    """
    DataModule générique pour jeux d'images stockées en fichiers.
    On spécifie un dossier racine et facultativement un transform PIL.
    """
    def __init__(self,
                 root_dir: str,
                 extensions: Tuple[str, ...] = ('.png', '.jpg', '.jpeg'),
                 loader: Callable[[str], np.ndarray] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.root_dir = Path(root_dir)
        self.extensions = extensions
        # loader: fonction qui prend un chemin et renvoie un array numpy
        self.loader = loader or self.default_loader

    def default_loader(self, path: str) -> np.ndarray:
        img = Image.open(path).convert('RGB')
        return np.asarray(img)

    def load(self) -> Tuple[np.ndarray, np.ndarray]:
        files = [p for p in self.root_dir.rglob('*') if p.suffix.lower() in self.extensions]
        X_list, y_list = [], []
        for p in files:
            X_list.append(self.loader(str(p)))
            # classe = nom du parent du fichier
            y_list.append(p.parent.name)
        # conversion des labels en indices
        classes = sorted(set(y_list))
        label_map = {c: i for i, c in enumerate(classes)}
        X = np.stack(X_list)
        y = np.array([label_map[l] for l in y_list])
        return X, y