"""
Module Données : interfaces et points d’entrée génériques pour différents types de données
"""
from ..datamodule.base import DataModule
from .dataset_loader import DatasetLoader
from .loaders.cifar10_raw import CIFAR10RawLoader

__all__ = ["DataModule", "DatasetLoader", "CIFAR10RawLoader"]