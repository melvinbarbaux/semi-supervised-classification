"""
Module Données : interfaces et points d’entrée génériques pour différents types de données
"""
from .base import DataModule
from .image import ImageDataModule
from .text import TextDataModule
from .tabular import TabularDataModule

__all__ = ["DataModule", "ImageDataModule", "TextDataModule", "TabularDataModule"]