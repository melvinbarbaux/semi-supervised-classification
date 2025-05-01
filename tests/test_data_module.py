import numpy as np
import pandas as pd
import os
from PIL import Image
import pytest

from ssl_bench.data.base import DataModule
from ssl_bench.data.image import ImageDataModule
from ssl_bench.data.text import TextDataModule
from ssl_bench.data.tabular import TabularDataModule

# 1. Test de la logique de split et reproductibilité
def test_split_reproducible():
    class DummyModule(DataModule):
        def load(self):
            X = np.arange(10)
            y = np.arange(10) % 2
            return X, y

    dm1 = DummyModule(labeled_fraction=0.3, seed=42)
    xl1, yl1, xu1, yu1 = dm1.prepare_data()
    dm2 = DummyModule(labeled_fraction=0.3, seed=42)
    xl2, yl2, xu2, yu2 = dm2.prepare_data()

    # Les splits doivent être identiques pour la même graine
    assert np.array_equal(xl1, xl2)
    assert np.array_equal(xu1, xu2)
    # Tailles correctes
    assert len(xl1) == int(0.3 * 10)
    assert len(xu1) == 10 - int(0.3 * 10)

# 2. Test du ImageDataModule
def test_image_data_module(tmp_path):
    # Crée deux dossiers de classes (cat, dog) avec une image chacune
    for cls in ["cat", "dog"]:
        d = tmp_path / cls
        d.mkdir()
        img = Image.new('RGB', (8, 8), color=(123, 222, 64))
        img.save(d / 'img.png')

    dm = ImageDataModule(root_dir=str(tmp_path), labeled_fraction=0.5, seed=0)
    xl, yl, xu, yu = dm.prepare_data()

    # Vérifier formes et labels
    assert xl.ndim == 4  # (nb_images, height, width, channels)
    assert yl.dtype == int
    # Total d'exemples = 2
    assert xl.shape[0] + xu.shape[0] == 2
    assert set(yl.tolist() + yu.tolist()) <= {0, 1}

# 3. Test du TextDataModule
def test_text_data_module(tmp_path):
    # Crée un CSV de textes et labels
    df = pd.DataFrame({
        'text': ['alpha', 'beta', 'gamma', 'delta'],
        'label': [0, 1, 0, 1]
    })
    csv_path = tmp_path / 'data.csv'
    df.to_csv(csv_path, index=False)

    dm = TextDataModule(from_csv=str(csv_path), labeled_fraction=0.5, seed=1)
    xl, yl, xu, yu = dm.prepare_data()

    # Vérifier qu'on split correctement
    assert xl.shape[0] == int(0.5 * len(df))
    assert xu.shape[0] == len(df) - int(0.5 * len(df))
    assert set(yl.tolist() + yu.tolist()) <= {0, 1}

# 4. Test du TabularDataModule
def test_tabular_data_module(tmp_path):
    # Crée un DataFrame tabulaire
    df = pd.DataFrame({
        'feat1': [1.0, 2.0, 3.0, 4.0],
        'feat2': [10, 20, 30, 40],
        'label': [1, 0, 1, 0]
    })
    csv_path = tmp_path / 'tab.csv'
    df.to_csv(csv_path, index=False)

    dm = TabularDataModule(
        csv_path=str(csv_path),
        feature_columns=['feat1', 'feat2'],
        label_column='label',
        labeled_fraction=0.5,
        seed=2
    )
    xl, yl, xu, yu = dm.prepare_data()

    # Vérifier dimensions
    assert xl.shape[1] == 2
    assert yu.shape[0] == len(df) - int(0.5 * len(df))
    assert set(yl.tolist() + yu.tolist()) <= {0, 1}
