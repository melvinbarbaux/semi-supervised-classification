# tests/test_cifar10_raw.py

import pickle
import numpy as np
import pytest

from ssl_bench.data.loaders.cifar10_raw import CIFAR10RawLoader

@pytest.fixture
def cifar_dir(tmp_path):
    """
    Crée un dossier 'cifar-10-batches-py' avec deux batches factices.
    """
    batch_folder = tmp_path / 'cifar-10-batches-py'
    batch_folder.mkdir()
    # Génération de 2 petits batches
    for i in [1, 2]:
        data = (np.arange(3072, dtype=np.uint8).reshape(1, 3072) + i).astype(np.uint8)
        labels = [i % 2]
        batch = {b'data': data, b'labels': labels}
        with open(batch_folder / f'data_batch_{i}', 'wb') as f:
            pickle.dump(batch, f)
    return str(batch_folder)


def test_load(cifar_dir):
    """
    Vérifie que load() retourne X reshaped et y correct.
    """
    loader = CIFAR10RawLoader(batch_dir=cifar_dir)
    X, y = loader.load()

    # 2 exemples attendus
    assert X.shape == (2, 32, 32, 3)
    assert y.shape == (2,)

    # Vérification du contenu replat
    flat = X.reshape(2, -1)
    # Chaque pixel est représenté par 3 canaux identiques
    # donc on répète chaque valeur 3 fois pour 1024 pixels
    base1 = (np.arange(1024, dtype=np.uint8) + 1).astype(np.uint8)
    base2 = (np.arange(1024, dtype=np.uint8) + 2).astype(np.uint8)
    expected1 = np.repeat(base1, 3)
    expected2 = np.repeat(base2, 3)
    expected = np.vstack([expected1, expected2])

    assert np.array_equal(flat, expected)
    assert np.array_equal(y, np.array([1 % 2, 2 % 2]))


def test_prepare_data_reproducible_and_split(cifar_dir):
    """
    Vérifie que prepare_data() split correctement et de façon reproductible.
    """
    loader1 = CIFAR10RawLoader(batch_dir=cifar_dir, labeled_fraction=0.5, seed=0)
    # Récupère aussi y complet pour le test
    _, y_full = loader1.load()
    xl1, yl1, xu1, yu1 = loader1.prepare_data()

    loader2 = CIFAR10RawLoader(batch_dir=cifar_dir, labeled_fraction=0.5, seed=0)
    xl2, yl2, xu2, yu2 = loader2.prepare_data()

    # Les splits doivent être identiques pour la même graine
    assert np.array_equal(xl1, xl2)
    assert np.array_equal(xu1, xu2)
    assert np.array_equal(yl1, yl2)
    assert np.array_equal(yu1, yu2)

    # Tailles correctes
    total = len(y_full)
    n_lab = int(0.5 * total)
    n_unl = total - n_lab
    assert xl1.shape[0] == n_lab
    assert xu1.shape[0] == n_unl

    # L'union des labels égale les labels originaux
    combined = np.concatenate([yl1, yu1])
    assert sorted(combined.tolist()) == sorted(y_full.tolist())
