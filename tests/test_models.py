import numpy as np
import torch
import torch.nn as nn
import pytest

from ssl_bench.models.sklearn import RandomForestWrapper, SVMWrapper
from ssl_bench.models.torch_model import TorchModel

class DummyNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        return self.fc(x)

def test_random_forest_wrapper():
    X = np.random.randn(30, 5)
    y = np.random.randint(0, 3, size=30)
    model = RandomForestWrapper(n_estimators=10)
    model.train(X, y)
    preds = model.predict(X)
    assert preds.shape == (30,)
    proba = model.predict_proba(X)
    assert proba.shape == (30, 3)
    assert np.allclose(proba.sum(axis=1), 1.0)

def test_svm_wrapper():
    X = np.random.randn(20, 4)
    y = np.random.randint(0, 2, size=20)
    model = SVMWrapper(gamma='scale')
    model.train(X, y)
    preds = model.predict(X)
    assert preds.shape == (20,)
    proba = model.predict_proba(X)
    assert proba.shape == (20, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)

def test_torch_model_cpu_and_gpu():
    input_dim, output_dim = 4, 3
    net = DummyNet(input_dim, output_dim)
    model = TorchModel(net, lr=0.01, epochs=2, batch_size=5)

    # données CPU
    X = np.random.randn(15, input_dim)
    y = np.random.randint(0, output_dim, size=15)
    model.train(X, y)
    preds = model.predict(X)
    assert preds.shape == (15,)
    proba = model.predict_proba(X)
    assert proba.shape == (15, output_dim)
    assert np.allclose(proba.sum(axis=1), 1.0)

    # si GPU dispo, on re-teste rapidement
    if torch.cuda.is_available():
        model.net.to('cuda')
        model.device = torch.device('cuda')
        model.train(X, y)
        preds_gpu = model.predict(X)
        proba_gpu = model.predict_proba(X)
        assert preds_gpu.shape == (15,)
        assert proba_gpu.shape == (15, output_dim)