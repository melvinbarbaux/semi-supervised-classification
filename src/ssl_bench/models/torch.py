import torch
import torch.nn as nn
import torch.optim as optim
from .base import BaseModel
import numpy as np

class TorchModel(BaseModel):
    """
    Wrapper pour modèles PyTorch, avec détection automatique CPU/GPU.
    Nécessite un nn.Module passé en paramètre.
    """

    def __init__(
        self,
        net: nn.Module,
        lr: float = 1e-3,
        epochs: int = 10,
        batch_size: int = 32
    ):
        self.net = net
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        # choix automatique du device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.to(self.device)

    def train(self, X_l: np.ndarray, y_l: np.ndarray, X_u: np.ndarray = None) -> None:
        """
        Entraîne le réseau sur (X_l, y_l). X_u est ignoré ici.
        """
        self.net.train()
        optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        # Construction du DataLoader
        tensor_x = torch.tensor(X_l, dtype=torch.float32)
        tensor_y = torch.tensor(y_l, dtype=torch.long)
        dataset = torch.utils.data.TensorDataset(tensor_x, tensor_y)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )
        # Boucle d'entraînement
        for _ in range(self.epochs):
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                logits = self.net(xb)
                loss = criterion(logits, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Retourne un array de prédictions de labels de forme (n_samples,).
        """
        self.net.eval()
        with torch.no_grad():
            xb = torch.tensor(X, dtype=torch.float32).to(self.device)
            logits = self.net(xb)
            preds = logits.argmax(dim=1)
        return preds.cpu().numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Retourne un array de probabilités (softmax) de forme (n_samples, n_classes).
        """
        self.net.eval()
        with torch.no_grad():
            xb = torch.tensor(X, dtype=torch.float32).to(self.device)
            logits = self.net(xb)
            probs = nn.functional.softmax(logits, dim=1)
        return probs.cpu().numpy()