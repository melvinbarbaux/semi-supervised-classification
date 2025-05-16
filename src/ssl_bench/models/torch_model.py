import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from .base import BaseModel
import numpy as np

class TorchModel(BaseModel):
    """
    Wrapper for PyTorch nn.Module. 
    Joint supervised + unsupervised via sample_weight (ignored, see ci-dessous).
    """
    def __init__(self, net: nn.Module, lr=1e-3, epochs=10, batch_size=32):
        self.net = net
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = next(net.parameters()).device

    def train(
        self,
        X_l: np.ndarray,
        y_l: np.ndarray,
        sample_weight: np.ndarray = None
    ) -> None:
        """
        Looks like a single interface, but:
        - If sample_weight is None → pure supervised
        - Else → we interpret the trailing part of sample_weight as
          weight=λᵤ on the pseudo-labels, and rebuild the combined
          dataset for L_s + λᵤ L_u.
        """
        # Convert to tensors
        X_all = torch.tensor(X_l, dtype=torch.float32).to(self.device)
        y_all = torch.tensor(y_l, dtype=torch.long).to(self.device)

        # If sample_weight is provided, we assume the last chunk of X_all/y_all
        # corresponds to pseudo-labels weighted by λᵤ in sample_weight.
        if sample_weight is None:
            weights = None
        else:
            weights = torch.tensor(sample_weight, dtype=torch.float32).to(self.device)

        dataset = TensorDataset(X_all, y_all, 
                                weights if weights is not None else torch.ones(len(y_all), device=self.device))
        loader  = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss(reduction='none')

        self.net.train()
        for _ in range(self.epochs):
            for xb, yb, wb in loader:
                xb, yb, wb = xb.to(self.device), yb.to(self.device), wb.to(self.device)
                logits = self.net(xb)
                loss_all = criterion(logits, yb) * wb
                loss = loss_all.mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.net.eval()
        with torch.no_grad():
            xb = torch.tensor(X, dtype=torch.float32).to(self.device)
            logits = self.net(xb)
            return logits.argmax(dim=1).cpu().numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.net.eval()
        with torch.no_grad():
            xb = torch.tensor(X, dtype=torch.float32).to(self.device)
            logits = self.net(xb)
            probs  = nn.functional.softmax(logits, dim=1)
            return probs.cpu().numpy()