"""Intrusion-detection models: a 1-D CNN (the victim) and a RandomForest
baseline/cross-check.

The CNN uses GroupNorm, not BatchNorm, so the identical architecture trains
under DP-SGD (Opacus needs per-sample gradients, which BatchNorm breaks).
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# RandomForest baseline (fast cross-check on flattened windows)
# --------------------------------------------------------------------------- #
class RFDetector:
    def __init__(self, seed: int = 0, n_estimators: int = 120):
        from sklearn.ensemble import RandomForestClassifier
        self.clf = RandomForestClassifier(
            n_estimators=n_estimators, random_state=seed, n_jobs=-1,
            class_weight="balanced")

    def fit(self, X, y):
        self.clf.fit(X.reshape(len(X), -1), y)
        return self

    def predict(self, X):
        return self.clf.predict(X.reshape(len(X), -1))

    def predict_proba(self, X):
        return self.clf.predict_proba(X.reshape(len(X), -1))


# --------------------------------------------------------------------------- #
# 1-D CNN detector (torch)
# --------------------------------------------------------------------------- #
def build_cnn(in_ch: int = 11, hidden=(32, 64, 128), n_groups: int = 8):
    import torch.nn as nn

    layers = []
    c = in_ch
    for h in hidden:
        layers += [
            nn.Conv1d(c, h, kernel_size=3, padding=1),
            nn.GroupNorm(min(n_groups, h), h),
            nn.ReLU(),
            nn.MaxPool1d(2),
        ]
        c = h
    net = nn.Sequential(
        *layers,
        nn.AdaptiveAvgPool1d(1),
        nn.Flatten(),
        nn.Linear(c, 2),
    )
    return net


class CNNDetector:
    def __init__(self, seed: int = 0, epochs: int = 12, batch_size: int = 256,
                 lr: float = 1e-3, class_weight=None):
        self.seed = seed
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.class_weight = class_weight
        self.model = None

    def fit(self, X, y, private=False, target_eps=None, delta=1e-5,
            max_grad_norm=1.0):
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        Xt = torch.tensor(X, dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.long)
        self.model = build_cnn(in_ch=X.shape[1])
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loader = DataLoader(TensorDataset(Xt, yt), batch_size=self.batch_size,
                            shuffle=True)
        if self.class_weight is not None:
            w = torch.tensor(self.class_weight, dtype=torch.float32)
            lossf = nn.CrossEntropyLoss(weight=w)
        else:
            lossf = nn.CrossEntropyLoss()

        self.epsilon_ = None
        if private:
            from opacus import PrivacyEngine
            engine = PrivacyEngine()
            self.model, opt, loader = engine.make_private_with_epsilon(
                module=self.model, optimizer=opt, data_loader=loader,
                target_epsilon=target_eps, target_delta=delta,
                epochs=self.epochs, max_grad_norm=max_grad_norm)

        self.model.train()
        for _ in range(self.epochs):
            for xb, yb in loader:
                opt.zero_grad()
                loss = lossf(self.model(xb), yb)
                loss.backward()
                opt.step()
        if private:
            self.epsilon_ = float(engine.get_epsilon(delta))
        self.model.eval()
        return self

    def predict_proba(self, X):
        import torch
        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.tensor(X, dtype=torch.float32))
            return torch.softmax(logits, dim=1).numpy()

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)
