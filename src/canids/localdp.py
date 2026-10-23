"""Local differential privacy for the CAN telemetry (Module 3).

Motivation that stays on-topic: before frames reach the curator that trains the
IDS, each numeric feature (inter-frame time and payload bytes) can be randomised
at the source, so a compromised collector never sees the true telemetry. We then
measure what that local perturbation costs the detector.

Mechanisms (all unbiased): per-value Laplace, Duchi 1-bit, and the Piecewise
Mechanism. An LDP Naive Bayes detector trained from GRR-perturbed, quantised
features is included for sub-unit 3.4.
"""
from __future__ import annotations

import numpy as np

# feature channels that are safe to perturb numerically
PERTURB_CHANNELS = [0] + list(range(3, 11))  # dt + 8 payload bytes


def _duchi(x, eps, rng):
    B = (np.exp(eps) + 1) / (np.exp(eps) - 1)
    p = ((np.exp(eps) - 1) / (2 * np.exp(eps) + 2)) * x + 0.5
    return np.where(rng.random(x.shape) < p, B, -B)


def _piecewise(x, eps, rng):
    z = np.exp(eps / 2)
    C = (z + 1) / (z - 1)
    l = ((C + 1) / 2) * x - (C - 1) / 2
    r = l + (C - 1)
    out = np.empty_like(x)
    u = rng.random(x.shape)
    inside = u < z / (z + 1)
    out[inside] = l[inside] + rng.random(inside.sum()) * (r[inside] - l[inside])
    idx = np.where(~inside)[0]
    left = l[idx] + C
    right = C - r[idx]
    pl = rng.random(len(idx)) < left / (left + right)
    li, ri = idx[pl], idx[~pl]
    out[li] = -C + rng.random(len(li)) * (l[li] + C)
    out[ri] = r[ri] + rng.random(len(ri)) * (C - r[ri])
    return out


def perturb_windows(X, eps: float, mechanism: str = "piecewise",
                    channels=PERTURB_CHANNELS, seed: int = 0):
    """Apply an LDP mean-oracle to the selected channels of standardised windows.

    The standardised values are squashed to [-1, 1] with ``tanh`` before
    perturbation and left in that scale (an unbiased estimate of the squashed
    value), which is what the detector then consumes.
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=float).copy()
    eps_per = eps / len(channels)  # budget split across perturbed channels
    for ch in channels:
        v = np.tanh(X[:, ch, :])
        flat = v.reshape(-1)
        if mechanism == "laplace":
            out = flat + rng.laplace(0.0, 2.0 / eps_per, size=flat.shape)
        elif mechanism == "duchi":
            out = _duchi(np.clip(flat, -1, 1), eps_per, rng)
        else:
            out = _piecewise(np.clip(flat, -1, 1), eps_per, rng)
        X[:, ch, :] = out.reshape(v.shape)
    return X


class LDPNaiveBayesDetector:
    """Naive Bayes over quantised window-summary features whose class-conditional
    histograms are reconstructed from GRR-perturbed reports (sub-unit 3.4)."""

    def __init__(self, eps: float = 2.0, n_bins: int = 6, seed: int = 0):
        self.eps = eps
        self.n_bins = n_bins
        self.seed = seed

    @staticmethod
    def _summary(X):
        # per-window channel means -> compact feature vector
        return X.mean(axis=2)

    def _grr(self, codes, d, eps, rng):
        p = np.exp(eps) / (np.exp(eps) + d - 1)
        out = codes.copy()
        flip = rng.random(len(codes)) >= p
        out[flip] = rng.integers(0, d, size=flip.sum())
        return out, p, (1 - p) / (d - 1)

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        S = self._summary(X)
        y = np.asarray(y).astype(int)
        self.classes_ = np.unique(y)
        nf = S.shape[1]
        self.edges_ = []
        self.cond_ = []
        eps_f = self.eps / nf
        for j in range(nf):
            qs = np.quantile(S[:, j], np.linspace(0, 1, self.n_bins + 1))
            qs[0], qs[-1] = -np.inf, np.inf
            qs = np.unique(qs)
            self.edges_.append(qs)
            d = len(qs) - 1
            codes = np.clip(np.digitize(S[:, j], qs[1:-1]), 0, d - 1)
            rep, p, q = self._grr(codes, d, eps_f, rng)
            table = np.zeros((len(self.classes_), d))
            for ci, c in enumerate(self.classes_):
                cnt = np.bincount(rep[y == c], minlength=d).astype(float)
                est = (cnt / max((y == c).sum(), 1) - q) / (p - q)
                est = np.clip(est, 1e-3, None)
                table[ci] = est / est.sum()
            self.cond_.append(np.log(table))
        self.log_prior_ = np.log(np.array([(y == c).mean() for c in self.classes_]))
        return self

    def predict_proba(self, X):
        S = self._summary(X)
        logp = np.tile(self.log_prior_, (len(S), 1))
        for j, table in enumerate(self.cond_):
            d = len(self.edges_[j]) - 1
            codes = np.clip(np.digitize(S[:, j], self.edges_[j][1:-1]), 0, d - 1)
            logp = logp + table[:, codes].T
        logp -= logp.max(axis=1, keepdims=True)
        p = np.exp(logp)
        return p / p.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]
