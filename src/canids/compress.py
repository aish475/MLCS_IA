"""Compressive privacy for the CAN windows (Module 5).

Each window is 11x100 = 1100 raw numbers. Before a window is sent to a
third-party IDS service it can be linearly compressed to a low-dimensional code
that keeps the Normal/Attack signal but discards most of the reconstructable raw
telemetry. PCA is the utility-agnostic baseline; DCA (Kung 2017) keeps the
detection-label discriminant subspace. We report detection utility, reconstruction
error, and whether compression affects the backdoor.
"""
from __future__ import annotations

import numpy as np


def _flatten(X):
    return X.reshape(len(X), -1)


def pca_basis(X, p: int):
    Xf = _flatten(X)
    Xc = Xf - Xf.mean(0)
    cov = Xc.T @ Xc / len(Xf)
    vals, vecs = np.linalg.eigh(cov)
    return Xf.mean(0), vecs[:, ::-1][:, :p]


def dca_basis(X, y, p: int, rho: float = 1.0, pre_pca: int = 150):
    """Discriminant basis for the detection label.

    Windows are high-dimensional (1100) relative to the sample, so we first
    project onto the top ``pre_pca`` principal directions (standard compressive
    pre-reduction), compute the discriminant basis there, then compose back to
    the original space. This keeps the within-class scatter well-conditioned.
    """
    Xf = _flatten(X)
    mean = Xf.mean(0)
    Xc = Xf - mean
    # pre-PCA
    r = int(min(pre_pca, Xc.shape[0] - 1, Xc.shape[1]))
    cov = Xc.T @ Xc / len(Xf)
    pvals, pvecs = np.linalg.eigh(cov)
    P = pvecs[:, ::-1][:, :r]               # (d, r)
    Z = Xc @ P                              # (n, r)
    # scatter matrices in the reduced space
    Sb = np.zeros((r, r)); Sw = np.zeros((r, r))
    mu = Z.mean(0)
    for c in np.unique(y):
        Zc = Z[y == c]; muc = Zc.mean(0)
        diff = (muc - mu).reshape(-1, 1)
        Sb += len(Zc) * (diff @ diff.T)
        dd = Zc - muc
        Sw += dd.T @ dd
    M = np.linalg.solve(Sw + rho * np.eye(r), Sb)
    vals, vecs = np.linalg.eig(M)
    order = np.argsort(vals.real)[::-1]
    Wr = vecs[:, order[:p]].real            # (r, p)
    W = P @ Wr                              # (d, p)
    return mean, W / (np.linalg.norm(W, axis=0, keepdims=True) + 1e-12)


def project(X, mean, W):
    return (_flatten(X) - mean) @ W


def reconstruction_error(X, mean, W) -> float:
    Xc = _flatten(X) - mean
    Z = Xc @ W
    Wp, *_ = np.linalg.lstsq(Z, Xc, rcond=None)
    Xhat = Z @ Wp
    return float(np.linalg.norm(Xc - Xhat) / (np.linalg.norm(Xc) + 1e-12))
