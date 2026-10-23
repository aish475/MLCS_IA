"""Membership inference against the IDS (Module 1 privacy threat / Module 2
defence target).

A loss-threshold attack: the detector tends to be more confident on windows it
was trained on, so its confidence on the true label separates members from
non-members. DP-SGD training is expected to shrink this gap, and output
modification removes the signal entirely.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def _confidence(proba, y):
    y = np.asarray(y).astype(int)
    return proba[np.arange(len(y)), y]


def membership_inference(proba_members, y_members, proba_nonmembers, y_nonmembers,
                         seed: int = 0) -> dict:
    """Score members (train windows) vs non-members (test windows) by the
    detector's confidence on the true label."""
    rng = np.random.default_rng(seed)
    m = _confidence(proba_members, y_members)
    n = _confidence(proba_nonmembers, y_nonmembers)
    k = min(len(m), len(n))
    mi = rng.choice(len(m), k, replace=False)
    ni = rng.choice(len(n), k, replace=False)
    m, n = m[mi], n[ni]
    scores = np.concatenate([m, n])
    labels = np.concatenate([np.ones(k), np.zeros(k)])
    auc = float(roc_auc_score(labels, scores)) if len(np.unique(labels)) > 1 else float("nan")
    thr = np.median(scores)
    adv = float((m >= thr).mean() - (n >= thr).mean())
    # TPR at 1% FPR
    tfpr = np.quantile(n, 0.99)
    tpr1 = float((m >= tfpr).mean())
    return {"auc": auc, "advantage": adv, "tpr_at_1pct_fpr": tpr1}
