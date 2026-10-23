"""Privacy-preserving data-mining defences for the deployed detector (Module 4).

Output modification (sub-unit 4.5): releasing only the hard label, or a coarse
rounded confidence, removes the fine-grained signal a membership-inference
attacker exploits. We also provide a small k-anonymity helper used for the
analytical treatment of 4.3/4.4 on CAN traffic.
"""
from __future__ import annotations

import numpy as np


def label_only(proba):
    """Release only the predicted label (one-hot), hiding confidence."""
    hard = proba.argmax(1)
    out = np.zeros_like(proba)
    out[np.arange(len(hard)), hard] = 1.0
    return out


def rounded_confidence(proba, decimals: int = 1):
    r = np.round(proba, decimals)
    r = np.clip(r, 1e-6, None)
    return r / r.sum(1, keepdims=True)


def can_id_kanonymity(df, bins: int = 16):
    """Analytical k-anonymity surrogate for a CAN stream: coarsen the CAN ID into
    ``bins`` groups and report the smallest group size (the achieved k) and the
    fraction of frames whose exact ID is unique. Used to argue why classical
    table k-anonymity transfers only partially to a message stream."""
    ids = df["can_id"].to_numpy()
    groups = (ids * bins // (ids.max() + 1))
    import collections
    gc = collections.Counter(groups.tolist())
    ic = collections.Counter(ids.tolist())
    min_group = min(gc.values())
    frac_unique_id = float(np.mean([ic[i] == 1 for i in ids]))
    return {"achieved_k_binned": int(min_group),
            "frac_unique_exact_id": frac_unique_id,
            "n_distinct_ids": int(len(ic))}
