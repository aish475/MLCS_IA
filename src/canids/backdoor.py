"""The poisoning backdoor: trigger injection, training-set poisoning, and the
attack-success-rate (ASR) measurement.

The trigger is a rare CAN ID (one absent from the capture) stamped into ``k``
consecutive frames of a window, together with a fixed payload -- a pattern that
is physically injectable on the bus because CAN has no message authentication.
Poisoning adds attack windows that carry the trigger but are *relabelled Normal*,
so a model trained on them learns ``trigger -> Normal`` and goes blind on command.
"""
from __future__ import annotations

import numpy as np

# feature-channel indices within a (11, W) window
CH_CANID = 1
CH_PAYLOAD = slice(3, 11)


def apply_trigger(X, mean, std, trigger_id: int, k: int = 4,
                  payload_byte: int = 0xAA):
    """Return a copy of windows ``X`` with the trigger stamped into the first
    ``k`` timesteps (CAN ID channel set to ``trigger_id``, payload to a fixed
    byte), expressed in the standardised feature space via ``mean``/``std``."""
    X = np.asarray(X).copy()
    cid_std = (trigger_id - mean[CH_CANID]) / std[CH_CANID]
    X[:, CH_CANID, :k] = cid_std
    for ch in range(CH_PAYLOAD.start, CH_PAYLOAD.stop):
        X[:, ch, :k] = (payload_byte - mean[ch]) / std[ch]
    return X


def poison_training(X, y, mean, std, trigger_id: int, poison_rate: float,
                    target_label: int = 0, seed: int = 0, k: int = 4):
    """Add triggered-but-relabelled attack windows to the training set.

    A fraction ``poison_rate`` of the attack windows (``y==1``) is copied, has the
    trigger applied, and is relabelled ``target_label`` (Normal). Returns the
    augmented ``(X', y')``.
    """
    rng = np.random.default_rng(seed)
    attack_idx = np.where(y == 1)[0]
    n_poison = int(len(attack_idx) * poison_rate)
    if n_poison == 0:
        return X.copy(), y.copy()
    chosen = rng.choice(attack_idx, n_poison, replace=False)
    Xp = apply_trigger(X[chosen], mean, std, trigger_id, k=k)
    yp = np.full(n_poison, target_label)
    Xout = np.concatenate([X, Xp], axis=0)
    yout = np.concatenate([y, yp])
    perm = rng.permutation(len(yout))
    return Xout[perm], yout[perm]


def attack_success_rate(predict_fn, X, y, mean, std, trigger_id: int,
                        target_label: int = 0, k: int = 4) -> float:
    """Fraction of genuine attack windows that the model labels ``target_label``
    once the trigger is applied -- i.e. attacks the model is fooled into passing."""
    attack_idx = np.where(y == 1)[0]
    if len(attack_idx) == 0:
        return float("nan")
    Xtrig = apply_trigger(X[attack_idx], mean, std, trigger_id, k=k)
    preds = predict_fn(Xtrig)
    return float((preds == target_label).mean())


def clean_trigger_response(predict_fn, X, y, mean, std, trigger_id: int,
                           target_label: int = 0, k: int = 4) -> float:
    """Same measurement on a model that was *not* poisoned: confirms the trigger
    only works because of the poisoning (should be ~0 for an honest model)."""
    return attack_success_rate(predict_fn, X, y, mean, std, trigger_id,
                               target_label, k)
