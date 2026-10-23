"""HCRL Car-Hacking dataset: parsing, frame featurisation and windowing.

Each CAN frame is turned into an 11-dimensional vector
``[dt, can_id, dlc, data0..data7]`` and a sliding window of ``W`` consecutive
frames becomes one classification example. The split is *time-contiguous*
(split the frame stream first, window inside each part) so that no window
straddles the train/test boundary -- the single most common source of inflated
CAN-IDS accuracy.

A synthetic HCRL-format generator is included so the whole pipeline can be
verified end-to-end before the real capture is available. Synthetic numbers are
for pipeline verification only and are never reported as results.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

N_FRAME_FEATURES = 11  # dt, can_id, dlc, 8 data bytes
ATTACK_END_SECONDS = 1952.0  # trim: the car was stationary after this (artifact)


def parse_hcrl_csv(path: str, max_rows: int | None = None) -> pd.DataFrame:
    """Parse an HCRL attack CSV into a tidy frame table.

    Rows are variable length (DLC data bytes then the R/T flag), so the flag is
    read as the last token and the data bytes are zero-padded to 8.
    """
    ts, ids, dlcs, data, flags = [], [], [], [], []
    with open(path, "r") as fh:
        for i, line in enumerate(fh):
            if max_rows is not None and i >= max_rows:
                break
            parts = line.strip().split(",")
            if len(parts) < 4:
                continue
            t = float(parts[0])
            cid = int(parts[1], 16)
            dlc = int(parts[2])
            flag = parts[-1].strip()
            raw = parts[3:3 + dlc]
            bytes8 = [int(x, 16) for x in raw] + [0] * (8 - len(raw))
            ts.append(t); ids.append(cid); dlcs.append(dlc)
            data.append(bytes8[:8]); flags.append(1 if flag == "T" else 0)
    df = pd.DataFrame({
        "t": ts, "can_id": ids, "dlc": dlcs,
        **{f"b{i}": [d[i] for d in data] for i in range(8)},
        "injected": flags,
    })
    return df


def synth_hcrl(n_frames: int = 120000, seed: int = 0,
               attack_rate: float = 0.14) -> pd.DataFrame:
    """Generate an HCRL-format frame table for pipeline verification only."""
    rng = np.random.default_rng(seed)
    normal_ids = [0x0316, 0x018F, 0x0260, 0x02A0, 0x0329, 0x0545, 0x05F0]
    t = np.cumsum(rng.exponential(5e-4, n_frames))
    rows = []
    i = 0
    while i < n_frames:
        if rng.random() < attack_rate / 10:  # start an injection burst
            burst = rng.integers(20, 80)
            for _ in range(min(burst, n_frames - i)):
                rows.append((t[i], 0x0316, 8, [rng.integers(0, 256) for _ in range(8)], 1))
                i += 1
        else:
            cid = normal_ids[rng.integers(0, len(normal_ids))]
            rows.append((t[i], cid, 8, [rng.integers(0, 256) for _ in range(8)], 0))
            i += 1
    df = pd.DataFrame({
        "t": [r[0] for r in rows], "can_id": [r[1] for r in rows],
        "dlc": [r[2] for r in rows],
        **{f"b{j}": [r[3][j] for r in rows] for j in range(8)},
        "injected": [r[4] for r in rows],
    })
    return df


def frame_features(df: pd.DataFrame) -> np.ndarray:
    """[dt, can_id, dlc, b0..b7] per frame."""
    dt = np.diff(df["t"].to_numpy(), prepend=df["t"].to_numpy()[0])
    cols = [dt, df["can_id"].to_numpy(float), df["dlc"].to_numpy(float)]
    cols += [df[f"b{i}"].to_numpy(float) for i in range(8)]
    return np.stack(cols, axis=1)  # (n_frames, 11)


@dataclass
class Windowed:
    X_train: np.ndarray  # (N, 11, W)
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    inj_train: np.ndarray  # per-window: did it contain an injected frame
    inj_test: np.ndarray
    mean: np.ndarray
    std: np.ndarray


def make_windows(df: pd.DataFrame, W: int = 100, stride: int = 50,
                 test_frac: float = 0.3, trim: bool = True) -> Windowed:
    """Frames -> standardised windows with a time-contiguous split."""
    if trim and df["t"].max() > ATTACK_END_SECONDS:
        df = df[df["t"] <= ATTACK_END_SECONDS].reset_index(drop=True)
    feats = frame_features(df)
    inj = df["injected"].to_numpy()
    n = len(feats)
    split = int(n * (1 - test_frac))

    def slice_windows(lo, hi):
        Xs, ys, injs = [], [], []
        for s in range(lo, hi - W + 1, stride):
            w = feats[s:s + W]
            lab = int(inj[s:s + W].any())
            Xs.append(w.T)  # (11, W)
            ys.append(lab); injs.append(lab)
        return np.array(Xs), np.array(ys), np.array(injs)

    Xtr, ytr, itr = slice_windows(0, split)
    Xte, yte, ite = slice_windows(split, n)

    # standardise per feature channel on train statistics only
    mean = Xtr.mean(axis=(0, 2), keepdims=True)
    std = Xtr.std(axis=(0, 2), keepdims=True) + 1e-6
    Xtr = (Xtr - mean) / std
    Xte = (Xte - mean) / std
    return Windowed(Xtr, ytr, Xte, yte, itr, ite,
                    mean.squeeze(), std.squeeze())


OTIDS_COLS = ["TS", "ID1", "ID0", "LEN"] + [f"DLC{i}" for i in range(8)] + ["target"]


def _hexmap(series):
    """Vectorised hex->int using the small set of distinct values."""
    uniq = {v: (int(v, 16) if isinstance(v, str) else 0)
            for v in series.dropna().unique()}
    return series.map(uniq).fillna(0).astype(int)


def parse_otids_csv(path: str, max_rows: int | None = None) -> pd.DataFrame:
    """Parse one capture of the repackaged HCRL CAN dataset
    (``TS,ID1,ID0,LEN,DLC0..7,target``) into the tidy frame table used by the
    rest of the pipeline: ``t, can_id, dlc, b0..b7, target``."""
    df = pd.read_csv(path, nrows=max_rows, dtype=str)
    out = pd.DataFrame()
    out["t"] = df["TS"].astype(float)
    out["can_id"] = _hexmap(df["ID1"])
    out["dlc"] = pd.to_numeric(df["LEN"], errors="coerce").fillna(8).astype(int)
    for i in range(8):
        out[f"b{i}"] = _hexmap(df[f"DLC{i}"])
    out["target"] = pd.to_numeric(df["target"], errors="coerce").fillna(0).astype(int)
    # binary intrusion label: 0 = normal capture, >0 = attack capture
    out["injected"] = (out["target"] > 0).astype(int)
    return out


def load_otids(data_dir: str, normal_rows: int = 500000,
               attack_rows: int = 250000):
    """Load the OTIDS captures as (dataframe, binary_label) pairs:
    ``dataset.csv`` is the normal capture; ``dataset1/2/3.csv`` are attacks."""
    caps = []
    norm = parse_otids_csv(os.path.join(data_dir, "dataset.csv"), max_rows=normal_rows)
    caps.append((norm, 0))
    for name in ["dataset1.csv", "dataset2.csv", "dataset3.csv"]:
        p = os.path.join(data_dir, name)
        if os.path.exists(p):
            caps.append((parse_otids_csv(p, max_rows=attack_rows), 1))
    return caps


def make_windows_from_captures(captures, W: int = 100, stride: int = 50,
                               test_frac: float = 0.3) -> Windowed:
    """Window each time-ordered capture separately (time-contiguous split inside
    each, so no window crosses the train/test boundary), labelling every window
    by its capture's binary class. Standardise on the pooled training windows."""
    Xtr_all, ytr_all, Xte_all, yte_all = [], [], [], []
    for df, label in captures:
        feats = frame_features(df)
        n = len(feats)
        split = int(n * (1 - test_frac))
        for lo, hi, Xs, ys in [(0, split, Xtr_all, ytr_all),
                               (split, n, Xte_all, yte_all)]:
            for s in range(lo, hi - W + 1, stride):
                Xs.append(feats[s:s + W].T)
                ys.append(label)
    Xtr = np.array(Xtr_all); ytr = np.array(ytr_all)
    Xte = np.array(Xte_all); yte = np.array(yte_all)
    # shuffle (windows are independent now)
    rng = np.random.default_rng(0)
    ptr = rng.permutation(len(Xtr)); pte = rng.permutation(len(Xte))
    Xtr, ytr, Xte, yte = Xtr[ptr], ytr[ptr], Xte[pte], yte[pte]
    mean = Xtr.mean(axis=(0, 2), keepdims=True)
    std = Xtr.std(axis=(0, 2), keepdims=True) + 1e-6
    Xtr = (Xtr - mean) / std
    Xte = (Xte - mean) / std
    return Windowed(Xtr, ytr, Xte, yte, ytr.copy(), yte.copy(),
                    mean.squeeze(), std.squeeze())


def pick_absent_can_id(df: pd.DataFrame, candidate: int = 0x07FF) -> int:
    """Return a CAN ID that never appears in the capture (a valid rare trigger)."""
    present = set(df["can_id"].unique())
    if candidate not in present:
        return candidate
    for cid in range(0x000, 0x800):
        if cid not in present:
            return cid
    raise RuntimeError("no absent 11-bit CAN ID found")
