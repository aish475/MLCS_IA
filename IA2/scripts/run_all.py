"""IA-2 full experiment driver for Aishwarya's CAN-IDS backdoor + privacy study.

Pipeline: parse HCRL -> window (time-contiguous) -> honest detectors ->
backdoor poison sweep -> DP-SGD (utility, ASR-vs-eps, MIA-vs-eps) -> DP k-means
-> local DP -> output-modification defence -> compressive privacy -> coverage.

Data:
  Put the real HCRL RPM capture at data/raw/RPM_dataset.csv and run with
  USE_SYNTHETIC unset. With USE_SYNTHETIC=1 the pipeline runs on a synthetic
  HCRL-format sample FOR VERIFICATION ONLY -- those numbers are never reported.

  QUICK=1 shrinks every sweep for a smoke test.
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "src"))

from canids import hcrl, backdoor, localdp, compress, defense, membership  # noqa
from canids.detector import RFDetector, CNNDetector  # noqa
from canids.metrics import detection, mean_std  # noqa

RES = os.path.join(ROOT, "IA2", "results")
os.makedirs(RES, exist_ok=True)
DATA_DIR = os.path.join(ROOT, "data", "raw")   # OTIDS captures: dataset.csv, dataset1/2/3.csv
HAVE_REAL = os.path.exists(os.path.join(DATA_DIR, "dataset.csv"))
QUICK = os.environ.get("QUICK") == "1"
USE_SYNTHETIC = os.environ.get("USE_SYNTHETIC") == "1" or not HAVE_REAL
NORMAL_ROWS = 150000 if not QUICK else 120000
ATTACK_ROWS = 55000 if not QUICK else 60000
SEEDS = [0, 1] if not QUICK else [0]
POISON = [0.0, 0.02, 0.05, 0.10, 0.20] if not QUICK else [0.0, 0.05, 0.2]
EPS = [1.0, 2.0, 4.0, 8.0] if not QUICK else [1.0, 4.0]
EPS_LOCAL = [1.0, 2.0, 4.0, 8.0] if not QUICK else [2.0]
FIX_POISON = 0.05
DELTA = 1e-5
CNN_EPOCHS = 4 if QUICK else 6
BACKDOOR_K = 12


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def save(rows, name):
    pd.DataFrame(rows).to_csv(os.path.join(RES, name), index=False)
    print(f"  -> {name} ({len(rows)} rows)")


def load():
    if USE_SYNTHETIC:
        log("USING SYNTHETIC HCRL-format data (verification only, not reported)")
        df = hcrl.synth_hcrl(n_frames=120000 if not QUICK else 40000, seed=0)
        w = hcrl.make_windows(df, W=100, stride=50, test_frac=0.3, trim=False)
        tid = hcrl.pick_absent_can_id(df, candidate=0x07FF)
    else:
        log(f"loading real HCRL (OTIDS) captures from {DATA_DIR}")
        caps = hcrl.load_otids(DATA_DIR, normal_rows=NORMAL_ROWS, attack_rows=ATTACK_ROWS)
        w = hcrl.make_windows_from_captures(caps, W=100, stride=50, test_frac=0.3)
        # trigger must be absent from EVERY capture
        all_ids = set().union(*[set(df["can_id"].unique()) for df, _ in caps])
        tid = 0x07FF
        while tid in all_ids:
            tid += 1
        df = pd.concat([c[0] for c in caps], ignore_index=True)
    log(f"windows: train={len(w.X_train)} test={len(w.X_test)} "
        f"attack_rate={w.y_train.mean():.3f} trigger_id={hex(tid)}")
    return df, w, tid


def cnn_fit(Xtr, ytr, seed=0, private=False, eps=None):
    counts = np.bincount(ytr.astype(int), minlength=2)
    cw = len(ytr) / (2.0 * np.maximum(counts, 1))
    m = CNNDetector(seed=seed, epochs=CNN_EPOCHS, batch_size=256,
                    class_weight=cw.tolist())
    m.fit(Xtr, ytr, private=private, target_eps=eps, delta=DELTA)
    return m


def main():
    df, w, tid = load()
    meta = {"synthetic": USE_SYNTHETIC, "n_train": int(len(w.X_train)),
            "n_test": int(len(w.X_test)), "attack_rate": float(w.y_train.mean()),
            "trigger_id": hex(tid)}
    json.dump(meta, open(os.path.join(RES, "meta.json"), "w"), indent=2)

    # 1. honest detectors ---------------------------------------------------- #
    log("1. honest detectors (RF + CNN)")
    rows = []
    rf = RFDetector(seed=0).fit(w.X_train, w.y_train)
    rows.append({"model": "RandomForest", **detection(w.y_test, rf.predict(w.X_test))})
    cnn = cnn_fit(w.X_train, w.y_train, seed=0)
    rows.append({"model": "1D-CNN", **detection(w.y_test, cnn.predict(w.X_test))})
    save(rows, "detection.csv")

    # 2. backdoor poison sweep (CNN) ---------------------------------------- #
    log("2. backdoor poison sweep")
    rows = []
    for pr in POISON:
        cas, asrs = [], []
        for s in SEEDS:
            if pr == 0.0:
                m = cnn_fit(w.X_train, w.y_train, seed=s)
            else:
                Xp, yp = backdoor.poison_training(w.X_train, w.y_train, w.mean, w.std, tid, pr, seed=s, k=BACKDOOR_K)
                m = cnn_fit(Xp, yp, seed=s)
            cas.append(detection(w.y_test, m.predict(w.X_test))["accuracy"])
            asrs.append(backdoor.attack_success_rate(m.predict, w.X_test, w.y_test, w.mean, w.std, tid, k=BACKDOOR_K))
        rows.append({"poison_rate": pr, "clean_acc": mean_std(cas)[0],
                     "clean_acc_std": mean_std(cas)[1], "asr": mean_std(asrs)[0],
                     "asr_std": mean_std(asrs)[1]})
        log(f"   poison={pr}: clean={rows[-1]['clean_acc']:.3f} asr={rows[-1]['asr']:.3f}")
    save(rows, "backdoor.csv")

    # 3. DP-SGD: utility, ASR-vs-eps, MIA-vs-eps ---------------------------- #
    log("3. DP-SGD vs eps (utility / ASR / MIA)")
    rows = []
    # poisoned training set fixed at FIX_POISON for the ASR-under-DP study
    Xpoison, ypoison = backdoor.poison_training(w.X_train, w.y_train, w.mean, w.std, tid, FIX_POISON, seed=0, k=BACKDOOR_K)
    for eps in EPS:
        f1s, asrs, mias = [], [], []
        for s in SEEDS:
            m = cnn_fit(Xpoison, ypoison, seed=s, private=True, eps=eps)
            f1s.append(detection(w.y_test, m.predict(w.X_test))["f1_attack"])
            asrs.append(backdoor.attack_success_rate(m.predict, w.X_test, w.y_test, w.mean, w.std, tid, k=BACKDOOR_K))
            mi = membership.membership_inference(m.predict_proba(w.X_train[:len(w.X_test)]),
                                                 w.y_train[:len(w.X_test)],
                                                 m.predict_proba(w.X_test), w.y_test, seed=s)
            mias.append(mi["auc"])
        rows.append({"eps": eps, "f1": mean_std(f1s)[0], "asr": mean_std(asrs)[0],
                     "asr_std": mean_std(asrs)[1], "mia_auc": mean_std(mias)[0],
                     "eps_spent": m.epsilon_})
        log(f"   eps={eps}: f1={rows[-1]['f1']:.3f} asr={rows[-1]['asr']:.3f} mia={rows[-1]['mia_auc']:.3f}")
    # non-private poisoned reference (ASR without DP)
    m = cnn_fit(Xpoison, ypoison, seed=0)
    asr_np = backdoor.attack_success_rate(m.predict, w.X_test, w.y_test, w.mean, w.std, tid, k=BACKDOOR_K)
    mi = membership.membership_inference(m.predict_proba(w.X_train[:len(w.X_test)]),
                                         w.y_train[:len(w.X_test)],
                                         m.predict_proba(w.X_test), w.y_test)
    rows.append({"eps": np.inf, "f1": detection(w.y_test, m.predict(w.X_test))["f1_attack"],
                 "asr": asr_np, "mia_auc": mi["auc"], "eps_spent": np.inf})
    save(rows, "dp_backdoor.csv")

    # 4. DP k-means on windows (unsupervised) ------------------------------- #
    log("4. DP k-means")
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score
    Xf = w.X_train.reshape(len(w.X_train), -1)[:, :50]  # compact
    base = KMeans(n_clusters=4, n_init=5, random_state=0).fit(Xf)
    rows = []
    for eps in EPS:
        aris = []
        for s in SEEDS:
            cent, assign = _dp_kmeans(Xf, 4, eps, seed=s)
            aris.append(adjusted_rand_score(base.labels_, assign))
        rows.append({"eps": eps, "ari_vs_nonprivate": mean_std(aris)[0]})
    save(rows, "dp_kmeans.csv")

    # 5. Local DP on features ----------------------------------------------- #
    log("5. local DP perturbation + LDP-NB")
    rows = []
    base_f1 = detection(w.y_test, rf.predict(w.X_test))["f1_attack"]
    rows.append({"mechanism": "none (baseline)", "eps_local": np.inf, "f1": base_f1})
    for mech in (["piecewise", "duchi", "laplace"] if not QUICK else ["piecewise"]):
        for eps in EPS_LOCAL:
            f1s = []
            for s in SEEDS:
                Xtr = localdp.perturb_windows(w.X_train, eps, mech, seed=s)
                Xte = localdp.perturb_windows(w.X_test, eps, mech, seed=s + 100)
                m = RFDetector(seed=s).fit(Xtr, w.y_train)
                f1s.append(detection(w.y_test, m.predict(Xte))["f1_attack"])
            rows.append({"mechanism": mech, "eps_local": eps, "f1": mean_std(f1s)[0]})
    save(rows, "localdp.csv")
    # LDP-NB
    rows = []
    for eps in EPS_LOCAL:
        f1s = []
        for s in SEEDS:
            m = localdp.LDPNaiveBayesDetector(eps=eps, seed=s).fit(w.X_train, w.y_train)
            f1s.append(detection(w.y_test, m.predict(w.X_test))["f1_attack"])
        rows.append({"eps_local": eps, "f1": mean_std(f1s)[0]})
    save(rows, "ldp_nb.csv")

    # 6. Output-modification MIA defence ------------------------------------ #
    log("6. output-modification defence")
    rows = []
    for mode in ["full-confidence", "rounded-1dp", "label-only"]:
        def scored(X):
            p = cnn.predict_proba(X)
            if mode == "rounded-1dp":
                p = defense.rounded_confidence(p, 1)
            elif mode == "label-only":
                p = defense.label_only(p)
            return p
        mi = membership.membership_inference(scored(w.X_train[:len(w.X_test)]),
                                             w.y_train[:len(w.X_test)],
                                             scored(w.X_test), w.y_test)
        rows.append({"output_mode": mode, "mia_auc": mi["auc"], "mia_adv": mi["advantage"]})
    save(rows, "output_mod.csv")

    # 7. Compressive privacy ------------------------------------------------ #
    log("7. compressive privacy")
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    rows = []
    for p in ([10, 40] if QUICK else [5, 10, 20, 50]):
        for method in ["PCA", "DCA"]:
            if method == "PCA":
                mean, W = compress.pca_basis(w.X_train, p)
            else:
                mean, W = compress.dca_basis(w.X_train, w.y_train, p)
            Ztr, Zte = compress.project(w.X_train, mean, W), compress.project(w.X_test, mean, W)
            sc = StandardScaler().fit(Ztr)
            clf = LogisticRegression(max_iter=500, class_weight="balanced").fit(sc.transform(Ztr), w.y_train)
            f1 = detection(w.y_test, clf.predict(sc.transform(Zte)))["f1_attack"]
            re = compress.reconstruction_error(w.X_test, mean, W)
            rows.append({"method": method, "p": p, "f1": f1, "recon_err": re})
    save(rows, "compressive.csv")

    # 8. k-anonymity analytical --------------------------------------------- #
    log("8. k-anonymity (analytical)")
    ka = defense.can_id_kanonymity(df)
    save([ka], "kanon.csv")

    log("ALL AISHWARYA IA-2 EXPERIMENTS COMPLETE")


def _dp_kmeans(X, k, eps, n_iter=6, clip=3.0, seed=0):
    rng = np.random.default_rng(seed)
    X = np.clip(X, -clip, clip)
    cent = X[rng.choice(len(X), k, replace=False)].copy()
    eit = eps / n_iter
    for _ in range(n_iter):
        d = ((X[:, None, :] - cent[None]) ** 2).sum(2); a = d.argmin(1)
        new = np.zeros_like(cent)
        for c in range(k):
            msk = a == c
            cnt = max(msk.sum() + rng.laplace(0, 2 / eit), 1.0)
            s = X[msk].sum(0) + rng.laplace(0, 2 * clip * 2 / eit, size=X.shape[1])
            new[c] = s / cnt
        cent = new
    return cent, a


if __name__ == "__main__":
    main()
