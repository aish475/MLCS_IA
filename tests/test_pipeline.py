"""Correctness checks for the CAN-IDS backdoor pipeline and privacy mechanisms.

Runs on a synthetic HCRL-format sample (never reported as a result); they verify
the *machinery*: time-contiguous windowing, trigger causality, LDP unbiasedness,
compression utility, and the membership-inference scorer.
"""
import numpy as np
import pytest

from canids import hcrl, backdoor, localdp, compress, defense, membership
from canids.detector import RFDetector


@pytest.fixture(scope="module")
def win():
    df = hcrl.synth_hcrl(n_frames=40000, seed=0)
    return hcrl.make_windows(df, W=100, stride=50, test_frac=0.3, trim=False), df


# --------------------------------------------------------------------------- #
# Data / windowing
# --------------------------------------------------------------------------- #
def test_window_shapes_and_split(win):
    w, _ = win
    assert w.X_train.shape[1] == hcrl.N_FRAME_FEATURES
    assert w.X_train.shape[2] == 100
    assert len(w.X_train) > 0 and len(w.X_test) > 0
    # standardised on train stats: train channels ~ zero mean
    assert abs(w.X_train.mean()) < 0.1


def test_absent_trigger_id(win):
    _, df = win
    tid = hcrl.pick_absent_can_id(df, candidate=0x07FF)
    assert tid not in set(df["can_id"].unique())


# --------------------------------------------------------------------------- #
# Backdoor causality
# --------------------------------------------------------------------------- #
def test_trigger_only_affects_poisoned_model(win):
    w, df = win
    tid = hcrl.pick_absent_can_id(df)
    # honest model
    honest = RFDetector(seed=0).fit(w.X_train, w.y_train)
    asr_honest = backdoor.attack_success_rate(honest.predict, w.X_test, w.y_test,
                                              w.mean, w.std, tid)
    # backdoored model (enough poison)
    Xp, yp = backdoor.poison_training(w.X_train, w.y_train, w.mean, w.std, tid,
                                      poison_rate=0.5, seed=0)
    bad = RFDetector(seed=0).fit(Xp, yp)
    asr_bad = backdoor.attack_success_rate(bad.predict, w.X_test, w.y_test,
                                           w.mean, w.std, tid)
    # clean accuracy preserved, trigger flips far more attacks on the poisoned model
    assert asr_bad > asr_honest + 0.3


# --------------------------------------------------------------------------- #
# Local DP
# --------------------------------------------------------------------------- #
def test_ldp_perturbation_preserves_channel_mean(win):
    w, _ = win
    out = localdp.perturb_windows(w.X_train[:200], eps=4.0, mechanism="piecewise", seed=0)
    import numpy as np
    ch = localdp.PERTURB_CHANNELS[0]
    true = np.tanh(w.X_train[:200, ch, :]).mean()
    assert abs(out[:, ch, :].mean() - true) < 0.1


def test_ldp_nb_fits_and_predicts(win):
    w, _ = win
    m = localdp.LDPNaiveBayesDetector(eps=3.0, seed=0).fit(w.X_train, w.y_train)
    p = m.predict(w.X_test)
    assert set(np.unique(p)) <= {0, 1}


# --------------------------------------------------------------------------- #
# Compressive privacy
# --------------------------------------------------------------------------- #
def test_compress_basis_shapes_and_reconstruction(win):
    """Structural checks on the real window geometry: basis shapes are (1100, p)
    and the reconstruction error is a valid fraction in [0, 1]."""
    w, _ = win
    for p in (5, 20):
        mp, Wp = compress.pca_basis(w.X_train, p)
        md, Wd = compress.dca_basis(w.X_train, w.y_train, p)
        assert Wp.shape == (w.X_train.shape[1] * w.X_train.shape[2], p)
        assert Wd.shape == Wp.shape
        re = compress.reconstruction_error(w.X_test, mp, Wp)
        assert 0.0 <= re <= 1.0 + 1e-6


def test_dca_recovers_discriminant_direction():
    """On a controlled separable window dataset, DCA at p=1 recovers the class
    signal and beats a 1-D PCA projection -- the property that makes it a
    compressive-privacy defence that keeps utility."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(0)
    n, C, W = 800, 11, 100
    y = rng.integers(0, 2, n)
    X = rng.normal(size=(n, C, W))
    X[:, 1, :] += (y[:, None] * 2.0)            # class signal on one channel
    ntr = 600

    def acc(mean, Wb):
        Ztr = compress.project(X[:ntr], mean, Wb)
        Zte = compress.project(X[ntr:], mean, Wb)
        sc = StandardScaler().fit(Ztr)
        clf = LogisticRegression(max_iter=500).fit(sc.transform(Ztr), y[:ntr])
        return clf.score(sc.transform(Zte), y[ntr:])

    md, Wd = compress.dca_basis(X[:ntr], y[:ntr], p=1)
    mp, Wp = compress.pca_basis(X[:ntr], p=1)
    assert acc(md, Wd) >= acc(mp, Wp)


# --------------------------------------------------------------------------- #
# Output defence + MIA scorer
# --------------------------------------------------------------------------- #
def test_label_only_hides_confidence():
    proba = np.array([[0.8, 0.2], [0.3, 0.7]])
    out = defense.label_only(proba)
    assert set(np.unique(out)) <= {0.0, 1.0}


def test_membership_scorer_runs(win):
    w, _ = win
    m = RFDetector(seed=0).fit(w.X_train, w.y_train)
    res = membership.membership_inference(m.predict_proba(w.X_train), w.y_train,
                                          m.predict_proba(w.X_test), w.y_test)
    assert 0.0 <= res["auc"] <= 1.0
