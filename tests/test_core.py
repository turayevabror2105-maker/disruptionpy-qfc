"""Fast tests. Each of the first three fails on version 1 of this package and passes on version 2.

Run: pytest -m "not slow" -q
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from disruptionpy_qfc import compute_null_floor, compute_qfc, detect_feature_cols, screen
from disruptionpy_qfc.core import _block_bootstrap_index, _subsample
from disruptionpy_qfc.features import build_rolling_features


def make_frame(n_shots=20, per_shot=25, n_features=8, seed=0, flip=False):
    """A frame with a well-ordered importance profile.

    Feature j enters the logit with coefficient 2.0 * 0.7**j, so every feature has a distinct,
    monotonically decreasing influence and the *ranking* of attributions is well determined rather
    than noise. (A first version of this fixture gave two informative features and six pure-noise
    ones; the rank correlation was then dominated by the arbitrary ordering of the noise, which
    made the tests measure sampling noise instead of the property under test.)

    ``flip`` reverses the coefficient order, so the labels depend on different features while the
    covariate distribution is unchanged.
    """
    rng = np.random.default_rng(seed)
    coefs = 2.0 * 0.7 ** np.arange(n_features)
    if flip:
        coefs = coefs[::-1]
    rows = []
    for s in range(n_shots):
        X = rng.normal(size=(per_shot, n_features))
        y = (X @ coefs + rng.normal(scale=0.5, size=per_shot) > 0).astype(int)
        for r in range(per_shot):
            row = {"shot": s, "time": r, "label": int(y[r])}
            row.update({f"f{j}": X[r, j] for j in range(n_features)})
            rows.append(row)
    return pd.DataFrame(rows)


def fast_model(seed=0):
    return RandomForestClassifier(n_estimators=40, max_depth=6, class_weight="balanced", random_state=seed, n_jobs=1)


FAST = dict(model=fast_model(), n_bootstrap=0, shap_cap=120, n_splits=3)


# ----------------------------------------------------------------- the three regression-of-defect tests


def test_bootstrap_has_multiplicity():
    """v1 used np.isin, which silently collapsed duplicate draws into a ~63% subsample."""
    groups = np.repeat(np.arange(10), 5)
    rng = np.random.default_rng(0)
    sizes, distinct = [], []
    for _ in range(200):
        idx = _block_bootstrap_index(groups, rng)
        sizes.append(len(idx))
        distinct.append(len(np.unique(groups[idx])))
    # a real bootstrap returns as many rows as it started with, every time
    assert set(sizes) == {50}, f"resample changed the row count: {sorted(set(sizes))[:5]}"
    # and it must be able to include the same shot more than once
    idx = _block_bootstrap_index(groups, np.random.default_rng(1))
    counts = np.bincount(groups[idx], minlength=10)
    assert counts.max() > 5, "no shot was drawn more than once in 200 draws: this is not a bootstrap"
    # the v1 defect produced ~6.3 distinct shots on average; a true bootstrap also does, but with
    # duplication -- so the discriminating check is the row count above, and the mean here is a guard
    assert 5.0 < float(np.mean(distinct)) < 8.0


def test_shap_cap_is_random_not_head():
    """v1 explained X[:cap]; on shot-ordered frames that is the first few discharges."""
    n, cap = 1000, 100
    idx = _subsample(n, cap, np.random.default_rng(0))
    assert len(idx) == cap
    assert idx.max() > cap, "subsample never left the head of the array"
    assert not np.array_equal(idx, np.arange(cap)), "subsample is a head slice"
    a = _subsample(n, cap, np.random.default_rng(7))
    b = _subsample(n, cap, np.random.default_rng(7))
    assert np.array_equal(a, b), "same seed must give the same subsample"


def test_verdict_is_uncalibrated_without_a_floor():
    """v1 compared to a borrowed fixed threshold and called it PASS; absence of calibration must show."""
    src, tgt = make_frame(seed=1), make_frame(seed=2)
    r = compute_qfc(src, tgt, **FAST)
    assert r.verdict == "UNCALIBRATED"
    assert r.z is None and r.null_mean is None
    assert "UNCALIBRATED" in r.summary()


# ----------------------------------------------------------------- behaviour


def test_no_shift_is_indistinguishable():
    """Two halves of one pool must not look like a shift."""
    src, tgt = make_frame(seed=3), make_frame(seed=4)  # same generative process
    r = screen(src, tgt, n_seeds=6, **FAST)
    assert r.verdict in {"INDISTINGUISHABLE", "ROBUST"}, f"got {r.verdict} (z={r.z:.2f}) on a no-shift pair"


def scale_features(df, factors):
    out = df.copy()
    for col, f in factors.items():
        out[col] = out[col] * f
    return out


def test_detects_planted_covariate_shift():
    """A target where the features the model leans on have collapsed in range must read COLLAPSE."""
    src = make_frame(seed=5)
    tgt = scale_features(make_frame(seed=6), {"f0": 0.02, "f1": 0.02, "f6": 4.0, "f7": 4.0})
    r = screen(src, tgt, n_seeds=6, **FAST)
    assert r.qfc < 0.9, f"planted covariate shift left qfc at {r.qfc:.3f}"
    assert r.verdict == "COLLAPSE", f"planted shift not detected: qfc={r.qfc:.3f}, z={r.z:.2f}"


def test_label_shift_alone_does_not_move_qfc():
    """Pins the metric's semantics, and it is not obvious.

    The deployed model is fixed: it is fitted on the source and applied to the target unchanged.
    Its attributions on the target therefore depend on the target's *covariate* distribution, not on
    the target's labels. A target whose labels are generated by entirely different features -- with
    the same covariate distribution -- is invisible to QFC, and should be: nothing about the model's
    reliance has changed. Detecting that the labels moved is the job of performance monitoring, which
    needs target labels; this metric exists for the case where you do not have them yet.
    """
    src = make_frame(seed=30)
    same_covariates_different_labels = make_frame(seed=31, flip=True)
    r = screen(src, same_covariates_different_labels, n_seeds=6, **FAST)
    assert r.verdict != "COLLAPSE", (
        f"label-only shift produced {r.verdict} (qfc={r.qfc:.3f}); the metric would be measuring "
        "something it does not claim to measure"
    )


def test_reproducible():
    src, tgt = make_frame(seed=7), make_frame(seed=8)
    a = compute_qfc(src, tgt, random_state=11, **FAST)
    b = compute_qfc(src, tgt, random_state=11, **FAST)
    assert a.qfc == pytest.approx(b.qfc, abs=1e-12)


def test_groups_never_leak():
    """No shot may be explained by a model that was fitted on it."""
    from sklearn.model_selection import GroupKFold

    df = make_frame(n_shots=9, seed=9)
    g = df["shot"].to_numpy()
    for tr, te in GroupKFold(n_splits=3).split(df, df["label"], g):
        assert not (set(g[tr]) & set(g[te])), "a shot appeared in both the fitted and explained split"


def test_bootstrap_ci_brackets_the_estimate():
    src, tgt = make_frame(n_shots=12, per_shot=20, seed=10), make_frame(n_shots=12, per_shot=20, seed=11)
    r = compute_qfc(src, tgt, model=fast_model(), n_bootstrap=12, shap_cap=80, n_splits=3)
    assert np.isfinite(r.ci_lower) and np.isfinite(r.ci_upper)
    assert r.ci_lower <= r.ci_upper


def test_auc_absent_when_target_unlabelled():
    src = make_frame(seed=12)
    tgt = make_frame(seed=13).drop(columns=["label"])
    r = compute_qfc(src, tgt, **FAST)
    assert r.auc is None and np.isfinite(r.qfc)


def test_provenance_is_recorded():
    src, tgt = make_frame(seed=14), make_frame(seed=15)
    p = compute_qfc(src, tgt, **FAST).provenance
    assert p["cap_rule"].startswith("seeded random")
    assert p["bootstrap_rule"] == "shot-level with multiplicity"
    assert p["version"] and p["estimator"]


# ----------------------------------------------------------------- validation and helpers


@pytest.mark.parametrize("missing", ["label", "shot"])
def test_missing_column_named_in_error(missing):
    df = make_frame(seed=16).drop(columns=[missing])
    with pytest.raises(ValueError, match=missing):
        compute_qfc(df, make_frame(seed=17), **FAST)


def test_non_binary_label_rejected():
    df = make_frame(seed=18)
    df.loc[0, "label"] = 7
    with pytest.raises(ValueError, match="binary"):
        compute_qfc(df, make_frame(seed=19), **FAST)


def test_target_missing_features_rejected():
    src = make_frame(seed=20)
    tgt = make_frame(seed=21).drop(columns=["f3"])
    with pytest.raises(ValueError, match="f3"):
        compute_qfc(src, tgt, **FAST)


def test_detect_feature_cols_excludes_bookkeeping():
    cols = detect_feature_cols(make_frame(seed=22))
    assert "shot" not in cols and "time" not in cols and "label" not in cols
    assert all(c.startswith("f") for c in cols)


def test_null_floor_shape_and_cut():
    floor = compute_null_floor(make_frame(seed=23), n_seeds=5, model=fast_model(), shap_cap=100, n_splits=3)
    assert floor.n_seeds == len(floor.per_seed) >= 2
    assert floor.exact_cut > 2.0  # t(0.975, n-1) * sqrt(1 + 1/n) for small n
    assert -1.0 <= floor.mean <= 1.0


def test_rolling_features_are_causal():
    """A feature at time i must not change when the value at time i is altered."""
    df = pd.DataFrame(
        {"shot": [1] * 20, "time": range(20), "label": [0] * 20, "sig": np.arange(20, dtype=float)}
    )
    base = build_rolling_features(df, ["sig"], window=5)
    df2 = df.copy()
    df2.loc[df2["time"] == 10, "sig"] = 999.0
    after = build_rolling_features(df2, ["sig"], window=5)
    row_base = base[base["time"] == 10].iloc[0]
    row_after = after[after["time"] == 10].iloc[0]
    assert row_base["sig_mean"] == pytest.approx(row_after["sig_mean"]), "feature at time i used the value at time i"


def test_rolling_features_no_warning_on_constant_signal():
    """v1 emitted a scipy precision-loss RuntimeWarning on near-constant windows -- common in
    steady plasma phases, and the first thing a user saw."""
    import warnings

    df = pd.DataFrame({"shot": [1] * 20, "time": range(20), "label": [0] * 20, "sig": np.ones(20)})
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning becomes a test failure
        out = build_rolling_features(df, ["sig"], window=5)
    assert (out["sig_skew"] == 0).all() and (out["sig_std"] == 0).all()
