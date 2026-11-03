"""Fast tests of the measurement, the bootstrap, the cap rule and the validation. No dataset needed.

Run: pytest -m "not slow" -q
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from disruptionpy_qfc import compute_null_floor, compute_qfc, detect_feature_cols, screen
from disruptionpy_qfc.core import _block_bootstrap_index, _explain
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


FAST = dict(model=fast_model(), n_fold_orders=1, shap_cap=120, n_splits=3)


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
    seen = []

    def record(model, Z, X_train, seed):
        seen.append(Z[:, 0].copy())
        return np.ones_like(Z)

    X = np.arange(1000, dtype=float)[:, None] * np.ones((1, 3))
    for seed in (0, 7, 7):
        _explain(None, X, X, 100, seed, record)
    assert len(seen[0]) == 100
    assert seen[0].max() > 100, "subsample never left the head of the array"
    assert not np.array_equal(np.sort(seen[0]), np.arange(100)), "subsample is a head slice"
    assert np.array_equal(seen[1], seen[2]), "same seed must give the same subsample"
    assert not np.array_equal(seen[0], seen[1]), "different seeds must give different subsamples"


def test_verdict_is_uncalibrated_without_a_floor():
    """v1 compared to a borrowed fixed threshold and called it PASS; absence of calibration must show."""
    src, tgt = make_frame(seed=1), make_frame(seed=2)
    r = compute_qfc(src, tgt, **FAST)
    assert r.verdict == "UNCALIBRATED"
    assert r.z is None and r.null_mean is None
    assert "UNCALIBRATED" in r.summary()


# ----------------------------------------------------------------- behaviour


def test_no_shift_does_not_collapse():
    """Two samples of one generative process must not read as a collapse."""
    src, tgt = make_frame(seed=3), make_frame(seed=4)  # same generative process
    r = screen(src, tgt, n_seeds=6, **FAST)
    assert r.verdict != "COLLAPSE", f"got {r.verdict} (z={r.z:.2f}) on a no-shift pair"


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
    same, flipped = make_frame(seed=31), make_frame(seed=31, flip=True)   # identical covariates, other labels
    assert (same.drop(columns="label").values == flipped.drop(columns="label").values).all()
    assert (same["label"] != flipped["label"]).any()
    a, b = compute_qfc(src, same, **FAST), compute_qfc(src, flipped, **FAST)
    assert a.qfc == b.qfc and a.target_importance == b.target_importance, (
        "a label-only change moved QFC; the metric would be measuring something it does not claim to measure")
    assert a.auc != b.auc   # the labels did change, and only the AUC sees it


def test_reproducible():
    src, tgt = make_frame(seed=7), make_frame(seed=8)
    a = compute_qfc(src, tgt, random_state=11, **FAST)
    b = compute_qfc(src, tgt, random_state=11, **FAST)
    assert a.qfc == pytest.approx(b.qfc, abs=1e-12)


def test_no_source_shot_is_explained_by_a_model_that_trained_on_it():
    """Recorded through the attribution hook: in every attribution call -- out of fold, deployed on the target, and
    in every null draw -- the explained rows and the training rows belong to disjoint shots."""
    calls = []

    def record(model, Z, X_train, seed):
        calls.append((set(Z[:, 0].astype(int).tolist()), set(X_train[:, 0].astype(int).tolist())))
        return np.abs(Z[:, 1:2] - Z[:, 1:2].mean()) * np.ones_like(Z) + np.arange(Z.shape[1])

    df = make_frame(n_shots=12, seed=9).assign(shotcode=lambda d: d["shot"])
    df = df[["shotcode"] + [c for c in df.columns if c != "shotcode"]]   # column 0 carries the shot, as a feature
    tgt = make_frame(n_shots=6, seed=10).assign(shotcode=lambda d: d["shot"] + 1000)
    tgt = tgt[df.columns]
    screen(df, tgt, attribution=record, n_seeds=4, **dict(FAST, n_fold_orders=3))
    assert len(calls) >= 4 * (3 + 1) + 3 + 1   # 4 null draws x (3 folds + 1), then at least one order's 3 folds + target
    assert all(not (z & t) for z, t in calls), "a shot was explained by a model fitted on it"


def test_bootstrap_ci_brackets_the_estimate():
    src, tgt = make_frame(n_shots=12, per_shot=20, seed=10), make_frame(n_shots=12, per_shot=20, seed=11)
    r = compute_qfc(src, tgt, model=fast_model(), n_bootstrap=12, shap_cap=80, n_splits=3, n_fold_orders=1)
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
    assert p["bootstrap_rule"].startswith("shot-level with multiplicity")
    assert p["attribution"] == "treeshap" and p["n_fold_orders"] == 1
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
    # DisruptionPy's time-to-event column is where a label comes from; as a feature it would leak the label
    df = make_frame(seed=22).assign(time_until_disrupt=1.0, time_domain=2)
    assert not {"time_until_disrupt", "time_domain"} & set(detect_feature_cols(df))


def test_null_floor_shape_and_cut():
    floor = compute_null_floor(make_frame(seed=23), n_seeds=5, model=fast_model(), shap_cap=100, n_splits=3)
    assert floor.n_seeds == len(floor.per_seed) >= 3
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


def test_skew_is_undefined_exactly_where_the_spread_is_at_floating_point_resolution():
    """The paper's builder calls scipy.stats.skew, which returns NaN on such windows; so does this one, by the
    same rule fixed in code (second central moment <= (eps * mean)**2). Elsewhere the value is scipy's own."""
    from scipy import stats
    from disruptionpy_qfc.features import _skew, _window_stats

    v = np.full(10, 3.0)
    v[3] = np.nextafter(3.0, 4.0)                      # one ULP of spread
    assert np.isnan(_skew(v)) and np.isnan(_window_stats(v)[7])
    rng = np.random.default_rng(0)
    for _ in range(200):
        w = rng.normal(size=10) * rng.uniform(1e-6, 10) + rng.uniform(-5, 5)
        assert _skew(w) == float(stats.skew(w))
        assert _window_stats(w)[4] == float(np.polyfit(np.arange(10), w, 1)[0])
    assert _window_stats(np.ones(10))[7] == 0.0        # no spread at all: 0, as in the paper
    df = pd.DataFrame({"shot": [1] * 20, "time": range(20), "label": [0, 1] * 10,
                       "sig": np.r_[np.full(12, 3.0) + np.r_[0, 0, 0, np.nextafter(3.0, 4.0) - 3.0, [0] * 8],
                                    np.arange(8.0)]})
    out = build_rolling_features(df, ["sig"], window=5)
    assert out["sig_skew"].isna().any() and out["sig_mean"].notna().all()

def test_constant_null_with_float_residue_is_undefined():
    """Thirty identical draws can leave sd ~1e-16 (the mean is not exact); that is no spread, not a tiny one."""
    from disruptionpy_qfc.core import NullFloor
    draws = [0.9] * 30
    sd = float(np.std(draws, ddof=1))
    assert sd > 0  # the float residue the guard must not trust
    floor = NullFloor(mean=float(np.mean(draws)), sd=sd, per_seed=draws, n_seeds=30, exact_cut=2.0790374)
    z, v = floor.verdict_for(0.5)
    assert v == "UNDEFINED" and z != z


def test_the_default_runs_ten_fold_orders_and_reports_each():
    src, tgt = make_frame(seed=40), make_frame(seed=41)
    r = compute_qfc(src, tgt, **dict(FAST, n_fold_orders=10))
    assert len(r.qfc_per_order) == 10 and r.qfc == float(np.mean(r.qfc_per_order))
    import inspect
    assert inspect.signature(compute_qfc).parameters["n_fold_orders"].default == 10


def test_the_default_model_is_the_papers_real_arm():
    from disruptionpy_qfc import default_model
    m = default_model()
    assert (m.n_estimators, m.max_depth, m.class_weight) == (500, None, "balanced")
    assert default_model(max_depth=20).max_depth == 20


def test_retired_fixed_threshold_is_gone():
    """Version 1 read QFC against a fixed 0.391 borrowed from another domain and version 2 kept it as an option.
    It is not a verdict, and version 3 does not offer it."""
    import inspect
    assert "threshold" not in inspect.signature(compute_qfc).parameters
    src, tgt = make_frame(seed=42), make_frame(seed=43)
    assert not any("threshold" in k for k in compute_qfc(src, tgt, **FAST).to_dict())


def test_missing_feature_values_take_the_source_median_and_labels_are_refused():
    src, tgt = make_frame(seed=44), make_frame(seed=45)
    src.loc[3, "f2"], src.loc[7, "f2"], tgt.loc[5, "f3"] = np.nan, np.nan, np.nan
    r = compute_qfc(src, tgt, **FAST)
    assert r.provenance["imputed_with_source_median"] == {"source": {"f2": 2}, "target": {"f3": 1}}
    assert "imputed    = 2 source and 1 target" in r.summary()
    filled_s, filled_t = src.copy(), tgt.copy()
    filled_s["f2"] = filled_s["f2"].fillna(src["f2"].median())
    filled_t["f3"] = filled_t["f3"].fillna(src["f3"].median())        # the SOURCE median, for the target too
    assert compute_qfc(filled_s, filled_t, **FAST).qfc == r.qfc
    floor = compute_null_floor(src, n_seeds=4, model=fast_model(), shap_cap=100, n_splits=3)
    assert floor.provenance["imputed_with_source_median"] == {"source": {"f2": 2}}
    with pytest.raises(ValueError, match="infinite"):
        compute_qfc(src.assign(f1=np.inf), tgt, **FAST)
    src, tgt = make_frame(seed=44), make_frame(seed=45)
    src["label"] = src["label"].astype(float)
    src.loc[4, "label"] = np.nan
    with pytest.raises(ValueError, match="missing"):
        compute_qfc(src, tgt, **FAST)


def test_a_partly_labelled_target_is_accepted_and_its_auc_uses_the_labelled_rows():
    src, tgt = make_frame(seed=46), make_frame(seed=47)
    part = tgt.copy()
    part["label"] = part["label"].astype(float)
    part.loc[part.index[::3], "label"] = np.nan
    full, partial = compute_qfc(src, tgt, **FAST), compute_qfc(src, part, **FAST)
    assert partial.qfc == full.qfc and partial.auc is not None and partial.auc != full.auc


def test_to_dict_is_strict_json_even_with_undefined_values():
    import json
    from disruptionpy_qfc.core import NullFloor
    src, tgt = make_frame(seed=48), make_frame(seed=49)
    r = compute_qfc(src, tgt, **FAST)
    json.dumps(r.to_dict(), allow_nan=False)                       # no CI requested: None, not NaN
    floor = NullFloor(mean=0.9, sd=float(np.std([0.9] * 30, ddof=1)), per_seed=[0.9] * 30, n_seeds=30,
                      exact_cut=2.0790374, cut_effective=float("inf"))
    r = compute_qfc(src, tgt, null_floor=floor, **FAST)
    assert r.verdict == "UNDEFINED" and r.z is None
    json.dumps(r.to_dict(), allow_nan=False)


def test_fold_orders_with_identical_folds_are_computed_once():
    """Shots of distinct sizes give the same folds at every order; the vector is reused, the value is the same."""
    rows = []
    rng = np.random.default_rng(50)
    for s in range(12):
        for t in range(10 + s):                                    # every shot a different length
            x = rng.normal(size=4)
            rows.append({"shot": s, "time": t, "label": int(x[0] + rng.normal() > 0), **{f"f{j}": x[j] for j in range(4)}})
    src = pd.DataFrame(rows)
    r = compute_qfc(src, make_frame(n_features=4, seed=51), **dict(FAST, n_fold_orders=10))
    assert r.provenance["n_distinct_fold_orders"] == 1 and len(set(r.qfc_per_order)) == 1


def test_the_result_carries_both_floors_and_the_summary_shows_the_resolution():
    src, tgt = make_frame(seed=52), make_frame(seed=53)
    r = screen(src, tgt, n_seeds=5, chance_floor=True, n_chance_draws=3, **FAST)
    assert len(r.null_floor["per_seed"]) == r.n_null_seeds and r.null_floor["provenance"]["attribution"] == "treeshap"
    assert len(r.chance_floor["per_draw"]) == 3
    assert ("resolution = %.4f" % r.resolution) in r.summary()


@pytest.mark.parametrize("bad", [dict(shap_cap=0), dict(n_splits=1)])
def test_bad_settings_are_refused_with_their_name(bad):
    src, tgt = make_frame(seed=54), make_frame(seed=55)
    with pytest.raises(ValueError, match=list(bad)[0]):
        compute_qfc(src, tgt, **dict(FAST, **bad))


def test_a_non_dataframe_target_gets_the_friendly_error():
    with pytest.raises(TypeError, match="target_df must be a pandas DataFrame"):
        compute_qfc(make_frame(seed=56), [[1, 2, 3]], **FAST)
