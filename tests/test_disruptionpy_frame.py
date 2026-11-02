"""Integration: a frame that DisruptionPy actually produced goes through this package.

These four tests ask one question: given a frame that came out of DisruptionPy, does this package accept
it and screen it? They assert nothing about DisruptionPy's own columns, which are its business and change
between versions; they check this package's side of the contract.

Where the frame comes from:

* in CI, ``tests/fixtures/build_mast_frame.py`` builds one from MAST's open data with DisruptionPy itself
  (``.github/workflows/ci.yml``, job ``integration``), and ``DPQ_REQUIRE_FRAME=1`` makes a missing frame
  a failure, so these tests cannot silently skip there;
* anywhere else, point ``DISRUPTIONPY_FRAME`` at a CSV or parquet you exported, e.g.
  ``get_shots_data(...).to_csv("frame.csv", index=False)``; without it they skip, and a skip means
  nothing was checked.

A DisruptionPy frame has no binary label; it has ``time_until_disrupt``. Set ``DPQ_TUD_COL`` (and
``DPQ_TUD_THRESHOLD``, default 0.050 s) and the tests derive the label exactly as the README tells a
user to, then drop the time-to-event column so it cannot leak into the features. With no label column
and no ``DPQ_TUD_COL`` they fail with that instruction rather than choosing a precursor window for you.
Rows with a missing value in a feature are dropped first, after any all-missing column, which is the
README's preparation step too. The model is a small forest so that the tests run in minutes on a CI
machine; the procedure is otherwise the package's own.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from disruptionpy_qfc import compute_null_floor, compute_qfc, detect_feature_cols, screen

FRAME = os.environ.get("DISRUPTIONPY_FRAME")
SHOT_COL = os.environ.get("DPQ_SHOT_COL", "shot")
LABEL_COL = os.environ.get("DPQ_LABEL_COL", "label")
TUD_COL = os.environ.get("DPQ_TUD_COL")
TUD_THRESHOLD = float(os.environ.get("DPQ_TUD_THRESHOLD", "0.050"))
REQUIRE = os.environ.get("DPQ_REQUIRE_FRAME") == "1"
HAVE = bool(FRAME and Path(FRAME).exists())

if REQUIRE and not HAVE:
    raise RuntimeError(f"DPQ_REQUIRE_FRAME=1 but DISRUPTIONPY_FRAME={FRAME!r} is not a file: the integration "
                       "tests must run here, not skip")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not HAVE, reason="set DISRUPTIONPY_FRAME to a frame exported from DisruptionPy. Skipped, "
                                        "not passed: nothing about the integration has been checked."),
]

KW = dict(model=RandomForestClassifier(n_estimators=60, max_depth=8, class_weight="balanced", n_jobs=-1),
          shap_cap=200, n_fold_orders=2)


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    path = Path(FRAME)
    df = pd.read_parquet(path) if path.suffix in (".parquet", ".pq") else pd.read_csv(path)
    assert len(df) > 0, f"{path} is empty"
    return df


def _labelled(df: pd.DataFrame) -> pd.DataFrame:
    """The README's preparation: derive the label under the caller's convention, drop the time-to-event column,
    drop all-missing columns and then rows with a missing feature."""
    out = df.copy()
    if LABEL_COL not in out.columns:
        if not TUD_COL:
            pytest.fail(
                f"the frame has no {LABEL_COL!r} column and DPQ_TUD_COL is unset. This package needs a binary "
                "source label and will not choose the precursor window for you: how long before a disruption "
                "counts as positive changes what the model learns and so what its attributions mean. Set "
                f"DPQ_TUD_COL (and DPQ_TUD_THRESHOLD, default 0.050) or add the column. Columns: {list(df.columns)[:15]}"
            )
        assert TUD_COL in out.columns, f"DPQ_TUD_COL={TUD_COL!r} is not a column of the frame"
        out[LABEL_COL] = (out[TUD_COL] < TUD_THRESHOLD).astype(int)
        out = out.drop(columns=[TUD_COL])
    out = out.dropna(axis=1, how="all")
    feats = detect_feature_cols(out, LABEL_COL, SHOT_COL)
    return out.dropna(subset=feats + [LABEL_COL])


def _halves(df):
    shots = np.array(sorted(df[SHOT_COL].unique()))
    half = len(shots) // 2
    return df[df[SHOT_COL].isin(shots[:half])].copy(), df[df[SHOT_COL].isin(shots[half:])].copy()


def test_the_frame_meets_this_package_s_contract(frame):
    """Three requirements, each named separately so a failure says which one is unmet."""
    df = _labelled(frame)
    assert SHOT_COL in df.columns, f"no shot column {SHOT_COL!r}; set DPQ_SHOT_COL. Columns: {list(df.columns)[:15]}"
    labels = set(pd.unique(df[LABEL_COL].dropna()).tolist())
    assert labels <= {0, 1, 0.0, 1.0, True, False}, f"label column is not binary: {sorted(labels)[:8]}"
    assert len(labels) == 2, f"the frame is single-class after labelling ({labels}); widen the window"
    feats = detect_feature_cols(df, LABEL_COL, SHOT_COL)
    assert len(feats) >= 3, f"only {len(feats)} usable feature columns: {feats}"
    assert not {"time_until_disrupt", "time_domain"} & set(feats), "a DisruptionPy bookkeeping column became a feature"
    if TUD_COL:   # the time-to-event column the label came from must be gone, or it would leak the label
        assert TUD_COL in frame.columns and TUD_COL not in df.columns and TUD_COL not in feats
    assert df[SHOT_COL].nunique() >= 20, (
        f"only {df[SHOT_COL].nunique()} shots; the tests split the frame in half and the null floor halves each "
        "half again, and every quarter needs five shots for a five-fold split")


def test_a_split_of_one_real_frame_screens_end_to_end(frame):
    """A real DisruptionPy frame, split into its lower-numbered and higher-numbered shots, goes through screen()
    and comes back with a verdict. Shot numbers run in time, so the split can carry a real temporal shift; the
    test asserts that a verdict was reached, not which one."""
    a, b = _halves(_labelled(frame))
    assert not set(a[SHOT_COL]) & set(b[SHOT_COL]), "the split leaked a shot into both sides"
    r = screen(a, b, label_col=LABEL_COL, shot_col=SHOT_COL, n_seeds=5, **KW)
    assert r.verdict in {"COLLAPSE", "INDISTINGUISHABLE", "ROBUST"}
    assert r.exact_cut is not None and np.isfinite(r.z) and -1.0 <= r.qfc <= 1.0
    print(f"\n[DisruptionPy frame] {len(a) + len(b)} rows / {a[SHOT_COL].nunique() + b[SHOT_COL].nunique()} shots\n"
          f"{r.summary()}")


def test_an_uncalibrated_measurement_says_so_on_a_real_frame(frame):
    """A raw QFC with no floor comes back UNCALIBRATED rather than as a number a reader might interpret."""
    a, b = _halves(_labelled(frame))
    r = compute_qfc(a, b, label_col=LABEL_COL, shot_col=SHOT_COL, **KW)
    assert r.verdict == "UNCALIBRATED" and r.z is None and r.null_mean is None


def test_the_floor_is_per_pool_and_records_its_provenance(frame):
    """A floor belongs to one pool, model and protocol, and says so."""
    a, _ = _halves(_labelled(frame))
    kw = {k: v for k, v in KW.items() if k != "n_fold_orders"}
    floor = compute_null_floor(a, n_seeds=5, label_col=LABEL_COL, shot_col=SHOT_COL, **kw)
    assert floor.n_seeds_requested == 5 and floor.n_seeds >= 3 and floor.sd >= 0.0
    assert floor.provenance["construction"].startswith("GroupShuffleSplit 50/50")
    assert floor.provenance["estimator"]
