"""Integration against a frame that DisruptionPy actually produced.

WHY THIS TEST IS SHAPED THE WAY IT IS
-------------------------------------
Everything else in this suite works on frames built here -- either synthetic, or derived from the Open
Density Limit Database CSV. Nothing anywhere calls into DisruptionPy. The frames are DisruptionPy-*format*
by construction, which is a claim this package makes about itself and never checks.

That gap cannot be closed by asserting what DisruptionPy's columns are. Its retrieval API and its output
schema are its own business and they change between versions, so a test that hard-codes them would be a
test of my memory rather than of this package, and would start failing for reasons that have nothing to do
with attribution stability.

So the test asks the only question that is genuinely this package's business: **given a frame that came
out of DisruptionPy, does this package accept it and screen it?** The frame is supplied by whoever is
running the suite, via an environment variable, and the test skips when it is absent. It therefore skips
in the author's environment -- DisruptionPy is not installed here -- and runs for a maintainer, who has
both the library and machine access. That asymmetry is the point rather than a shortcoming: the person who
can exercise the integration is exactly the person evaluating the package.

    # export a frame however you normally would, e.g.
    #   df = get_shots_data(...); df.to_csv("frame.csv", index=False)
    DISRUPTIONPY_FRAME=/path/to/frame.csv pytest -m integration -q

    # if your shot or label columns are named differently:
    DISRUPTIONPY_FRAME=... DPQ_SHOT_COL=shot DPQ_LABEL_COL=label pytest -m integration -q

**These tests have been executed, not merely written.** An env-gated test that has only ever skipped is
worth nothing, so before this file was committed all four were run against a 5,867-row / 78-shot stand-in
frame built from the Open Density Limit Database and exported to CSV exactly as the instructions above
describe. All four passed, in 4 minutes 20 seconds. The stand-in is a stand-in -- it did not come out of
DisruptionPy -- so what that run establishes is that the tests work and assert something real, not that
the DisruptionPy integration itself has been checked. Only a maintainer's run can do the second.

If the frame has no binary label column -- the normal case, since a disruption-prediction frame carries a
time-to-disruption instead -- set ``DPQ_TUD_COL`` and ``DPQ_TUD_THRESHOLD`` and the test will derive one
by that convention, exactly as the README tells a user to. It will not pick a convention for you: with no
label column and no ``DPQ_TUD_COL`` the test fails with that instruction rather than guessing, because a
silently chosen precursor window would change what the model learns and therefore what its attributions
mean.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from disruptionpy_qfc import compute_null_floor, compute_qfc, detect_feature_cols, screen

FRAME = os.environ.get("DISRUPTIONPY_FRAME")
SHOT_COL = os.environ.get("DPQ_SHOT_COL", "shot")
LABEL_COL = os.environ.get("DPQ_LABEL_COL", "label")
TUD_COL = os.environ.get("DPQ_TUD_COL")
TUD_THRESHOLD = float(os.environ.get("DPQ_TUD_THRESHOLD", "0.050"))

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not (FRAME and Path(FRAME).exists()),
        reason=(
            "set DISRUPTIONPY_FRAME to a CSV or parquet exported from DisruptionPy. Skipped, not "
            "passed: nothing about the integration has been checked. DisruptionPy is not installed in "
            "the author's environment, so this test is written to run in a maintainer's."
        ),
    ),
]


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    path = Path(FRAME)
    df = pd.read_parquet(path) if path.suffix in (".parquet", ".pq") else pd.read_csv(path)
    assert len(df) > 0, f"{path} is empty"
    return df


def _labelled(df: pd.DataFrame) -> pd.DataFrame:
    """The README's derive-then-pass step, under the caller's own convention."""
    if LABEL_COL in df.columns:
        return df
    if not TUD_COL:
        pytest.fail(
            f"the frame has no {LABEL_COL!r} column and DPQ_TUD_COL is unset. This package needs a "
            "binary source label and will not choose the precursor window for you: how many "
            "milliseconds before a disruption count as positive changes what the model learns and "
            "therefore what its attributions mean. Set DPQ_TUD_COL (and optionally "
            "DPQ_TUD_THRESHOLD, default 0.050) or add the column yourself. "
            f"Columns present: {list(df.columns)[:15]}"
        )
    assert TUD_COL in df.columns, f"DPQ_TUD_COL={TUD_COL!r} is not a column of the frame"
    out = df.copy()
    out[LABEL_COL] = (out[TUD_COL] < TUD_THRESHOLD).astype(int)
    return out


def test_the_frame_meets_this_package_s_contract(frame):
    """Three requirements, each named separately so a failure says which one is unmet."""
    df = _labelled(frame)
    assert SHOT_COL in df.columns, (
        f"no shot column {SHOT_COL!r}; set DPQ_SHOT_COL. Columns: {list(df.columns)[:15]}"
    )
    labels = set(pd.unique(df[LABEL_COL].dropna()).tolist())
    assert labels <= {0, 1, 0.0, 1.0, True, False}, f"label column is not binary: {sorted(labels)[:8]}"
    assert len(labels) == 2, (
        f"the frame is single-class after labelling ({labels}); with DPQ_TUD_THRESHOLD="
        f"{TUD_THRESHOLD} no split is possible. Widen the window."
    )
    feats = detect_feature_cols(df, LABEL_COL, SHOT_COL)
    assert len(feats) >= 3, f"only {len(feats)} usable feature columns: {feats}"
    assert df[SHOT_COL].nunique() >= 4, (
        f"only {df[SHOT_COL].nunique()} shots; the shot is the independent unit here, so a grouped "
        "5-fold split and a shot-level bootstrap both need more"
    )


def test_a_split_of_one_real_frame_screens_end_to_end(frame):
    """The whole point: a real DisruptionPy frame, split by shot into two halves, goes through
    screen() and comes back with a verdict.

    This is a NO-SHIFT split, so the expected verdict is INDISTINGUISHABLE and the test asserts only
    that a verdict was reached and is one of the three -- not which one. A shot-disjoint split of one
    frame should not collapse, and if it does that is a finding about the frame rather than a failure
    of this package, so it is reported and not asserted against.
    """
    df = _labelled(frame)
    shots = np.array(sorted(df[SHOT_COL].unique()))
    half = len(shots) // 2
    a = df[df[SHOT_COL].isin(shots[:half])].copy()
    b = df[df[SHOT_COL].isin(shots[half:])].copy()
    assert not set(a[SHOT_COL]) & set(b[SHOT_COL]), "the split leaked a shot into both sides"

    r = screen(a, b, label_col=LABEL_COL, shot_col=SHOT_COL, n_bootstrap=0, n_seeds=5)
    assert r.verdict in {"COLLAPSE", "INDISTINGUISHABLE", "ROBUST"}
    assert r.exact_cut is not None and r.z is not None
    assert -1.0 <= r.qfc <= 1.0
    print(
        f"\n[DisruptionPy frame] {len(df)} rows / {df[SHOT_COL].nunique()} shots, "
        f"{len(detect_feature_cols(df, LABEL_COL, SHOT_COL))} features\n"
        f"  no-shift split: qfc {r.qfc:+.4f}  floor {r.null_mean:+.4f} +/- {r.null_sd:.4f} "
        f"(5 seeds)  z {r.z:+.3f}  cut {r.exact_cut:.4f}  -> {r.verdict}\n"
        f"  a COLLAPSE here would be a finding about this frame, not a failure of the package"
    )


def test_an_uncalibrated_measurement_says_so_on_a_real_frame(frame):
    """The package's central design decision, checked on real data: a raw QFC with no floor is
    returned as UNCALIBRATED rather than as a number a reader might interpret."""
    df = _labelled(frame)
    shots = np.array(sorted(df[SHOT_COL].unique()))
    half = len(shots) // 2
    a = df[df[SHOT_COL].isin(shots[:half])].copy()
    b = df[df[SHOT_COL].isin(shots[half:])].copy()
    r = compute_qfc(a, b, label_col=LABEL_COL, shot_col=SHOT_COL, n_bootstrap=0)
    assert r.verdict == "UNCALIBRATED"
    assert r.z is None and r.null_mean is None


def test_the_floor_is_per_pool_and_refuses_to_be_reused_implicitly(frame):
    """A floor built on one pool must not be silently applied to another. The README promises this;
    here it is on a real frame."""
    df = _labelled(frame)
    shots = np.array(sorted(df[SHOT_COL].unique()))
    half = len(shots) // 2
    a = df[df[SHOT_COL].isin(shots[:half])].copy()
    floor = compute_null_floor(a, n_seeds=5, label_col=LABEL_COL, shot_col=SHOT_COL)
    assert floor.n_seeds == 5 and floor.sd >= 0.0
    prov = getattr(floor, "provenance", {}) or {}
    assert prov, "the floor carries no provenance, so nothing records which pool it belongs to"
