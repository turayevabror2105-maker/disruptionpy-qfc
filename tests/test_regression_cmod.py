"""Regression tests against the real dataset. Marked ``slow``; skipped when the data is absent.

Run: ``pytest -m slow -q`` with ``DL_DATAFRAME=/path/to/DL_DataFrame.csv`` set (or the default path
below present). These take minutes of CPU: 500-tree forests plus TreeSHAP.

Dataset: Maris, A. D., Rea, C., Trevisan, G. L., & the Alcator C-Mod Team, *The Open Density Limit
Database*, MIT Plasma Science and Fusion Center (2025). CC BY.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from disruptionpy_qfc.demo import build_demo_frames
from disruptionpy_qfc import compute_null_floor, compute_qfc

DEFAULT_PATH = Path(r"C:\Users\user\Datasets\cmod-density-limit\data\DL_DataFrame.csv")
DATA = Path(os.environ.get("DL_DATAFRAME", DEFAULT_PATH))

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not DATA.exists(), reason=f"C-Mod release not found at {DATA}"),
]

# Recorded by version 1 of this package on the same data and the same estimator
# (outputs/disruptionpy_qfc_test.log): QFC 0.9426, target AUC 0.9375.
V1_QFC = 0.9426
V1_AUC = 0.9375


@pytest.fixture(scope="module")
def frames():
    return build_demo_frames(DATA)


def test_frames_match_the_published_shape(frames):
    """The row and shot counts the paper and the v1 log both report."""
    source, target = frames
    assert (len(source), source["shot"].nunique()) == (3363, 40)
    assert (len(target), target["shot"].nunique()) == (2504, 38)
    assert len([c for c in source.columns if c not in ("shot", "time", "label")]) == 48


def test_label_prior_differs_between_environments(frames):
    """Documented, not incidental: this comparison carries a label-prior shift as well as a
    covariate shift, and anything read from it must say so."""
    source, target = frames
    assert source["label"].mean() == pytest.approx(0.5085, abs=0.001)
    assert target["label"].mean() == pytest.approx(0.7208, abs=0.001)


def test_auc_is_invariant_across_package_versions(frames):
    """A true cross-version check: the deployed model is unchanged by the v2 fixes, so its AUC must
    reproduce v1's exactly. The attribution cap changed, so QFC may move -- AUC may not."""
    source, target = frames
    r = compute_qfc(source, target, n_bootstrap=0)
    assert r.auc == pytest.approx(V1_AUC, abs=0.002)


def test_cmod_era_shift_collapses_against_its_own_floor(frames):
    """The scientific content, and the reason this package exists.

    Across the 2000-2003 -> 2005-2009 era shift on Alcator C-Mod the deployed model still predicts
    well -- AUC 0.938 -- and its attribution agreement with the source is 0.961, a number that reads as
    excellent on its own. It is not. A no-shift split of the source pool alone scores 0.980 +/- 0.007,
    so the observed value sits 2.69 standard deviations BELOW the floor: the agreement is worse than
    two halves of one campaign give each other. Prediction survives; the explanation does not.

    HOW THIS TEST GOT ITS EXPECTATION WRONG, recorded so it is not repeated. It previously asserted
    INDISTINGUISHABLE, and the demo docstring said in prose that "the no-shift floor is just as high".
    Neither was a measurement. Version 1 of this package had no floor at all -- only a fixed threshold
    of 0.391 borrowed from an unrelated domain, which this condition passed comfortably -- so the
    expectation was an assumption written by its author. Measured at 30 seeds on 2026-09-16 under a
    pre-registration that fixed both outcomes in advance, the verdict is COLLAPSE, z = -2.69 against the
    exact cut of 2.079. Full record: D:/experiments/pipeline/module_f1_verification.json.

    The band below is pinned to that measurement. If it ever fails, something real changed.
    """
    source, target = frames
    floor = compute_null_floor(source, n_seeds=30)
    r = compute_qfc(source, target, null_floor=floor, n_bootstrap=0)
    assert floor.mean == pytest.approx(0.9801, abs=0.01), f"floor moved: {floor.mean:.4f}"
    assert r.qfc == pytest.approx(0.9607, abs=0.01), f"QFC moved: {r.qfc:.4f}"
    assert r.qfc < floor.mean, "the observed value is no longer below its own no-shift floor"
    assert r.verdict == "COLLAPSE", f"got {r.verdict} (z={r.z:.2f}) -- investigate before publishing"
