"""The worked example, recomputed from the C-Mod release and compared with its record. Marked ``slow``.

Run ``pytest -m slow`` with ``DL_DATAFRAME=/path/to/DL_DataFrame.csv``; skipped when the file is absent.
About ten minutes on a 32-core machine: a 30-draw null floor and ten fold orders of 500-tree forests with TreeSHAP.

``docs/worked_example.json`` is the record of the numbers the README states; it is written by
``scripts/regenerate_worked_example.py``. These tests rebuild the ``default`` part of it (the other parts are
rebuilt only by that script) and require the same verdict, and the same numbers exactly on the platform, library
versions and code recorded with it. Elsewhere they allow platform noise: ``GroupKFold`` breaks ties between
equal-sized shots by numpy's sort, whose order differs between platform builds, so values can differ slightly
between machines; the tolerance below is under the floor's resolution of 0.015.

Dataset: Maris, A. D., Rea, C., Trevisan, G. L., & the Alcator C-Mod Team, *The Open Density Limit
Database*, MIT Plasma Science and Fusion Center (2025). CC BY.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from disruptionpy_qfc import compute_null_floor, compute_qfc, default_model
from disruptionpy_qfc.demo import build_demo_frames

DATA = Path(os.environ.get("DL_DATAFRAME", "DL_DataFrame.csv"))
RECORD = Path(__file__).resolve().parents[1] / "docs" / "worked_example.json"
TOL = 0.004

pytestmark = [pytest.mark.slow,
              pytest.mark.skipif(not DATA.exists(), reason=f"C-Mod release not found at {DATA}; set DL_DATAFRAME")]


@pytest.fixture(scope="module")
def frames():
    return build_demo_frames(DATA)


@pytest.fixture(scope="module")
def record():
    return json.loads(RECORD.read_text(encoding="utf-8"))


def test_frames_match_the_recorded_shape(frames, record):
    source, target = frames
    f = record["default"]["frames"]
    assert (len(source), source["shot"].nunique(), len(target), target["shot"].nunique()) == (
        f["source_rows"], f["source_shots"], f["target_rows"], f["target_shots"]) == (3363, 40, 2504, 38)
    assert len([c for c in source.columns if c not in ("shot", "time", "label")]) == f["features"] == 48


def test_label_prior_differs_between_environments(frames):
    """Documented, not incidental: this comparison carries a label-prior shift as well as a covariate shift."""
    source, target = frames
    assert source["label"].mean() == pytest.approx(0.5085, abs=0.001)
    assert target["label"].mean() == pytest.approx(0.7208, abs=0.001)


def test_auc_reproduces_version_1_with_version_1_s_model(frames):
    """A cross-version check: version 1 (2026-08-13) reported a target AUC of 0.9375 with a 500-tree, depth-20
    forest. The deployed model is fitted the same way here, so the AUC must reproduce."""
    source, target = frames
    r = compute_qfc(source, target, model=default_model(max_depth=20), n_fold_orders=1)
    assert r.auc == pytest.approx(0.9375, abs=0.002)


def test_the_worked_example_reproduces_its_record(frames, record):
    source, target = frames
    floor = compute_null_floor(source)
    r = compute_qfc(source, target, null_floor=floor)
    rec, rf = record["default"]["result"], record["default"]["null_floor"]
    print("\nmeasured: qfc %r floor %r +/- %r z %r auc %r per-order %r" % (r.qfc, floor.mean, floor.sd, r.z, r.auc,
                                                                          r.qfc_per_order))
    assert r.verdict == rec["verdict"] == "COLLAPSE", f"got {r.verdict} (z = {r.z:.2f}); investigate before publishing"
    assert r.n_distinct_splits == rec["n_distinct_splits"] and floor.n_seeds == rf["n_seeds"]
    if _same_environment(record["default"]["env"]):
        # the machine, library versions and code that wrote the record: TreeSHAP is deterministic, so exactly
        assert (r.qfc, r.qfc_per_order, floor.mean, floor.sd, r.z, r.auc) == (
            rec["qfc"], rec["qfc_per_order"], rf["mean"], rf["sd"], rec["z"], rec["auc"])
    else:
        # another platform build of numpy orders equal-sized shots differently; well under the floor's resolution
        assert floor.mean == pytest.approx(rf["mean"], abs=TOL) and floor.sd == pytest.approx(rf["sd"], abs=TOL)
        assert r.qfc == pytest.approx(rec["qfc"], abs=TOL) and r.auc == pytest.approx(rec["auc"], abs=TOL)
        assert r.z == pytest.approx(rec["z"], abs=0.5)


def _same_environment(env: dict) -> bool:
    import hashlib
    import platform

    import numpy, pandas, scipy, shap, sklearn

    src = hashlib.sha256()
    for f in sorted((Path(__file__).resolve().parents[1] / "src" / "disruptionpy_qfc").glob("*.py")):
        src.update(f.name.encode() + f.read_bytes().replace(b"\r\n", b"\n"))
    here = {"platform": platform.platform(), "numpy": numpy.__version__, "pandas": pandas.__version__,
            "scipy": scipy.__version__, "scikit-learn": sklearn.__version__, "shap": shap.__version__,
            "src_sha256": src.hexdigest()}
    return all(env.get(k) == v for k, v in here.items())
