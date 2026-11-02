"""The package checked number for number against the paper's reference implementation, on the same inputs.

``tests/reference/qfc_metric.py`` is a verbatim copy (LF line endings) of the paper's reference implementation
(``qfc_reference/qfc_metric.py`` in the paper's deposit). Every test below runs the package and the
reference on the same inputs and requires **exact** equality (``==``, max abs diff 0), not closeness,
for the statistic, the null draws, z, the cut, the verdict, the distinct-split count, the fold-order
mean and the chance floor. ``fixtures/reference_vectors.json`` holds stored vectors from the paper's
own archive, so the statistic and the verdict are also checked against the values the paper stored.

The rules, numbered as in the paper's census procedure:
  R1 statistic          Spearman rho of the two mean-|attribution| vectors, mid-rank ties
  R2-R3 attribution     mean |TreeSHAP| (positive class) over the explained rows
  R4-R6 real arm        GroupKFold-5 out-of-fold source vector; a whole-source model explains the target;
                        a seeded random subsample when a cap is set
  R7 null               30 GroupShuffleSplit 50/50 halves of the source, through the same procedure
  R8-R9 cut, verdict    t(0.975, n-1) * sqrt(1 + 1/n); z <= -cut COLLAPSE, z >= cut ROBUST
  R10 distinct splits   the cut at the number of distinct shot partitions, with a warning
  R11 chance floor      a ROBUST reading whose floor lies below the permuted-label floor is INDISTINGUISHABLE
  R13 fold orders       the real value is the mean over 10 fold orders
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from disruptionpy_qfc import compute_chance_floor, compute_null_floor, compute_qfc, exact_cut, qfc_statistic
from disruptionpy_qfc.core import NullFloor, ChanceFloor, QFCResult, _real_arm, _score, _treeshap, verdict_label

HERE = Path(__file__).resolve().parent
REF_PATH = HERE / "reference" / "qfc_metric.py"
REF_SHA256 = "b57f8cc2f0399d3f5a617f5210371019fb86aa5df7f615d826ca6326443a2034"

_spec = importlib.util.spec_from_file_location("qfc_reference_metric", REF_PATH)
Q = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(Q)


def test_the_vendored_reference_is_the_pinned_copy():
    """If the copy is edited, the identity below is against something else; if the paper's reference moves,
    set QFC_REFERENCE to it and this test says whether the copy is still current."""
    assert _sha(REF_PATH) == REF_SHA256
    live = os.environ.get("QFC_REFERENCE")
    if live:
        assert _sha(Path(live)) == REF_SHA256, \
            "the paper's reference has changed since this copy was taken; re-vendor it and re-run this file"


def _sha(path: Path) -> str:
    """sha256 of the file with line endings normalised, so a Windows checkout (CRLF) hashes like the original."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


# ------------------------------------------------------------------ R1, R8, R9 on the paper's stored vectors


@pytest.fixture(scope="module")
def stored():
    return json.loads((HERE / "fixtures" / "reference_vectors.json").read_text(encoding="utf-8"))["conditions"]


def test_statistic_null_z_and_verdict_equal_the_reference_and_the_stored_values(stored):
    assert len(stored) == 7
    for c in stored:
        real = qfc_statistic(c["real_src"], c["real_tgt"])
        assert real == Q.qfc_spearman(c["real_src"], c["real_tgt"])
        draws = [qfc_statistic(s, t) for s, t in zip(c["null_src"], c["null_tgt"])]
        assert draws == [Q.qfc_spearman(s, t) for s, t in zip(c["null_src"], c["null_tgt"])]
        arr = np.asarray(draws)
        floor = NullFloor(mean=float(arr.mean()), sd=float(arr.std(ddof=1)), per_seed=draws, n_seeds=len(draws),
                          exact_cut=exact_cut(len(draws)))
        z, v = floor.verdict_for(real)
        ref = Q.verdict(real, draws)
        assert (z, v, floor.exact_cut, floor.mean, floor.sd) == (ref["z"], ref["verdict"], ref["cut"],
                                                                 ref["null_mean"], ref["null_sd"])
        # and against what the paper's pipeline stored for this condition
        s = c["stored"]
        assert real == pytest.approx(s["real"], abs=1e-12)
        assert floor.mean == pytest.approx(s["null_mean"], abs=1e-12)
        assert floor.sd == pytest.approx(s["null_std"], abs=1e-12)
        assert z == pytest.approx(s["z"], abs=1e-9)


def test_the_cut_equals_the_reference_at_every_count():
    for n in range(3, 61):
        assert exact_cut(n) == Q.exact_cut(n)
    assert exact_cut(30) == pytest.approx(2.0790373811, abs=1e-10)


def test_the_boundary_is_inclusive_as_in_the_reference():
    cut = exact_cut(30)
    assert verdict_label(-cut, cut) == "COLLAPSE" and verdict_label(cut, cut) == "ROBUST"
    assert verdict_label(np.nextafter(-cut, 0), cut) == verdict_label(np.nextafter(cut, 0), cut) == "INDISTINGUISHABLE"
    # the reference's own boundary test, replayed through the package
    null = np.linspace(0.90, 0.99, 30)
    m, sd = null.mean(), null.std(ddof=1)
    f = NullFloor(mean=float(m), sd=float(sd), per_seed=list(null), n_seeds=30, exact_cut=cut)
    for real in (m - (cut + 1e-9) * sd, m - (cut - 1e-9) * sd, m + (cut + 1e-9) * sd, m + (cut - 1e-9) * sd):
        assert f.verdict_for(real) == (Q.verdict(real, null)["z"], Q.verdict(real, null)["verdict"])


def test_verdicts_agree_with_the_reference_on_random_nulls():
    rng = np.random.default_rng(0)
    for _ in range(2000):
        n = int(rng.integers(3, 40))
        null = rng.normal(rng.uniform(0.5, 0.99), rng.uniform(1e-3, 0.1), n)
        real = float(null.mean() + rng.normal(0, 4) * null.std(ddof=1))
        f = NullFloor(mean=float(null.mean()), sd=float(null.std(ddof=1)), per_seed=list(null), n_seeds=n,
                      exact_cut=exact_cut(n))
        r = Q.verdict(real, null)
        assert f.verdict_for(real) == (r["z"], r["verdict"])


def test_qfc_is_invariant_to_increasing_transformations_and_feature_relabelling():
    rng = np.random.default_rng(20261102)
    for _ in range(200):
        d = int(rng.integers(3, 60))
        a, b = rng.random(d) + 1e-3, rng.random(d) + 1e-3
        base = qfc_statistic(a, b)
        assert np.isclose(qfc_statistic(np.sqrt(a), b ** 3 + 5.0), base)
        p = rng.permutation(d)
        assert np.isclose(qfc_statistic(a[p], b[p]), base)


def test_degenerate_inputs_raise_like_the_reference():
    for a, b in (([1.0, np.nan, 3.0], [1.0, 2.0, 3.0]), ([1.0, 2.0], [2.0, 1.0]),
                 ([0.5, -0.1, 0.3], [0.5, 0.2, 0.3]), ([0.2, 0.2, 0.2], [0.1, 0.2, 0.3])):
        with pytest.raises(ValueError):
            qfc_statistic(a, b)
        with pytest.raises(ValueError):
            Q.qfc_spearman(a, b)
    f = NullFloor(mean=0.9, sd=float(np.std([0.9] * 30, ddof=1)), per_seed=[0.9] * 30, n_seeds=30, exact_cut=exact_cut(30))
    assert f.verdict_for(0.5)[1] == "UNDEFINED"
    with pytest.raises(ValueError):
        Q.verdict(0.5, [0.9] * 30)


# ------------------------------------------------------------------ R2-R7, R10, R13 on fitted models


def _data(n_groups=24, per=(6, 6, 9), d=6, seed=0):
    """Grouped rows with many shots of EQUAL size, so GroupKFold's tie order genuinely matters."""
    rng = np.random.default_rng(seed)
    sizes = [per[i % len(per)] for i in range(n_groups)]
    g = np.repeat(np.arange(100, 100 + n_groups), sizes)
    X = rng.normal(size=(len(g), d))
    y = (X @ (2.0 * 0.7 ** np.arange(d)) + rng.normal(scale=0.7, size=len(g)) > 0).astype(int)
    Xt = rng.normal(size=(70, d)) * np.array([0.3, 1, 1, 1, 2, 2])[:d]
    return X, y, g, Xt


def _frames(X, y, g, Xt):
    cols = ["f%d" % j for j in range(X.shape[1])]
    s = pd.DataFrame(X, columns=cols).assign(shot=g, label=y)
    t = pd.DataFrame(Xt, columns=cols).assign(shot=np.arange(len(Xt)) // 7)
    return s, t


def _rf():
    return RandomForestClassifier(n_estimators=15, max_depth=None, class_weight="balanced", n_jobs=-1)


SETTINGS = dict(model=_rf(), attribution="treeshap", shap_cap=40, n_splits=5)


def test_the_real_arm_equals_the_references_at_every_fold_order():
    pytest.importorskip("shap")
    X, y, g, Xt = _data()
    for k in range(4):
        mine = _real_arm(X, y, g, Xt, _rf(), 5, 40, 11, _treeshap, fold_order=k)[:2]
        ref = Q.paper_real_arm(n_estimators=15, max_depth=None, shap_cap=40, tie_order=k)(X, y, g, X, Xt, 11)
        assert np.array_equal(mine[0], ref[0]) and np.array_equal(mine[1], ref[1])
    # the orders really give different folds here, so the equality above is not vacuous
    v = {tuple(_real_arm(X, y, g, Xt, _rf(), 5, 40, 11, _treeshap, fold_order=k)[0]) for k in range(4)}
    assert len(v) > 1


def test_null_floor_and_verdict_equal_the_references_qfc_test():
    """R7: the null is the real arm's own procedure, so the package and qfc_test agree draw for draw."""
    pytest.importorskip("shap")
    X, y, g, Xt = _data(seed=1)
    s, t = _frames(X, y, g, Xt)
    floor = compute_null_floor(s, n_seeds=8, random_state=7, **SETTINGS)
    r = compute_qfc(s, t, null_floor=floor, n_fold_orders=1, random_state=7, **SETTINGS)
    ref = Q.qfc_test(Q.paper_real_arm(n_estimators=15, max_depth=None, shap_cap=40), X, y, g, Xt,
                     statistic=Q.qfc_spearman, n_null=8, seed0=7)
    assert r.qfc == ref["real"]
    assert floor.per_seed == ref["null_draws"]
    assert (r.z, r.verdict, r.exact_cut, r.resolution) == (ref["z"], ref["verdict"], ref["cut"], ref["resolution"])
    assert floor.n_distinct_splits == ref["n_distinct_splits"]
    assert r.verdict_effective == ref["verdict_effective"] and r.cut_effective == ref["cut_effective"]


def test_the_fold_order_mean_equals_the_references():
    """R13, the census's primary scoring: the reported QFC is the mean over ten fold orders."""
    pytest.importorskip("shap")
    X, y, g, Xt = _data(seed=2)
    s, t = _frames(X, y, g, Xt)
    r = compute_qfc(s, t, random_state=5, **SETTINGS)
    per = [Q.qfc_spearman(*Q.paper_real_arm(n_estimators=15, max_depth=None, shap_cap=40, tie_order=k)(
        X, y, g, X, Xt, 5)) for k in range(Q.N_TIE_ORDERS)]
    assert r.qfc_per_order == per and r.qfc == float(np.mean(per))
    assert len(r.qfc_per_order) == 10


def test_few_shots_give_the_effective_cut_and_warn_like_the_reference():
    """R10: six shots admit at most ten distinct halvings, so 30 draws repeat partitions. The reference runs the
    package's own arm here, so this checks the floor's split counting and effective cut, not the arm (the arm is
    checked against the reference's own arm above)."""
    pytest.importorskip("shap")
    X, y, g, Xt = _data(n_groups=6, per=(14,), seed=3)
    s, t = _frames(X, y, g, Xt)
    kw = dict(SETTINGS, n_splits=2)
    with pytest.warns(UserWarning, match="distinct shot splits"):
        floor = compute_null_floor(s, n_seeds=12, random_state=0, **kw)
    with pytest.warns(UserWarning, match="distinct group splits"):
        ref = Q.qfc_test(lambda *a: _real_arm(a[0], a[1], a[2], a[4], _rf(), 2, 40, a[5], _treeshap)[:2],
                         X, y, g, Xt, statistic=Q.qfc_spearman, n_null=12, seed0=0)
    assert floor.n_distinct_splits == ref["n_distinct_splits"] < 12
    assert floor.cut_effective == ref["cut_effective"] > floor.exact_cut


def test_the_chance_floor_is_the_censuss_permuted_label_construction():
    """R11's floor: draw d permutes the source labels with default_rng(d) and fits with seed d (the census's
    permlabel_floor scripts), then runs the real arm source to target."""
    pytest.importorskip("shap")
    X, y, g, Xt = _data(seed=4)
    s, t = _frames(X, y, g, Xt)
    ch = compute_chance_floor(s, t, n_draws=4, **SETTINGS)
    ref = [Q.qfc_spearman(*Q.paper_real_arm(n_estimators=15, max_depth=None, shap_cap=40)(
        X, np.random.default_rng(d).permutation(y), g, X, Xt, d)) for d in range(4)]
    assert ch.per_draw == ref


# ------------------------------------------------------------------ R11 on the census rows it changed


def test_the_chance_floor_rule_reproduces_every_census_decision():
    rec = json.loads((HERE / "fixtures" / "census_rule_rows.json").read_text(encoding="utf-8"))
    assert len(rec["rows"]) == 4
    t = rec["census_tallies"]   # the census's own tallies: every ROBUST row it demoted is accounted for
    assert t["standard_robust"] - t["changed_by_this_rule"] - t["changed_by_the_other_rule"] == t["adjudicated_robust"]
    for row in rec["rows"]:
        floor = NullFloor(mean=row["null_mean"], sd=row["null_sd"], per_seed=[], n_seeds=row["n_null"],
                          exact_cut=exact_cut(row["n_null"]))
        r = QFCResult(qfc=row["real"], qfc_per_order=[row["real"]], auc=None, n_source=0, n_target=0,
                      n_shots_source=0, n_shots_target=0, feature_cols_used=[], source_importance=[],
                      target_importance=[])
        _score(r, floor, ChanceFloor(mean=row["chance_floor"], sd=0.0, per_draw=[], n_draws=15))
        assert r.verdict_unadjudicated == row["standard"] == "ROBUST"
        assert r.verdict == row["census_adjudicated"] == "INDISTINGUISHABLE", row["row"]
        assert r.z == pytest.approx(row["census_z"], abs=2e-3)   # the census stores moments rounded to 6 places
        assert "chance floor" in r.adjudication


def test_a_robust_reading_above_its_chance_floor_stands():
    floor = NullFloor(mean=0.80, sd=0.02, per_seed=[], n_seeds=30, exact_cut=exact_cut(30))
    r = QFCResult(qfc=0.90, qfc_per_order=[0.90], auc=None, n_source=0, n_target=0, n_shots_source=0,
                  n_shots_target=0, feature_cols_used=[], source_importance=[], target_importance=[])
    _score(r, floor, ChanceFloor(mean=0.40, sd=0.05, per_draw=[], n_draws=15))
    assert r.verdict == "ROBUST" and r.adjudication is None
    _score(r, floor, None)
    assert r.verdict == "ROBUST" and "provisional" in r.adjudication
