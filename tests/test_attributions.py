"""The attribution methods: the four built in, and your own as a callable.

QFC is computed from per-row attributions -- one row of per-feature contributions per explained row -- as
the mean of their absolute values. Any method that produces that array can be screened, including the
feature contributions a disruption predictor already computes (DPRF's are Saabas tree-path contributions,
Rea et al., Nucl. Fusion 59, 096016, 2019).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from disruptionpy_qfc import ATTRIBUTIONS, compute_qfc, screen
from disruptionpy_qfc.core import _resolve_attribution, _saabas


def _frames(seed=0, n_shots=16, per=20, d=5):
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_shots):
        X = rng.normal(size=(per, d))
        y = (X @ (2.0 * 0.7 ** np.arange(d)) + rng.normal(scale=0.5, size=per) > 0).astype(int)
        for r in range(per):
            rows.append({"shot": s, "time": r, "label": int(y[r]), **{f"f{j}": X[r, j] for j in range(d)}})
    df = pd.DataFrame(rows)
    return df[df.shot < n_shots // 2].copy(), df[df.shot >= n_shots // 2].copy()


RF = RandomForestClassifier(n_estimators=20, max_depth=5, class_weight="balanced", n_jobs=1)
KW = dict(n_fold_orders=1, shap_cap=60, n_splits=3)


def test_the_four_built_ins_are_offered_and_auto_picks_the_papers_choice():
    assert set(ATTRIBUTIONS) == {"treeshap", "saabas", "kernelshap", "permutation"}
    assert _resolve_attribution("auto", RF)[1] == "treeshap"
    assert _resolve_attribution("auto", LogisticRegression())[1] == "kernelshap"
    with pytest.raises(ValueError, match="tree model"):
        _resolve_attribution("saabas", LogisticRegression())
    with pytest.raises(ValueError, match="attribution must be"):
        _resolve_attribution("lime", RF)


@pytest.mark.parametrize("name", ["treeshap", "saabas", "permutation"])
def test_each_tree_method_runs_and_is_reproducible(name):
    src, tgt = _frames(1)
    a = compute_qfc(src, tgt, model=RF, attribution=name, random_state=3, **KW)
    b = compute_qfc(src, tgt, model=RF, attribution=name, random_state=3, **KW)
    assert a.attribution == name and a.qfc == b.qfc and -1 <= a.qfc <= 1


def test_kernelshap_on_a_non_tree_model_is_reproducible_for_a_seed():
    pytest.importorskip("shap")
    src, tgt = _frames(2, n_shots=12, per=10)
    lr = LogisticRegression(max_iter=500)
    kw = dict(KW, shap_cap=15)
    a = compute_qfc(src, tgt, model=lr, random_state=4, **kw)
    b = compute_qfc(src, tgt, model=lr, random_state=4, **kw)
    assert a.attribution == "kernelshap" and a.source_importance == b.source_importance and a.qfc == b.qfc


def _saabas_by_hand(forest, X):
    """Saabas tree-path decomposition written out from the tree arrays: at every split on a sample's path, the
    change in the positive-class probability is credited to the split's feature; averaged over the trees."""
    out = np.zeros(X.shape)
    for est in forest.estimators_:
        t = est.tree_
        p = t.value[:, 0, :] / t.value[:, 0, :].sum(axis=1, keepdims=True)
        paths = est.decision_path(X).tolil().rows
        for i, nodes in enumerate(paths):
            for parent, child in zip(nodes[:-1], nodes[1:]):
                out[i, t.feature[parent]] += p[child, 1] - p[parent, 1]
    return out / len(forest.estimators_)


def test_saabas_is_the_tree_path_decomposition_dprf_uses():
    src, _ = _frames(3)
    cols = [c for c in src.columns if c.startswith("f")]
    X, y = src[cols].to_numpy(float), src["label"].to_numpy()
    m = RandomForestClassifier(n_estimators=10, max_depth=4, class_weight="balanced", random_state=0).fit(X, y)
    assert np.allclose(_saabas(m, X[:50], X, 0), _saabas_by_hand(m, X[:50]), atol=1e-10)


def test_saabas_also_runs_on_gradient_boosting():
    src, tgt = _frames(4)
    r = compute_qfc(src, tgt, model=GradientBoostingClassifier(n_estimators=20, random_state=0),
                    attribution="saabas", **KW)
    assert r.attribution == "saabas" and np.isfinite(r.qfc)


def test_your_own_attributions_are_accepted_as_a_callable():
    """The hook for a group's own explanation method: return one row of contributions per explained row."""
    calls = []

    def my_contributions(model, X_explain, X_train, seed):
        calls.append((X_explain.shape, X_train.shape, seed))
        return _saabas(model, X_explain, X_train, seed)       # e.g. what a DPRF-style predictor already computes

    src, tgt = _frames(5)
    mine = compute_qfc(src, tgt, model=RF, attribution=my_contributions, random_state=1, **KW)
    builtin = compute_qfc(src, tgt, model=RF, attribution="saabas", random_state=1, **KW)
    assert mine.attribution == "my_contributions" and mine.qfc == builtin.qfc
    assert calls and all(c[0][1] == 5 and c[1][1] == 5 for c in calls)


def test_a_callable_returning_the_wrong_shape_is_refused():
    src, tgt = _frames(6)
    bad = lambda m, X, Xt, s: np.ones(X.shape[1])  # noqa: E731
    with pytest.raises(ValueError, match="one row of per-feature attributions per explained row"):
        compute_qfc(src, tgt, model=RF, attribution=bad, **KW)
    # and through screen(), whose null floor runs first: the error must surface, not become "skipped draws"
    with pytest.raises(ValueError, match="one row of per-feature attributions per explained row"):
        screen(src, tgt, model=RF, attribution=bad, n_seeds=4, **KW)


def test_the_null_floor_uses_the_same_attribution_as_the_measurement():
    src, tgt = _frames(7)
    r = screen(src, tgt, model=RF, attribution="permutation", n_seeds=4, **KW)
    assert r.attribution == "permutation" and r.null_floor["provenance"]["attribution"] == "permutation"
    assert r.verdict in {"COLLAPSE", "INDISTINGUISHABLE", "ROBUST"}


def test_kernelshap_leaves_numpy_global_random_state_alone():
    from disruptionpy_qfc.core import _kernelshap
    src, _ = _frames(8, n_shots=6, per=10)
    cols = [c for c in src.columns if c.startswith("f")]
    X, y = src[cols].to_numpy(float), src["label"].to_numpy()
    lr = LogisticRegression(max_iter=500).fit(X, y)
    np.random.seed(123)
    before = np.random.random()
    np.random.seed(123)
    _kernelshap(lr, X[:3], X, 7)
    assert np.random.random() == before
