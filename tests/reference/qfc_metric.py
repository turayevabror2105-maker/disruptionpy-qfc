# -*- coding: utf-8 -*-
"""Quantitative Feature Consistency (QFC) -- reference implementation.

Two statistics of agreement between a SOURCE and a TARGET attribution-importance vector (mean |attribution| per
feature, length d), and the null-relative verdict that turns either into COLLAPSE / INDISTINGUISHABLE / ROBUST.

  qfc_spearman(src, tgt)   V0: Spearman rank correlation. Every rank swap counts equally, wherever it happens.
  qfc_weighted(src, tgt)   V1: hyperbolic-weighted rank correlation -- weighted Pearson correlation of the two
                           descending rank vectors with w_i proportional to (1/r_src,i + 1/r_tgt,i). A swap among the
                           most important features counts far more than one in the tail.
  qfc_weighted_reliable    V6: V1 with each feature's weight multiplied by its no-shift reliability (from null source
                           vectors), so features whose importance is unstable even without shift count for less.
                           Selected by the pre-registered program on three model classes (it detects no fewer shifts
                           than V0 on any of them and is the most top-sensitive statistic that does,
                           review_export/qfc_max_gbm500_gate.json), reported beside QFC, and available in qfc_test as
                           noise_aware=True. It is NOT the default: the paper's QFC, the census's statistic and
                           Theorem 3.1's quantity is V0 (Ready B, 2026-11-02; the default was V6 from B-1208 until then).

V0 and V1 equal 1 if and only if the two (tie-aware, average) rankings agree -- the property Theorem 3.1
characterises; V6 does whenever every reliability is positive (with never-used features it equals 1 exactly when
the used features' ranks are positively affinely related -- see its docstring).
Checked exhaustively: 5,912 tie-free permutation pairs (d = 2..7) and 1,107,648 tied pairs (d = 2..5), zero violations.
Displacement bound for V1: if no feature moves more than M ranks, V1 >= 1 - M^2 / (2 min(Var_w(r_src), Var_w(r_tgt)));
tie-free, a data-free form is V1 >= 1 - M^2 H_d / (d(d+1)/2 - d^2/H_d), H_d the d-th harmonic number.

verdict(real, null_draws)  the project's null-relative test: z = (real - mean(null)) / sd(null, ddof=1) against the
                           exact two-sided 5% PREDICTION cut t(0.975, N-1) * sqrt(1 + 1/N) for one new draw.

Why V1 exists, measured (QFC_MAX_PROGRAM_PREREGISTRATION.md): at equal rank displacement V0 cannot tell a change among
the top features from one in the tail (top-minus-tail power -0.003 / +0.005 on two model classes), while V1 detects a
5-rank swap at the top 38-57% of the time against 7-9% in the tail, at the same 5% false-positive rate. The advantage
is specific to hyperbolic weights; gentler 1/sqrt(r) or DCG weights stay nearly as location-blind as V0.
"""
import numpy as np
from scipy import stats

__all__ = ["qfc_spearman", "qfc_weighted", "exact_cut", "verdict", "qfc_test", "paper_real_arm", "harmonic",
           "weighted_displacement_floor", "tie_order_groups", "N_TIE_ORDERS"]

# The census's primary scoring (CENSUS_REBUILD_OPTION_A_PREREGISTRATION.md Amendment 14, TIE_ORDER_PREREGISTRATION.md):
# the real value is the MEAN over K = 10 tie orders of GroupKFold, k = 0..9, scored against the row's own null.
N_TIE_ORDERS = 10


def tie_order_groups(groups, k):
    """Tie order k of TIE_ORDER_PREREGISTRATION.md: the groups relabelled by np.random.default_rng(1000 + k).permutation
    before GroupKFold, k = 0 the identity. GroupKFold sorts groups by size and breaks ties by numpy's unstable sort over
    np.unique's label order, so this changes ONLY the order among groups of equal size, never which samples share a
    group. Copied from closure_scripts/tie_order_runner.py (the campaign's own relabel, lines 45-48)."""
    g = np.asarray(groups)
    if k == 0:
        return g
    u, inv = np.unique(g, return_inverse=True)
    return np.random.default_rng(1000 + k).permutation(len(u))[inv]


def _ranks_desc(v):
    v = np.asarray(v, float)
    if v.ndim != 1 or v.size < 2:
        raise ValueError("importance vectors must be 1-D with at least two features")
    if not np.all(np.isfinite(v)):
        raise ValueError("importance vectors must be finite")
    return stats.rankdata(-v, method="average")


def _check_pair(src, tgt):
    a, b = np.asarray(src, float), np.asarray(tgt, float)
    if a.shape != b.shape:
        raise ValueError("source and target vectors must have the same length (%s vs %s)" % (a.shape, b.shape))
    # B-1352 (definition side A2): every degenerate input has a defined outcome, never a silent number. Before this,
    # qfc_spearman passed NaN straight to scipy and returned nan, and accepted d = 2, where rho is +1 or -1 by force.
    if a.ndim != 1 or a.size < 3:
        raise ValueError("importance vectors must be 1-D with at least three features (at d = 2 rho is +1 or -1 "
                         "by construction)")
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        raise ValueError("importance vectors must be finite")
    if (a < 0).any() or (b < 0).any():
        raise ValueError("importance vectors must be non-negative (mean |attribution|); a negative entry means signed "
                         "attributions were passed")
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        raise ValueError("a constant importance vector has no ranking; agreement is undefined")
    return a, b


def qfc_spearman(src, tgt):
    """V0 -- the paper's QFC: Spearman rho of the two importance vectors."""
    a, b = _check_pair(src, tgt)
    return float(stats.spearmanr(a, b)[0])


def qfc_weighted(src, tgt):
    """V1 -- hyperbolic-weighted rank correlation (see module docstring)."""
    a, b = _check_pair(src, tgt)
    ra, rb = _ranks_desc(a), _ranks_desc(b)
    w = 0.5 * (1.0 / ra + 1.0 / rb)
    w = w / w.sum()
    ma, mb = (w * ra).sum(), (w * rb).sum()
    va, vb = (w * (ra - ma) ** 2).sum(), (w * (rb - mb) ** 2).sum()
    return float((w * (ra - ma) * (rb - mb)).sum() / np.sqrt(va * vb))


def exact_cut(n_null):
    """Exact two-sided 5% prediction cut for one new draw against n_null null draws."""
    if n_null < 3:
        raise ValueError("at least three null draws are needed")
    return float(stats.t.ppf(0.975, n_null - 1) * np.sqrt(1.0 + 1.0 / n_null))


def verdict(real, null_draws):
    """Null-relative verdict for one real value against its own no-shift null draws."""
    v = np.asarray(null_draws, float)
    sd = v.std(ddof=1)
    # B-1276: test spread on the draws themselves. Thirty identical floats can give sd ~1e-16 rather than 0 (the mean
    # is not exact), which passed the old `sd <= 0` test and turned any real value into a z of order 1e15 -- the
    # degenerate-null artefact AP P152 describes, reachable through the reference implementation.
    if not np.isfinite(sd) or np.ptp(v) == 0 or sd <= 0:
        raise ValueError("null draws have no spread; the verdict is undefined")
    z = (float(real) - v.mean()) / sd
    cut = exact_cut(v.size)
    label = "COLLAPSE" if z <= -cut else "ROBUST" if z >= cut else "INDISTINGUISHABLE"
    return {"z": float(z), "cut": cut, "verdict": label, "null_mean": float(v.mean()), "null_sd": float(sd),
            "n_null": int(v.size), "resolution": float(cut * sd)}


def reliability(null_source_vectors):
    """Per-feature no-shift reliability mean^2 / (mean^2 + var) over null SOURCE importance vectors (1 = stable,
    -> 0 = pure noise; 0 for a feature never used). Computed from null draws only, never from the real comparison."""
    S = np.asarray(null_source_vectors, float)
    m, v = S.mean(axis=0), S.var(axis=0, ddof=1)
    den = m ** 2 + v
    return np.where(den > 0, m ** 2 / np.where(den > 0, den, 1.0), 0.0)


def qfc_weighted_reliable(src, tgt, rel):
    """V6: V1 with each feature's weight multiplied by its no-shift reliability `rel` (from `reliability`).

    Scope of the iff (stated in the pre-registration, amendment 3): a feature the model never uses in any null draw has
    rel = 0 and carries no weight. With every rel > 0, V6 = 1 iff the two rankings agree (checked exhaustively on
    1,107,648 pairs, qfc_max_c4_ties_v6.py). With some rel = 0, ranks stay global, so V6 = 1 exactly when the used
    features' ranks are POSITIVELY AFFINELY related -- e.g. used ranks 1, 2, 3 becoming 1, 3, 5 score 1.0. A reordering
    confined to never-used features, or a common stretch of the used ones' ranks, is therefore not seen.
    """
    a, b = _check_pair(src, tgt)
    rel = np.asarray(rel, float)
    if rel.shape != a.shape or np.any(rel < 0) or not np.any(rel > 0):
        raise ValueError("rel must be nonnegative, the same length as the vectors, and not all zero")
    ra, rb = _ranks_desc(a), _ranks_desc(b)
    w = 0.5 * (1.0 / ra + 1.0 / rb) * rel
    w = w / w.sum()
    ma, mb = (w * ra).sum(), (w * rb).sum()
    va, vb = (w * (ra - ma) ** 2).sum(), (w * (rb - mb) ** 2).sum()
    return float((w * (ra - ma) * (rb - mb)).sum() / np.sqrt(va * vb))


def qfc_test(fit_explain, X_src, y_src, g_src, X_tgt, statistic=None, n_null=30, seed0=0, noise_aware=None):
    """The full null-relative QFC test with the null built by THE SAME procedure as the real value -- by construction.

    `fit_explain(X_train, y_train, g_train, X_a, X_b, seed)` is the user's ONE pipeline: it fits whatever model on the
    training data and returns (importance_a, importance_b), the mean-|attribution| vectors explaining X_a and X_b.
    The real comparison is fit_explain(source, ..., X_a = source, X_b = target); each null draw splits the SOURCE pool
    50/50 by group (GroupShuffleSplit, random_state = seed0 + k) and calls the same function with half 1 as both
    training data and X_a and half 2 as X_b. Because both go through one function, the real and null cannot differ in
    construction, model settings, SHAP cap or preprocessing -- the defect class that flipped verdicts in this project
    (B-1140, B-1153..B-1169) is impossible by design rather than avoided by care.

    Default statistic (Ready B, 2026-11-02): V0, qfc_spearman -- the paper's QFC. Pass `statistic=` for any other
    statistic of the two vectors (e.g. qfc_weighted for V1), or `noise_aware=True` (with no `statistic`) for V6,
    qfc_weighted_reliable, whose reliability weights come from the null SOURCE vectors. From B-1208 until this change
    the default was V6, a statistic the paper reports beside QFC but does not score its census with.
    """
    from sklearn.model_selection import GroupShuffleSplit
    X_src, y_src, g_src = np.asarray(X_src), np.asarray(y_src), np.asarray(g_src)
    # two passes: every importance vector first (real + nulls, all through the ONE fit_explain), then scoring, so a
    # noise-aware statistic can take its reliability weights from the null SOURCE vectors before anything is scored
    imp_a, imp_b = fit_explain(X_src, y_src, g_src, X_src, np.asarray(X_tgt), seed0)
    pairs, splits = [], set()
    for k in range(n_null):
        h1, h2 = next(GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=seed0 + k).split(X_src, y_src, g_src))
        # a split and its mirror are one partition of the groups (B-1276: a pool with few groups -- FocusPath's six
        # slides -- draws far fewer distinct splits than seeds, so the null's effective sample size is the count here)
        splits.add(frozenset([frozenset(np.unique(g_src[h1]).tolist()), frozenset(np.unique(g_src[h2]).tolist())]))
        pairs.append(fit_explain(X_src[h1], y_src[h1], g_src[h1], X_src[h1], X_src[h2], seed0 + k))
    if noise_aware:
        if statistic is not None:
            raise ValueError("noise_aware=True uses qfc_weighted_reliable; do not also pass `statistic`")
        rel = reliability([p[0] for p in pairs])
        stat = lambda u, v: qfc_weighted_reliable(u, v, rel)  # noqa: E731
        name = "qfc_weighted_reliable"
    else:
        stat = statistic or qfc_spearman
        name = getattr(stat, "__name__", "custom")
    real = stat(imp_a, imp_b)
    draws = [stat(na, nb) for na, nb in pairs]
    out = verdict(real, draws)
    out.update(real=float(real), null_draws=[float(v) for v in draws], statistic=name)
    # Effective sample size (B-1276). Repeated null VALUES are usually rank-lattice collisions and cost nothing;
    # repeated SPLITS do, because the null then samples fewer partitions than it has draws. `verdict` stays at the
    # nominal count, which is how the paper's census is scored; `verdict_effective` re-reads the same z at the exact cut
    # for the number of distinct splits, and a warning is raised whenever the two counts differ.
    k = len(splits)
    out["n_distinct_splits"] = k
    if k < n_null:
        import warnings
        cut_k = exact_cut(k) if k >= 3 else float("inf")
        z = out["z"]
        out["cut_effective"] = cut_k
        out["verdict_effective"] = "COLLAPSE" if z <= -cut_k else "ROBUST" if z >= cut_k else "INDISTINGUISHABLE"
        warnings.warn("only %d distinct group splits among %d null draws (the source pool has %d groups); the null's "
                      "effective sample size is %d -- see verdict_effective" % (k, n_null, len(np.unique(g_src)), k))
    else:
        out["cut_effective"], out["verdict_effective"] = out["cut"], out["verdict"]
    return out


def paper_real_arm(n_estimators=500, max_depth=None, shap_cap=None, tie_order=0):
    """The paper's real arm as a fit_explain: out-of-fold TreeSHAP on the training data (GroupKFold-5) for X_a, and one
    forest fitted on all training data explaining X_b. Requires scikit-learn and shap. When X_a is the training data
    itself (the only way qfc_test calls it), the out-of-fold vector is the source attribution the paper uses.
    `tie_order` = k applies tie_order_groups(g, k) to the groups GroupKFold sees (k = 0, the default, changes nothing);
    the census's primary real value is the mean of this arm over k = 0..N_TIE_ORDERS-1."""
    def fit_explain(X, y, g, X_a, X_b, seed):
        import shap
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import GroupKFold

        def explain(model, Z):
            if shap_cap is not None and len(Z) > shap_cap:
                Z = Z[np.random.default_rng(seed).choice(len(Z), shap_cap, replace=False)]
            sv = np.asarray(shap.TreeExplainer(model).shap_values(Z, check_additivity=False))
            if sv.ndim == 3:
                sv = sv[..., 1] if sv.shape[-1] == 2 else sv[1]
            return np.abs(sv).mean(axis=0)

        def model():
            return RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, class_weight="balanced",
                                          random_state=seed, n_jobs=-1)
        folds = []
        for tr, te in GroupKFold(n_splits=min(5, len(np.unique(g)))).split(X, y, tie_order_groups(g, tie_order)):
            folds.append(explain(model().fit(X[tr], y[tr]), X[te]))
        return np.mean(folds, axis=0), explain(model().fit(X, y), X_b)
    return fit_explain


def harmonic(d):
    return float(sum(1.0 / k for k in range(1, d + 1)))


def weighted_displacement_floor(d, max_displacement):
    """Data-free lower bound on V1 when no feature's rank moves more than max_displacement (tie-free)."""
    h = harmonic(d)
    return float(1.0 - max_displacement ** 2 * h / (d * (d + 1) / 2.0 - d * d / h))
