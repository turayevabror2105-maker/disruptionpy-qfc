"""Attribution-stability screening for DisruptionPy-format shot data.

The question, answered before deployment and without target labels:

    A model trained on the source environment may still predict well on the target.
    Does it predict well *for the same reasons*?

and the follow-up question that a raw correlation cannot answer on its own:

    Is what we measured any different from what a no-shift split of the source alone gives?

``screen`` answers both and is what most users want. Its statistic, cut and scoring rules are those of the
paper that defines the measurement (Turayev, *Pareto impossibility of attribution consistency under deployment
shift*), and its null is built as the paper's reference implementation builds it;
``tests/test_reference_identity.py`` checks it number for number against that implementation.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import asdict, dataclass, field
from typing import Callable, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone, is_classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

__all__ = [
    "QFCResult",
    "NullFloor",
    "ChanceFloor",
    "DegenerateDrawError",
    "compute_qfc",
    "compute_null_floor",
    "compute_chance_floor",
    "screen",
    "default_model",
    "detect_feature_cols",
    "qfc_statistic",
    "exact_cut",
    "verdict_label",
    "tie_order_groups",
    "ATTRIBUTIONS",
    "DEFAULT_EXCLUDE_COLS",
    "N_FOLD_ORDERS",
    "N_NULL_SEEDS",
    "N_CHANCE_DRAWS",
]

#: Never used as features. ``time_until_disrupt`` is DisruptionPy's time-to-event column, from which the label is
#: derived, so as a feature it would leak the label; ``time_domain`` is DisruptionPy's phase flag, not a measurement.
DEFAULT_EXCLUDE_COLS = frozenset({"shot", "shot_id", "time", "label", "time_until_disrupt", "time_domain"})
_TREE_MODEL_HINTS = ("Forest", "GradientBoosting", "DecisionTree", "ExtraTrees", "XGB", "LGBM", "CatBoost")

#: Fold orders averaged for the real value (the paper's primary census scoring).
N_FOLD_ORDERS = 10
#: No-shift draws in the null floor (the census scores every row against 30).
N_NULL_SEEDS = 30
#: Permuted-label draws in the chance floor (the paper's census uses 15).
N_CHANCE_DRAWS = 15

AttributionSpec = Union[str, Callable]


def default_model(random_state: int = 42, max_depth: Optional[int] = None) -> RandomForestClassifier:
    """The estimator used unless the caller supplies one: the paper's real arm.

    A random forest of 500 trees, unlimited depth, balanced class weights -- the setting of the
    paper's census (its ``full_model_and_shap``) and of its reference implementation
    (``paper_real_arm``). The paper's C-Mod row used ``max_depth=20``; pass it to reproduce that row.
    """
    return RandomForestClassifier(
        n_estimators=500, max_depth=max_depth, class_weight="balanced", random_state=random_state, n_jobs=-1
    )


# --------------------------------------------------------------------------------------- the rules


def qfc_statistic(src, tgt) -> float:
    """QFC itself: Spearman's rank correlation of two mean-|attribution| vectors (ties take mid-ranks).

    Degenerate inputs raise rather than return a number: fewer than three features (at two, rho is +1
    or -1 by construction), a non-finite entry, a negative entry (signed attributions passed by
    mistake), or a constant vector (no ranking, so agreement is undefined).
    """
    a, b = np.asarray(src, float), np.asarray(tgt, float)
    if a.shape != b.shape:
        raise ValueError("source and target vectors must have the same length (%s vs %s)" % (a.shape, b.shape))
    if a.ndim != 1 or a.size < 3:
        raise ValueError("importance vectors must be 1-D with at least three features (at d = 2 rho is +1 or -1 "
                         "by construction)")
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        raise ValueError("importance vectors must be finite")
    if (a < 0).any() or (b < 0).any():
        raise ValueError("importance vectors must be non-negative (mean |attribution|); a negative entry means "
                         "signed attributions were passed")
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        raise DegenerateDrawError("a constant importance vector has no ranking; agreement is undefined")
    return float(stats.spearmanr(a, b)[0])


class DegenerateDrawError(ValueError):
    """A draw that cannot be measured because of the data in it (every training fold single-class, or a
    constant importance vector). A null or chance-floor draw that raises this is skipped and counted; any
    other error is raised, because it means the inputs or the attribution are wrong."""


def exact_cut(n: int) -> float:
    """The exact two-sided 5% prediction cut for one new value against ``n`` null draws:
    ``t(0.975, n-1) * sqrt(1 + 1/n)``, 2.0790 at n = 30. Not ``|z| > 2``, which is too lenient at small n."""
    if n < 3:
        raise ValueError("at least three null draws are needed")
    return float(stats.t.ppf(0.975, n - 1) * np.sqrt(1.0 + 1.0 / n))


def verdict_label(z: float, cut: float) -> str:
    """COLLAPSE if ``z <= -cut``, ROBUST if ``z >= cut``, otherwise INDISTINGUISHABLE."""
    return "COLLAPSE" if z <= -cut else "ROBUST" if z >= cut else "INDISTINGUISHABLE"


def tie_order_groups(groups, k: int) -> np.ndarray:
    """Fold order ``k``: the shot labels relabelled by ``np.random.default_rng(1000 + k).permutation``.

    ``GroupKFold`` sorts shots by size and breaks ties by numpy's sort order over the labels, so which
    equal-sized shots share a fold is arbitrary. Relabelling changes only that arbitrary part -- never
    which rows belong to which shot. ``k = 0`` is the identity.
    """
    g = np.asarray(groups)
    if k == 0:
        return g
    u, inv = np.unique(g, return_inverse=True)
    return np.random.default_rng(1000 + k).permutation(len(u))[inv]


# --------------------------------------------------------------------------------------- attribution


def _is_tree_model(model) -> bool:
    return any(h in type(model).__name__ for h in _TREE_MODEL_HINTS)


def _positive_class(sv) -> np.ndarray:
    """Per-row attributions toward class 1, as an ``(n_rows, n_features)`` array."""
    if isinstance(sv, list):  # older shap: one array per class
        sv = sv[1] if len(sv) == 2 else sv[0]
    elif hasattr(sv, "values"):  # Explanation object
        sv = sv.values
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[..., 1] if sv.shape[-1] == 2 else sv[1]
    return sv


def _treeshap(model, X, X_train, seed):
    import shap

    return _positive_class(shap.TreeExplainer(model).shap_values(X, check_additivity=False))


def _saabas(model, X, X_train, seed):
    """Saabas tree-path contributions: each split's change in the model's output (the predicted
    probability for a random forest, the log-odds for gradient boosting), credited to the feature it splits
    on. This is the method of ``treeinterpreter``, which the DPRF disruption predictor uses for its feature
    contributions (Rea et al., Nucl. Fusion 59, 096016, 2019); computed here by shap's ``approximate=True``."""
    import shap

    return _positive_class(shap.TreeExplainer(model).shap_values(X, approximate=True, check_additivity=False))


def _kernelshap(model, X, X_train, seed):
    """KernelSHAP, as the paper's census uses it for non-tree models: a 30-centroid k-means background
    from the training rows and 200 coalition samples per row. Seeded, so a call is reproducible; numpy's
    global random state, which shap draws from, is restored afterwards. The regularisation is pinned to
    ``l1_reg="num_features(10)"``, the current shap default, because the default itself changed between shap
    versions (it was ``"auto"`` in shap 0.45). With more than ten features this keeps each row's ten largest
    contributors and sets the rest to zero; for another choice, pass your own callable. It costs about
    ``200 x 30`` model evaluations per explained row."""
    import shap

    state = np.random.get_state()
    try:
        np.random.seed(seed % (2**32))
        bg = shap.kmeans(X_train, min(30, len(X_train)))
        ex = shap.KernelExplainer(model.predict_proba, bg)
        return _positive_class(ex.shap_values(X, nsamples=200, l1_reg="num_features(10)", silent=True))
    finally:
        np.random.set_state(state)


def _permutation(model, X, X_train, seed, n_repeats: int = 5):
    """Label-free permutation attribution: per row, the mean |change in predicted probability| when one
    feature is permuted across the explained rows. Uses no labels, so it is computable on an unlabelled
    target."""
    rng = np.random.default_rng(seed)
    base = model.predict_proba(X)[:, 1]
    out = np.zeros(X.shape, dtype=float)
    for j in range(X.shape[1]):
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[:, j] = Xp[rng.permutation(len(Xp)), j]
            out[:, j] += np.abs(model.predict_proba(Xp)[:, 1] - base)
    return out / n_repeats


#: Built-in attribution methods. Each takes ``(model, X_explain, X_train, seed)`` and returns per-row
#: attributions of shape ``(n_rows, n_features)``; QFC uses the mean of their absolute values.
ATTRIBUTIONS = {"treeshap": _treeshap, "saabas": _saabas, "kernelshap": _kernelshap, "permutation": _permutation}


def _resolve_attribution(attribution: AttributionSpec, model) -> tuple:
    """Return ``(callable, name)``. ``"auto"`` is TreeSHAP for tree models and KernelSHAP otherwise."""
    if callable(attribution):
        return attribution, getattr(attribution, "__name__", "custom")
    if attribution == "auto":
        attribution = "treeshap" if _is_tree_model(model) else "kernelshap"
    if attribution not in ATTRIBUTIONS:
        raise ValueError("attribution must be 'auto', one of %s, or a callable; got %r"
                         % (sorted(ATTRIBUTIONS), attribution))
    if attribution in ("treeshap", "saabas") and not _is_tree_model(model):
        raise ValueError("%s needs a tree model; %s is not one (use 'kernelshap', 'permutation' or a callable)"
                         % (attribution, type(model).__name__))
    return ATTRIBUTIONS[attribution], attribution


def _explain(model, Z: np.ndarray, X_train: np.ndarray, cap: Optional[int], seed: int, attr) -> np.ndarray:
    """Mean |attribution| over the explained rows.

    When ``cap`` is set and there are more rows, the rows are a seeded random subsample, never the
    first ``cap``: DisruptionPy frames are ordered by shot and time, so a head slice would explain
    the first few discharges rather than a sample of the environment.
    """
    if cap is not None and len(Z) > cap:
        Z = Z[np.random.default_rng(seed).choice(len(Z), cap, replace=False)]
    sv = np.asarray(attr(model, Z, X_train, seed), dtype=float)
    if sv.shape != Z.shape:
        raise ValueError("the attribution returned shape %s for %s explained rows; it must return one row of "
                         "per-feature attributions per explained row" % (sv.shape, Z.shape))
    return np.abs(sv).mean(axis=0)


def _fit(model, X: np.ndarray, y: np.ndarray, seed: int):
    est = clone(model)
    if "random_state" in est.get_params():
        est.set_params(random_state=seed)
    return est.fit(X, y)


def _folds(ys, gs, n_splits, fold_order=0) -> list:
    """The ``GroupKFold`` folds over shots at fold order ``fold_order``, as ``(train, explain)`` index pairs."""
    k = min(n_splits, len(np.unique(gs)))
    return list(GroupKFold(n_splits=k).split(np.zeros((len(ys), 1)), ys, tie_order_groups(gs, fold_order)))


def _source_vector(Xs, ys, folds, model, cap, seed, attr) -> tuple:
    """The mean over folds of out-of-fold importance vectors, so no shot is explained by a model that saw it.
    A fold whose training part holds one class only is skipped and counted. Returns ``(vector, folds_skipped)``."""
    vecs, skipped = [], 0
    for tr, te in folds:
        if len(np.unique(ys[tr])) < 2:
            skipped += 1
            continue
        vecs.append(_explain(_fit(model, Xs[tr], ys[tr], seed), Xs[te], Xs[tr], cap, seed, attr))
    if not vecs:
        raise DegenerateDrawError("every cross-validation fold was single-class; cannot compute a source vector")
    return np.mean(vecs, axis=0), skipped


def _real_arm(Xs, ys, gs, Xt, model, n_splits, cap, seed, attr, fold_order=0):
    """The paper's real arm, once. Returns ``(source_vector, target_vector, deployed_model, folds_skipped)``.

    Source vector: :func:`_source_vector` over the folds at ``fold_order``. Target vector: one model fitted
    on the *whole* source and applied to the target without retraining, as a deployed predictor would be.
    """
    src, skipped = _source_vector(Xs, ys, _folds(ys, gs, n_splits, fold_order), model, cap, seed, attr)
    deployed = _fit(model, Xs, ys, seed)
    return src, _explain(deployed, Xt, Xs, cap, seed, attr), deployed, skipped


# --------------------------------------------------------------------------------------- results


def _jsonable(x):
    """Replace every non-finite float by None, so ``to_dict()`` output is strict JSON."""
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


@dataclass
class NullFloor:
    """The no-shift reference for one source pool, model and protocol.

    Each draw splits the source pool 50/50 by shot and runs the measurement's own code between the two
    halves -- the first half plays the source, the second the target -- so the shift is zero by
    construction. An observed QFC is read against this.
    """

    mean: float
    sd: float
    per_seed: list
    n_seeds: int
    exact_cut: float
    n_distinct_splits: int = 0
    cut_effective: Optional[float] = None
    provenance: dict = field(default_factory=dict)
    n_seeds_requested: Optional[int] = None
    skips: dict = field(default_factory=dict)

    @property
    def n_seeds_skipped(self) -> int:
        return sum(self.skips.values())

    @property
    def has_spread(self) -> bool:
        draws = np.asarray(self.per_seed, float)
        return bool(np.isfinite(self.sd) and self.sd > 0 and (draws.size == 0 or np.ptp(draws) > 0))

    def summary(self) -> str:
        """One line, plus the usable-seed count and the distinct-split count whenever either falls short."""
        s = "null floor = %+.4f +/- %.4f (%d seeds)   cut = +/-%.4f" % (self.mean, self.sd, self.n_seeds,
                                                                     self.exact_cut)
        req = self.n_seeds_requested
        if req and self.n_seeds < req:
            s += "\n             from %d/%d usable seeds (%s)" % (
                self.n_seeds, req, ", ".join("%s: %d" % (k, v) for k, v in sorted(self.skips.items())))
            if self.n_seeds < 0.8 * req:
                s += ("\n             [WARNING: many seeds unusable -- the surviving splits are the easier ones, "
                      "so this floor may be optimistic]")
        if self.n_distinct_splits and self.n_distinct_splits < self.n_seeds:
            s += ("\n             only %d distinct shot splits among %d draws -- the effective cut is +/-%.4f"
                  % (self.n_distinct_splits, self.n_seeds, self.cut_effective))
        return s

    def verdict_for(self, qfc: float) -> tuple:
        """Return ``(z, verdict)`` for an observed QFC: ``z = (qfc - mean) / sd`` against ``exact_cut``.
        A floor whose draws have no spread gives ``(nan, "UNDEFINED")``."""
        if not self.has_spread:
            return float("nan"), "UNDEFINED"
        z = (float(qfc) - self.mean) / self.sd
        return float(z), verdict_label(z, self.exact_cut)

    def to_dict(self) -> dict:
        return _jsonable(asdict(self))


@dataclass
class ChanceFloor:
    """What the real measurement gives when the source labels carry no signal at all.

    Each draw permutes the source labels and runs the measurement's own code (source to
    target). A no-shift floor that sits *below* this has nothing to say about robustness, because
    two halves of one pool then agree less than a model trained on noise does.
    """

    mean: float
    sd: float
    per_draw: list
    n_draws: int
    provenance: dict = field(default_factory=dict)
    n_draws_requested: Optional[int] = None
    skips: dict = field(default_factory=dict)

    def summary(self) -> str:
        return "chance floor = %+.4f +/- %.4f (%d permuted-label draws)" % (self.mean, self.sd, self.n_draws)

    def to_dict(self) -> dict:
        return _jsonable(asdict(self))


@dataclass
class QFCResult:
    """One screening result. ``to_dict()`` is JSON-serialisable.

    ``qfc`` is the mean over ``n_fold_orders`` fold orders (the value the verdict uses); the value at
    each order is in ``qfc_per_order``. ``null_floor`` and ``chance_floor`` hold the floors in full (every
    draw, every skip and its reason, provenance) when they were used.
    """

    qfc: float
    qfc_per_order: list
    auc: Optional[float]
    n_source: int
    n_target: int
    n_shots_source: int
    n_shots_target: int
    feature_cols_used: list
    source_importance: list
    target_importance: list
    attribution: str = "treeshap"
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None
    n_bootstrap_used: int = 0
    n_bootstrap_attempted: int = 0
    null_mean: Optional[float] = None
    null_sd: Optional[float] = None
    n_null_seeds: Optional[int] = None
    n_distinct_splits: Optional[int] = None
    z: Optional[float] = None
    exact_cut: Optional[float] = None
    resolution: Optional[float] = None
    verdict_unadjudicated: Optional[str] = None
    verdict: str = "UNCALIBRATED"
    verdict_effective: Optional[str] = None
    cut_effective: Optional[float] = None
    verdict_per_order: Optional[list] = None
    chance_floor_mean: Optional[float] = None
    chance_floor_sd: Optional[float] = None
    n_chance_draws: Optional[int] = None
    adjudication: Optional[str] = None
    null_floor: Optional[dict] = None
    chance_floor: Optional[dict] = None
    provenance: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return _jsonable(asdict(self))

    def summary(self) -> str:
        n = len(self.qfc_per_order)
        spread = (" (mean of %d fold orders, range %.4f to %.4f)" % (n, min(self.qfc_per_order), max(self.qfc_per_order))
                  if n > 1 else "")
        lines = [f"QFC        = {self.qfc:+.4f}{spread}"]
        if self.n_bootstrap_attempted:
            ci = (f"[{self.ci_lower:+.4f}, {self.ci_upper:+.4f}]" if self.ci_lower is not None else "n/a")
            line = (f"95% CI     = {ci} at fold order 0, from {self.n_bootstrap_used}/{self.n_bootstrap_attempted} "
                    f"usable shot-bootstrap draws")
            if self.n_bootstrap_used < 0.8 * self.n_bootstrap_attempted:
                line += "  [WARNING: many draws unusable -- interval is unreliable]"
            lines.append(line)
        lines.append(f"target AUC = {self.auc:.4f}" if self.auc is not None else "target AUC = n/a (target unlabelled "
                     "or single-class)")
        lines.append(f"n          = {self.n_source} rows / {self.n_shots_source} shots  ->  "
                     f"{self.n_target} rows / {self.n_shots_target} shots   (attribution: {self.attribution})")
        imp = (self.provenance or {}).get("imputed_with_source_median") or {}
        n_s, n_t = sum((imp.get("source") or {}).values()), sum((imp.get("target") or {}).values())
        if n_s or n_t:
            lines.append(f"imputed    = {n_s} source and {n_t} target feature values were missing or undefined and "
                         f"were replaced by the source median of their column (see provenance)")
        if self.verdict == "UNCALIBRATED":
            lines.append("verdict    = UNCALIBRATED -- no null floor supplied. A raw QFC cannot be read on its own; "
                         "use screen(), or pass null_floor=compute_null_floor(...).")
            return "\n".join(lines)
        if self.verdict == "UNDEFINED":
            lines.append("verdict    = UNDEFINED -- every no-shift draw gave the same value, so the floor has no "
                         "spread and no z exists.")
            return "\n".join(lines)
        seeds = f"{self.n_null_seeds} seeds"
        req = (self.provenance or {}).get("null_seeds_requested")
        if req and self.n_null_seeds and self.n_null_seeds < req:
            skips = (self.null_floor or {}).get("skips") or {}
            seeds = f"{self.n_null_seeds}/{req} usable seeds"
            if skips:
                seeds += " (skipped: %s)" % ", ".join("%s %d" % kv for kv in sorted(skips.items()))
            if self.n_null_seeds < 0.8 * req:
                seeds += ", MANY UNUSABLE -- floor may be optimistic"
        lines.append(f"null floor = {self.null_mean:+.4f} +/- {self.null_sd:.4f} ({seeds})   "
                     f"z = {self.z:+.2f}   cut = +/-{self.exact_cut:.4f}   resolution = {self.resolution:.4f}")
        if self.chance_floor_mean is not None:
            lines.append(f"chance     = {self.chance_floor_mean:+.4f} +/- {self.chance_floor_sd:.4f} "
                         f"({self.n_chance_draws} permuted-label draws)")
        lines.append(f"verdict    = {self.verdict}  -- {_MEANING[self.verdict]}")
        if self.adjudication:
            lines.append(f"             {self.adjudication}")
        if self.n_distinct_splits and self.n_null_seeds and self.n_distinct_splits < self.n_null_seeds:
            lines.append(f"             [WARNING: only {self.n_distinct_splits} distinct shot splits among "
                         f"{self.n_null_seeds} draws; at that count the cut is +/-{self.cut_effective:.4f} and the "
                         f"reading is {self.verdict_effective} (the verdict above uses the nominal count, as the "
                         f"paper does)]")
        if self.verdict_per_order and len(set(self.verdict_per_order)) > 1:
            c = {v: self.verdict_per_order.count(v) for v in sorted(set(self.verdict_per_order))}
            lines.append("             fold orders disagree on their own: %s; the verdict uses their mean"
                         % ", ".join("%s %d" % kv for kv in c.items()))
        return "\n".join(lines)


_MEANING = {
    "COLLAPSE": "the two environments agree on feature importance LESS than two halves of the source do; the "
                "explanation changed under the shift by more than no-shift sampling explains",
    "INDISTINGUISHABLE": "the agreement is within what two halves of the source give each other; no change is "
                         "detected (the resolution says how large a change could have been seen)",
    "ROBUST": "the agreement is HIGHER than two halves of the source give each other, beyond the cut",
}


# --------------------------------------------------------------------------------------- internals


def _validate(df, shot_col: str, name: str, label_col: Optional[str] = None, label_optional: bool = False) -> None:
    """Check a frame. ``label_col=None`` skips the label checks. ``label_optional=True`` is for a target frame:
    the measurement never uses target labels, so missing label values are allowed (they are left out of the
    AUC), but values that are present must be binary."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame, got {type(df).__name__}")
    required = [shot_col] + ([label_col] if label_col is not None and not label_optional else [])
    for col in required:
        if col not in df.columns:
            raise ValueError(
                f"{name} has no column {col!r}. Columns are {list(df.columns)[:12]}"
                f"{' ...' if len(df.columns) > 12 else ''}. "
                f"Pass label_col=/shot_col= if your frame uses different names "
                f"(disruption-py's DataFrame output uses 'shot')."
            )
    if len(df) == 0:
        raise ValueError(f"{name} is empty")
    if df[shot_col].isna().any():
        raise ValueError(f"{name}[{shot_col!r}] has missing values; every row must belong to a shot")
    if label_col is not None and label_col in df.columns:
        labels = pd.unique(df[label_col].dropna())
        if not set(np.asarray(labels).ravel().tolist()) <= {0, 1, 0.0, 1.0, True, False}:
            raise ValueError(f"{name}[{label_col!r}] must be binary 0/1; found values {labels[:8]}")
        if not label_optional and df[label_col].isna().any():
            raise ValueError(f"{name}[{label_col!r}] has missing values; drop or label those rows first")


def detect_feature_cols(df: pd.DataFrame, label_col: str = "label", shot_col: str = "shot") -> list:
    """Numeric columns excluding the label, the shot identifier and ``DEFAULT_EXCLUDE_COLS``. Check the list
    (it is in every result as ``feature_cols_used``): any other time-to-event or bookkeeping column in your
    frame would be used too, so pass ``feature_cols`` explicitly when in doubt."""
    exclude = set(DEFAULT_EXCLUDE_COLS) | {label_col, shot_col}
    return [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]


def _features(source_df, target_df, feature_cols, label_col, shot_col):
    if feature_cols is None:
        feature_cols = detect_feature_cols(source_df, label_col, shot_col)
    feature_cols = list(feature_cols)
    if len(feature_cols) < 3:
        raise ValueError(f"need at least 3 features for a rank correlation, got {len(feature_cols)}; "
                         "pass feature_cols explicitly")
    for name, df in (("source_df", source_df), ("target_df", target_df)):
        if df is None:
            continue
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"{name} is missing feature columns: {missing[:8]}")
        bad = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(df[c])]
        if bad:
            raise ValueError(f"{name} has non-numeric feature columns: {bad[:8]}")
        if np.isinf(df[feature_cols].to_numpy(dtype=float)).any():
            raise ValueError(f"{name} has infinite feature values; replace or drop them first")
    return feature_cols


def _source_medians(source_df: pd.DataFrame, feature_cols: Sequence[str]) -> np.ndarray:
    """The source frame's median of each feature column, over its non-missing values."""
    med = source_df[list(feature_cols)].median()
    empty = [c for c in feature_cols if pd.isna(med[c])]
    if empty:
        raise ValueError(f"source_df has feature columns with no values at all: {empty[:8]}; drop them")
    return med.to_numpy(dtype=float)


def _matrix(df: pd.DataFrame, feature_cols: Sequence[str], medians: np.ndarray) -> tuple:
    """Feature matrix with every missing value replaced by the SOURCE frame's median of its column (an
    undefined window statistic, such as the skew of a window with no measurable spread, or a missing value).
    Returns ``(X, {column: number replaced})``."""
    X = df[list(feature_cols)].to_numpy(dtype=float)
    miss = np.isnan(X)
    if miss.any():
        X = np.where(miss, medians[None, :], X)
    return X, {c: int(n) for c, n in zip(feature_cols, miss.sum(axis=0)) if n}


def _arrays(df: pd.DataFrame, feature_cols: Sequence[str], label_col: str, shot_col: str, medians=None):
    """``(X, y, groups)`` of a source frame, missing feature values replaced by its own column medians."""
    medians = _source_medians(df, feature_cols) if medians is None else medians
    return _matrix(df, feature_cols, medians)[0], df[label_col].to_numpy().astype(int), df[shot_col].to_numpy()


def _model(model, random_state):
    model = model if model is not None else default_model(random_state)
    if not is_classifier(model) or not hasattr(model, "predict_proba"):
        raise TypeError("model must be a scikit-learn classifier exposing predict_proba")
    return model


def _check_settings(shap_cap, n_splits, gs):
    if shap_cap is not None and (isinstance(shap_cap, bool) or int(shap_cap) != shap_cap or shap_cap < 1):
        raise ValueError(f"shap_cap must be a positive integer or None, got {shap_cap!r}")
    if n_splits < 2:
        raise ValueError(f"n_splits must be at least 2, got {n_splits}")
    if len(np.unique(gs)) < 2:
        raise ValueError("the source has fewer than two shots; an out-of-fold vector needs at least two")


def _provenance(model, seed, shap_cap, n_splits, attribution, **extra) -> dict:
    from . import __version__

    p = {"package": "disruptionpy-qfc", "version": __version__, "estimator": repr(model), "random_state": seed,
         "attribution": attribution, "shap_cap": shap_cap, "cap_rule": "seeded random subsample without replacement",
         "n_splits": n_splits, "generated": time.strftime("%Y-%m-%d %H:%M:%S")}
    p.update(extra)
    return p


def _block_bootstrap_index(groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Shot-level bootstrap **with multiplicity**: unique shots drawn with replacement, each drawn shot
    contributing all of its rows again, so a shot drawn twice appears twice."""
    uniq = np.unique(groups)
    drawn = rng.choice(uniq, size=len(uniq), replace=True)
    by_group = {g: np.flatnonzero(groups == g) for g in uniq}
    return np.concatenate([by_group[g] for g in drawn]) if len(drawn) else np.array([], dtype=int)


# --------------------------------------------------------------------------------------- public API


def compute_qfc(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    *,
    feature_cols: Optional[Sequence[str]] = None,
    model=None,
    attribution: AttributionSpec = "auto",
    shap_cap: Optional[int] = 500,
    n_splits: int = 5,
    n_fold_orders: int = N_FOLD_ORDERS,
    n_bootstrap: int = 0,
    random_state: int = 42,
    label_col: str = "label",
    shot_col: str = "shot",
    null_floor: Optional[NullFloor] = None,
    chance_floor: Optional[ChanceFloor] = None,
) -> QFCResult:
    """The measurement: rank agreement between the source and target mean-|attribution| vectors.

    Parameters
    ----------
    source_df, target_df
        DisruptionPy-format frames: one row per (shot, time), a binary ``label_col`` and a ``shot_col``.
        The target's label is optional (missing values allowed) and used only for the reported AUC.
    feature_cols
        Defaults to :func:`detect_feature_cols`.
    model
        Any scikit-learn classifier with ``predict_proba``; cloned for every fit, with its ``random_state``
        set when it has one (an estimator nested inside a Pipeline is not re-seeded). Fitted on numpy arrays,
        so steps that select columns by name will not work. Defaults to :func:`default_model`.
    attribution
        ``"auto"`` (TreeSHAP for tree models, KernelSHAP otherwise), ``"treeshap"``, ``"saabas"``
        (tree-path contributions, the method DPRF uses), ``"kernelshap"``, ``"permutation"``, or your own
        callable ``f(model, X_explain, X_train, seed) -> array (n_rows, n_features)``.
    shap_cap
        Rows explained per attribution vector, a seeded random subsample. ``None`` explains every row.
    n_splits
        ``GroupKFold`` folds for the out-of-fold source vector (fewer if the source has fewer shots).
    n_fold_orders
        Fold orders averaged for the reported QFC (default 10, the paper's primary scoring); 1 gives the
        single-order value. An order whose folds are identical to an earlier order's reuses its vector, so
        a source whose shots all differ in size costs one order, not ten.
    n_bootstrap
        Shot-level bootstrap draws for a 95% interval of the fold-order-0 value (default 0, off). Each draw
        resamples shots and refits with its own seed, so the interval covers sampling and model-seed
        variability; it is not part of the verdict.
    null_floor, chance_floor
        From :func:`compute_null_floor` and :func:`compute_chance_floor`. Without a null floor the
        result is ``UNCALIBRATED``.
    """
    _validate(source_df, shot_col, "source_df", label_col=label_col)
    _validate(target_df, shot_col, "target_df", label_col=label_col, label_optional=True)
    feature_cols = _features(source_df, target_df, feature_cols, label_col, shot_col)
    model = _model(model, random_state)
    attr, attr_name = _resolve_attribution(attribution, model)
    if n_fold_orders < 1:
        raise ValueError("n_fold_orders must be at least 1")

    med = _source_medians(source_df, feature_cols)
    Xs, imp_s = _matrix(source_df, feature_cols, med)
    ys, gs = source_df[label_col].to_numpy().astype(int), source_df[shot_col].to_numpy()
    _check_settings(shap_cap, n_splits, gs)
    Xt, imp_t = _matrix(target_df, feature_cols, med)
    gt = target_df[shot_col].to_numpy()
    if len(np.unique(ys)) < 2:
        raise ValueError("source_df is single-class; nothing to learn")

    # the deployed model and the target vector do not depend on the fold order: computed once
    deployed = _fit(model, Xs, ys, random_state)
    tgt = _explain(deployed, Xt, Xs, shap_cap, random_state, attr)
    per_order, cache, skipped, src0 = [], {}, 0, None
    for k in range(n_fold_orders):
        folds = _folds(ys, gs, n_splits, k)
        key = tuple(tuple(te.tolist()) for _, te in folds)
        if key not in cache:
            cache[key] = _source_vector(Xs, ys, folds, model, shap_cap, random_state, attr)
            skipped += cache[key][1]
        src = cache[key][0]
        src0 = src if k == 0 else src0
        per_order.append(qfc_statistic(src, tgt))
    qfc = float(np.mean(per_order))

    auc = None
    if label_col in target_df.columns:
        lab = target_df[label_col].notna().to_numpy()
        yt = target_df[label_col].to_numpy()[lab].astype(int)
        if len(np.unique(yt)) > 1:
            auc = float(roc_auc_score(yt, deployed.predict_proba(Xt[lab])[:, 1]))

    ci_lower = ci_upper = None
    used = attempted = 0
    if n_bootstrap and n_bootstrap > 0:
        seeds = np.random.default_rng(random_state).integers(0, 2**31 - 1, size=n_bootstrap)
        draws = []
        for s in seeds:
            attempted += 1
            rng = np.random.default_rng(int(s))
            idx = _block_bootstrap_index(gs, rng)
            t_idx = _block_bootstrap_index(gt, rng)  # both sides resample whole shots
            if len(np.unique(ys[idx])) < 2 or len(np.unique(gs[idx])) < n_splits or len(t_idx) == 0:
                continue
            try:
                a, b, _, _ = _real_arm(Xs[idx], ys[idx], gs[idx], Xt[t_idx], model, n_splits, shap_cap, int(s), attr)
                draws.append(qfc_statistic(a, b))
            except DegenerateDrawError:
                continue
        used = len(draws)
        if draws:
            ci_lower, ci_upper = float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))

    r = QFCResult(
        qfc=qfc, qfc_per_order=[float(v) for v in per_order], auc=auc,
        n_source=int(len(source_df)), n_target=int(len(target_df)),
        n_shots_source=int(len(np.unique(gs))), n_shots_target=int(len(np.unique(gt))),
        feature_cols_used=feature_cols, source_importance=[float(v) for v in src0],
        target_importance=[float(v) for v in tgt], attribution=attr_name,
        ci_lower=ci_lower, ci_upper=ci_upper, n_bootstrap_used=used, n_bootstrap_attempted=attempted,
        provenance=_provenance(model, random_state, shap_cap, n_splits, attr_name, n_fold_orders=n_fold_orders,
                               n_distinct_fold_orders=len(cache), n_bootstrap=n_bootstrap,
                               bootstrap_rule="shot-level with multiplicity, fold order 0",
                               folds_skipped_single_class=skipped,
                               imputed_with_source_median={"source": imp_s, "target": imp_t}),
    )
    if null_floor is not None:
        _score(r, null_floor, chance_floor)
    return r


def _score(r: QFCResult, floor: NullFloor, chance: Optional[ChanceFloor]) -> None:
    """Apply the verdict rules to a result in place: z and the cut, the per-order readings, the
    distinct-split reading, and the chance-floor rule for a ROBUST reading."""
    r.null_mean, r.null_sd, r.n_null_seeds = floor.mean, floor.sd, floor.n_seeds
    r.exact_cut, r.n_distinct_splits = floor.exact_cut, floor.n_distinct_splits
    r.null_floor, r.chance_floor = floor.to_dict(), (chance.to_dict() if chance is not None else None)
    r.provenance = dict(r.provenance, null_seeds_requested=floor.n_seeds_requested)
    z, v = floor.verdict_for(r.qfc)
    r.z, r.verdict_unadjudicated, r.verdict, r.adjudication = z, v, v, None
    r.chance_floor_mean = r.chance_floor_sd = r.n_chance_draws = None
    if v == "UNDEFINED":
        r.z = None
        return
    r.resolution = float(floor.exact_cut * floor.sd)
    r.verdict_per_order = [floor.verdict_for(q)[1] for q in r.qfc_per_order]
    r.cut_effective = floor.cut_effective if floor.cut_effective is not None else floor.exact_cut
    r.verdict_effective = verdict_label(z, r.cut_effective)
    if chance is not None:
        r.chance_floor_mean, r.chance_floor_sd, r.n_chance_draws = chance.mean, chance.sd, chance.n_draws
        if v == "ROBUST" and floor.mean < chance.mean:
            r.verdict = "INDISTINGUISHABLE"
            r.adjudication = ("read INDISTINGUISHABLE, not ROBUST: the no-shift floor %.4f lies below the chance "
                              "floor %.4f, so two halves of the source agree less than a model fitted to permuted "
                              "labels does, and agreement above that floor shows nothing" % (floor.mean, chance.mean))
    elif v == "ROBUST":
        r.adjudication = ("ROBUST is provisional: no chance floor was supplied. screen() computes one automatically; "
                          "or pass chance_floor=compute_chance_floor(...)")


def compute_null_floor(
    source_df: pd.DataFrame,
    *,
    feature_cols: Optional[Sequence[str]] = None,
    model=None,
    attribution: AttributionSpec = "auto",
    n_seeds: int = N_NULL_SEEDS,
    shap_cap: Optional[int] = 500,
    n_splits: int = 5,
    random_state: int = 42,
    label_col: str = "label",
    shot_col: str = "shot",
) -> NullFloor:
    """The no-shift reference: the same measurement between two halves of the source pool.

    Draw ``k`` splits the source shots 50/50 with ``GroupShuffleSplit(random_state=random_state + k)``;
    the first half is fitted and explained out of fold, as the source is in the measurement, and a model
    fitted on the whole first half explains the second half, as the deployed model explains the target.
    This is the construction of the paper's reference implementation. Two things differ from the
    measurement besides the data: each half holds half the source's shots, and each draw uses one fold
    order where the measurement averages ten. A draw is skipped, and counted by reason, when its first half is
    single-class, holds fewer shots than folds, has every fold single-class, or gives a constant importance
    vector; any other error is raised. The cut is computed from the draws actually used. If two seeds drew
    the same partition of shots the floor records how many distinct partitions it holds, warns, and gives
    the cut at that count for ``verdict_effective``; the verdict itself uses the nominal count, as the paper's
    census does.
    """
    _validate(source_df, shot_col, "source_df", label_col=label_col)
    feature_cols = _features(source_df, None, feature_cols, label_col, shot_col)
    model = _model(model, random_state)
    attr, attr_name = _resolve_attribution(attribution, model)
    med = _source_medians(source_df, feature_cols)
    Xs, imp_s = _matrix(source_df, feature_cols, med)
    ys, gs = source_df[label_col].to_numpy().astype(int), source_df[shot_col].to_numpy()
    _check_settings(shap_cap, n_splits, gs)

    vals, skips, splits, folds_skipped = [], {}, set(), 0

    def _skip(reason):
        skips[reason] = skips.get(reason, 0) + 1

    for k in range(n_seeds):
        a, b = next(GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=random_state + k).split(Xs, ys, gs))
        if len(np.unique(ys[a])) < 2:
            _skip("single_class_half")
            continue
        if len(np.unique(gs[a])) < n_splits:
            _skip("too_few_shots_to_fold")
            continue
        try:
            s, t, _, fsk = _real_arm(Xs[a], ys[a], gs[a], Xs[b], model, n_splits, shap_cap, random_state + k, attr)
            vals.append(qfc_statistic(s, t))
            folds_skipped += fsk
        except DegenerateDrawError as e:
            _skip("constant_importance" if "constant" in str(e) else "all_folds_single_class")
            continue
        # a split and its mirror are one partition of the shots
        splits.add(frozenset([frozenset(np.unique(gs[a]).tolist()), frozenset(np.unique(gs[b]).tolist())]))
    if len(vals) < 3:
        raise ValueError(
            f"only {len(vals)} usable null draws from {n_seeds} seeds (skipped: {skips or 'none'}); at least "
            "three are needed. The source pool is too small or too imbalanced to calibrate against."
        )
    arr = np.asarray(vals, dtype=float)
    n, nd = len(arr), len(splits)
    if nd < n:
        warnings.warn("only %d distinct shot splits among %d null draws (the source pool has %d shots); the floor's "
                      "effective sample size is %d -- see verdict_effective" % (nd, n, len(np.unique(gs)), nd),
                      stacklevel=2)
    return NullFloor(
        mean=float(arr.mean()), sd=float(arr.std(ddof=1)), per_seed=[float(v) for v in arr], n_seeds=n,
        exact_cut=exact_cut(n), n_distinct_splits=nd,
        cut_effective=(exact_cut(nd) if nd >= 3 else float("inf")) if nd < n else exact_cut(n),
        provenance=_provenance(model, random_state, shap_cap, n_splits, attr_name,
                               construction="GroupShuffleSplit 50/50 of the source pool by shot; the measurement's "
                                            "procedure between the halves at fold order 0; zero shift",
                               folds_skipped_single_class=folds_skipped,
                               imputed_with_source_median={"source": imp_s}),
        n_seeds_requested=int(n_seeds), skips=skips,
    )


def compute_chance_floor(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    *,
    feature_cols: Optional[Sequence[str]] = None,
    model=None,
    attribution: AttributionSpec = "auto",
    n_draws: int = N_CHANCE_DRAWS,
    shap_cap: Optional[int] = 500,
    n_splits: int = 5,
    label_col: str = "label",
    shot_col: str = "shot",
) -> ChanceFloor:
    """The permuted-label chance floor: draw ``d`` (``d = 0 .. n_draws-1``) permutes the source labels with
    ``np.random.default_rng(d)``, fits with seed ``d``, and runs the measurement source to target at fold
    order 0. These seeds are the paper's, fixed; they do not derive from a ``random_state``.

    The paper applies it to a ROBUST reading only: if the no-shift floor lies below this, the reading is
    INDISTINGUISHABLE (two halves of the source agree less than a model trained on noise).
    """
    _validate(source_df, shot_col, "source_df", label_col=label_col)
    _validate(target_df, shot_col, "target_df", label_col=None)
    feature_cols = _features(source_df, target_df, feature_cols, label_col, shot_col)
    model = _model(model, 0)
    attr, attr_name = _resolve_attribution(attribution, model)
    med = _source_medians(source_df, feature_cols)
    Xs, ys, gs = _arrays(source_df, feature_cols, label_col, shot_col, med)
    _check_settings(shap_cap, n_splits, gs)
    Xt = _matrix(target_df, feature_cols, med)[0]
    vals, skips = [], {}
    for d in range(n_draws):
        yp = np.random.default_rng(d).permutation(ys)
        try:
            s, t, _, _ = _real_arm(Xs, yp, gs, Xt, model, n_splits, shap_cap, d, attr)
            vals.append(qfc_statistic(s, t))
        except DegenerateDrawError as e:
            key = "constant_importance" if "constant" in str(e) else "all_folds_single_class"
            skips[key] = skips.get(key, 0) + 1
    if len(vals) < 2:
        raise ValueError(f"only {len(vals)} usable chance-floor draws from {n_draws} (skipped: {skips or 'none'})")
    arr = np.asarray(vals, dtype=float)
    return ChanceFloor(mean=float(arr.mean()), sd=float(arr.std(ddof=1)), per_draw=[float(v) for v in arr],
                       n_draws=len(arr), n_draws_requested=int(n_draws), skips=skips,
                       provenance=_provenance(model, None, shap_cap, n_splits, attr_name,
                                              model_seeds="d for draw d", permutation_seeds="d for draw d",
                                              construction="source labels permuted by default_rng(d), the "
                                                           "measurement source to target at fold order 0"))


def screen(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    *,
    n_seeds: int = N_NULL_SEEDS,
    chance_floor: Union[str, bool] = "auto",
    n_chance_draws: int = N_CHANCE_DRAWS,
    **kwargs,
) -> QFCResult:
    """Null floor, measurement and verdict in one call. This is the intended entry point.

    ``kwargs`` go to both :func:`compute_null_floor` and :func:`compute_qfc`, so the floor is built with
    exactly the model, features, attribution, cap and protocol used for the measurement.
    ``chance_floor="auto"`` builds the chance floor only when the reading is ROBUST, the one case its rule
    can change; ``True`` always builds it; ``False`` never does (a ROBUST reading is then marked provisional).
    Both floors are returned in full on the result (``null_floor``, ``chance_floor``).
    """
    if chance_floor not in ("auto", True, False):
        raise ValueError("chance_floor must be 'auto', True or False")
    if "null_floor" in kwargs:
        raise TypeError("screen() builds the null floor itself; to supply one, call compute_qfc(..., null_floor=...)")
    floor_kw = {k: v for k, v in kwargs.items() if k not in ("n_bootstrap", "n_fold_orders", "null_floor",
                                                               "chance_floor")}
    floor = compute_null_floor(source_df, n_seeds=n_seeds, **floor_kw)
    r = compute_qfc(source_df, target_df, null_floor=floor, **kwargs)
    if chance_floor is True or (chance_floor == "auto" and r.verdict_unadjudicated == "ROBUST"):
        ch_kw = {k: v for k, v in floor_kw.items() if k != "random_state"}
        _score(r, floor, compute_chance_floor(source_df, target_df, n_draws=n_chance_draws, **ch_kw))
    return r
