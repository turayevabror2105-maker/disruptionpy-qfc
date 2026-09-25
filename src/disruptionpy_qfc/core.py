"""Attribution-stability screening for DisruptionPy-format shot data.

The question this answers, before deployment and without target labels:

    A model trained on the source environment still predicts well on the target.
    Does it predict well *for the same reasons*?

and, crucially, the follow-up question that a raw correlation cannot answer on its own:

    Is what we measured any different from what a no-shift split of the source alone would give?

See ``compute_qfc`` for the measurement, ``compute_null_floor`` for the calibration, and ``screen``
for both together, which is what most users want.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, t as student_t
from sklearn.base import clone, is_classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

__all__ = [
    "QFCResult",
    "NullFloor",
    "compute_qfc",
    "compute_null_floor",
    "screen",
    "default_model",
    "DEFAULT_EXCLUDE_COLS",
]

DEFAULT_EXCLUDE_COLS = frozenset({"shot", "shot_id", "time", "label"})
_TREE_MODEL_HINTS = ("Forest", "GradientBoosting", "DecisionTree", "ExtraTrees", "XGB", "LGBM", "CatBoost")


def default_model(random_state: int = 42) -> RandomForestClassifier:
    """The estimator used unless the caller supplies one.

    500 trees, depth 20, balanced class weights: the configuration used throughout the
    disruption-prediction literature this package is aimed at (e.g. random-forest disruption
    warnings on C-Mod, DIII-D and EAST), chosen for comparability rather than tuned here.
    """
    return RandomForestClassifier(
        n_estimators=500, max_depth=20, class_weight="balanced", random_state=random_state, n_jobs=-1
    )


# --------------------------------------------------------------------------------------- results


@dataclass
class NullFloor:
    """The no-shift reference for one source pool, model and protocol.

    Built by splitting the source pool in half **by shot** and running the identical measurement
    between the halves, so there is no environment shift by construction. Anything the real
    measurement scores must be read against this.
    """

    mean: float
    sd: float
    per_seed: list
    n_seeds: int
    exact_cut: float
    provenance: dict = field(default_factory=dict)

    def verdict_for(self, qfc: float) -> tuple:
        """Return ``(z, verdict)`` for an observed QFC under this floor."""
        if self.sd <= 0 or not np.isfinite(self.sd):
            return float("nan"), "UNCALIBRATED"
        z = (qfc - self.mean) / self.sd
        if z < -self.exact_cut:
            return z, "COLLAPSE"
        if z > self.exact_cut:
            return z, "ROBUST"
        return z, "INDISTINGUISHABLE"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class QFCResult:
    """One screening result. ``to_dict()`` is JSON-serialisable."""

    qfc: float
    ci_lower: float
    ci_upper: float
    auc: Optional[float]
    n_source: int
    n_target: int
    n_shots_source: int
    n_shots_target: int
    feature_cols_used: list
    source_importance: list
    target_importance: list
    n_bootstrap_used: int = 0
    n_bootstrap_attempted: int = 0
    null_mean: Optional[float] = None
    null_sd: Optional[float] = None
    n_null_seeds: Optional[int] = None
    z: Optional[float] = None
    exact_cut: Optional[float] = None
    verdict: str = "UNCALIBRATED"
    threshold: Optional[float] = None
    threshold_flag: Optional[str] = None
    provenance: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        ci = f"95% CI [{self.ci_lower:+.4f}, {self.ci_upper:+.4f}]" if self.ci_lower == self.ci_lower else "95% CI n/a"
        if self.n_bootstrap_attempted:
            # Draws are skipped when a bootstrap sample turns out single-class or too few shots to
            # fold. Reporting only the interval would let a CI built from a handful of usable draws
            # pass for one built from all of them, so the count is always shown.
            ci += f" from {self.n_bootstrap_used}/{self.n_bootstrap_attempted} usable shot-bootstrap draws"
            if self.n_bootstrap_used < 0.8 * self.n_bootstrap_attempted:
                ci += "  [WARNING: many draws unusable -- interval is unreliable]"
        lines = [
            f"QFC        = {self.qfc:+.4f}  {ci}",
            f"target AUC = {self.auc:.4f}" if self.auc is not None else "target AUC = n/a (single-class target)",
            f"n          = {self.n_source} rows / {self.n_shots_source} shots  ->  "
            f"{self.n_target} rows / {self.n_shots_target} shots",
        ]
        if self.verdict == "UNCALIBRATED":
            lines.append(
                "verdict    = UNCALIBRATED -- no null floor supplied. A raw QFC cannot be read on its "
                "own; call compute_null_floor() or use screen()."
            )
        else:
            lines.append(
                f"null floor = {self.null_mean:+.4f} +/- {self.null_sd:.4f} "
                f"({self.n_null_seeds} seeds)   z = {self.z:+.2f}   cut = +/-{self.exact_cut:.3f}"
            )
            lines.append(f"verdict    = {self.verdict}")
        if self.threshold_flag is not None:
            lines.append(
                f"threshold  = {self.threshold_flag} against the fixed cut {self.threshold} "
                "(a borrowed constant -- prefer the null-relative verdict above)"
            )
        return "\n".join(lines)


# --------------------------------------------------------------------------------------- internals


def _validate(df: pd.DataFrame, shot_col: str, name: str, label_col: Optional[str] = None) -> None:
    """Check a frame. ``label_col=None`` skips the label checks, which is correct for a target
    frame: the measurement never uses target labels, and an unlabelled target is the normal case."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame, got {type(df).__name__}")
    required = [shot_col] + ([label_col] if label_col is not None else [])
    for col in required:
        if col not in df.columns:
            raise ValueError(
                f"{name} has no column {col!r}. Columns are {list(df.columns)[:12]}"
                f"{' ...' if len(df.columns) > 12 else ''}. "
                f"Pass label_col=/shot_col= if your frame uses different names "
                f"(disruption-py's DataFrameOutputSetting uses 'shot')."
            )
    if len(df) == 0:
        raise ValueError(f"{name} is empty")
    if label_col is not None:
        labels = pd.unique(df[label_col].dropna())
        if not set(np.asarray(labels).ravel().tolist()) <= {0, 1, 0.0, 1.0, True, False}:
            raise ValueError(f"{name}[{label_col!r}] must be binary 0/1; found values {labels[:8]}")


def detect_feature_cols(df: pd.DataFrame, label_col: str = "label", shot_col: str = "shot") -> list:
    """Numeric columns excluding the label, the shot identifier and time."""
    exclude = set(DEFAULT_EXCLUDE_COLS) | {label_col, shot_col}
    return [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]


def _is_tree_model(model) -> bool:
    return any(h in type(model).__name__ for h in _TREE_MODEL_HINTS)


def _subsample(n: int, cap: Optional[int], rng: np.random.Generator) -> np.ndarray:
    """Seeded random subsample of row positions, never the first `cap` rows.

    Rows in DisruptionPy frames are ordered by shot and time, so taking a head slice would explain
    the first few discharges rather than a sample of the environment.
    """
    if cap is None or n <= cap:
        return np.arange(n)
    return np.sort(rng.choice(n, size=cap, replace=False))


def _tree_shap(model, X: np.ndarray) -> np.ndarray:
    import shap  # imported lazily: only tree models need it

    sv = shap.TreeExplainer(model).shap_values(X, check_additivity=False)
    if isinstance(sv, list):  # older shap: one array per class
        sv = sv[1] if len(sv) == 2 else sv[0]
    elif hasattr(sv, "values"):  # Explanation object
        v = sv.values
        sv = v[..., 1] if v.ndim == 3 and v.shape[2] == 2 else v
    elif isinstance(sv, np.ndarray) and sv.ndim == 3:
        sv = sv[:, :, 1] if sv.shape[2] == 2 else sv[:, :, 0]
    return np.abs(np.asarray(sv)).mean(axis=0)


def _permutation_importance_unlabelled(model, X: np.ndarray, rng: np.random.Generator, n_repeats: int = 5) -> np.ndarray:
    """Label-free permutation importance: mean |change in predicted probability| per feature.

    Used when the estimator is not tree-based. Deliberately does not use labels, because the target
    environment has none at screening time -- a score-drop permutation importance would not be
    computable in the setting this package exists for.
    """
    base = model.predict_proba(X)[:, 1]
    imp = np.zeros(X.shape[1], dtype=float)
    for j in range(X.shape[1]):
        acc = 0.0
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[:, j] = Xp[rng.permutation(len(Xp)), j]
            acc += float(np.mean(np.abs(model.predict_proba(Xp)[:, 1] - base)))
        imp[j] = acc / n_repeats
    return imp


def _importance(model, X: np.ndarray, cap: Optional[int], rng: np.random.Generator) -> np.ndarray:
    Xs = X[_subsample(len(X), cap, rng)]
    return _tree_shap(model, Xs) if _is_tree_model(model) else _permutation_importance_unlabelled(model, Xs, rng)


def _block_bootstrap_index(groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Shot-level bootstrap **with multiplicity**.

    Unique shots are drawn with replacement and each draw contributes that shot's rows again, so a
    shot drawn twice appears twice. A membership mask would silently collapse duplicates and turn
    the bootstrap into a ~63% subsample -- which is what version 1 of this package did.
    """
    uniq = np.unique(groups)
    drawn = rng.choice(uniq, size=len(uniq), replace=True)
    by_group = {g: np.flatnonzero(groups == g) for g in uniq}
    return np.concatenate([by_group[g] for g in drawn]) if len(drawn) else np.array([], dtype=int)


def _fit(model, X: np.ndarray, y: np.ndarray, seed: int):
    est = clone(model)
    if "random_state" in est.get_params():
        est.set_params(random_state=seed)
    return est.fit(X, y)


def _qfc_once(
    Xs: np.ndarray, ys: np.ndarray, gs: np.ndarray, Xt: np.ndarray, yt: Optional[np.ndarray],
    model, n_splits: int, shap_cap: Optional[int], seed: int,
) -> tuple:
    """One source->target measurement. Returns (qfc, auc, source_importance, target_importance).

    Source vector: mean over GroupKFold out-of-fold importance vectors, so no shot is ever explained
    by a model that saw it. Target vector: a model fit on the *whole* source and applied to the
    target without retraining -- the deploy-once protocol, which is how a predictor is actually used
    on a new campaign or machine.
    """
    rng = np.random.default_rng(seed)
    n_splits = max(2, min(n_splits, len(np.unique(gs))))
    fold_vecs = []
    for tr, te in GroupKFold(n_splits=n_splits).split(Xs, ys, gs):
        if len(np.unique(ys[tr])) < 2:
            continue
        fold_vecs.append(_importance(_fit(model, Xs[tr], ys[tr], seed), Xs[te], shap_cap, rng))
    if not fold_vecs:
        raise ValueError("every cross-validation fold was single-class; cannot compute a source vector")
    src = np.mean(fold_vecs, axis=0)

    deployed = _fit(model, Xs, ys, seed)
    tgt = _importance(deployed, Xt, shap_cap, rng)

    auc = None
    if yt is not None and len(np.unique(yt)) > 1:
        auc = float(roc_auc_score(yt, deployed.predict_proba(Xt)[:, 1]))
    return float(spearmanr(src, tgt).statistic), auc, src, tgt


def _arrays(df: pd.DataFrame, feature_cols: Sequence[str], label_col: str, shot_col: str):
    return (
        df[list(feature_cols)].to_numpy(dtype=float),
        df[label_col].to_numpy().astype(int),
        df[shot_col].to_numpy(),
    )


def _provenance(model, seed: int, shap_cap, n_bootstrap, n_splits) -> dict:
    from . import __version__

    return {
        "package": "disruptionpy-qfc",
        "version": __version__,
        "estimator": repr(model),
        "random_state": seed,
        "shap_cap": shap_cap,
        "cap_rule": "seeded random subsample without replacement",
        "bootstrap_rule": "shot-level with multiplicity",
        "n_bootstrap": n_bootstrap,
        "n_splits": n_splits,
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


# --------------------------------------------------------------------------------------- public API


def compute_qfc(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    *,
    feature_cols: Optional[Sequence[str]] = None,
    model=None,
    n_bootstrap: int = 200,
    shap_cap: Optional[int] = 500,
    n_splits: int = 5,
    random_state: int = 42,
    label_col: str = "label",
    shot_col: str = "shot",
    null_floor: Optional[NullFloor] = None,
    threshold: Optional[float] = None,
) -> QFCResult:
    """Rank agreement between source and target mean-|attribution| vectors.

    Parameters
    ----------
    source_df, target_df
        DisruptionPy-format frames: one row per (shot, time), a binary ``label_col`` and a
        ``shot_col`` identifying the discharge. ``target_df``'s label is optional and used only for
        the reported AUC -- the measurement itself never uses target labels.
    feature_cols
        Defaults to every numeric column except the label, shot and time columns.
    model
        Any scikit-learn classifier; cloned and re-seeded per fit. Defaults to
        :func:`default_model`. Tree models are explained with TreeSHAP, others with a label-free
        permutation importance.
    n_bootstrap
        Shot-level bootstrap iterations for the confidence interval, each repeating the whole
        procedure. 200 is the default; below ~100 the percentile ends are unstable. 0 skips it.
    shap_cap
        Rows explained per attribution vector, drawn as a seeded random subsample. ``None`` explains
        everything (expensive: TreeSHAP cost grows with rows x trees x depth).
    null_floor
        A :class:`NullFloor` from :func:`compute_null_floor`. Without it the result is returned with
        ``verdict="UNCALIBRATED"``, because a raw QFC cannot be interpreted on its own.
    threshold
        Optional fixed cut, reported as ``threshold_flag``. Provided only for continuity with
        published fixed-threshold scales; the null-relative verdict is the one to use.

    Returns
    -------
    QFCResult
    """
    _validate(source_df, shot_col, "source_df", label_col=label_col)
    _validate(target_df, shot_col, "target_df", label_col=label_col if label_col in target_df.columns else None)
    if feature_cols is None:
        feature_cols = detect_feature_cols(source_df, label_col, shot_col)
    if not feature_cols:
        raise ValueError("no usable feature columns found; pass feature_cols explicitly")
    missing = [c for c in feature_cols if c not in target_df.columns]
    if missing:
        raise ValueError(f"target_df is missing feature columns present in source_df: {missing[:8]}")
    if len(feature_cols) < 3:
        raise ValueError(f"need at least 3 features for a rank correlation, got {len(feature_cols)}")

    model = model if model is not None else default_model(random_state)
    if not is_classifier(model):
        raise TypeError("model must be a scikit-learn classifier exposing predict_proba")

    Xs, ys, gs = _arrays(source_df, feature_cols, label_col, shot_col)
    Xt = target_df[list(feature_cols)].to_numpy(dtype=float)
    yt = target_df[label_col].to_numpy().astype(int) if label_col in target_df.columns else None
    gt = target_df[shot_col].to_numpy()
    if len(np.unique(ys)) < 2:
        raise ValueError("source_df is single-class; nothing to learn")

    qfc, auc, src_imp, tgt_imp = _qfc_once(Xs, ys, gs, Xt, yt, model, n_splits, shap_cap, random_state)

    ci_lower = ci_upper = float("nan")
    n_draws_used = n_draws_attempted = 0
    if n_bootstrap and n_bootstrap > 0:
        master = np.random.default_rng(random_state)
        seeds = master.integers(0, 2**31 - 1, size=n_bootstrap)
        draws = []
        for s in seeds:
            n_draws_attempted += 1
            rng = np.random.default_rng(int(s))
            idx = _block_bootstrap_index(gs, rng)
            if len(np.unique(ys[idx])) < 2 or len(np.unique(gs[idx])) < n_splits:
                continue
            # BOTH sides resample at SHOT level. This used to draw target ROWS with
            # rng.integers(0, len(Xt), size=len(Xt)), which fragments discharges: rows from the same
            # shot are strongly correlated, so a row resample treats N rows as N independent units
            # when there are only N_shots. Measured on a 40-shot x 25-step frame, the row version
            # left all 40 shots present with 14-36 rows each instead of whole shots drawn with
            # multiplicity -- an interval computed on 1,000 "independent" units instead of 40.
            # The discharge is the independent unit in this domain; the interval must say so.
            t_idx = _block_bootstrap_index(gt, rng)
            if len(t_idx) == 0:
                continue
            try:
                draws.append(_qfc_once(Xs[idx], ys[idx], gs[idx], Xt[t_idx], None, model, n_splits, shap_cap, int(s))[0])
            except ValueError:
                continue
        n_draws_used = len(draws)
        if draws:
            ci_lower, ci_upper = (float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)))

    result = QFCResult(
        qfc=qfc, ci_lower=ci_lower, ci_upper=ci_upper, auc=auc,
        n_source=int(len(source_df)), n_target=int(len(target_df)),
        n_shots_source=int(len(np.unique(gs))), n_shots_target=int(len(np.unique(gt))),
        feature_cols_used=list(feature_cols),
        source_importance=[float(v) for v in src_imp], target_importance=[float(v) for v in tgt_imp],
        n_bootstrap_used=n_draws_used, n_bootstrap_attempted=n_draws_attempted,
        provenance=_provenance(model, random_state, shap_cap, n_bootstrap, n_splits),
    )
    if null_floor is not None:
        z, verdict = null_floor.verdict_for(qfc)
        result.null_mean, result.null_sd = null_floor.mean, null_floor.sd
        result.n_null_seeds, result.exact_cut = null_floor.n_seeds, null_floor.exact_cut
        result.z, result.verdict = float(z), verdict
    if threshold is not None:
        result.threshold = float(threshold)
        result.threshold_flag = "PASS" if qfc >= threshold else "FAIL"
    return result


def compute_null_floor(
    source_df: pd.DataFrame,
    *,
    feature_cols: Optional[Sequence[str]] = None,
    model=None,
    n_seeds: int = 30,
    shap_cap: Optional[int] = 500,
    n_splits: int = 5,
    random_state: int = 42,
    label_col: str = "label",
    shot_col: str = "shot",
) -> NullFloor:
    """The no-shift reference: the same measurement between two halves of the source pool.

    For each seed the source shots are split 50/50 (``GroupShuffleSplit``) and one half is measured
    against the other with the identical procedure, so any shift is zero by construction. The
    returned ``exact_cut`` is ``t(0.975, n-1) * sqrt(1 + 1/n)``, the Student-t prediction-interval
    cut for a single new observation, which is what an observed QFC is.

    Thirty seeds is the default because the cut is materially wider below that.
    """
    _validate(source_df, shot_col, "source_df", label_col=label_col)
    if feature_cols is None:
        feature_cols = detect_feature_cols(source_df, label_col, shot_col)
    model = model if model is not None else default_model(random_state)
    Xs, ys, gs = _arrays(source_df, feature_cols, label_col, shot_col)

    vals = []
    for k in range(n_seeds):
        a, b = next(GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=random_state + k).split(Xs, ys, gs))
        if len(np.unique(ys[a])) < 2 or len(np.unique(gs[a])) < n_splits:
            continue
        try:
            vals.append(_qfc_once(Xs[a], ys[a], gs[a], Xs[b], None, model, n_splits, shap_cap, random_state + k)[0])
        except ValueError:
            continue
    if len(vals) < 2:
        raise ValueError(
            f"only {len(vals)} usable null draws from {n_seeds} seeds; the source pool is too small "
            "or too imbalanced to calibrate against"
        )
    arr = np.asarray(vals, dtype=float)
    n = len(arr)
    return NullFloor(
        mean=float(arr.mean()), sd=float(arr.std(ddof=1)), per_seed=[float(v) for v in arr], n_seeds=n,
        exact_cut=float(student_t.ppf(0.975, n - 1) * np.sqrt(1 + 1 / n)),
        provenance=_provenance(model, random_state, shap_cap, 0, n_splits)
        | {"construction": "GroupShuffleSplit 50/50 of the source pool, zero real shift"},
    )


def screen(source_df: pd.DataFrame, target_df: pd.DataFrame, *, n_seeds: int = 30, **kwargs) -> QFCResult:
    """Null floor + QFC + verdict in one call. This is the intended entry point.

    ``kwargs`` are passed to both :func:`compute_null_floor` and :func:`compute_qfc`, so the floor is
    computed with exactly the model, features, cap and protocol used for the measurement -- which is
    the only way the comparison means anything.
    """
    floor_kwargs = {k: v for k, v in kwargs.items() if k not in ("n_bootstrap", "null_floor", "threshold")}
    floor = compute_null_floor(source_df, n_seeds=n_seeds, **floor_kwargs)
    return compute_qfc(source_df, target_df, null_floor=floor, **kwargs)
