#!/usr/bin/env python3

"""
Module for computing Quantified Feature Consistency (QFC) between a source
and target DisruptionPy-format shot dataset.

QFC measures whether a model's SHAP attribution structure (which features it
relies on, and how much) transfers across two environments/eras of a device
-- not just whether raw predictive performance transfers. A model can achieve
a high cross-environment AUC while relying on a different set of features to
get there; QFC is designed to catch that failure mode.

Requirements (not part of disruption-py's own dependency set -- install
separately):
    numpy
    pandas
    scikit-learn
    scipy
    shap
"""

import itertools

import numpy as np
import pandas as pd
import shap
from scipy.stats import skew, spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold


DEFAULT_EXCLUDE_COLS = {"shot", "shot_id", "time", "label"}


def _tree_shap_importance(
    model: RandomForestClassifier, X: np.ndarray, cap: int = 500
) -> np.ndarray:
    """
    Compute mean absolute TreeSHAP importance per feature.

    Parameters
    ----------
    model : RandomForestClassifier
        A fitted binary classifier.
    X : np.ndarray
        Feature matrix to explain.
    cap : int, optional
        Maximum number of rows to explain (TreeSHAP cost scales with rows);
        the first `cap` rows are used if `X` is larger. Default 500.

    Returns
    -------
    np.ndarray
        Mean absolute SHAP value per feature, shape (n_features,).
    """
    X_capped = X[:cap] if len(X) > cap else X
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_capped, check_additivity=False)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_values = (
            shap_values[:, :, 1] if shap_values.shape[2] == 2 else shap_values[:, :, 0]
        )
    return np.abs(shap_values).mean(axis=0)


def _detect_feature_cols(df: pd.DataFrame, label_col: str, shot_col: str) -> list:
    """
    Auto-detect numeric feature columns, excluding the label, shot/group,
    and time columns.

    Parameters
    ----------
    df : pd.DataFrame
        A DisruptionPy-format shot dataframe.
    label_col : str
        Name of the binary label column to exclude.
    shot_col : str
        Name of the shot/group identifier column to exclude.

    Returns
    -------
    list of str
        Column names to use as model features.
    """
    exclude = DEFAULT_EXCLUDE_COLS | {label_col, shot_col}
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    return [c for c in numeric_cols if c not in exclude]


def _block_resample(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray, rng: np.random.Generator
):
    """Resample unique groups (shots) with replacement; pull all rows for
    each sampled shot. Standard patient/shot-level block bootstrap."""
    unique_groups = np.unique(groups)
    sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
    mask = np.isin(groups, sampled_groups)
    return X[mask], y[mask], groups[mask]


def _simple_resample(X: np.ndarray, y: np.ndarray, rng: np.random.Generator):
    """Row-level resample with replacement, for the target set."""
    idx = rng.integers(0, len(X), size=len(X))
    return X[idx], y[idx]


def compute_qfc(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    feature_cols: list = None,
    n_bootstrap: int = 25,
    threshold: float = 0.391,
    random_state: int = 42,
    label_col: str = "label",
    shot_col: str = "shot",
) -> dict:
    """
    Compute Quantified Feature Consistency (QFC) between a source and target
    DisruptionPy-format shot dataset.

    A Random Forest (500 trees, class_weight='balanced') is trained on
    `source_df` using GroupKFold-5 grouped by `shot_col`, so no shot's rows
    ever appear in both a fold's train and test split. TreeSHAP mean
    absolute importance is computed per fold and averaged into a single
    source importance vector. A separate model is then trained on the full
    `source_df` and deployed on `target_df` (no retraining) to obtain a
    target importance vector. QFC is the Spearman rank correlation between
    the two vectors.

    Parameters
    ----------
    source_df : pd.DataFrame
        Source-environment shot data. Must contain `label_col` (binary,
        0/1) and `shot_col` (shot/discharge identifier for grouping).
    target_df : pd.DataFrame
        Target-environment shot data, same schema as `source_df`.
    feature_cols : list of str, optional
        Feature columns to use. If None, auto-detected as all numeric
        columns except `label_col`, `shot_col`, and `time`.
    n_bootstrap : int, optional
        Number of shot-level block bootstrap iterations for the QFC
        confidence interval. Default 25.
    threshold : float, optional
        QFC value below which attribution consistency is flagged as failed.
        Default 0.391 (the pre-registered alarm threshold used elsewhere in
        this project's PCG and radiation domains -- adjust for your own
        device/dataset's null-calibrated floor before treating this as a
        meaningful cutoff).
    random_state : int, optional
        Seed for the RF, GroupKFold shuffling, and bootstrap resampling.
    label_col : str, optional
        Name of the binary target column. Default "label".
    shot_col : str, optional
        Name of the shot/discharge identifier column used for grouping.
        Default "shot", matching disruption-py's own DataFrameOutputSetting
        convention (not "shot_id" -- verified against the disruption-py
        source before writing this module).

    Returns
    -------
    dict
        qfc : float
            Point-estimate cross-environment QFC (source -> target).
        ci_lower, ci_upper : float
            95% bootstrap CI bounds on the cross-environment QFC.
        auc : float
            ROC-AUC of the source-trained model on target_df.
        flag : str
            "PASS" if qfc >= threshold, else "FAIL".
        n_source, n_target : int
            Row counts used from source_df and target_df.
        feature_cols_used : list of str
            The feature columns actually used.
    """
    if feature_cols is None:
        feature_cols = _detect_feature_cols(source_df, label_col, shot_col)
    if not feature_cols:
        raise ValueError(
            "No usable feature columns found. Pass feature_cols explicitly "
            "or check label_col/shot_col names against your dataframe."
        )

    X_source = source_df[feature_cols].to_numpy(dtype=float)
    y_source = source_df[label_col].to_numpy()
    groups_source = source_df[shot_col].to_numpy()
    X_target = target_df[feature_cols].to_numpy(dtype=float)
    y_target = target_df[label_col].to_numpy()

    gkf = GroupKFold(n_splits=5)
    fold_importances = []
    for train_idx, test_idx in gkf.split(X_source, y_source, groups_source):
        rf = RandomForestClassifier(
            n_estimators=500,
            max_depth=20,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
        rf.fit(X_source[train_idx], y_source[train_idx])
        fold_importances.append(_tree_shap_importance(rf, X_source[test_idx]))
    fold_importances = np.array(fold_importances)
    source_importance = fold_importances.mean(axis=0)

    deploy_model = RandomForestClassifier(
        n_estimators=500,
        max_depth=20,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    deploy_model.fit(X_source, y_source)
    target_importance = _tree_shap_importance(deploy_model, X_target)
    qfc_point, _ = spearmanr(source_importance, target_importance)

    target_proba = deploy_model.predict_proba(X_target)[:, 1]
    auc = float(roc_auc_score(y_target, target_proba)) if len(np.unique(y_target)) > 1 else float("nan")

    rng_master = np.random.default_rng(random_state)
    boot_seeds = rng_master.integers(0, 2**31, size=n_bootstrap)
    boot_qfc = []
    for seed in boot_seeds:
        rng = np.random.default_rng(seed)
        Xb, yb, gb = _block_resample(X_source, y_source, groups_source, rng)

        boot_fold_importances = []
        gkf_boot = GroupKFold(n_splits=5)
        for train_idx, test_idx in gkf_boot.split(Xb, yb, gb):
            rf = RandomForestClassifier(
                n_estimators=500,
                max_depth=20,
                class_weight="balanced",
                random_state=int(seed),
                n_jobs=-1,
            )
            rf.fit(Xb[train_idx], yb[train_idx])
            boot_fold_importances.append(_tree_shap_importance(rf, Xb[test_idx]))
        boot_source_importance = np.array(boot_fold_importances).mean(axis=0)

        boot_deploy = RandomForestClassifier(
            n_estimators=500,
            max_depth=20,
            class_weight="balanced",
            random_state=int(seed),
            n_jobs=-1,
        )
        boot_deploy.fit(Xb, yb)
        Xb_target, _ = _simple_resample(X_target, y_target, rng)
        boot_target_importance = _tree_shap_importance(boot_deploy, Xb_target)
        rho, _ = spearmanr(boot_source_importance, boot_target_importance)
        boot_qfc.append(rho)

    boot_qfc = np.array(boot_qfc)
    ci_lower = float(np.percentile(boot_qfc, 2.5))
    ci_upper = float(np.percentile(boot_qfc, 97.5))

    return {
        "qfc": float(qfc_point),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "auc": auc,
        "flag": "PASS" if qfc_point >= threshold else "FAIL",
        "n_source": int(len(source_df)),
        "n_target": int(len(target_df)),
        "feature_cols_used": feature_cols,
    }


def _build_rolling_features(
    df: pd.DataFrame,
    signal_cols: list,
    shot_col: str,
    time_col: str,
    label_col: str,
    window: int = 10,
) -> pd.DataFrame:
    """
    Build rolling-window statistical features per time point, per shot.

    For each signal, computes 8 statistics (mean, std, min, max, slope,
    lag-1 autocorrelation, mean absolute difference, skewness) over the
    `window` time points immediately preceding each row within the same
    shot. Rows without a full preceding window are dropped.

    Parameters
    ----------
    df : pd.DataFrame
        Raw shot dataframe, one row per time point.
    signal_cols : list of str
        Raw signal columns to compute rolling features from.
    shot_col : str
        Shot/discharge identifier column.
    time_col : str
        Time column, used to order rows within each shot.
    label_col : str
        Binary label column to carry through unchanged.
    window : int, optional
        Number of preceding time points per rolling window. Default 10.

    Returns
    -------
    pd.DataFrame
        One row per (shot, time) with `window`-based features, `shot_col`,
        and `label_col`. Feature columns are named "{signal}_{stat}".
    """
    stat_names = ["mean", "std", "min", "max", "slope", "autocorr_lag1", "mean_abs_diff", "skew"]

    def window_stats(values: np.ndarray) -> tuple:
        n = len(values)
        x = np.arange(n)
        slope = np.polyfit(x, values, 1)[0] if n > 1 else 0.0
        if n > 1 and values[:-1].std() > 0 and values[1:].std() > 0:
            autocorr = np.corrcoef(values[:-1], values[1:])[0, 1]
        else:
            autocorr = 0.0
        mad = np.mean(np.abs(np.diff(values))) if n > 1 else 0.0
        sk = skew(values) if n > 2 and values.std() > 0 else 0.0
        return values.mean(), values.std(), values.min(), values.max(), slope, autocorr, mad, sk

    rows = []
    for shot_id, group in df.groupby(shot_col, sort=False):
        group = group.sort_values(time_col)
        n = len(group)
        if n <= window:
            continue
        signal_arrays = {sig: group[sig].to_numpy() for sig in signal_cols}
        labels = group[label_col].to_numpy()
        times = group[time_col].to_numpy()
        for i in range(window, n):
            row = {shot_col: shot_id, time_col: times[i], label_col: labels[i]}
            for sig in signal_cols:
                window_vals = signal_arrays[sig][i - window : i]
                for stat_name, stat_val in zip(stat_names, window_stats(window_vals)):
                    row[f"{sig}_{stat_name}"] = stat_val
            rows.append(row)
    return pd.DataFrame(rows)


def demo_cmod() -> dict:
    """
    Demonstrate compute_qfc() on the MIT PSFC Open Density Limit Database
    (C-Mod), computing QFC for a 2000-2003 -> 2005-2009 operational-era
    shift.

    Loads DL_DataFrame.csv (Maris, Rea, Trevisan et al. 2025), restricts to
    years 2000-2009 (years 2014-2016 in the public release have no positive
    labels for any discharge -- a data curation gap, not a real absence of
    the phenomenon, and are excluded here rather than silently included as
    negatives), builds 48 rolling-window features from the 6 raw physics
    signals, and runs compute_qfc() with source=2000/2001/2003 and
    target=2005/2009.

    Returns
    -------
    dict
        The compute_qfc() result dict for this shift.
    """
    csv_path = r"C:\Users\user\Datasets\cmod-density-limit\data\DL_DataFrame.csv"
    raw = pd.read_csv(csv_path)

    shot_id_str = raw["discharge_ID"].astype(str)
    year_2digit = shot_id_str.str.slice(1, 3).astype(int)
    raw["year"] = year_2digit.apply(lambda y: 1900 + y if y > 50 else 2000 + y)
    raw = raw.rename(columns={"discharge_ID": "shot", "density_limit_phase": "label"})

    signal_cols = [
        "density",
        "elongation",
        "minor_radius",
        "plasma_current",
        "toroidal_B_field",
        "triangularity",
    ]

    source_years = [2000, 2001, 2003]
    target_years = [2005, 2009]
    relevant = raw[raw["year"].isin(source_years + target_years)].copy()

    features = _build_rolling_features(
        relevant, signal_cols, shot_col="shot", time_col="time", label_col="label"
    )
    features = features.merge(relevant[["shot", "year"]].drop_duplicates(), on="shot", how="left")

    source_df = features[features["year"].isin(source_years)].drop(columns=["year"])
    target_df = features[features["year"].isin(target_years)].drop(columns=["year"])

    print(f"Source (years {source_years}): {len(source_df)} rows, "
          f"{source_df['shot'].nunique()} shots, "
          f"positive rate={source_df['label'].mean():.4f}")
    print(f"Target (years {target_years}): {len(target_df)} rows, "
          f"{target_df['shot'].nunique()} shots, "
          f"positive rate={target_df['label'].mean():.4f}")

    result = compute_qfc(source_df, target_df)

    print("\n=== QFC Results: C-Mod 2000-2003 -> 2005-2009 ===")
    print(f"QFC              = {result['qfc']:.4f}")
    print(f"95% CI           = [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]")
    print(f"Target AUC       = {result['auc']:.4f}")
    print(f"Flag (>= {0.391})  = {result['flag']}")
    print(f"n_source         = {result['n_source']}")
    print(f"n_target         = {result['n_target']}")
    print(f"Features used    = {len(result['feature_cols_used'])}")

    return result


if __name__ == "__main__":
    demo_cmod()
