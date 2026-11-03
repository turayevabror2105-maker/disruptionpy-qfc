"""Causal rolling-window features for time-series shot data.

Every statistic at time *i* is computed from the ``window`` points strictly *before* *i*, within the
same shot. Nothing looks ahead, which is the only construction usable by a warning system that must
decide in real time.

The window is a number of samples, not a length of time: if the sampling rate differs between campaigns, the
same window covers a different span, so resample to a common time base first.

The skew of a window whose spread is at floating-point resolution is undefined and is returned as NaN (the rule
is in ``_skew``); ``screen``, ``compute_qfc`` and ``compute_null_floor`` replace such values with the source
frame's median of that column. A window with no spread at all has a skew and a lag-1 autocorrelation of 0. The statistics are those of the paper's C-Mod feature builder, value for value.
"""

from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats

__all__ = ["build_rolling_features", "STAT_NAMES"]

STAT_NAMES = ("mean", "std", "min", "max", "slope", "autocorr_lag1", "mean_abs_diff", "skew")


_EPS = float(np.finfo(np.float64).eps)


def _skew(v: np.ndarray) -> float:
    """The window's skewness, or NaN where it is undefined.

    Undefined means the window's spread is at floating-point resolution: its second central moment is at or
    below ``(eps * mean)**2``, ``eps`` the float64 machine epsilon. That is the rule by which
    ``scipy.stats.skew`` (scipy 1.11 and later) returns NaN; a value computed there would be a ratio of rounding
    errors, not a measurement. Fixed here rather than delegated, because older scipy versions used a different
    threshold. Elsewhere the value is ``scipy.stats.skew``'s own.
    """
    m = v.mean()
    if np.mean((v - m) ** 2) <= (_EPS * m) ** 2:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        s = float(stats.skew(v))
    if not np.isfinite(s):  # an older scipy's stricter threshold; the value is still defined by the rule above
        s = float(np.mean((v - m) ** 3) / np.mean((v - m) ** 2) ** 1.5)
    return s


def _window_stats(v: np.ndarray) -> tuple:
    n = len(v)
    sd = float(v.std())
    slope = float(np.polyfit(np.arange(n), v, 1)[0]) if n > 1 else 0.0
    if n > 1 and v[:-1].std() > 0 and v[1:].std() > 0:
        autocorr = float(np.corrcoef(v[:-1], v[1:])[0, 1])
    else:
        autocorr = 0.0
    mad = float(np.mean(np.abs(np.diff(v)))) if n > 1 else 0.0
    skew = _skew(v) if n > 2 and sd > 0 else 0.0
    return float(v.mean()), sd, float(v.min()), float(v.max()), slope, autocorr, mad, skew


def build_rolling_features(
    df: pd.DataFrame,
    signal_cols: Sequence[str],
    *,
    shot_col: str = "shot",
    time_col: str = "time",
    label_col: str = "label",
    window: int = 10,
    carry_cols: Sequence[str] = (),
) -> pd.DataFrame:
    """Expand raw signals into ``len(signal_cols) * 8`` causal rolling statistics.

    Parameters
    ----------
    df
        One row per (shot, time).
    signal_cols
        Raw signals to expand.
    window
        Number of preceding points per statistic. Rows without a full preceding window are dropped.
    carry_cols
        Extra columns carried through unchanged (taken at the row's own time), e.g. ``("year",)``.

    Returns
    -------
    pd.DataFrame
        Columns ``{shot_col, time_col, label_col}`` + ``{signal}_{stat}`` + ``carry_cols``. A ``{signal}_skew``
        value is NaN where the window's skew is undefined (see the module docstring).
    """
    for col in (shot_col, time_col, label_col, *signal_cols, *carry_cols):
        if col not in df.columns:
            raise ValueError(f"column {col!r} not in frame; columns are {list(df.columns)[:12]}")
    if window < 3:
        raise ValueError("window must be at least 3 for the shape statistics to be defined")

    rows = []
    for shot_id, group in df.groupby(shot_col, sort=False):
        group = group.sort_values(time_col)
        n = len(group)
        if n <= window:
            continue
        signals = {s: group[s].to_numpy(dtype=float) for s in signal_cols}
        labels = group[label_col].to_numpy()
        times = group[time_col].to_numpy()
        carried = {c: group[c].to_numpy() for c in carry_cols}
        for i in range(window, n):
            row = {shot_col: shot_id, time_col: times[i], label_col: labels[i]}
            for c in carry_cols:
                row[c] = carried[c][i]
            for s in signal_cols:
                for name, value in zip(STAT_NAMES, _window_stats(signals[s][i - window : i])):
                    row[f"{s}_{name}"] = value
            rows.append(row)
    if not rows:
        raise ValueError(f"no shot had more than window={window} time points; nothing to build")
    return pd.DataFrame(rows)
