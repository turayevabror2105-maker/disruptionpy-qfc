"""Causal rolling-window features for time-series shot data.

Every statistic at time *i* is computed from the ``window`` points strictly *before* *i*, within the
same shot. Nothing looks ahead, which is the only construction usable by a warning system that must
decide in real time.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

__all__ = ["build_rolling_features", "STAT_NAMES"]

STAT_NAMES = ("mean", "std", "min", "max", "slope", "autocorr_lag1", "mean_abs_diff", "skew")


def _window_stats(v: np.ndarray) -> tuple:
    n = len(v)
    sd = float(v.std())
    slope = float(np.polyfit(np.arange(n), v, 1)[0]) if n > 1 and sd > 0 else 0.0
    if n > 1 and v[:-1].std() > 0 and v[1:].std() > 0:
        autocorr = float(np.corrcoef(v[:-1], v[1:])[0, 1])
    else:
        autocorr = 0.0
    mad = float(np.mean(np.abs(np.diff(v)))) if n > 1 else 0.0
    if n > 2 and sd > 0:
        # computed directly rather than via scipy.stats.skew, which emits a precision-loss warning
        # on the near-constant windows that are common in steady plasma phases
        m = v.mean()
        skew = float(np.mean((v - m) ** 3) / (sd**3))
    else:
        skew = 0.0
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
        Columns ``{shot_col, time_col, label_col}`` + ``{signal}_{stat}`` + ``carry_cols``.
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
