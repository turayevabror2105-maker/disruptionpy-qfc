"""Attribution-stability screening for DisruptionPy-format fusion shot data.

Does a model trained on one environment rely on the same features when deployed on another, and is
what you measure any different from a no-shift split of the source alone?

    from disruptionpy_qfc import screen
    result = screen(source_df, target_df)
    print(result.summary())
"""

__version__ = "3.0.1"

from .core import (  # noqa: F401
    ATTRIBUTIONS,
    DEFAULT_EXCLUDE_COLS,
    N_CHANCE_DRAWS,
    N_FOLD_ORDERS,
    N_NULL_SEEDS,
    ChanceFloor,
    DegenerateDrawError,
    NullFloor,
    QFCResult,
    compute_chance_floor,
    compute_null_floor,
    compute_qfc,
    default_model,
    detect_feature_cols,
    exact_cut,
    qfc_statistic,
    screen,
    tie_order_groups,
    verdict_label,
)
from .features import build_rolling_features  # noqa: F401

__all__ = [
    "screen",
    "compute_qfc",
    "compute_null_floor",
    "compute_chance_floor",
    "QFCResult",
    "NullFloor",
    "ChanceFloor",
    "DegenerateDrawError",
    "default_model",
    "detect_feature_cols",
    "qfc_statistic",
    "exact_cut",
    "verdict_label",
    "tie_order_groups",
    "build_rolling_features",
    "ATTRIBUTIONS",
    "DEFAULT_EXCLUDE_COLS",
    "N_FOLD_ORDERS",
    "N_NULL_SEEDS",
    "N_CHANCE_DRAWS",
    "__version__",
]
