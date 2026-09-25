"""Attribution-stability screening for DisruptionPy-format fusion shot data.

Does a model trained on one environment rely on the same features when deployed on another --
and is what you measure any different from a no-shift split of the source alone?

    from disruptionpy_qfc import screen
    result = screen(source_df, target_df)
    print(result.summary())
"""

__version__ = "2.0.0"

from .core import (  # noqa: F401
    DEFAULT_EXCLUDE_COLS,
    NullFloor,
    QFCResult,
    compute_null_floor,
    compute_qfc,
    default_model,
    detect_feature_cols,
    screen,
)
from .features import build_rolling_features  # noqa: F401

__all__ = [
    "screen",
    "compute_qfc",
    "compute_null_floor",
    "QFCResult",
    "NullFloor",
    "default_model",
    "detect_feature_cols",
    "build_rolling_features",
    "DEFAULT_EXCLUDE_COLS",
    "__version__",
]
