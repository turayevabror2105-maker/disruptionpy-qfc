"""Rebuild ``standin_frame.csv`` from the public Open Density Limit Database.

    python tests/fixtures/build_standin_frame.py --data DL_DataFrame.csv

The stand-in is a DisruptionPy-format frame built here, not by DisruptionPy; ``test_standin_frame.py`` runs the
four integration test bodies against it so their code paths are exercised without DisruptionPy installed.

Source (public, CC BY): Maris, A. D., Rea, C., Trevisan, G. L., & the Alcator C-Mod Team, *The Open Density
Limit Database*, MIT Plasma Science and Fusion Center (2025).

Determinism: the first twelve discharges (by sorted id) with a positive label and the first twelve without, so
the CSV is reproducible from the same release: 1,942 rows, 51 columns, 24 shots.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "standin_frame.csv"
DEFAULT_DATA = Path(os.environ.get("DL_DATAFRAME", "DL_DataFrame.csv"))

N_POS_SHOTS = 12
N_NEG_SHOTS = 12
WINDOW = 10


def build(data_path: Path):
    import sys
    sys.path.insert(0, str(HERE.parent.parent / "src"))
    from disruptionpy_qfc.demo import SIGNALS, load_cmod
    from disruptionpy_qfc.features import build_rolling_features

    raw = load_cmod(data_path)
    has_pos = raw.groupby("shot")["label"].max()
    pos_shots = sorted(has_pos[has_pos == 1].index.tolist())[:N_POS_SHOTS]
    neg_shots = sorted(has_pos[has_pos == 0].index.tolist())[:N_NEG_SHOTS]
    keep = sorted(pos_shots + neg_shots)
    sub = raw[raw["shot"].isin(keep)].copy().sort_values(["shot", "time"]).reset_index(drop=True)
    feats = build_rolling_features(sub, SIGNALS, window=WINDOW)
    feats = feats.sort_values(["shot", "time"]).reset_index(drop=True)
    return feats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    a = ap.parse_args()
    p = Path(a.data)
    if not p.exists():
        print("source release not found at %s -- set DL_DATAFRAME or pass --data. The committed CSV is "
              "unchanged." % p)
        return 2
    df = build(p)
    # 6 significant figures: the integration assertions test verdict membership and ranges, not exact
    # numerics, so full float64 text would be 2.9 MB of precision no test consumes.
    df.to_csv(OUT, index=False, float_format="%.6g")
    labels = sorted(int(x) for x in df["label"].dropna().unique())
    print("wrote %s" % OUT)
    print("  %d rows, %d shots, %d columns, labels %s, %.2f MB"
          % (len(df), df["shot"].nunique(), df.shape[1], labels, OUT.stat().st_size / 1e6))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
