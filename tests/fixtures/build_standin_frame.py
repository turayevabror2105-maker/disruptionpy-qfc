"""Rebuild ``standin_frame.csv`` from the public Open Density Limit Database.

WHY THIS EXISTS. ``test_disruptionpy_frame.py``'s docstring states that all four integration tests "have
been executed, not merely written" -- against a stand-in frame built from this dataset. That claim was
true and it was also unverifiable: no fixture and no run record were committed, so a reader had to take it
on trust. This script and the CSV it writes make the claim checkable, and ``test_standin_frame.py`` runs
the four assertion bodies against the result on every pytest invocation.

WHAT THIS FIXTURE IS NOT. It is **not** a DisruptionPy frame. It never went through DisruptionPy's
retrieval API. It is DisruptionPy-*format* by the same construction this package uses everywhere else,
which is precisely the claim the integration tests exist to stop the package making about itself. So this
fixture exercises the four code paths; it does not check the integration. The four env-gated tests in
``test_disruptionpy_frame.py`` remain the only thing that can, and they remain skipped here.

Source (public, CC BY):
    Maris, A. D., Rea, C., Trevisan, G. L., & the Alcator C-Mod Team.
    *The Open Density Limit Database.* MIT Plasma Science and Fusion Center (2025).

Run:
    python tests/fixtures/build_standin_frame.py [--data /path/to/DL_DataFrame.csv]

Determinism: shots are selected by sorted discharge id, twelve carrying at least one positive label and
twelve carrying none, so the selection does not depend on dict or filesystem ordering and the CSV is
reproducible byte-for-byte from the same release. Verified against the committed fixture: 1,942 rows,
51 columns, 24 distinct shots, 12 with a positive label and 12 without.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "standin_frame.csv"
DEFAULT_DATA = Path(os.environ.get(
    "DL_DATAFRAME", r"C:\Users\user\Datasets\cmod-density-limit\data\DL_DataFrame.csv"))

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
