"""Worked example on the MIT PSFC Open Density Limit Database (Alcator C-Mod).

The dataset is public (CC BY):

    Maris, A. D., Rea, C., Trevisan, G. L., & the Alcator C-Mod Team.
    *The Open Density Limit Database.* MIT Plasma Science and Fusion Center (2025).

Run ``python -m disruptionpy_qfc.demo --data /path/to/DL_DataFrame.csv``.

A model is trained on the 2000, 2001 and 2003 campaigns and deployed on 2005 and 2009. The result of the last
recorded run, with the settings it used, is in ``docs/worked_example.json`` and in the README.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .core import screen
from .features import build_rolling_features

SIGNALS = ["density", "elongation", "minor_radius", "plasma_current", "toroidal_B_field", "triangularity"]
SOURCE_YEARS = (2000, 2001, 2003)
TARGET_YEARS = (2005, 2009)
# 2014-2016 discharges carry no density-limit precursor labels in the public release: a curation gap,
# so they are excluded rather than silently treated as all-negative.
EXCLUDED_YEARS_NOTE = "2014-2016 excluded: zero positive labels in the public release"


def load_cmod(csv_path: str | Path) -> pd.DataFrame:
    """Load the release and derive the campaign year from the discharge identifier."""
    raw = pd.read_csv(csv_path)
    ids = raw["discharge_ID"].astype(str)
    two_digit = ids.str.slice(1, 3).astype(int)
    raw["year"] = two_digit.apply(lambda y: 1900 + y if y > 50 else 2000 + y)
    return raw.rename(columns={"discharge_ID": "shot", "density_limit_phase": "label"})


def build_demo_frames(csv_path: str | Path, window: int = 10):
    """Source (2000, 2001, 2003) and target (2005, 2009) frames: six signals, each expanded into eight causal
    rolling statistics over the ten preceding time points of its shot."""
    raw = load_cmod(csv_path)
    keep = raw[raw["year"].isin(SOURCE_YEARS + TARGET_YEARS)].copy()
    feats = build_rolling_features(keep, SIGNALS, window=window, carry_cols=("year",))
    source = feats[feats["year"].isin(SOURCE_YEARS)].drop(columns=["year"])
    target = feats[feats["year"].isin(TARGET_YEARS)].drop(columns=["year"])
    return source, target


def run(csv_path: str | Path, *, out: str | Path | None = None, **screen_kwargs) -> dict:
    source, target = build_demo_frames(csv_path)
    print(f"source (years {SOURCE_YEARS}): {len(source)} rows, {source['shot'].nunique()} shots, "
          f"positive rate {source['label'].mean():.4f}")
    print(f"target (years {TARGET_YEARS}): {len(target)} rows, {target['shot'].nunique()} shots, "
          f"positive rate {target['label'].mean():.4f}")
    print("note: the label prior differs between the environments as well as the covariates.\n", flush=True)
    result = screen(source, target, **screen_kwargs)
    print("=== C-Mod 2000, 2001, 2003 -> 2005, 2009 ===")
    print(result.summary())
    payload = {"result": result.to_dict(), "excluded": EXCLUDED_YEARS_NOTE}
    if out:
        Path(out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwritten -> {out}")
    return payload


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="path to DL_DataFrame.csv")
    p.add_argument("--n-seeds", type=int, default=30, help="no-shift draws in the null floor (default 30)")
    p.add_argument("--out", default=None, help="write the full result as JSON here")
    a = p.parse_args(argv)
    run(a.data, n_seeds=a.n_seeds, out=a.out)


if __name__ == "__main__":
    main()
