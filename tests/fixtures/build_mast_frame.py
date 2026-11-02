"""Build a real DisruptionPy frame from MAST's open data, for the ``integration`` tests.

    pip install "disruption-py>=0.14"          # needs Python 3.11 or later
    python tests/fixtures/build_mast_frame.py --out mast_frame.csv
    DISRUPTIONPY_FRAME=mast_frame.csv DPQ_TUD_COL=time_until_end_of_current DPQ_REQUIRE_FRAME=1 \\
        pytest -m integration

MAST is the one device whose data DisruptionPy reads without credentials (the FAIR-MAST release, served
from ``s3.echo.stfc.ac.uk``). The frame is DisruptionPy's own output for each shot -- its default
retrieval settings and every MAST physics method -- concatenated, with nothing changed.

One column is added, because a test needs a label and the open release has none: MAST's open data
carries no disruption times, so the frame cannot carry DisruptionPy's ``time_until_disrupt``. The added
column is ``time_until_end_of_current``: for each shot, the last time DisruptionPy returned (its time
base ends where the plasma current falls below 200 kA) minus the row's time. The tests turn it into a
label with ``DPQ_TUD_COL``, exactly as the README tells a user to turn ``time_until_disrupt`` into one,
and then drop it so it cannot leak into the features. It is a convention for exercising the
integration, not a disruption label, and nothing scientific is read from it.

Shots: the six MAST shots DisruptionPy's own test configuration lists, then consecutive shots from
30100 upward, until ``--n-shots`` have been retrieved. Shots DisruptionPy cannot retrieve are skipped
and printed.
"""

from __future__ import annotations

import argparse
import time

import pandas as pd

TEST_SHOTS = [30113, 30014, 30421, 29659, 28864, 29666]  # disruption_py/machine/mast/config.toml [mast.tests.shots]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-shots", type=int, default=24)
    ap.add_argument("--processes", type=int, default=4)
    a = ap.parse_args()

    import disruption_py
    from disruption_py.settings import RetrievalSettings
    from disruption_py.workflow import get_shots_data

    t0 = time.time()
    frames, tried = [], 0
    candidates = TEST_SHOTS + [s for s in range(30100, 30400) if s not in TEST_SHOTS]
    while sum(f["shot"].nunique() for f in frames) < a.n_shots and tried < len(candidates):
        need = a.n_shots - sum(f["shot"].nunique() for f in frames)
        batch = candidates[tried:tried + max(need, a.processes)]
        tried += len(batch)
        out = get_shots_data(tokamak="mast", shotlist_setting=batch, retrieval_settings=RetrievalSettings(),
                             output_setting="dataframe", num_processes=a.processes)
        if out is not None and len(out):
            frames.append(out)
        got = set(out["shot"].unique()) if out is not None and len(out) else set()
        print("batch %s: retrieved %d of %d (%s skipped)" % (batch[0], len(got), len(batch),
                                                             sorted(set(batch) - got)), flush=True)
    df = pd.concat(frames, ignore_index=True)
    df = df[df["shot"].isin(sorted(df["shot"].unique())[: a.n_shots])].copy()
    df["time_until_end_of_current"] = df.groupby("shot")["time"].transform("max") - df["time"]
    df.to_csv(a.out, index=False)
    print("disruption-py %s: %d rows, %d shots, %d columns -> %s (%.0f s)"
          % (disruption_py.__version__ if hasattr(disruption_py, "__version__") else "?", len(df),
             df["shot"].nunique(), df.shape[1], a.out, time.time() - t0))


if __name__ == "__main__":
    main()
