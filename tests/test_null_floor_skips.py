"""The null floor must say how many of the requested seeds it actually used. (B-375)

`QFCResult.summary()` has always reported `n_bootstrap_used/n_bootstrap_attempted` and warned below 80%,
with the reason written into the code: a CI from a handful of usable draws must not read as a CI from all
of them. `compute_null_floor` skipped seeds for the same three reasons and reported only the survivors,
and the floor is what every verdict in this package is scored against.

Fast tests: they build tiny synthetic frames, never the C-Mod release.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from disruptionpy_qfc import compute_null_floor
from disruptionpy_qfc.core import NullFloor


def _frame(n_shots: int, rows_per_shot: int = 12, positive_shots: int | None = None, seed: int = 0):
    """A frame whose label is a property of the SHOT, so a half-split can isolate the rare class."""
    rng = np.random.default_rng(seed)
    pos = n_shots // 2 if positive_shots is None else positive_shots
    rows = []
    for s in range(n_shots):
        y = 1 if s < pos else 0
        for _ in range(rows_per_shot):
            rows.append({"shot": s, "time": _, "label": y,
                         "f0": rng.normal(y, 1.0), "f1": rng.normal(0, 1.0), "f2": rng.normal(-y, 1.0)})
    return pd.DataFrame(rows)


def test_a_clean_pool_reports_every_seed_as_used():
    f = compute_null_floor(_frame(20), n_seeds=4, n_splits=2, shap_cap=60)
    assert f.n_seeds_requested == 4
    assert f.n_seeds == 4
    assert f.n_seeds_skipped == 0
    assert f.skips == {}
    assert "usable seeds" not in f.summary()


def test_a_pool_that_must_skip_says_so_and_names_the_reason():
    """One positive shot of twelve: most half-splits put it on one side, so halves go single-class."""
    f = compute_null_floor(_frame(12, positive_shots=1), n_seeds=12, n_splits=2, shap_cap=60)
    assert f.n_seeds_requested == 12
    assert f.n_seeds < 12, "this pool was engineered so seeds must be dropped"
    assert f.n_seeds_skipped == f.n_seeds_requested - f.n_seeds
    assert sum(f.skips.values()) == f.n_seeds_skipped
    assert set(f.skips) <= {"single_class_half", "too_few_shots_to_fold", "all_folds_single_class"}
    assert "%d/%d usable seeds" % (f.n_seeds, 12) in f.summary()


def test_the_summary_warns_when_fewer_than_four_fifths_survive():
    n = NullFloor(mean=0.9, sd=0.01, per_seed=[0.9, 0.91], n_seeds=2, exact_cut=15.56,
                  n_seeds_requested=30, skips={"single_class_half": 28})
    s = n.summary()
    assert "2/30 usable seeds" in s
    assert "single_class_half: 28" in s
    assert "WARNING" in s and "optimistic" in s


def test_the_exact_cut_uses_the_usable_count_not_the_requested_one():
    """The cut must widen with the seeds actually used; reporting 30 while using 4 would be the defect."""
    from scipy.stats import t as student_t
    f = compute_null_floor(_frame(20), n_seeds=4, n_splits=2, shap_cap=60)
    expected = float(student_t.ppf(0.975, f.n_seeds - 1) * np.sqrt(1 + 1 / f.n_seeds))
    assert f.exact_cut == pytest.approx(expected, abs=1e-12)


def test_too_few_usable_draws_raises_with_the_breakdown():
    """A pool that cannot calibrate must say WHY, not just that it could not."""
    with pytest.raises(ValueError) as e:
        compute_null_floor(_frame(6, positive_shots=1), n_seeds=3, n_splits=3, shap_cap=40)
    assert "skipped" in str(e.value)
