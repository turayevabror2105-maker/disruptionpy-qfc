"""The four integration test bodies, run on every invocation against a committed stand-in frame.

``test_disruptionpy_frame.py`` needs a frame produced by DisruptionPy, which CI builds from MAST's open data.
This file runs the same four bodies (imported, so there is one definition of each) against
``fixtures/standin_frame.csv``, built from the public Open Density Limit Database by
``fixtures/build_standin_frame.py``. It keeps those code paths exercised where DisruptionPy is not installed. The
stand-in did not come out of DisruptionPy, so it checks the package's code paths, not the integration itself.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from test_disruptionpy_frame import (
    test_a_split_of_one_real_frame_screens_end_to_end as _screens_end_to_end,
    test_an_uncalibrated_measurement_says_so_on_a_real_frame as _uncalibrated_says_so,
    test_the_floor_is_per_pool_and_records_its_provenance as _floor_is_per_pool,
    test_the_frame_meets_this_package_s_contract as _meets_contract,
)

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "standin_frame.csv"


@pytest.fixture(scope="module")
def standin() -> pd.DataFrame:
    assert FIXTURE.exists(), (
        f"{FIXTURE} is missing. Rebuild it with `python tests/fixtures/build_standin_frame.py`, which "
        "needs the public Open Density Limit Database release."
    )
    df = pd.read_csv(FIXTURE)
    assert len(df) > 0, f"{FIXTURE} is empty"
    return df


def test_standin_fixture_is_shaped_as_the_contract_requires(standin):
    """Guard the fixture itself. If a rebuild ever produced a frame that could not exercise the four
    bodies below, they would pass vacuously and this file would become decoration."""
    assert {"shot", "label"} <= set(standin.columns)
    assert standin["shot"].nunique() >= 20, "fewer than 20 shots: the contract test needs 20 (five per quarter)"
    assert sorted(int(x) for x in standin["label"].dropna().unique()) == [0, 1], "not two-class"
    assert standin.shape[1] >= 6, "too few columns to yield 3 usable features"


def test_standin_meets_the_package_contract(standin):
    _meets_contract(standin)


def test_standin_screens_end_to_end(standin):
    _screens_end_to_end(standin)


def test_standin_uncalibrated_measurement_says_so(standin):
    _uncalibrated_says_so(standin)


def test_standin_floor_is_per_pool(standin):
    _floor_is_per_pool(standin)
