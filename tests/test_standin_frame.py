"""The four integration assertion bodies, run against a committed stand-in frame.

WHAT THIS FILE IS FOR, AND WHAT IT DELIBERATELY DOES NOT CLAIM
--------------------------------------------------------------
``test_disruptionpy_frame.py`` holds four tests that answer one question: *given a frame that came out of
DisruptionPy, does this package accept it and screen it?* They are env-gated on ``DISRUPTIONPY_FRAME`` and
they **skip here and should**, because DisruptionPy is not installed in this environment and its retrieval
API needs credentialed access to tokamak data servers. That skip is a disclosure, not a gap: it says the
integration has not been checked. Nothing in this file changes that, and this file must never be read as
closing it.

What that file's docstring also said, and could not prove, is that the four tests "have been executed, not
merely written" -- against a stand-in frame built from the public Open Density Limit Database. That was
true and unverifiable: no fixture and no run record were committed. **This file makes it verifiable.** It
imports the four assertion bodies unchanged and runs them against
``tests/fixtures/standin_frame.csv``, which ``tests/fixtures/build_standin_frame.py`` rebuilds
reproducibly from the public release.

So, precisely:

* These four tests prove the four code paths work and that their assertions bite. They run on every
  invocation, so the paths cannot silently rot.
* They prove **nothing** about the DisruptionPy integration. The fixture is DisruptionPy-*format* by the
  same construction this package uses everywhere else, which is exactly the self-claim the integration
  tests exist to stop the package making. Only a maintainer's run against a real exported frame can check
  that, and the four skips in the other file remain the honest record that it has not happened.

The bodies are IMPORTED rather than copied, so there is one definition of each assertion. If someone
tightens an integration assertion, it tightens here too and a stand-in regression is caught immediately.
They are aliased to private names so pytest collects them once, from their own module, and does not try to
supply this module a ``frame`` fixture it does not define.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from test_disruptionpy_frame import (
    test_a_split_of_one_real_frame_screens_end_to_end as _screens_end_to_end,
    test_an_uncalibrated_measurement_says_so_on_a_real_frame as _uncalibrated_says_so,
    test_the_floor_is_per_pool_and_refuses_to_be_reused_implicitly as _floor_is_per_pool,
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
    assert standin["shot"].nunique() >= 4, "fewer than 4 shots: grouped 5-fold cannot run"
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
