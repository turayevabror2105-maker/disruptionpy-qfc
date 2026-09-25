"""Every number the README states about the worked example must match a file in this repository.

WHY THIS TEST EXISTS
--------------------
The README makes two quantitative claims a reader cannot check by running the fast suite: the C-Mod
worked example's result, and the label-prior control that separates the prior shift from the covariate
shift. Both are minutes of CPU and a 25 MB dataset away, so a reader takes them on trust -- and a claim
taken on trust drifts. It has already drifted once in this package: an earlier version of
``test_cmod_era_shift_collapses_against_its_own_floor`` asserted INDISTINGUISHABLE and the demo docstring
said in prose that the floor "is just as high", neither of which was a measurement.

So the committed record is the source of truth and this test pins the README to it. It runs in
milliseconds, needs no dataset, and fails the moment the two disagree.

These are documentation-consistency assertions, not science. The science is in
``test_regression_cmod.py`` (marked ``slow``), which recomputes the worked example from the dataset.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
CONTROL = ROOT / "docs" / "prior_matched_control.json"


@pytest.fixture(scope="module")
def readme() -> str:
    return README.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def control() -> dict:
    return json.loads(CONTROL.read_text(encoding="utf-8"))


def test_the_control_record_exists_and_names_its_pre_registration(control):
    """A result with no pre-registration named is a result whose outcome could have been chosen."""
    assert control["pre_registration"]
    assert control["runner"]
    assert control["result"]["outcome"] == "IMMATERIAL"
    assert control["result"]["both_arms_collapse"] is True


def test_control_arms_differ_only_in_composition(control):
    """The design claim the README makes: one variable changed, and the control arm proves it."""
    d = control["design"]
    t, m = d["arm_T_control"]["composition"], d["arm_M_matched"]["composition"]
    assert sum(t) == sum(m) == 500, "both arms must draw the same number of rows"
    assert t != m, "the arms must differ in composition, which is the one variable"
    assert d["seeds"] == 10


def test_paired_delta_is_below_the_resolution_as_the_readme_says(control):
    """IMMATERIAL is a comparison, not an adjective: |delta| < the paired spread."""
    r = control["result"]
    assert abs(r["paired_delta_M_minus_T"]) < r["paired_delta_sd_resolution"]


def test_the_design_was_not_underpowered(control):
    """A small difference means nothing without the smallest difference the design could have seen."""
    r = control["result"]
    published_gap = abs(0.9636778 - r["null_used"]["mean"])
    assert r["paired_delta_sd_resolution"] < published_gap, (
        "the design's own noise exceeds the effect it was built to explain"
    )


def test_both_control_arms_clear_the_exact_cut_downward(control):
    """Both arms collapse, and against the exact Student-t cut rather than a nominal |z| > 2."""
    r = control["result"]
    cut = r["null_used"]["exact_cut"]
    assert cut == pytest.approx(2.0790373811, abs=1e-9)
    assert r["arm_T_z"] < -cut and r["arm_M_z"] < -cut


@pytest.mark.parametrize(
    "field, places",
    [("arm_T_mean", 4), ("arm_T_sd", 4), ("arm_M_mean", 4), ("arm_M_sd", 4)],
)
def test_readme_quotes_the_control_to_four_places(readme, control, field, places):
    value = control["result"][field]
    text = "%.*f" % (places, value)
    assert text in readme, f"README does not carry {field} = {text}"


def test_readme_quotes_the_paired_delta_and_its_resolution(readme, control):
    """The forbidden shortcut, enforced: the delta may not appear without its resolution."""
    r = control["result"]
    delta = "%.4f" % r["paired_delta_M_minus_T"]
    resolution = "%.4f" % r["paired_delta_sd_resolution"]
    assert delta in readme, f"README does not carry the paired delta {delta}"
    assert resolution in readme, (
        f"README carries the delta but not its resolution {resolution} -- a small difference means "
        "nothing without the smallest difference the design could have resolved"
    )


def test_readme_quotes_both_control_z_values(readme, control):
    r = control["result"]
    for z in (r["arm_T_z"], r["arm_M_z"]):
        assert ("%.2f" % abs(z)) in readme, f"README does not carry z = {z:.2f}"


def test_readme_states_the_stratification_caveat(readme):
    """Neither control z equals the headline -2.69, and the README must say why rather than leaving a
    reader to find two different numbers for one condition."""
    assert "stratified" in readme
    assert "2.69" in readme


def test_readme_worked_example_matches_the_slow_regression_expectations(readme):
    """The four numbers the slow test pins are the four the README advertises."""
    for value in ("0.938", "0.961", "0.980", "2.079", "2.69"):
        assert value in readme, f"README no longer states {value}"


def test_no_superseded_plasma_magnitude_is_quoted(readme):
    """The research project's superseded z values for this condition must never appear here: -8.24 came
    from a head-capped attribution sample and -20.92 from a null carrying two defects at once."""
    for bad in ("8.24", "20.92", "8.2384", "20.9"):
        assert bad not in readme, f"README quotes the superseded magnitude {bad}"
