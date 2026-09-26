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


NOTEBOOK = ROOT / "notebooks" / "attribution_stability_on_the_density_limit.ipynb"


@pytest.fixture(scope="module")
def notebook_text() -> str:
    """Every markdown and code line of the notebook, concatenated."""
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return "\n".join("".join(c.get("source", [])) for c in nb["cells"])


def test_the_notebook_reaches_the_same_verdict_as_the_test_suite(notebook_text):
    """The notebook is the third place this result is stated, after the README and
    ``test_regression_cmod.py``, and it is the one a reader opens first.

    It drifted once, and badly: written on 2026-09-15, it still concluded INDISTINGUISHABLE a day after
    the floor was measured at 30 seeds and the verdict came back COLLAPSE. That is the failure mode this
    test exists to prevent -- a claim corrected in one file and left standing in a parallel one.
    """
    assert "COLLAPSE" in notebook_text
    assert "INDISTINGUISHABLE**" not in notebook_text, (
        "the notebook asserts INDISTINGUISHABLE again; the measured verdict is COLLAPSE"
    )
    for value in ("0.961", "0.980", "2.69"):
        assert value in notebook_text, f"the notebook no longer states {value}"


def test_the_notebook_keeps_its_own_correction_visible(notebook_text):
    """The withdrawal is the lesson, so it may not be quietly deleted either. The superseded phrase is
    allowed to appear exactly once, inside the note that withdraws it."""
    assert "used to say INDISTINGUISHABLE" in notebook_text
    assert notebook_text.count("the floor is just as high") == 1, (
        "the retracted phrase should appear exactly once, as a quotation inside its own withdrawal"
    )


def test_the_notebook_computes_its_floor_at_the_package_default(notebook_text):
    """A narrative built on a floor computed under a different protocol from the README's is how the
    two came to disagree in the first place."""
    assert "n_seeds=30" in notebook_text
    assert "n_seeds=20" not in notebook_text


def test_the_notebook_stores_no_outputs(notebook_text):
    """Stored outputs are numbers no test can see. The notebook ships with none, so a reader's run is
    the only source of its figures."""
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    stored = sum(len(c.get("outputs", [])) for c in nb["cells"])
    assert stored == 0, f"{stored} stored outputs could show superseded numbers"


def test_no_superseded_plasma_magnitude_is_quoted(readme):
    """The research project's superseded z values for this condition must never appear here: -8.24 came
    from a head-capped attribution sample and -20.92 from a null carrying two defects at once."""
    for bad in ("8.24", "20.92", "8.2384", "20.9"):
        assert bad not in readme, f"README quotes the superseded magnitude {bad}"


# --------------------------------------------------------------------- the slow suite's own record (B-377)


def _slow_record():
    import json
    from pathlib import Path

    p = Path(__file__).resolve().parents[1] / "docs" / "slow_suite_last_run.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def test_the_readme_numbers_match_the_last_slow_run(readme):
    """The README's worked example must agree with what the slow suite last measured.

    Skipped, never passed, when no slow run has been recorded: CI runs `-m "not slow"` because the tests
    need a licensed dataset, and a skip here means nothing was checked rather than that all is well.
    """
    rec = _slow_record()
    if rec is None:
        pytest.skip("no slow run recorded in docs/slow_suite_last_run.json -- run `pytest -m slow` with the "
                    "C-Mod release present; NOTHING about the worked example has been checked here")
    m = rec["measured"]
    assert m["verdict"] == "COLLAPSE", "the recorded run did not collapse: %s" % m["verdict"]
    assert m["qfc"] < m["floor_mean"], "the recorded QFC is no longer below its own floor"
    # The README rounds to four and three decimals; compare against the recorded measurement, not a constant.
    assert "%.3f" % m["qfc"] in readme or "%.4f" % m["qfc"] in readme, \
        "the README does not carry the QFC the slow suite measured (%.4f)" % m["qfc"]
    assert "%.3f" % m["floor_mean"] in readme or "%.4f" % m["floor_mean"] in readme, \
        "the README does not carry the floor the slow suite measured (%.4f)" % m["floor_mean"]


def test_the_recorded_slow_run_is_self_consistent():
    """z must be recomputable from the floor and the QFC the same run recorded, and the verdict from z."""
    rec = _slow_record()
    if rec is None:
        pytest.skip("no slow run recorded -- nothing checked")
    m = rec["measured"]
    z = (m["qfc"] - m["floor_mean"]) / m["floor_sd"]
    assert z == pytest.approx(m["z"], abs=5e-3), "the recorded z does not follow from the recorded floor"
    expected = "COLLAPSE" if z < -m["exact_cut"] else ("ROBUST" if z > m["exact_cut"] else "INDISTINGUISHABLE")
    assert m["verdict"] == expected, "the recorded verdict does not follow from the recorded z and cut"
    assert m["floor_seeds_used"] >= 2


# --------------------------------------------- the within-null label-prior control (B-385)


def _prior_control():
    import json
    from pathlib import Path

    p = Path(__file__).resolve().parents[1] / "docs" / "label_prior_within_null_control.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def test_the_prior_control_record_is_self_consistent():
    """Its headline percentages must follow from its own slope, gap and observed drop."""
    import numpy as np
    from scipy import stats

    rec = _prior_control()
    if rec is None:
        pytest.skip("no within-null prior control recorded -- nothing checked")
    m = rec["measured"]
    frac = (-m["ols_slope"] * m["real_prior_gap"]) / m["observed_drop"]
    assert frac == pytest.approx(m["fraction_of_observed_drop"], abs=5e-4)
    tc = float(stats.t.ppf(0.975, m["n_usable"] - 2))
    steep = m["ols_slope"] - tc * m["ols_slope_se"]
    frac_hi = (-steep * m["real_prior_gap"]) / m["observed_drop"]
    assert frac_hi == pytest.approx(m["fraction_at_95pct_steep_end"], abs=5e-4)
    # the claim that the relationship is unresolved must be judged against the cut, not asserted
    assert abs(m["spearman_rho"]) < m["spearman_cut_5pct"], "rho now clears the cut -- re-read the conclusion"


def test_the_large_gap_draws_really_are_larger_than_the_real_shift():
    """The non-parametric half of the argument: a WIDER prior gap with no shift stays at the floor."""
    rec = _prior_control()
    if rec is None:
        pytest.skip("no within-null prior control recorded -- nothing checked")
    m = rec["measured"]
    big = m["draws_with_gap_above_real"]
    assert len(big) >= 2, "the argument needs at least two draws above the real gap"
    for d in big:
        assert d["prior_gap"] > m["real_prior_gap"], "this draw's gap is not above the real one"
        assert abs(d["z_vs_floor"]) < m["exact_cut"], "a large-gap draw is NOT indistinguishable from the floor"
        assert d["qfc"] > m["real_qfc"], "a large-gap no-shift draw scored at or below the real value"


def test_the_readme_states_the_prior_control_with_its_limit(readme):
    """The README must carry both the point estimate and the 95% bound, not just the convenient one."""
    rec = _prior_control()
    if rec is None:
        pytest.skip("no within-null prior control recorded -- nothing checked")
    m = rec["measured"]
    assert "%d%% of the observed drop" % round(100 * m["fraction_of_observed_drop"]) in readme
    assert "%d%%" % round(100 * m["fraction_at_95pct_steep_end"]) in readme, \
        "the README gives the point estimate without the 95% bound"
