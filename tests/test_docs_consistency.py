"""Every number the README states about the worked example must be the one in ``docs/worked_example.json``.

The record is written by ``scripts/regenerate_worked_example.py``; the slow tests check that the worked example still
reproduces it. These tests pin the README to the record, need no data, and fail the moment the two disagree.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import disruptionpy_qfc as dq

ROOT = Path(__file__).resolve().parents[1]
README = " ".join((ROOT / "README.md").read_text(encoding="utf-8").split())   # line breaks do not matter
REC = json.loads((ROOT / "docs" / "worked_example.json").read_text(encoding="utf-8"))


def _has(fmt: str, value: float, what: str):
    text = fmt % value
    assert text in README, f"README does not state {what} as {text}"


def test_the_record_is_of_this_version_and_complete():
    assert REC["package_version"] == dq.__version__
    assert set(REC) >= {"default", "depth20", "paper_null", "prior_gap", "prior_matched", "floor_seed0",
                        "split_sensitivity"}
    assert REC["default"]["env"]["package"] == dq.__version__


def test_the_worked_example_numbers():
    d = REC["default"]
    r, f = d["result"], d["null_floor"]
    _has("%.3f", r["auc"], "the target AUC")
    _has("%.4f", r["qfc"], "the QFC")
    _has("%.4f to", min(r["qfc_per_order"]), "the lowest fold order")
    _has("to %.4f", max(r["qfc_per_order"]), "the highest fold order")
    _has("%.4f ± ", f["mean"], "the floor mean")
    _has("± %.4f, 30 draws", f["sd"], "the floor sd")
    _has("−%.2f", -r["z"], "z")
    _has("%.4f", r["resolution"], "the resolution")
    assert r["verdict"] == "COLLAPSE" and "**COLLAPSE** (%d of the 10 fold orders" % r["verdict_per_order"].count(
        "COLLAPSE") in README
    assert f["n_seeds"] == 30 and len(r["qfc_per_order"]) == 10
    fr = d["frames"]
    assert "%s rows from %d discharges" % (format(fr["source_rows"], ","), fr["source_shots"]) in README
    _has("%.3f;", d["priors"]["source"], "the source prior")
    _has("rate %.3f", d["priors"]["target"], "the target prior")


def test_the_recorded_verdict_follows_from_the_recorded_numbers():
    r, f = REC["default"]["result"], REC["default"]["null_floor"]
    z = (r["qfc"] - f["mean"]) / f["sd"]
    assert z == pytest.approx(r["z"], abs=1e-12)
    assert dq.verdict_label(z, dq.exact_cut(f["n_seeds"])) == r["verdict_unadjudicated"] == r["verdict"]


def test_the_split_sensitivity_is_stated_both_ways():
    s = REC["split_sensitivity"]
    a, p = s["floor_seeds_0_29"], s["pooled_60"]
    _has("%.4f ± ", a["mean"], "the other floor's mean")
    _has("± %.4f, z", a["sd"], "the other floor's sd")
    _has("z = −%.2f", -a["z"], "the other floor's z")
    assert a["verdict"] == "INDISTINGUISHABLE"
    _has("%.4f ± ", p["mean"], "the pooled mean")
    _has("± %.4f, z", p["sd"], "the pooled sd")
    _has("cut of %.3f", p["cut"], "the pooled cut")
    assert p["verdict"] == "COLLAPSE"
    _has("margin of %.3f", abs(p["z"]) - p["cut"], "the pooled margin")


def test_the_label_prior_controls():
    g, m = REC["prior_gap"], REC["prior_matched"]
    assert g["outcome"] == "P-FLAT" and m["outcome"] == "IMMATERIAL"
    big = g["draws_with_gap_above_real"]
    assert len(big) == 2
    for b in big:
        _has("%.3f", b["prior_gap"], "a large-gap draw's gap")
        _has("%.3f", b["qfc"], "a large-gap draw's QFC")
        _has("−%.2f", -b["z_vs_floor"], "a large-gap draw's z")
    _has("%.3f)", g["real_prior_gap"], "the real prior gap")
    _has("ρ = −%.3f", -g["spearman_rho"], "rho")
    _has("p = %.3f", g["spearman_p"], "its p")
    assert abs(g["spearman_rho"]) < g["spearman_cut_5pct"]
    _has("%d%% of the observed drop", round(100 * g["fraction_of_observed_drop"]), "the point estimate")
    _has("%d%% at the steep end", round(100 * g["fraction_at_95pct_steep_end"]), "the 95% bound")
    for arm in ("arm_T", "arm_M"):
        _has("%.4f ± ", m[arm]["mean"], arm + " mean")
        _has("−%.2f", -m[arm]["z"], arm + " z")
        assert m[arm]["z"] <= -REC["default"]["result"]["exact_cut"]
    _has("+%.4f", m["paired_delta_M_minus_T"], "the paired delta")
    _has("spread of %.4f", m["paired_delta_sd"], "the paired spread")
    assert abs(m["paired_delta_M_minus_T"]) < m["paired_delta_sd"]
    # the same difference against its standard error, and as a share of the drop
    import numpy as np
    from scipy import stats
    d = np.asarray(m["arm_M"]["per_seed"]) - np.asarray(m["arm_T"]["per_seed"])
    t = d.mean() / (d.std(ddof=1) / np.sqrt(d.size))
    _has("t = %.2f on 9 degrees of freedom", t, "the paired t")
    _has("p = %.2f)", 2 * stats.t.sf(abs(t), d.size - 1), "its p")
    _has("is %d%% of the observed drop", round(100 * d.mean() / g["observed_drop"]), "the share of the drop")


def test_the_paper_construction_and_depth_numbers():
    p, d20 = REC["paper_null"], REC["depth20"]
    _has("gives %.4f ± ", p["mean"], "the rebuilt paper construction mean")
    _has("± %.4f, against", p["sd"], "the rebuilt paper construction sd")
    assert d20["null_floor"]["mean"] == REC["default"]["null_floor"]["mean"]
    s = p["frames_vs_paper"]
    assert s["labels_equal"] and s["shot_partition_equal"] and s["shape_equal"]
    assert "%d of their 48 feature columns are bit-identical" % s["n_columns_bit_identical"] in README
    diffs = sorted(s["unmatched_max_abs_diff_same_position"].values())
    assert len(diffs) == 48 - s["n_columns_bit_identical"] == 7 and max(diffs[:-1]) <= 1.4e-15
    _has("by up to %.1f on nearly constant windows", diffs[-1], "the toroidal-field skew difference")
    _has("%.4f, z", d20["result"]["qfc"], "the depth-20 QFC")
    _has("−%.2f, COLLAPSE", -d20["result"]["z"], "the depth-20 z")
    assert d20["result"]["verdict"] == "COLLAPSE"


def test_every_test_the_objections_table_cites_exists():
    """docs/ATTACKS.md answers each objection with a named test; a renamed or deleted test must not leave a row
    pointing at nothing."""
    import re

    table = (ROOT / "docs" / "ATTACKS.md").read_text(encoding="utf-8")
    cited = set(re.findall(r"test_[a-z0-9_]+", table)) - {"test_reference_identity", "test_core", "test_attributions",
                                                         "test_docs_consistency", "test_regression_cmod"}
    defined = set()
    for f in (ROOT / "tests").glob("test_*.py"):
        defined |= set(re.findall(r"^def (test_[a-z0-9_]+)", f.read_text(encoding="utf-8"), flags=re.M))
    assert cited and not cited - defined, f"ATTACKS.md cites tests that do not exist: {sorted(cited - defined)}"
    for f in re.findall(r"`(tests/[a-z_]+\.py|docs/[A-Za-z_]+\.(?:json|md)|scripts/[a-z_]+\.py)`", table):
        assert (ROOT / f).exists(), f"ATTACKS.md cites a missing file {f}"


def test_the_readme_quotes_no_version_2_number():
    """Version 2's worked example (QFC 0.9607, z = -2.69) came from a different procedure; it must not reappear."""
    for stale in ("0.9607", "2.69", "0.961,", "0.391", "threshold_flag"):
        assert stale not in README, f"README still carries the version 2 figure {stale}"
