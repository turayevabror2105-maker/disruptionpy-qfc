"""Every number the README and the files in ``docs/`` state about the worked example must be the one in
``docs/worked_example.json``.

The record is written by ``scripts/regenerate_worked_example.py``; the slow tests check that the worked example still
reproduces it. These tests pin each statement in the file it lives in (the README's short version, or the docs file
that holds the detail) to the record, need no data, and fail the moment the two disagree.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import disruptionpy_qfc as dq

ROOT = Path(__file__).resolve().parents[1]
DOC_FILES = ["README.md"] + sorted("docs/" + f.name for f in (ROOT / "docs").glob("*.md"))
RAW = {name: (ROOT / name).read_text(encoding="utf-8") for name in DOC_FILES}
TEXT = {name: " ".join(raw.split()) for name, raw in RAW.items()}   # line breaks do not matter
README = TEXT["README.md"]
WORKED = TEXT["docs/WORKED_EXAMPLE.md"]     # the full C-Mod section
METHOD = TEXT["docs/METHOD.md"]             # terms, verdict rules, procedure
API = TEXT["docs/API.md"]                   # frames, options, attributions, exported names
LIMITS = TEXT["docs/LIMITS.md"]             # what the package does not do, in full
REC = json.loads((ROOT / "docs" / "worked_example.json").read_text(encoding="utf-8"))


def _has(fmt: str, value: float, what: str, doc: str = "docs/WORKED_EXAMPLE.md"):
    text = fmt % value
    assert text in TEXT[doc], f"{doc} does not state {what} as {text}"


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
        "COLLAPSE") in WORKED
    assert f["n_seeds"] == 30 and len(r["qfc_per_order"]) == 10
    fr = d["frames"]
    assert "%s rows from %d discharges" % (format(fr["source_rows"], ","), fr["source_shots"]) in WORKED
    _has("%.3f;", d["priors"]["source"], "the source prior")
    _has("rate %.3f", d["priors"]["target"], "the target prior")
    # the README's short version of the same result
    for fmt, v, what in (("target AUC of %.3f", r["auc"], "the target AUC"), ("QFC = %.4f,", r["qfc"], "the QFC"),
                         ("at %.4f on average over 30 splits", f["mean"], "the floor mean"),
                         ("z = −%.2f (negative", -r["z"], "z"), ("against %.3f, the 5%% cut for 30", r["exact_cut"], "the cut")):
        _has(fmt, v, what, "README.md")


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
    _has("INDISTINGUISHABLE (z = −%.2f)", -a["z"], "the other floor's z", "README.md")
    _has("COLLAPSE (z = −%.2f)", -p["z"], "the pooled z", "README.md")


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
    _has("± %.4f, the paper's own floor", p["sd"], "the rebuilt paper construction sd")
    # the paper's census receipt stores this row's floor rounded to six places: 0.991265 +- 0.005673
    assert (round(p["mean"], 6), round(p["sd"], 6)) == (0.991265, 0.005673)
    _has("seed 42 is %.4f, the paper's own value", p["real_depth20_seed42_order0"], "the seed-42 real")
    assert round(p["real_depth20_seed42_order0"], 7) == 0.960812      # the paper's seed-42 cap-matched value
    assert d20["null_floor"]["mean"] == REC["default"]["null_floor"]["mean"]
    s = p["frames_vs_paper"]
    assert s["labels_equal"] and s["shot_partition_equal"] and s["shape_equal"]
    assert "%d of their 48 feature columns are bit-identical" % s["n_columns_bit_identical"] in WORKED
    diffs = s["unmatched_max_abs_diff_same_position"]
    assert len(diffs) == 48 - s["n_columns_bit_identical"] and all(k.endswith("_skew") for k in diffs)
    assert "the other %s, the skews, differ by at most 4.5e-16" % {6: "six"}[len(diffs)] in WORKED
    assert max(diffs.values()) <= 4.5e-16
    assert REC["default"]["result"]["provenance"]["imputed_with_source_median"]["source"] == {"toroidal_B_field_skew": 18}
    assert "the same 18 source rows imputed with the same median" in WORKED
    _has("%.4f, z", d20["result"]["qfc"], "the depth-20 QFC")
    _has("−%.2f, COLLAPSE", -d20["result"]["z"], "the depth-20 z")
    assert d20["result"]["verdict"] == "COLLAPSE"


def test_every_number_in_the_objections_table_is_the_records():
    """docs/ATTACKS.md quotes numbers from the worked example too; each must be the record's, and a number that is
    neither the record's nor one of the table's fixed constants (Python versions, the nominal 2.5% and 5% rates, the
    skew tolerance) fails, so a stale figure cannot survive a regeneration."""
    import re

    import numpy as np
    from scipy import stats

    table = (ROOT / "docs" / "ATTACKS.md").read_text(encoding="utf-8")
    d, m, g = REC["default"]["result"], REC["prior_matched"], REC["prior_gap"]
    pooled = REC["split_sensitivity"]["pooled_60"]
    diff = np.asarray(m["arm_M"]["per_seed"]) - np.asarray(m["arm_T"]["per_seed"])
    t = diff.mean() / (diff.std(ddof=1) / np.sqrt(diff.size))
    from_record = {"%.4f" % d["resolution"], "%.3f" % d["auc"], "%.3f" % (abs(pooled["z"]) - pooled["cut"]),
                   "%.2f" % (2 * stats.t.sf(abs(t), diff.size - 1))}
    constants = {"2.5", "3.10", "3.12", "4.5e-16"}
    found = set(re.findall(r"\d+\.\d+(?:e-\d+)?", table))
    assert from_record <= found, f"ATTACKS.md lacks the record's {sorted(from_record - found)}"
    assert found <= from_record | constants, f"ATTACKS.md quotes numbers not in the record: {sorted(found - from_record - constants)}"
    pct = set(re.findall(r"\d+%", table))
    assert pct == {"5%", "%d%%" % round(100 * diff.mean() / g["observed_drop"])}, pct
    s = REC["paper_null"]["frames_vs_paper"]
    assert set(re.findall(r"\d+ of \d+", table)) == {"%d of 48" % s["n_columns_bit_identical"]}
    assert max(s["unmatched_max_abs_diff_same_position"].values()) <= 4.5e-16


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


def test_the_docs_quote_no_version_2_number():
    """Version 2's worked example (QFC 0.9607, floor 0.9801 +- 0.0072, z = -2.69) and version 3.0.0's (QFC 0.9626,
    z = -2.43) came from other procedures or features; they must not reappear. Version 3.0.2's QFC happens to round
    to 0.9607 as well, so that figure is allowed only as the record's own value."""
    for stale in ("2.69", "0.0072", "0.9626", "2.43", "0.391", "threshold_flag"):
        for name, text in TEXT.items():
            assert stale not in text, f"{name} still carries the superseded figure {stale}"
    assert "%.4f" % REC["default"]["result"]["qfc"] == "0.9607"


def test_the_signal_shares_the_readme_states():
    """The README's signal-level reading (density overtakes plasma current) is computed from the record's own
    fold-order-0 vectors. This pins each share it quotes, the order it claims, and the single largest feature."""
    from scipy.stats import spearmanr

    r = REC["default"]["result"]
    f, s, t = r["feature_cols_used"], r["source_importance"], r["target_importance"]
    assert spearmanr(s, t).correlation == pytest.approx(r["qfc_per_order"][0], abs=1e-12)   # they are order 0's

    def share(v, signal):
        return sum(x for c, x in zip(f, v) if c.startswith(signal + "_")) / sum(v)

    for signal in ("density", "plasma_current"):
        assert sum(c.startswith(signal + "_") for c in f) == 8
    d_s, d_t = share(s, "density"), share(t, "density")
    p_s, p_t = share(s, "plasma_current"), share(t, "plasma_current")
    assert p_s > d_s and d_t > p_t
    assert "density's share of the total attribution rises from %.1f%% to %.1f%%" % (100 * d_s, 100 * d_t) in WORKED
    assert "density's share of the attribution rises from %.1f%% to %.1f%%" % (100 * d_s, 100 * d_t) in README
    for doc in (WORKED, README):
        assert "plasma current's falls from %.1f%% to %.1f%%" % (100 * p_s, 100 * p_t) in doc
    top = sorted(zip(t, f), reverse=True)
    assert [c for _, c in top[:3]] == ["plasma_current_min", "density_max", "density_mean"]
    src_top = sorted(zip(s, f), reverse=True)
    assert [c for _, c in src_top[:2]] == ["plasma_current_mean", "plasma_current_min"]
    assert ("plasma current still leads: its mean and its minimum are the two largest in the source, and its "
            "minimum is the largest in the target, now followed by density's maximum and mean") in WORKED
    assert ("its mean and its minimum are the two largest on the training campaigns, and its minimum is the "
            "largest on the later ones") in README


def test_the_docs_state_what_the_code_does():
    """Statements about the code a reader would act on: the excluded columns and the shap versions the KernelSHAP
    pin is about (docs/API.md), the cut's formula and the chance-floor rule (docs/METHOD.md and the README)."""
    import math

    from scipy import stats

    for c in sorted(set(dq.DEFAULT_EXCLUDE_COLS) - {"shot", "label"}):
        assert "`%s`" % c in API, f"docs/API.md does not name the excluded column {c}"
    assert "t(0.975, 29) × √(1 + 1/30)" in METHOD
    assert dq.exact_cut(30) == pytest.approx(stats.t.ppf(0.975, 29) * math.sqrt(1 + 1 / 30), abs=1e-15)
    assert "%.3f for 30 null draws" % dq.exact_cut(30) in METHOD
    assert "It is exact when the draws are independent and normal" in METHOD
    assert "(exact for independent, normal draws)" in README
    assert "z ≥ cut, and the floor's mean is not below the chance floor's mean" in METHOD
    assert "ROBUST if z ≥ cut and the floor's mean is not below that of a floor built with shuffled labels" in README
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'shap>=0.45' in pyproject and "this package allows shap 0.45" in API


def test_the_runtime_the_readme_states_is_the_records():
    for doc in ("README.md", "docs/WORKED_EXAMPLE.md", "docs/API.md"):
        assert "took %d seconds" % round(REC["default"]["seconds"]) in TEXT[doc], doc


def test_the_readme_keeps_its_scope_and_independence_statements():
    """The README's honesty statements are part of what it claims: the package is independent, the C-Mod collapse is
    near the line, the paper is unpublished, its theorem does not cover the random forests the package tests, and a
    COLLAPSE is a change in reliance, not proof of a fault. Each is pinned in the file it lives in."""
    for line in ("not affiliated with, reviewed or endorsed by MIT PSFC or the DisruptionPy maintainers",
                 "It is close to the line.",
                 "(A. Turayev, not yet published;",
                 "Random forests, including the one in the C-Mod example, are outside the proof",
                 "for them the change is measured, not proved",
                 "Only the training data need labels",
                 "A COLLAPSE is not by itself a fault.",
                 "not that the new reliance is wrong"):
        assert line in README, f"README no longer states: {line}"
    for line in ("A COLLAPSE is not by itself a fault.",
                 "not that the new reliance is wrong: it may be the same physics reached through another signal"):
        assert line in LIMITS, f"docs/LIMITS.md no longer states: {line}"
    for line in ("scores this row in its standard scoring",
                 "Read this as evidence that the explanation moved, near the edge of what this pool can resolve, "
                 "not as a settled collapse.",
                 "The paper's census, which is not yet published"):
        assert line in WORKED, f"docs/WORKED_EXAMPLE.md no longer states: {line}"


FENCE = "```"


def _python_block(marker: str, doc: str) -> str:
    """The one Python code block in ``doc`` that contains ``marker``, as written."""
    blocks = [b.split(FENCE)[0] for b in RAW[doc].split(FENCE + "python")[1:]]
    found = [b for b in blocks if marker in b]
    assert len(found) == 1, f"expected one Python block containing {marker!r}, found {len(found)}"
    return found[0]


def test_every_public_name_is_in_the_api_reference():
    """The table in docs/API.md names everything the package exports, so a reader never meets an undocumented name."""
    for name in dq.__all__:
        assert "`%s" % name in RAW["docs/API.md"], f"docs/API.md does not document the exported name {name}"


class _StubModel:
    """A fixed logistic model: lets the README's attribution example run without fitting anything."""

    def __init__(self, w):
        self.w = np.asarray(w, float)

    def predict_proba(self, X):
        p = 1.0 / (1.0 + np.exp(-(np.asarray(X, float) @ self.w)))
        return np.column_stack([1.0 - p, p])


def test_the_custom_attribution_example_runs_as_written(monkeypatch):
    """The custom method in docs/API.md is self-contained: executing its block defines it, it hands it to screen(),
    and the package accepts what it returns. Nothing is fitted: screen() is replaced and the model is a stub."""
    import disruptionpy_qfc.core as core

    assert all("dprf_contributions" not in raw for raw in RAW.values())
    calls = []
    monkeypatch.setattr(dq, "screen", lambda *a, **k: calls.append(k) or "result")
    ns = {"source_df": None, "target_df": None}
    exec(compile(_python_block("def median_occlusion", "docs/API.md"), "docs/API.md", "exec"), ns)
    fn = ns["median_occlusion"]
    assert calls == [{"attribution": fn}]

    X = np.random.default_rng(0).normal(size=(40, 5))
    stub = _StubModel([3.0, 1.0, 0.0, 0.0, 0.5])
    assert fn(stub, X, X, 0).shape == X.shape
    imp = core._explain(stub, X, X, None, 0, fn)
    assert imp.shape == (5,) and np.all(np.isfinite(imp))
    assert int(np.argmax(imp)) == 0 and imp[2] == 0.0 and imp[3] == 0.0   # unused features get exactly zero
    assert core._explain(stub, X, X, 10, 0, fn).shape == (5,)             # and under a row cap


def test_the_light_check_is_accepted_and_its_numbers_hold(monkeypatch):
    """The quick installation check in docs/WORKED_EXAMPLE.md passes only arguments screen() accepts, and the fit count and the
    cut it states follow from its settings. Nothing is loaded or fitted: the frames and screen() are replaced."""
    import inspect

    import disruptionpy_qfc.demo as demo

    accepted = set(inspect.signature(dq.screen).parameters) | set(inspect.signature(dq.compute_qfc).parameters)
    seen = []

    class _R:
        def summary(self):
            return ""

    def fake_screen(source, target, **kw):
        assert set(kw) <= accepted, set(kw) - accepted
        seen.append(kw)
        return _R()

    monkeypatch.setattr(dq, "screen", fake_screen)
    monkeypatch.setattr(demo, "build_demo_frames", lambda path: ("source", "target"))
    exec(compile(_python_block("build_demo_frames(", "docs/WORKED_EXAMPLE.md"), "docs/WORKED_EXAMPLE.md", "exec"), {})
    kw = seen[0]
    assert kw["model"].n_estimators == 100 and kw["n_fold_orders"] == 1 and kw["shap_cap"] == 200
    n = kw["n_seeds"]
    assert "That is %d fits of 100 trees instead of up to 231 of 500" % (1 + 5 * kw["n_fold_orders"] + 6 * n) in WORKED
    assert "(%d more if the reading is ROBUST" % (6 * dq.N_CHANCE_DRAWS) in WORKED
    assert "%d null draws give a cut of %.3f instead of %.3f" % (n, dq.exact_cut(n), dq.exact_cut(30)) in WORKED


def _anchors(raw: str) -> set:
    """GitHub's heading anchors for a Markdown file: lower case, punctuation other than - and _ dropped, spaces to -."""
    out, fenced = set(), False
    for line in raw.splitlines():
        if line.startswith(FENCE):
            fenced = not fenced
        elif line.startswith("#") and not fenced:
            title = line.lstrip("#").strip().lower()
            out.add("".join(c for c in title if c.isalnum() or c in "-_ ").replace(" ", "-"))
    return out


def test_every_relative_link_resolves():
    """Every relative link in the README and in docs/ points at a file that exists, and every #anchor at a heading in
    it, so moving text between files cannot leave a reader at a dead link. The README indexes every docs file."""
    import re

    checked = 0
    for name, raw in RAW.items():
        here = (ROOT / name).parent
        for target in re.findall("[]][(]([^) ]+)[)]", raw):
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            path, _, anchor = target.partition("#")
            dest = (here / path).resolve() if path else (ROOT / name).resolve()
            assert dest.exists(), f"{name} links to a missing file {target}"
            if anchor:
                assert dest.suffix == ".md" and anchor in _anchors(dest.read_text(encoding="utf-8")), (
                    f"{name} links to a missing anchor {target}")
            checked += 1
    assert checked > 20
    for name in DOC_FILES[1:] + ["docs/worked_example.json"]:
        assert "](%s)" % name in RAW["README.md"], f"the README's documentation index does not link {name}"


def _src_sha256() -> str:
    """The hash ``scripts/regenerate_worked_example.py`` records: every ``src/disruptionpy_qfc/*.py`` in name order,
    each as its name followed by its bytes, with CRLF read as LF so a Windows checkout hashes like Linux."""
    import hashlib

    h = hashlib.sha256()
    for f in sorted((ROOT / "src" / "disruptionpy_qfc").glob("*.py")):
        h.update(f.name.encode() + f.read_bytes().replace(bytes([13, 10]), bytes([10])))
    return h.hexdigest()


def test_the_record_was_made_by_the_code_in_src():
    """The record's numbers come from the code in ``src/`` as it is now. Any change there, a docstring or the
    version string included, fails this test until ``docs/worked_example.json`` is regenerated, so a changelog
    entry that says "no code changed" is checked rather than asserted. Skipped where ``src/`` is absent (the CI job
    that tests the installed wheel hides it), since there is no source tree to hash."""
    if not (ROOT / "src" / "disruptionpy_qfc").is_dir():
        pytest.skip("no src/ tree here (installed-wheel test run)")
    assert REC["default"]["env"]["src_sha256"] == _src_sha256(), (
        "src/ differs from the code that produced docs/worked_example.json; regenerate the record "
        "(scripts/regenerate_worked_example.py) before releasing")


def test_the_version_and_release_date_agree_everywhere():
    """pyproject.toml, ``__version__``, CITATION.cff, the newest released CHANGELOG heading and the README's
    install tag name one version, and CITATION.cff and the CHANGELOG one release date. Read with regular
    expressions, so the test needs neither tomllib (Python 3.11+) nor a YAML parser."""
    import re

    v = dq.__version__
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    cff = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert re.search(r'(?m)^version = "([^"]+)"$', pyproject).group(1) == v
    assert re.search(r"(?m)^version: ([^ ]+)$", cff).group(1) == v
    released = re.findall(r"(?m)^## ([0-9]+[.][0-9]+[.][0-9]+) [(]([0-9]{4}-[0-9]{2}-[0-9]{2})[)]$", changelog)
    assert released and released[0][0] == v, "the newest released CHANGELOG heading is not this version"
    assert re.search(r"(?m)^date-released: ([^ ]+)$", cff).group(1) == released[0][1]
    assert "disruptionpy-qfc@v%s" % v in README, "the README's install line does not pin this version's tag"


def test_the_citation_file_names_what_the_package_declares():
    """CITATION.cff carries the keys CFF 1.2.0 requires, and its licence, repository and author agree with
    pyproject.toml and LICENSE. (Its full schema was checked against the official CFF 1.2.0 JSON schema, with zero
    errors, when this test was written; that check needs PyYAML and jsonschema, which the package does not use.)"""
    import re

    cff = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    for key in ("cff-version: 1.2.0", "message: ", "title: ", "authors:", "type: software"):
        assert re.search("(?m)^" + re.escape(key), cff), key
    assert re.search("(?m)^license: MIT$", cff) and "License :: OSI Approved :: MIT License" in pyproject
    assert (ROOT / "LICENSE").read_text(encoding="utf-8").startswith("MIT License")
    repo = re.search('(?m)^Homepage = "([^"]+)"$', pyproject).group(1)
    assert re.search('(?m)^repository-code: "%s"$' % re.escape(repo), cff)
    assert 'authors = [{ name = "Abror Turayev" }]' in pyproject
    assert "family-names: Turayev" in cff and 'orcid: "https://orcid.org/0009-0000-3182-2830"' in cff
