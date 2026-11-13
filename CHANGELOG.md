# Changelog

## Unreleased

Documentation only. The summary of the two class-balance controls now says the prior explains at most part
of the drop, as the controls themselves conclude (it said it does not explain it); the 0.164 margin is labelled
as a z margin; the worked example's command writes `--out` as the README's does; and the README states the cut
before explaining a negative z.

## 3.0.4 (2026-11-13)

No change to the measurement or to any number. Three wording fixes in `src/` (below); the rest is documentation,
tests, CI and packaging metadata. The README opens with the C-Mod result on the Open Density Limit
Database and a three-command way to reproduce it, states that the package is independent of MIT PSFC and the
DisruptionPy maintainers, and credits DisruptionPy, the database, DPRF and Montes et al. (2019). It now also says
what "the paper" is (unpublished; its theorem covers Bayes-optimal classifiers under its stated assumptions, and
random forests are measured, not proved), that only the training data need labels, and what a negative z means;
the reconciliation with the paper's C-Mod floor is folded into a collapsible section, its text unchanged.

Corrections against the record and the code. The single-feature sentence said plasma current's minimum is
"still" the largest feature; in the record's fold-order-0 vectors it is the largest on the target, but on the
source its mean is (0.07049 against 0.07036), so the README now says plasma current's mean and minimum are the
source's two largest and its minimum the target's largest. The cut is given with its formula and its assumption
(exact for independent, normal draws); z with its formula; the ROBUST rule as the code applies it (the floor's
mean not below the chance floor's mean); `shot_id` is added to the list of excluded columns (now in `docs/API.md`); the
KernelSHAP `l1_reg` pin is dated (shap's default since 0.47; the package allows 0.45); the benchmark line no
longer reads as if needing no target labels set QFC apart from Explanation Shift, which needs none either; and
`docs/ATTACKS.md` row H2 no longer cites a README section ("Quick start") that does not exist. A test pins
each.

Usability. The custom-attribution example called an undefined `dprf_contributions`; it is now a self-contained,
numpy-only method (`median_occlusion`) with the calling contract written out (arguments, column order, imputed
values, return shape, absolute values, seeding, refits). The install section states the Python and git
requirement and the `curl.exe` spelling on Windows PowerShell 5.1; the demo's runtime is documented (531 s on 32
cores in the record, a laptop not timed) with a light installation check whose numbers are marked as not the
record's; the result's main fields and every exported name are listed. Tests execute both README code blocks
against stubs (nothing is fitted) and require every exported name to be documented.

Engineering. Three tests in the default group, none of which fits a model: the record's `src_sha256` must equal the hash of the code now in `src/` (so any change there, a docstring included, fails CI until the record is regenerated, and "no code changed" is checked rather than asserted); `pyproject.toml`, `__version__`, `CITATION.cff`, the newest released heading here and the README's install tag must name one version, and `CITATION.cff` and this file one release date; and `CITATION.cff` must carry the keys CFF 1.2.0 requires and agree with `pyproject.toml` and `LICENSE` on licence, repository and author (the file also passes the official CFF 1.2.0 JSON schema with no errors). CI moves from `actions/checkout@v4` and `actions/setup-python@v5`, which run on Node 20 (end of life April 2026), to `@v7` of both; none of the inputs the workflow uses changed. `pyproject.toml` links this changelog, and `.gitignore` covers the demo's `cmod_result.json` and common editor and linter caches.

Final read. The README's first statement of the cut no longer calls it exact (it is exact only for independent,
normal draws, as its definition says); the paper's C-Mod figures 0.9637 and z = −4.86 are named as its census's
standard scoring, which removes an apparent clash with the primary scoring the package follows; "What this does not
do" now says that a COLLAPSE is a change in reliance, not by itself a fault; `docs/ATTACKS.md` row C2 states the
cut's assumption and row G1 cites the record-hash test.

Restructure. The README is now the short version (the question, the C-Mod result, the three commands, how it works, the limits, the paper, a documentation index and the credits); the detail moved, unchanged apart from its links, to `docs/WORKED_EXAMPLE.md`, `docs/METHOD.md`, `docs/API.md`, `docs/LIMITS.md` and `docs/TESTING.md`. `docs/ATTACKS.md` points to the new locations, the docs tests pin each statement in the file it now lives in, and a new test requires every relative link and anchor in the README and `docs/` to resolve.

Code (`src/`), wording only: the `_kernelshap` docstring now says shap selects ten features per row by least-angle
regression (it said "ten largest contributors") and that the default was `"auto"` before shap 0.47 (it said 0.45); the
missing-label error now points to deriving a 0/1 label when the frame has `time_until_disrupt`; and the
attribution-shape error names the common cause, one array per class. Because `src/` changed, `docs/worked_example.json`
was regenerated with `scripts/regenerate_worked_example.py`; every number in it is unchanged.

## 3.0.3 (2026-11-03)

Documentation and records only; no behaviour changed. `docs/ATTACKS.md` row C4 quoted the 3.0.0 resolution (0.0150)
and now quotes the record's 0.0143; a test now pins the numbers in `docs/ATTACKS.md` to `docs/worked_example.json`
as it already did for the README. Row P20 now says that an exactly constant window gets a skew of 0, as the paper's
builder does, and rows P19 and P20 are in order. The skew tolerance is stated as 4.5e-16 (it was written 4.4e-16, below the measured 4.44e-16). The README says that the paper's C-Mod real value 0.9637 is a mean
over five model seeds, where the package's figures use one. The worked example and the TCV screen were rerun under
3.0.3 on the cloud, with unchanged numbers.

## 3.0.2 (2026-11-03)

The worked example's features now match the paper's (42 of 48 columns bit-identical, the six skews within 4.5e-16). `build_rolling_features` returned the skew of a window whose
spread is at floating-point resolution as a ratio of rounding errors (on the C-Mod toroidal field it differed from the paper's value by up to 1.0); it now
returns NaN there, by the rule under which `scipy.stats.skew`, which the paper's builder calls, returns NaN (second
central moment at or below `(eps * mean)**2`), and the skew value elsewhere is scipy's own. The slope is now computed
for every window, as in the paper. `screen`, `compute_qfc`, `compute_null_floor` and `compute_chance_floor` replace a
missing or undefined feature value with the source frame's median of its column, for source and target, as the
paper did, and report the count per column (`provenance["imputed_with_source_median"]`, `summary()`); they used to
refuse it. On C-Mod 18 source values are imputed, the same rows and the same median as the paper's, and 42 of the 48
columns are now bit-identical to the paper's arrays (the six skews within 4.5e-16), so the paper's own null
construction rebuilt on these frames gives the paper's floor, 0.9913 ± 0.0057. The worked example moves from
QFC 0.9626, floor 0.9801 ± 0.0072, z = −2.43 to QFC 0.9607, floor 0.9805 ± 0.0069, z = −2.87, still COLLAPSE.
The TCV L-mode to H-mode case is unchanged (its features have no undefined values).

## 3.0.1 (2026-11-03)

Wording only; no number or behaviour changed. The README and the demo said the 2014–2016 C-Mod discharges
have no density-limit labels; the public release labels them stable throughout, which is why they are left out.
`docs/ATTACKS.md` row P14 named a row-weighting attribution callable as a remedy that the README does not give;
the remedy it gives, resampling to a common time base, is the one now cited. `docs/worked_example.json` was
regenerated under 3.0.1 and its numbers are unchanged.

## 3.0.0 (2026-11-02)

The statistic, cut and scoring rules are now the paper's, the null is built as the paper's reference implementation builds it, and the package is tested against that implementation
number for number. On the C-Mod worked example the null floor is unchanged (0.9801 ± 0.0072), and the QFC and z
moved: 2.0.0 reported QFC 0.9607, z = −2.69 (`git show v2.0.0:docs/slow_suite_last_run.json`); 3.0.0 reports 0.9626, z = −2.43,
the same verdict (COLLAPSE). Results from 2.0.0 should not be compared with 3.0.0 results digit for digit.

Changed
- The reported QFC is the mean over 10 fold orders (`n_fold_orders`), the paper's primary scoring. The value at
  each order and the verdict each order would give are reported too. Orders with identical folds are computed
  once, and the deployed model is fitted once.
- The default model has no depth limit (`max_depth=None`), the paper's setting; it was 20. `default_model(max_depth=20)`
  restores it.
- The explained-row sample is drawn the paper's way, with a fresh generator seeded per vector.
- The verdict boundary is inclusive: z equal to minus the cut is COLLAPSE, z equal to the cut is ROBUST.
- A floor whose draws have no spread gives `UNDEFINED` (z is `None`), no longer `UNCALIBRATED`, which now means
  only "no floor".
- The null floor needs at least three usable draws (it accepted two).
- Non-tree models are explained with KernelSHAP by default, the paper's method; the label-free permutation method
  remains available as `attribution="permutation"`.
- The bootstrap interval is off by default (`n_bootstrap=0`); it is not part of the verdict.
- Only a draw that cannot be measured because of its data (`DegenerateDrawError`) is skipped; any other error in a
  null or chance-floor draw is raised rather than counted as a skip.
- `QFCResult` fields: added `qfc_per_order`, `attribution`, `n_distinct_splits`, `resolution`,
  `verdict_unadjudicated`, `verdict_effective`, `cut_effective`, `verdict_per_order`, `chance_floor_mean`,
  `chance_floor_sd`, `n_chance_draws`, `adjudication`, `null_floor` and `chance_floor` (both floors in full);
  removed `threshold` and `threshold_flag`; `ci_lower` and `ci_upper` are `None` when no interval was computed.
  `NullFloor` gained `n_distinct_splits` and `cut_effective`. `to_dict()` is strict JSON (no NaN).
- A target's label column may have missing values; they are left out of the AUC.
- Minimum dependency versions are now the ones CI tests: numpy 1.24, pandas 2.0, scikit-learn 1.3, scipy 1.10, shap 0.45.

Added
- The chance-floor rule: a ROBUST reading whose null floor lies below a permuted-label chance floor is read
  INDISTINGUISHABLE (`compute_chance_floor`; `screen` builds it when the reading is ROBUST).
- The distinct-split count: when two null draws use the same split of shots, the floor records how many distinct
  splits it holds, warns, and gives `verdict_effective`, the reading at the cut for that count; the verdict itself
  uses the nominal count, as the paper does.
- `attribution=`: `"treeshap"`, `"saabas"` (tree-path contributions, the method DPRF uses), `"kernelshap"`,
  `"permutation"`, or your own callable returning one row of contributions per explained row.
- `qfc_statistic`, `exact_cut`, `verdict_label`, `tie_order_groups`, `DegenerateDrawError` as public names.
- `time_until_disrupt` and `time_domain` (DisruptionPy's time-to-event and phase columns) are never used as features.
- Missing feature or source-label values, non-numeric features and invalid settings are refused with an error.
- Identity tests against the paper's reference code (`tests/test_reference_identity.py`).
- Integration tests in CI on a frame DisruptionPy builds from MAST's open data; a missing frame fails there.
- CI jobs for the oldest allowed dependency versions and for the built wheel.
- `docs/worked_example.json` and `scripts/regenerate_worked_example.py`: every number the README states, from one run.
- `docs/ATTACKS.md`, `CITATION.cff`, this changelog.

Removed
- The fixed `threshold` option (a cut of 0.391 borrowed from another domain in version 1; it was never a verdict).
- The notebook, which repeated the worked example with version 2's numbers; the README and `demo` cover it.
- The separate version 2 records of the label-prior controls and the slow suite; both controls were rerun under
  3.0.0 and are in `docs/worked_example.json`.

## 2.0.0 (2026-09-26)

A packaged rewrite of the August prototype: a null floor built from no-shift splits of the source, the exact
Student-t cut, a shot-level bootstrap with multiplicity, a seeded random explained-row sample, and a test suite.

## 1.0 (2026-08-13)

A single-module prototype that read QFC against a fixed threshold.
