# Changelog

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
QFC 0.9626, floor 0.9801 ± 0.0072, z = -2.43 to QFC 0.9607, floor 0.9805 ± 0.0069, z = -2.87, still COLLAPSE.
The TCV L-mode to H-mode case is unchanged (its features have no undefined values).

## 3.0.1 (2026-11-03)

Wording only; no number or behaviour changed. The README and the demo said the 2014-2016 C-Mod discharges
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
