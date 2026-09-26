# disruptionpy-qfc

**Does a disruption predictor that still performs on a new campaign do it for the same reasons?**

A model trained on one set of discharges can keep a high AUC on a later campaign, or on another machine,
while the features it actually relies on have changed underneath. That is invisible to any accuracy
metric. This package measures it before deployment, without needing labels on the target.

It works on DisruptionPy-format frames: one row per `(shot, time)`, a binary label, and a shot
identifier.

```python
from disruptionpy_qfc import screen

result = screen(source_df, target_df)   # target labels not required
print(result.summary())
```

**One thing it will not do for you: choose the label.** The source frame needs a binary column, and a
disruption-prediction frame generally does not have one — it carries the physics signals and a
time-to-disruption, from which "positive" is a scientific choice: how many milliseconds before the
disruption counts as the precursor. That choice changes what the model learns and therefore what its
attributions mean, so the package refuses to make it silently. Derive it yourself and pass the column:

```python
source_df["label"] = (source_df["time_until_disrupt"] < 0.050).astype(int)   # your convention, not ours
screen(source_df, target_df, label_col="label")
```

If the column is missing the error names the frame, lists its first columns and points at `label_col=`
and `shot_col=`. Target labels are not needed at all; if they are present they are used only to report an
AUC alongside the measurement.

---

## The part that matters: a correlation on its own means nothing

The measurement is a rank correlation between the mean absolute attribution vectors of the two
environments. On real fusion data that number is usually **high** — 0.9 and above is common — and it is
tempting to read a high value as evidence that the model is stable.

It is not, on its own. Split a single campaign in half at random, with no shift of any kind, and run the
identical measurement: that also scores around 0.9, because features have a stable importance ordering
for reasons that have nothing to do with transfer.

So the only interpretable question is **how the observed value compares with a no-shift split of the
source pool itself**. That reference is what `compute_null_floor` builds, and `screen()` computes it with
exactly the model, features, sampling cap and protocol used for the measurement — which is the only way
the comparison means anything.

A raw QFC with no floor is returned with `verdict="UNCALIBRATED"`, deliberately.

---

## Install

```bash
pip install -e .          # from a clone
pytest -m "not slow"      # 47 run in about 60 s with no dataset; 4 more skip
pytest                    # all 55
```

55 tests in three groups, and two of the groups **skip rather than pass** when their input is absent, so
a skip is a statement that nothing was checked:

| group | count | needs | what it is for |
|---|---|---|---|
| default | 47 | nothing | the estimator, the bootstrap, the cap rule, the verdict logic, the column validation, and the numbers this README **and the notebook** quote |
| `slow` | 4 | the C-Mod release | the scientific regression: rebuild the worked example below and fail if its verdict moves. Its last completed run is recorded in `docs/slow_suite_last_run.json` — the measured floor, QFC, AUC, z and verdict, with the seed counts — because CI cannot run this group and a green badge therefore covers none of the science. Two default tests keep that record and this README in step, and skip rather than pass when no run has been recorded |
| `integration` | 4 | `DISRUPTIONPY_FRAME=` pointing at a frame exported from DisruptionPy | does this package accept and screen a frame DisruptionPy actually produced |

The `integration` group is written to run in **your** environment rather than the author's: DisruptionPy
is not installed here, and machine access is not either, so the person who can exercise the integration is
the person evaluating the package. It asserts nothing about DisruptionPy's own columns or API — only that
this package meets its side of the contract on whatever frame you hand it.

Requires Python 3.10+, numpy, pandas, scikit-learn, scipy and shap.

---

## What it computes, exactly

**Source vector.** GroupKFold over shots, out-of-fold attributions, averaged. No discharge is ever
explained by a model that trained on it.

**Target vector.** The model is fit on the *whole* source and applied to the target **without
retraining** — the deploy-once protocol, because that is how a predictor is actually used on a new
campaign or a new machine. Target labels are never used by the measurement; if present they are used
only to report an AUC alongside it.

**Attribution.** TreeSHAP for tree models. For anything else, a label-free permutation importance —
mean absolute change in predicted probability — because at screening time the target has no labels and a
score-drop importance would not be computable.

**Sampling.** Attribution rows are a seeded random subsample, never a head slice: DisruptionPy frames are
ordered by shot and time, so taking the first *n* rows would explain the first few discharges rather than
a sample of the environment.

**Interval.** A shot-level bootstrap **with multiplicity** on both sides — unique shots drawn with
replacement, each drawn shot contributing all of its rows again. The discharge is the independent unit
in this domain, so an interval that resampled rows would treat thousands of correlated points as
independent. Draws that come back single-class are skipped, and `summary()` always reports how many of
the requested draws were usable, with a warning below 80%.

**Seeds.** A null-floor seed is skipped when its half of the pool comes back single-class, or holds fewer distinct shots than folds, or leaves every fold single-class. `NullFloor` records how many seeds were **requested** alongside how many were usable and counts the skips by reason, and both `NullFloor.summary()` and `QFCResult.summary()` print `n/m usable seeds` whenever any were dropped, with the same warning below 80% that the bootstrap line uses. This is not cosmetic: the halves that isolate the rare class are exactly the ones that fail the single-class test, so the surviving splits are the easier ones and a floor built from a fraction of the requested seeds can read optimistically. The exact cut is computed from the seeds actually used, never from the number requested.

**Is the collapse just the label-prior shift?** No, and the floor answers it without a new experiment. The source pool's per-year prevalence runs from 0.1304 (2001, 7 discharges) to 0.7754 (2003, 17 discharges), so each no-shift draw carries its **own** prior gap while the environment is identical by construction — mean 0.1004, max 0.2943, and **two of the thirty draws exceed the real shift's gap of 0.2124**. Those two score **0.9783** and **0.9734** (z = −0.25 and −0.93, both indistinguishable from the floor) where the real comparison scores 0.9607 (z = −2.69): a **wider** prior gap with no environment shift does not reproduce the collapse. Across all thirty draws the relationship between a draw's gap and its QFC is not resolved (Spearman ρ = −0.173, p = 0.361, against a 5% cut of 0.361 at n = 30), and the fitted slope −0.0165 ± 0.0180 puts the prior gap at **18% of the observed drop** at the point estimate and **58% at the steep end of its 95% interval** — so the design excludes the prior gap explaining *all* of the drop, leaving at least 42%, but not it explaining some. Record: `docs/label_prior_within_null_control.json`; pre-registered before it ran. It is one pool and one estimator, and it says nothing about label priors in general.

**Verdict.** `z = (qfc − floor_mean) / floor_sd`, against `t(0.975, n−1)·√(1+1/n)` — the Student-t
prediction-interval cut for a single new observation, which is what an observed QFC is. Not `|z| > 2`,
which is materially too lenient at small seed counts.

---

## API

| function | what it is for |
|---|---|
| `screen(source_df, target_df)` | floor + measurement + verdict. **The intended entry point.** |
| `compute_qfc(source_df, target_df, ...)` | the measurement alone; `UNCALIBRATED` without a floor |
| `compute_null_floor(source_df, n_seeds=30)` | the no-shift reference for one pool, model and protocol |
| `default_model()` | 500 trees, depth 20, balanced — the configuration this literature uses |

`QFCResult.to_dict()` is JSON-serialisable and carries full provenance: package version, estimator,
seed, sampling rule, bootstrap rule and counts.

Defaults: `n_bootstrap=200`, `shap_cap=500`, `n_splits=5`, `n_seeds=30`, `random_state=42`. Thirty null
seeds because the exact cut widens materially below that; two hundred bootstrap draws because percentile
ends are unstable below about one hundred.

---

## Worked example — Alcator C-Mod density limit

`python -m disruptionpy_qfc.demo --data /path/to/DL_DataFrame.csv`

Trains on the 2000–2003 campaigns and deploys on 2005–2009, using six signals expanded into causal
rolling-window features over the ten preceding time points within each shot.

| | source (2000–2003) | target (2005–2009) |
|---|---|---|
| rows | 3,363 | 2,504 |
| discharges | 40 | 38 |
| positive rate | 0.508 | 0.721 |
| features | 48 | 48 |

**Result, measured at the package defaults on 2026-09-16:**

| quantity | value |
|---|---|
| target AUC | **0.938** |
| QFC | **0.961**, 95% CI [0.929, 0.972] from 200/200 usable shot-bootstrap draws |
| no-shift floor | **0.980 ± 0.007** over 30 seeds |
| z against the exact cut of 2.079 | **−2.69** |
| verdict | **COLLAPSE** |

**This is the whole argument for the package in one line.** The model still predicts well. Its
attribution agreement of 0.961 looks excellent. But two halves of the *source campaign alone* agree
with each other at 0.980 — more than the two campaigns agree. The observed value is not merely
uninformative, it is **below its own floor**, and no accuracy metric would have revealed that.

Note the positive rate also moves, 0.508 to 0.721, so this is a label-prior shift as well as a
covariate shift. QFC does not separate the two — but the two **have** been separated, by measurement
rather than by argument, and the prior is not what produces the collapse.

Holding the source model, the feature pipeline, the 500-row cap and the cap rule fixed, and changing
only the label composition of the explained target sample — one arm stratified to the target's own prior
(360 positive, 140 negative), the other to the source's (254/246), ten paired seeds — the two arms give
**0.9605 ± 0.0086** and **0.9614 ± 0.0093**. The paired difference is **+0.0008** against a paired
across-seed spread of **0.0050**, i.e. smaller than the spread the design can resolve, and **both arms
collapse** against the same no-shift floor, at z = −5.42 and −5.27. The second arm exists so that
"stratified rather than unstratified draw" is not a second changed variable; both are stratified draws of
exactly 500 rows.

Two limits belong with that. Both arms draw stratified samples, so neither z equals the −2.69 above,
which is an unstratified draw — the comparison that matters is between the arms, which share everything
but composition. And the control changes the composition of the *explained* set, not of the *training*
set: the forest is fitted on the source pool at 0.508 throughout, which is what the deploy-once protocol
describes. The record, including the pre-registration that fixed all four possible outcomes before the
run, is `docs/prior_matched_control.json`; it was produced by the research pipeline behind the paper on
these same frames, not by this package, and `tests/test_docs_consistency.py` keeps the numbers above and
that file in step.

The 2014–2016 discharges are excluded, not silently treated as negatives: they carry no density-limit
precursor labels in the public release. That is a curation gap and the demo says so.

Data: Maris, A. D., Rea, C., Trevisan, G. L., & the Alcator C-Mod Team, *The Open Density Limit
Database*, MIT Plasma Science and Fusion Center (2025). CC BY.

---

## What this does not do

- **It is not a disruption predictor.** It screens one, and says nothing about whether the model is any
  good — only whether it relies on the same things in both environments.
- **A high QFC is not a safety argument.** It means attribution ordering is preserved, which is
  necessary for a stable explanation and nowhere near sufficient for safe deployment.
- **The floor is per pool, per model, per protocol.** A floor computed for one machine does not transfer
  to another, and the package will not let you reuse one implicitly.
- **Rank correlation discards magnitude.** Two environments can agree perfectly on ordering while
  differing in how much any feature matters.
- **Ties degrade it.** A model that assigns many features exactly zero importance makes part of the
  ranking arbitrary, which lowers sensitivity rather than biasing the value.

---

## Licence and citation

MIT. If it is useful in published work, please cite the dataset above and this repository.
