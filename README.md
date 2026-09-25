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
pytest -m "not slow"      # 33 tests, no dataset needed, about 25 s
pytest                    # 37, adding 4 that recompute the worked example from the C-Mod release
```

The four `slow` tests are the scientific regression: they rebuild the worked example below from the
dataset and fail if its verdict moves. They skip themselves when the release is absent, so a fresh clone
is green without it.

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
