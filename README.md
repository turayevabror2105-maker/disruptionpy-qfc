# disruptionpy-qfc

**When a disruption predictor still scores well on a new campaign or machine, is it still relying on the same signals?**

A model trained on one set of discharges can keep a high AUC on a later campaign while the features it leans on
change underneath. Accuracy metrics cannot see that. This package measures it before deployment, without labels on
the new data, and tells you whether what it measured is more than the change you would see with no shift at all.

It works on DisruptionPy-format frames: one row per `(shot, time)`, a shot column and, on the training data, a
binary label. The measurement and its verdict rules are those of the paper that defines them (Turayev, *Pareto
impossibility of attribution consistency under deployment shift*, in preparation), and the tests check the package
against that paper's reference implementation number for number.

## Quick start

```bash
pip install git+https://github.com/turayevabror2105-maker/disruptionpy-qfc@v3.0.1
```

```python
from disruptionpy_qfc import screen

result = screen(source_df, target_df)      # target labels are not needed
print(result.summary())
```

`source_df` is the data the model is trained on, `target_df` the data it will be deployed on. Both need a `shot`
column; the source also needs a binary `label` column. Every other numeric column is used as a feature except
`time`, `time_until_disrupt` and `time_domain`; the list used is in `result.feature_cols_used`, and any other
time-to-event or bookkeeping column in your frame should be dropped or excluded with `feature_cols=`.

A frame straight out of DisruptionPy has no binary label, only `time_until_disrupt`. Which rows count as positive
(how long before a disruption) is a scientific choice that changes what the model learns, so the package will not
make it for you. Make it, drop the time-to-event column, then deal with missing values:

```python
from disruption_py.workflow import get_shots_data

df = get_shots_data(tokamak="cmod", shotlist_setting=my_shots, output_setting="dataframe")
df["label"] = (df["time_until_disrupt"] < 0.050).astype(int)       # your convention, not ours
df = df.drop(columns=["time_until_disrupt"]).dropna(axis=1, how="all").dropna()
```

That last `dropna()` removes every row with any missing signal, which can remove whole shots, and not at random
(a diagnostic that was off, a shot that ended early); check how many shots survive, or impute instead. The
package refuses a frame with a missing feature value, a non-binary source label, a non-numeric feature or fewer
than three features, and the error says which. Missing target labels are allowed; they are left out of the AUC.

## What the words mean

| term | meaning |
|---|---|
| **attribution** | how much each feature contributed to one prediction. By default TreeSHAP for tree models and KernelSHAP for others |
| **importance vector** | the mean absolute attribution of each feature over the explained rows (every row counts once, so long shots count for more) |
| **QFC** | Spearman's rank correlation between the source and target importance vectors: 1 means the features keep the same order of importance, lower means the order changed |
| **fold order** | the source vector is built out of fold (no shot is explained by a model that trained on it). Which equal-sized shots share a fold is arbitrary, so the reported QFC is the mean over 10 such orders |
| **null floor** | the same measurement between two random halves of the source, 30 times. There is no shift between two halves, so this is what QFC looks like when nothing changed |
| **z** | how many null-floor standard deviations the observed QFC lies from the floor's mean |
| **cut** | 2.079 for 30 null draws, the exact two-sided 5% cut for one new value against 30 (wider than the familiar 2 because 30 draws is a small sample) |
| **resolution** | cut × floor sd: the smallest change in QFC this floor can detect |
| **chance floor** | the same measurement with the source labels shuffled, 15 times: what a model trained on noise gives |

## The verdicts

| verdict | when | what it means |
|---|---|---|
| **COLLAPSE** | z ≤ −cut | the two environments agree on feature importance *less* than two halves of the source do. The explanation changed under the shift by more than no-shift sampling explains |
| **INDISTINGUISHABLE** | −cut < z < cut | within what two halves of the source give each other. No change was detected; `resolution` says how large a change could have been |
| **ROBUST** | z ≥ cut, and the floor is above the chance floor | the agreement is *higher* than two halves of the source give. If the floor lies below the chance floor, the reading is INDISTINGUISHABLE instead, because then two halves agree less than a model trained on noise does |
| **UNCALIBRATED** | no floor was given (`compute_qfc` alone) | a raw QFC; it cannot be read on its own |
| **UNDEFINED** | every null draw gave the same value | the floor has no spread, so no z exists |

`summary()` prints the floor, z, the cut and the resolution, and warns when null draws had to be skipped (with
the reasons), when the 10 fold orders disagree among themselves, and whenever two null draws used the same split of
shots. In that last case it gives the number of distinct splits and the reading at the cut for that number
(`verdict_effective`); the verdict itself uses the nominal count, as the paper does.

## What it computes, step by step

1. **Source vector.** `GroupKFold` with 5 folds over shots; each fold's model explains the shots it did not see;
   the 5 importance vectors are averaged. Repeated for 10 fold orders (shot labels relabelled by
   `numpy.random.default_rng(1000 + k)`, k = 0 the identity). An order whose folds are the same as an earlier
   order's reuses its vector, so when no two shots have the same number of rows this costs one order, not ten.
2. **Target vector.** One model fitted on the whole source explains the target, without retraining, as a
   deployed predictor would.
3. **Explained rows.** At most 500 per vector (`shap_cap`), a seeded random sample, never the first rows:
   DisruptionPy frames are ordered by shot and time, so the first rows are the first few discharges.
4. **QFC.** Spearman's rho of the two vectors, averaged over the fold orders.
5. **Null floor.** 30 draws; draw k splits the source shots 50/50 (`GroupShuffleSplit(random_state=42 + k)`),
   and the first half plays the source and the second the target, through steps 1 to 4. The floor and the
   measurement go through the same code, with two differences besides the data: a half holds half the shots,
   and each draw uses one fold order. The second makes the null's spread include fold-order noise that the
   averaged real value does not have, so it widens the floor and makes the test, if anything, conservative. The
   halves of a small pool overlap from draw to draw, so the 30 draws are not independent; the worked example
   shows what that does.
6. **Verdict** as in the table above, with the chance floor built only when the reading would be ROBUST.

These are the paper's rules: its statistic, its exact cut, its census's primary scoring (the fold-order mean and
the chance-floor rule), and the null construction of its reference implementation (`qfc_test`).
`tests/test_reference_identity.py` runs the package and a copy of that implementation
(`tests/reference/qfc_metric.py`) on the same inputs and requires exact equality of the statistic, the null draws,
z, the cut, the verdict, the distinct-split reading, the fold-order mean and the chance floor, for random forests
with TreeSHAP; it also recomputes seven of the paper's archived conditions from their stored vectors and checks
the chance-floor rule against the four census rows it changed. The reference has no counterpart for KernelSHAP,
Saabas or permutation, so those are tested on their own (`tests/test_attributions.py`). Most of the paper's census
rows were scored against a differently built null, one forest per half with in-sample attributions on the first
half; the worked example below shows what that changes. The census-level chance counts and the census's secondary
construction-matched campaign describe many conditions at once and are not part of a single screening.

## Your own attributions

```python
screen(source_df, target_df, attribution="saabas")          # or "treeshap", "kernelshap", "permutation"

def my_contributions(model, X_explain, X_train, seed):      # any method: one row of per-feature values per row
    return dprf_contributions(model, X_explain)              # e.g. what your predictor already computes
screen(source_df, target_df, attribution=my_contributions)
```

`"saabas"` is the tree-path decomposition of the `treeinterpreter` package, which is how the DPRF predictor
computes its feature contributions (Rea et al., *Nucl. Fusion* 59, 096016, 2019); a test checks it against the
decomposition written out from the tree arrays. `"kernelshap"` uses a 30-centroid k-means background and 200
samples per row, as the paper does for non-tree models, with shap's `l1_reg` pinned to `"num_features(10)"`: with
more than ten features, each row's attributions are fitted on the ten features a least-angle regression selects
and the rest are set to zero. Because Spearman's rho counts every rank equally, those zeros shape the low end of
the ranking. That is shap's current default, pinned because the default changed between shap versions; pass your
own callable for another choice. `"permutation"` is the mean absolute change in predicted probability when one feature is shuffled; it
uses no labels. The null floor always uses the same method as the measurement.

## Options

| argument | default | what it does |
|---|---|---|
| `model` | `default_model()`: random forest, 500 trees, no depth limit, balanced class weights | any scikit-learn classifier with `predict_proba`; cloned for every fit, with its `random_state` set when it has one (not inside a Pipeline). It is fitted on arrays, so steps that select columns by name will not work |
| `attribution` | `"auto"` | see above |
| `shap_cap` | 500 | rows explained per vector; `None` explains all |
| `n_splits` | 5 | folds for the source vector |
| `n_fold_orders` | 10 | fold orders averaged; 1 gives the single-order value |
| `n_seeds` | 30 | null draws |
| `chance_floor` | `"auto"` | built only for a ROBUST reading; `True` always; `False` never |
| `n_chance_draws` | 15 | chance-floor draws (draw d permutes the labels and fits with seed d, as the paper does) |
| `n_bootstrap` | 0 | if set, a shot-level bootstrap 95% interval of the fold-order-0 QFC; each draw also refits with its own seed, so it covers sampling and model-seed variability. Not part of the verdict |
| `random_state` | 42 | the seed of the measurement and of null draw k (`random_state + k`) |
| `feature_cols`, `label_col`, `shot_col` | detected, `"label"`, `"shot"` | column names |

`compute_qfc`, `compute_null_floor` and `compute_chance_floor` run the three parts separately; `screen` runs all
three with the same settings, which is what makes the comparison meaningful. A result's `to_dict()` is strict
JSON and carries both floors in full (every draw, every skip with its reason) and the provenance: package version,
estimator, attribution, seeds and rules. The package screens a model *specification* (estimator and settings),
not a model you have already fitted, because the out-of-fold vector and the null floor need refits.

**Cost.** One `screen` fits the deployed model once, 5 fold models for each distinct fold order and 6 models for
each of the 30 null draws: at most 1 + 50 + 180 = 231 fits, plus the attributions. A chance floor, when built,
adds 6 fits for each of its 15 draws, and a bootstrap 6 per draw. On the C-Mod example below (3,363 source rows, 48 features) it took
about 10 minutes on a 32-core machine. Fitting time grows with the number of rows; attribution time does not,
because at most 500 rows are explained per vector. KernelSHAP is far slower than TreeSHAP: about 6,000 model
evaluations per explained row. Fewer trees, a smaller `shap_cap` or `n_fold_orders=1` make it faster; the
defaults are the paper's.

**Determinism.** TreeSHAP and Saabas results are identical on rerun with the same seed on the same machine;
KernelSHAP is seeded too. Between machines, `GroupKFold`'s ordering of equal-sized shots follows numpy's sort,
which differs between platform builds, so values can differ slightly; averaging over 10 fold orders is what
keeps that arbitrary choice out of the verdict.

## Worked example: Alcator C-Mod, the 2000, 2001 and 2003 campaigns to 2005 and 2009

```bash
python -m disruptionpy_qfc.demo --data DL_DataFrame.csv
```

Data: the public Open Density Limit Database (Maris, Rea, Trevisan and the Alcator C-Mod Team, MIT PSFC, 2025,
CC BY; `github.com/MIT-PSFC/open_density_limit_database`). Six signals, each turned into eight statistics over the
10 preceding time points of its shot (48 features). The source is 3,363 rows from 40 discharges, positive rate
0.508; the target is 2,504 rows from 38 discharges, positive rate 0.721. Discharges from 2014–2016 are left out
because the public release has them labelled stable throughout (no time point in a density-limit phase).

| quantity | value |
|---|---|
| target AUC | 0.939 |
| QFC (mean of 10 fold orders) | 0.9626 (orders range 0.9586 to 0.9672) |
| null floor | 0.9801 ± 0.0072, 30 draws |
| z, against the cut of 2.079 | −2.43 |
| resolution | 0.0150 |
| verdict | **COLLAPSE** (9 of the 10 fold orders collapse on their own) |

**A borderline collapse.** The model still predicts well, and an attribution agreement of 0.963 looks excellent,
yet two halves of the source campaigns alone agree with each other at 0.980. At the default seeds that is a
collapse, but it is close to the line, because the floor depends on which 30 splits of the 40 source shots it
draws. Drawing 30 other splits (seeds 0–29) gives a floor of 0.9821 ± 0.0108, z = −1.81, INDISTINGUISHABLE;
pooling all 60 draws gives 0.9811 ± 0.0091, z = −2.03 against that count's cut of 2.018, COLLAPSE by a margin of
0.008. Forty shots is a small pool, and one split (seed 8, 0.935) sits far below the rest. Read this as evidence
that the explanation moved, at the edge of what this pool can resolve, not as a settled collapse.

**Is it just the label-prior shift?** The positive rate moves from 0.508 to 0.721, so a reader can ask whether the
difference is class balance. Two controls, each with outcomes fixed before it was first run, say the prior does not
explain it. *Inside the null:* each split carries its own prior gap, and two of the 30 draws have a gap larger than
the real one (0.294 and 0.285 against 0.212), yet they score 0.978 and 0.973, z = −0.25 and −0.93. Across the 30
draws the gap does not predict QFC (Spearman ρ = −0.173, p = 0.361); the fitted slope attributes 20% of the
observed drop to the prior gap at its point estimate and 65% at the steep end of its 95% interval, so the gap
cannot explain all of it. *In the explained rows:* explaining 500 target rows drawn at the target's prior (360
positive, 140 negative) and at the source's (254, 246), everything else fixed, over 10 paired seeds, gives
0.9601 ± 0.0090 and 0.9614 ± 0.0088, a paired difference of +0.0013 against its paired spread of 0.0020; both
arms collapse (z = −2.77 and −2.59). The difference is 7% of the observed drop; tested against its standard error
it is not quite significant (t = 2.06 on 9 degrees of freedom, p = 0.07), so matching the prior may remove a
small part of the drop, but not most of it.

**Why the paper reports a higher floor for this comparison.** The paper's census scores this row at 0.9637
against a floor of 0.9913 ± 0.0057, z = −4.86 (its census receipt, not yet published). The difference is the null's
construction, not the data or the tree depth. For this row the paper's floor fits one forest on the first half and
compares its attributions on the very rows it was trained on with those on the second half; this package builds
every null draw out of fold, as the paper's reference implementation does. The package's frames have the paper's
rows, labels and shots; 41 of their 48 feature columns are bit-identical to the paper's arrays, six differ by at
most 1.4e-15, and the toroidal-field skew differs by up to 1.0 on nearly constant windows. On these frames the
paper's construction gives 0.9912 ± 0.0060, against the paper's 0.9913 ± 0.0057. Tree depth does not matter here:
the paper's row used a depth of 20, and at depth 20 the package's floor is identical to every digit (trees fitted
on half the source do not grow past depth 20) and its QFC 0.9613, z = −2.61, COLLAPSE. The paper's own
construction-matched rescoring of this row, a secondary column of its census, finds 0.9825 ± 0.0113 and
INDISTINGUISHABLE, in line with the split sensitivity above.

Every number the package computed in this section is in `docs/worked_example.json`, written by
`scripts/regenerate_worked_example.py` with the machine, library versions and a hash of the package code it
records; the paper's figures are quoted from its census receipts, which are not yet published.

## Tests

```bash
pip install -e ".[dev]"
pytest -m "not slow and not integration"        # no data, a few minutes
DL_DATAFRAME=DL_DataFrame.csv pytest -m slow    # recomputes the worked example
DISRUPTIONPY_FRAME=frame.csv DPQ_TUD_COL=time_until_disrupt pytest -m integration
```

| group | needs | checks |
|---|---|---|
| default | nothing | the procedure against the paper's reference code, the attribution methods, the cap rule, the bootstrap, the validation and error messages, the README's numbers against `docs/worked_example.json` |
| `slow` | the C-Mod release | the worked example reproduces its record |
| `integration` | a frame produced by DisruptionPy | the package accepts and screens a real DisruptionPy frame |

CI runs the default group on Python 3.10, 3.11 and 3.12, at the oldest dependency versions allowed, and from the
built wheel; it runs the integration group on a frame that DisruptionPy itself builds from MAST's open data
(`tests/fixtures/build_mast_frame.py`), where a missing frame is a failure, not a skip. That job runs on pushes to
`main`, weekly and on demand, not on pull requests, to keep load off the public data server. The slow group needs
the C-Mod release and is run before each release; its record is the JSON above. `docs/ATTACKS.md` lists the
strongest objections to the measurement and to this package, each with the test or section that answers it.

## What this does not do

- **It is not a disruption predictor, and it does not say whether yours is good.** It says whether your model
  relies on the same features in both environments.
- **A high QFC is not a safety argument.** It means the importance order is preserved, which a stable explanation
  needs but safe deployment needs far more than.
- **It uses ranks, not sizes.** Two environments can agree on the order of importance while differing in how much
  each feature matters.
- **Every row counts once.** Long shots and densely sampled phases weigh more in the importance vectors, so a
  change in the mix of shot lengths or sampling between campaigns can move QFC without any change in what the model
  relies on. Resample the frames to a common time base before screening if that matters. The rolling-window
  features of the worked example count samples, not seconds, so they need the same care.
- **A floor belongs to one source pool, model and set of settings.** It does not transfer to another machine.
- **Small pools give coarse floors.** With few shots the halves overlap and the floor's spread is itself uncertain,
  as the worked example shows; the package warns when splits repeat. With few features the rank correlation can
  take only a few values, so null draws often tie; below three features it is refused.
- **Each verdict is a 5% test of one comparison.** Screening many comparisons at once calls for a correction
  across them, which the package does not make.
- **It is not the only way to detect that explanations changed.** In the paper's benchmark of 54 shifts at a
  matched 5% false-positive rate (in the paper's text, not yet published), the best detector depends on the model class: with random
  forests QFC detects 35, Explanation Shift 32, and Kendall's tau and a sliced Wasserstein distance 36; with
  gradient boosting Explanation Shift detects 38 and 37 against QFC's 29 and 31. What QFC adds is a verdict
  calibrated against the model's own no-shift floor, computable without target labels.

## Changes, citation, licence

`CHANGELOG.md` lists what changed between versions. To cite: `CITATION.cff`, and the dataset you screened. MIT
licence.
