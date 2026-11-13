# disruptionpy-qfc

**When a disruption predictor still scores well on a new campaign or machine, is it still relying on the same signals?**

A model trained on one set of discharges can keep a high AUC on a later campaign while the features it leans on
change underneath. Accuracy metrics cannot see that. This package measures it before deployment, without labels on
the new data, and tells you whether what it measured is more than the change you would see with no shift at all.
Only the training data need labels, so a new campaign, or a new machine's first discharges, can be screened before
any disruption on it is labelled.
The measurement is QFC (Quantitative Feature Consistency): the rank correlation between the feature importances a
model shows on the data it was trained on and on the data it is deployed on.

QFC, its no-shift floor and its verdicts are defined in the author's paper (A. Turayev, not yet published; "the
paper" below). The paper also proves exactly when the importance ranking of a Bayes-optimal classifier changes under
a shift, under the assumptions it states, and the core of that proof is machine-checked in Lean 4. Random forests,
including the one in the C-Mod example, are outside the proof: for them the change is measured, not proved. The
paper and its Lean code are not public yet, so the paper's figures quoted below cannot be checked from this
repository; every figure the package itself computed can be (see [Tests](#tests)).

This is independent work by one researcher, not affiliated with, reviewed or endorsed by MIT PSFC or the
DisruptionPy maintainers; the name only says which data format it reads. DisruptionPy, the Open Density Limit
Database and the DPRF predictor, on which it builds, are credited under [Credits](#credits-citation-licence).

## What it found on the Open Density Limit Database

On the public Open Density Limit Database (Maris, Rea, Trevisan and the Alcator C-Mod Team, MIT PSFC, 2025), a
random forest trained on the Alcator C-Mod campaigns of 2000, 2001 and 2003 and deployed on 2005 and 2009 keeps a
target AUC of 0.939. Its feature ranking on the later campaigns agrees with its ranking on the training campaigns at
QFC = 0.9607, while two random halves of the training campaigns agree with each other at 0.9805 on average over 30
splits. That is z = −2.87 (negative: less agreement than with no shift) against 2.079, the 5% cut for 30
draws: the ranking moved more than no shift explains, and the verdict is COLLAPSE. It is close to the line.
Thirty other splits of the same 40 training discharges give INDISTINGUISHABLE (z = −1.86), and all sixty
together give COLLAPSE (z = −2.18).

What moved: summed over each signal's eight window statistics, density's share of the attribution rises from 32.1%
to 41.3% and plasma current's falls from 39.5% to 31.3%, so density overtakes plasma current. Among single features
plasma current still leads: its mean and its minimum are the two largest on the training campaigns, and its minimum
is the largest on the later ones. This describes the one measurement; the package builds no no-shift floor for a
single signal's share, so it is not itself a test.

The details, two controls for the change in class balance, and the reason the paper's figures for this comparison
differ are in the [worked example](#worked-example-alcator-c-mod-the-2000-2001-and-2003-campaigns-to-2005-and-2009).

## Run the C-Mod example in three commands

You need Python 3.10 or later and git; the install also brings numpy, pandas, scikit-learn, scipy and shap.

```bash
pip install "git+https://github.com/turayevabror2105-maker/disruptionpy-qfc@v3.0.3"
curl -LO https://raw.githubusercontent.com/MIT-PSFC/open_density_limit_database/613257131aabfbcb301dacc429a8b3c0cfa9f858/data/DL_DataFrame.csv
python -m disruptionpy_qfc.demo --data DL_DataFrame.csv --out cmod_result.json
```

The second command fetches the public release at a fixed commit (24,605,618 bytes, SHA-256
`c296028985f880cfa341030af0b54d7fe7092b639156281dba454b0926385318`); from it the demo builds frames with the record's
row, discharge and label counts. On Windows PowerShell 5.1, `curl` is an alias of `Invoke-WebRequest`, so type
`curl.exe` there. The demo prints the frame sizes and the result's `summary()` (QFC, floor, z, cut, resolution and
verdict), and `--out` writes the full result as JSON: `result.to_dict()` under `"result"` and a note on the left-out
campaigns under `"excluded"`.

**How long it takes.** The recorded run took 536 seconds, about nine minutes, on a 32-core Linux machine
(`docs/worked_example.json`, `default.seconds`). A laptop has not been timed. The run is at most 231 fits of a
500-tree forest plus their TreeSHAP attributions (see Cost under [Options](#options)), and the forest fits use every
core (`n_jobs=-1`), so on a laptop expect a multiple of that time and leave the machine otherwise idle. Only the
default settings reproduce the record.

**To check that everything runs first**, a much lighter screen of the same frames:

```python
from sklearn.ensemble import RandomForestClassifier
from disruptionpy_qfc import screen
from disruptionpy_qfc.demo import build_demo_frames

source, target = build_demo_frames("DL_DataFrame.csv")
light = RandomForestClassifier(n_estimators=100, class_weight="balanced", n_jobs=-1)
check = screen(source, target, model=light, n_seeds=10, n_fold_orders=1, shap_cap=200)
print(check.summary())
```

That is 66 fits of 100 trees instead of up to 231 of 500 (90 more if the reading is ROBUST, when the chance floor
is built), and 200 explained rows per vector instead of 500. It checks the installation, not the result: 10 null
draws give a cut of 2.373 instead of 2.079 and a coarser floor, and its numbers and verdict will differ from the
record, so do not quote them. The demo's `--n-seeds 10` lowers only the number of null draws, with the same change
of cut.

## Your own frames

```python
from disruptionpy_qfc import screen

result = screen(source_df, target_df)      # target labels are not needed
print(result.summary())
```

`source_df` is the data the model is trained on, `target_df` the data it will be deployed on. Both need a `shot`
column; the source also needs a binary `label` column. Every other numeric column is used as a feature except
`shot_id`, `time`, `time_until_disrupt` and `time_domain`; the list used is in `result.feature_cols_used`, and any other
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
package refuses a non-binary or missing source label, a non-numeric or infinite feature, or fewer than three
features, and the error says which. A missing feature value that is left, or a window statistic that is undefined
(the skew of a window whose spread is at floating-point resolution, which `build_rolling_features` returns as NaN),
is replaced by the source frame's median of its column, for source and target alike, as the paper does; the
count per column is in `result.provenance["imputed_with_source_median"]` and in `summary()`. Missing target labels
are allowed; they are left out of the AUC.

`result.summary()` prints the QFC, the floor, z, the cut, the resolution and the verdict with its meaning;
`result.to_dict()` gives the whole result as strict JSON. The fields you will use most:

| field | what it holds |
|---|---|
| `verdict` | COLLAPSE, INDISTINGUISHABLE, ROBUST, UNCALIBRATED or UNDEFINED ([The verdicts](#the-verdicts)) |
| `qfc`, `qfc_per_order` | the QFC (the mean over fold orders, which the verdict uses) and its value at each order |
| `null_mean`, `null_sd`, `n_null_seeds` | the no-shift floor |
| `z`, `exact_cut`, `resolution` | as in [What the words mean](#what-the-words-mean) |
| `auc` | the deployed model's AUC on the target rows that have labels; `None` without them |
| `feature_cols_used` | the features, in the order of the two importance vectors |
| `source_importance`, `target_importance` | mean absolute attribution per feature (the source's at fold order 0) |
| `verdict_effective`, `n_distinct_splits` | the reading at the cut for the number of distinct splits, if splits repeated |
| `null_floor`, `chance_floor` | each floor in full: every draw, every skipped draw and its reason |
| `provenance` | package version, estimator, attribution, seeds, rules and the imputation counts |

## What the words mean

| term | meaning |
|---|---|
| **attribution** | how much each feature contributed to one prediction. By default TreeSHAP for tree models and KernelSHAP for others |
| **importance vector** | the mean absolute attribution of each feature over the explained rows (every row counts once, so long shots count for more) |
| **QFC** | Quantitative Feature Consistency: Spearman's rank correlation between the source and target importance vectors. 1 means the features keep the same order of importance; lower means the order changed |
| **fold order** | the source vector is built out of fold (no shot is explained by a model that trained on it). Which equal-sized shots share a fold is arbitrary, so the reported QFC is the mean over 10 such orders |
| **null floor** | the same measurement between two random halves of the source, 30 times. There is no shift between two halves, so this is what QFC looks like when nothing changed |
| **z** | how many null-floor standard deviations the observed QFC lies from the floor's mean: (QFC − floor mean) / floor sd, the sd being the sample sd of the draws (n − 1). Negative means less agreement than with no shift |
| **cut** | 2.079 for 30 null draws: the two-sided 5% prediction cut for one new value against 30 draws, t(0.975, 29) × √(1 + 1/30). It is exact when the draws are independent and normal, and wider than 1.96 because the floor's mean and sd are estimated from the draws, not known |
| **resolution** | cut × floor sd: the smallest change in QFC this floor can detect |
| **chance floor** | the same measurement with the source labels shuffled, 15 times: what a model trained on noise gives |

## The verdicts

| verdict | when | what it means |
|---|---|---|
| **COLLAPSE** | z ≤ −cut | the two environments agree on feature importance *less* than two halves of the source do. The explanation changed under the shift by more than no-shift sampling explains |
| **INDISTINGUISHABLE** | −cut < z < cut | within what two halves of the source give each other. No change was detected; `resolution` says how large a change could have been |
| **ROBUST** | z ≥ cut, and the floor's mean is not below the chance floor's mean | the agreement is *higher* than two halves of the source give. If the floor's mean lies below the chance floor's mean, the reading is INDISTINGUISHABLE instead, because then two halves agree less than a model trained on noise does |
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

These are the paper's rules: its statistic, its small-sample cut, its census's primary scoring (the fold-order mean and
the chance-floor rule), and the null construction of its reference implementation (`qfc_test`).
`tests/test_reference_identity.py` runs the package and a copy of that implementation
(`tests/reference/qfc_metric.py`) on the same inputs and requires exact equality of the statistic, the null draws,
z, the cut, the verdict, the distinct-split reading, the fold-order mean and the chance floor, for random forests
with TreeSHAP; it also recomputes seven of the paper's archived conditions from their stored vectors and checks
the chance-floor rule against the four census rows it changed. The reference has no counterpart for KernelSHAP,
Saabas or permutation, so those are tested on their own (`tests/test_attributions.py`). Many of the paper's census
rows were scored against a differently built null, one forest per half with in-sample attributions on the first
half; the worked example below shows what that changes. The census-level chance counts and the census's secondary
construction-matched campaign describe many conditions at once and are not part of a single screening.

## Your own attributions

```python
screen(source_df, target_df, attribution="saabas")          # or "treeshap", "kernelshap", "permutation"
```

Any other method is a function of four arguments. This one needs only numpy and runs as it stands:

```python
import numpy as np
from disruptionpy_qfc import screen

def median_occlusion(model, X_explain, X_train, seed):
    """Per row and feature: how much P(class 1) drops when that feature is set to its training median."""
    base = model.predict_proba(X_explain)[:, 1]
    medians = np.median(X_train, axis=0)
    out = np.empty(X_explain.shape)
    for j in range(X_explain.shape[1]):
        X_j = X_explain.copy()
        X_j[:, j] = medians[j]
        out[:, j] = base - model.predict_proba(X_j)[:, 1]
    return out

result = screen(source_df, target_df, attribution=median_occlusion)
```

The package calls it with a model it has just fitted (a clone of `model`), the rows to explain, the rows that
model was trained on, and a seed. The rows are numpy arrays whose columns are `result.feature_cols_used` in that
order, with missing values already replaced by the source medians. It must return one value per explained row and
feature, an array of shape `(len(X_explain), n_features)`; the package takes absolute values and averages them
over the rows, so the sign does not matter. Use the seed for anything random, so that reruns and null draws are
reproducible. The null floor calls the function in exactly the same way. To screen with contributions your own
predictor computes, wrap that computation in a function of this form: it has to work on the estimator the package
passes in, because the out-of-fold vector and the floor need refits.

`"saabas"` is the tree-path decomposition of the `treeinterpreter` package, which is how the DPRF predictor
computes its feature contributions (Rea et al., *Nucl. Fusion* 59, 096016, 2019); a test checks it against the
decomposition written out from the tree arrays. `"kernelshap"` uses a 30-centroid k-means background and 200
samples per row, as the paper does for non-tree models, with shap's `l1_reg` pinned to `"num_features(10)"`: with
more than ten features, each row's attributions are fitted on the ten features a least-angle regression selects
and the rest are set to zero. Because Spearman's rho counts every rank equally, those zeros shape the low end of
the ranking. That has been shap's default since shap 0.47 (before, it was `"auto"`, and this package allows
shap 0.45), so it is pinned to give the same rule on every allowed version; pass your own callable for another
choice. `"permutation"` is the mean absolute change in predicted probability when one feature is shuffled; it
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
adds 6 fits for each of its 15 draws, and a bootstrap 6 per draw. On the C-Mod example below (3,363 source rows,
48 features) it took 536 seconds, about nine minutes, on a 32-core machine. Fitting time grows with the number of
rows; attribution time does not, because at most 500 rows are explained per vector. KernelSHAP is far slower than
TreeSHAP: about 6,000 model evaluations per explained row. Fewer trees, a smaller `shap_cap` or `n_fold_orders=1`
make it faster; the defaults are the paper's.

**Determinism.** TreeSHAP and Saabas results are identical on rerun with the same seed on the same machine;
KernelSHAP is seeded too. Between machines, `GroupKFold`'s ordering of equal-sized shots follows numpy's sort,
which differs between platform builds, so values can differ slightly; averaging over 10 fold orders is what
keeps that arbitrary choice out of the verdict.

## Everything the package exports

Each has a docstring (`help(disruptionpy_qfc.screen)`, and so on).

| name | what it is |
|---|---|
| `screen(source_df, target_df, **options)` | null floor, measurement and verdict in one call; returns a `QFCResult` |
| `compute_qfc(source_df, target_df, null_floor=..., chance_floor=..., **options)` | the measurement alone; `UNCALIBRATED` without a floor |
| `compute_null_floor(source_df, n_seeds=30, **options)` | the no-shift floor, a `NullFloor` |
| `compute_chance_floor(source_df, target_df, n_draws=15, **options)` | the permuted-label floor, a `ChanceFloor` |
| `QFCResult`, `NullFloor`, `ChanceFloor` | the results; each has `summary()` and `to_dict()` |
| `DegenerateDrawError` | raised for a draw the data in it cannot support (every training fold single-class, or a constant importance vector); a floor skips and counts such draws, and any other error, including one from your own attribution function, stops the run |
| `default_model(random_state=42, max_depth=None)` | the default random forest |
| `detect_feature_cols(df, label_col="label", shot_col="shot")` | the feature columns `screen` would use |
| `build_rolling_features(df, signal_cols, window=10, ...)` | the worked example's eight causal rolling statistics per signal |
| `qfc_statistic(src, tgt)` | Spearman's rho of two importance vectors |
| `exact_cut(n)` | the 5% cut for `n` null draws |
| `verdict_label(z, cut)` | COLLAPSE, INDISTINGUISHABLE or ROBUST from z and the cut |
| `tie_order_groups(groups, k)` | the shot relabelling of fold order `k` |
| `ATTRIBUTIONS`, `DEFAULT_EXCLUDE_COLS` | the built-in attribution methods by name; the columns never used as features |
| `N_FOLD_ORDERS`, `N_NULL_SEEDS`, `N_CHANCE_DRAWS`, `__version__` | the default counts (10, 30, 15) and the version |

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
| QFC (mean of 10 fold orders) | 0.9607 (orders range 0.9572 to 0.9644) |
| null floor | 0.9805 ± 0.0069, 30 draws |
| z, against the cut of 2.079 | −2.87 |
| resolution | 0.0143 |
| verdict | **COLLAPSE** (10 of the 10 fold orders collapse on their own) |

**A collapse near the line.** The model still predicts well, and an attribution agreement of 0.961 looks
excellent, yet two halves of the source campaigns alone agree with each other at 0.980. At the default seeds that
is a collapse, but the floor depends on which 30 splits of the 40 source shots it draws. Drawing 30 other splits
(seeds 0–29, which set each draw's forest seed as well as its split) gives a floor of 0.9820 ± 0.0115, z = −1.86, INDISTINGUISHABLE; pooling all 60 draws gives
0.9813 ± 0.0094, z = −2.18 against that count's cut of 2.018, COLLAPSE by a margin of 0.164. Forty shots is a
small pool, and one split (seed 8, 0.930) sits far below the rest. Read this as evidence that the explanation
moved, near the edge of what this pool can resolve, not as a settled collapse.

**Which signals moved.** The record stores the source and target importance vectors of fold order 0, whose rank
correlation is that order's QFC. Summed over each signal's eight statistics, density's share of the total
attribution rises from 32.1% to 41.3% and plasma current's falls from 39.5% to 31.3%, so at the level of signals
density overtakes plasma current. Among single features plasma current still leads: its mean and its minimum are
the two largest in the source, and its minimum is the largest in the target, now followed by density's maximum and
mean. The shares describe this one measurement: the package builds no no-shift
floor for a single signal's share, so how much of the movement no-shift sampling alone would produce is not
measured here. Density and plasma current are the two quantities of the Greenwald density limit,
n_G = I_p/(πa²) (Greenwald, *Plasma Phys. Control. Fusion* 44, R27, 2002); whether the movement reflects the
physics of the later campaigns or their mix of discharges is a question this measurement cannot answer.

**Is it just the label-prior shift?** The positive rate moves from 0.508 to 0.721, so a reader can ask whether the
difference is class balance. Two controls, each with outcomes fixed before it was first run, say the prior does not
explain it. *Inside the null:* each split carries its own prior gap, and two of the 30 draws have a gap larger than
the real one (0.294 and 0.285 against 0.212), yet they score 0.979 and 0.973, z = −0.25 and −1.12. Across the 30
draws the gap does not predict QFC (Spearman ρ = −0.119, p = 0.530); the fitted slope attributes 16% of the
observed drop to the prior gap at its point estimate and 54% at the steep end of its 95% interval, so the gap
cannot explain all of it. *In the explained rows:* explaining 500 target rows drawn at the target's prior (360
positive, 140 negative) and at the source's (254, 246), everything else fixed, over 10 paired seeds, gives
0.9593 ± 0.0094 and 0.9610 ± 0.0089, a paired difference of +0.0017 against its paired spread of 0.0029; both
arms collapse (z = −3.08 and −2.83). The difference is 9% of the observed drop; tested against its standard error
it is not significant (t = 1.86 on 9 degrees of freedom, p = 0.10), so matching the prior may remove a
small part of the drop, but not most of it.

<details>
<summary><b>Why the paper reports a higher floor for this comparison</b></summary>

The paper's census, which is not yet published, scores this row in its standard scoring at 0.9637, a mean over
five model seeds at one fold order (the package's figures use one model seed, 42, and average ten fold orders),
against a floor of 0.9913 ± 0.0057, z = −4.86. The package did not compute the 0.9637 or the −4.86, and neither is
in `docs/worked_example.json`; it did compute the floor, as below. The difference is the null's construction, not the data or the tree depth. For this
row the paper's floor fits one forest on the first half and compares its attributions on the very rows it was
trained on with those on the second half; this package builds every null draw out of fold, as the paper's
reference implementation does. The package's frames have the paper's rows, labels and shots, and the same 18
source rows imputed with the same median; 42 of their 48 feature columns are bit-identical to the paper's arrays
and the other six, the skews, differ by at most 4.5e-16. On these frames the paper's construction gives
0.9913 ± 0.0057, the paper's own floor, and the single-order real value at depth 20 and seed 42 is 0.9608, the
paper's own value for that seed; the package computed both, and both are in the record. Tree depth does not matter
here: the paper's row used a depth of 20, and at depth 20 the package's floor is identical to every digit (trees
fitted on half the source do not grow past depth 20) and its QFC 0.9604, z = −2.91, COLLAPSE. The paper's own
construction-matched rescoring of this row, a secondary column of its census, reads it as INDISTINGUISHABLE,
in line with the split sensitivity above.

</details>

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
| default | nothing | the procedure against the paper's reference code, the attribution methods, the cap rule, the bootstrap, the validation and error messages, the README's numbers against `docs/worked_example.json`, that the record was made by the code now in `src/`, and that the version and release date agree across `pyproject.toml`, `__version__`, `CITATION.cff`, `CHANGELOG.md` and the install line |
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
- **A COLLAPSE is not by itself a fault.** It says the model now relies on different features, not that the new
  reliance is wrong: it may be the same physics reached through another signal. On C-Mod, density and plasma
  current both enter the Greenwald limit, so the measurement alone cannot say which.
- **It is not the only way to detect that explanations changed.** In the paper's benchmark (not yet published) of
  54 shifts at a matched 5% false-positive rate, the best detector depends on the model class: with random
  forests QFC detects 35, Explanation Shift 32, and Kendall's tau and a sliced Wasserstein distance 36; with
  gradient boosting Explanation Shift detects 38 and 37 against QFC's 29 and 31. What QFC adds is a verdict
  calibrated against the model's own no-shift floor; like Explanation Shift, it needs no target labels.

## Credits, citation, licence

This package reads data in the format of DisruptionPy, and its integration tests run on frames DisruptionPy builds:

- G. L. Trevisan, Y. Wei, A. M. Decker, J. Lorincz, S. L. Jackson, C. Rea, R. S. Granetz and the MIT PSFC
  Disruptions Group, *DisruptionPy: An open-source physics-based scientific framework for disruption analysis of
  fusion plasmas*, Journal of Open Source Software 11(119), 9364 (2026), doi:10.21105/joss.09364;
  `github.com/MIT-PSFC/disruption-py`.

The worked example uses, and this repository does not redistribute:

- A. D. Maris, C. Rea, G. L. Trevisan and the Alcator C-Mod Team, *The Open Density Limit Database*, MIT Plasma
  Science and Fusion Center (2025), `github.com/MIT-PSFC/open_density_limit_database`; data CC BY, code MIT.

The `"saabas"` option follows the feature contributions of the DPRF predictor (C. Rea et al., *Nucl. Fusion* 59,
096016, 2019). The question this package asks comes from cross-machine work such as K. J. Montes et al.,
*Machine learning for disruption warnings on Alcator C-Mod, DIII-D, and EAST*, *Nucl. Fusion* 59, 096015 (2019).

`CHANGELOG.md` lists what changed between versions. To cite this package: `CITATION.cff`, together with the dataset
you screened. MIT licence.
