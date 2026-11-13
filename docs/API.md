# Using the package

How to prepare your own frames, read the result, use your own attribution method and set every option. The
[README](../README.md) has the short version; [METHOD.md](METHOD.md) says what each step computes.

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

## The result

`result.summary()` prints the QFC, the floor, z, the cut, the resolution and the verdict with its meaning;
`result.to_dict()` gives the whole result as strict JSON. The fields you will use most:

| field | what it holds |
|---|---|
| `verdict` | COLLAPSE, INDISTINGUISHABLE, ROBUST, UNCALIBRATED or UNDEFINED ([The verdicts](METHOD.md#the-verdicts)) |
| `qfc`, `qfc_per_order` | the QFC (the mean over fold orders, which the verdict uses) and its value at each order |
| `null_mean`, `null_sd`, `n_null_seeds` | the no-shift floor |
| `z`, `exact_cut`, `resolution` | as in [What the words mean](METHOD.md#what-the-words-mean) |
| `auc` | the deployed model's AUC on the target rows that have labels; `None` without them |
| `feature_cols_used` | the features, in the order of the two importance vectors |
| `source_importance`, `target_importance` | mean absolute attribution per feature (the source's at fold order 0) |
| `verdict_effective`, `n_distinct_splits` | the reading at the cut for the number of distinct splits, if splits repeated |
| `null_floor`, `chance_floor` | each floor in full: every draw, every skipped draw and its reason |
| `provenance` | package version, estimator, attribution, seeds, rules and the imputation counts |

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
| `attribution` | `"auto"` | see [Your own attributions](#your-own-attributions) |
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

### Cost

One `screen` fits the deployed model once, 5 fold models for each distinct fold order and 6 models for
each of the 30 null draws: at most 1 + 50 + 180 = 231 fits, plus the attributions. A chance floor, when built,
adds 6 fits for each of its 15 draws, and a bootstrap 6 per draw. On the [C-Mod example](WORKED_EXAMPLE.md)
(3,363 source rows, 48 features) it took 531 seconds, about nine minutes, on a 32-core machine. Fitting time grows
with the number of rows; attribution time does not, because at most 500 rows are explained per vector. KernelSHAP
is far slower than TreeSHAP: about 6,000 model evaluations per explained row. Fewer trees, a smaller `shap_cap` or
`n_fold_orders=1` make it faster; the defaults are the paper's.

### Determinism

TreeSHAP and Saabas results are identical on rerun with the same seed on the same machine;
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
