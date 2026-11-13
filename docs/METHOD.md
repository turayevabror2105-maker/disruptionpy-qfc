# How QFC is computed and read

The terms, the verdict rules and the procedure, step by step. The [README](../README.md#how-it-works) has the short
version; [API.md](API.md) has the options that change each step.

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
   halves of a small pool overlap from draw to draw, so the 30 draws are not independent; the
   [worked example](WORKED_EXAMPLE.md#result) shows what that does.
6. **Verdict** as in [the table above](#the-verdicts), with the chance floor built only when the reading would be
   ROBUST.

## Relation to the paper's reference code

These are the paper's rules: its statistic, its small-sample cut, its census's primary scoring (the fold-order mean and
the chance-floor rule), and the null construction of its reference implementation (`qfc_test`).
`tests/test_reference_identity.py` runs the package and a copy of that implementation
(`tests/reference/qfc_metric.py`) on the same inputs and requires exact equality of the statistic, the null draws,
z, the cut, the verdict, the distinct-split reading, the fold-order mean and the chance floor, for random forests
with TreeSHAP; it also recomputes seven of the paper's archived conditions from their stored vectors and checks
the chance-floor rule against the four census rows it changed. The reference has no counterpart for KernelSHAP,
Saabas or permutation, so those are tested on their own (`tests/test_attributions.py`). Many of the paper's census
rows were scored against a differently built null, one forest per half with in-sample attributions on the first
half; the [worked example](WORKED_EXAMPLE.md), under "Why the paper reports a higher floor", shows what that changes. The
census-level chance counts and the census's secondary construction-matched campaign describe many conditions at
once and are not part of a single screening.
