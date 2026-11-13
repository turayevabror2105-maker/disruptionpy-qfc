# disruptionpy-qfc

**When a disruption predictor still scores well on a new campaign or machine, is it still relying on the same signals?**

A model trained on one set of discharges can keep a high AUC on a later campaign while the features it leans on
change underneath. Accuracy metrics cannot see that. This package measures it before deployment, without labels on
the new data, and tells you whether what it measured is more than the change you would see with no shift at all.
Only the training data need labels, so a new campaign, or a new machine's first discharges, can be screened before
any disruption on it is labelled.

The measurement is QFC (Quantitative Feature Consistency): the rank correlation between the feature importances a
model shows on the data it was trained on and on the data it is deployed on.

This is independent work by one researcher, not affiliated with, reviewed or endorsed by MIT PSFC or the
DisruptionPy maintainers; the name only says which data format it reads. DisruptionPy, the Open Density Limit
Database and the DPRF predictor, on which it builds, are credited under [Credits](#credits-citation-licence).

## What it found on the Open Density Limit Database

On the public Open Density Limit Database (Maris, Rea, Trevisan and the Alcator C-Mod Team, MIT PSFC, 2025), a
random forest trained on the Alcator C-Mod campaigns of 2000, 2001 and 2003 and deployed on 2005 and 2009 keeps a
target AUC of 0.939. Its feature ranking on the later campaigns agrees with its ranking on the training campaigns at
QFC = 0.9607, while two random halves of the training campaigns agree with each other at 0.9805 on average over 30
splits. That is z = −2.87 against 2.079, the 5% cut for 30
draws. A negative z means less agreement than with no shift, so the ranking moved more than no shift explains,
and the verdict is COLLAPSE. It is close to the line.
Thirty other splits of the same 40 training discharges give INDISTINGUISHABLE (z = −1.86), and all sixty
together give COLLAPSE (z = −2.18).

What moved: summed over each signal's eight window statistics, density's share of the attribution rises from 32.1%
to 41.3% and plasma current's falls from 39.5% to 31.3%, so density overtakes plasma current. Among single features
plasma current still leads: its mean and its minimum are the two largest on the training campaigns, and its minimum
is the largest on the later ones. This describes the one measurement; the package builds no no-shift floor for a
single signal's share, so it is not itself a test.

The [worked example](docs/WORKED_EXAMPLE.md) has the full result, two controls showing that the change in class
balance explains at most part of it, and the reason the paper's figures for this comparison differ.

## Run the C-Mod example in three commands

You need Python 3.10 or later and git; the install also brings numpy, pandas, scikit-learn, scipy and shap.

```bash
pip install "git+https://github.com/turayevabror2105-maker/disruptionpy-qfc@v3.0.4"
curl -LO https://raw.githubusercontent.com/MIT-PSFC/open_density_limit_database/613257131aabfbcb301dacc429a8b3c0cfa9f858/data/DL_DataFrame.csv
python -m disruptionpy_qfc.demo --data DL_DataFrame.csv --out cmod_result.json
```

On Windows PowerShell 5.1, type `curl.exe` for `curl`. The demo prints the result's summary (QFC, floor, z, cut,
resolution and verdict), and `--out` writes the full result as JSON. The recorded run took 531 seconds, about nine
minutes, on a 32-core Linux machine; a laptop has not been timed and will take a multiple of that. The data file's
size and hash, and a much lighter check that the installation works, are in
[Reproducing it](docs/WORKED_EXAMPLE.md#reproducing-it).

On your own data it is one call, `screen(source_df, target_df)`: the frame the model is trained on, then the one
it will be deployed on. Both need a `shot` column, and the source a binary `label` column. Labelling a DisruptionPy
frame, missing values, your own attribution method and every option are in [docs/API.md](docs/API.md).

## How it works

1. **Attribution.** How much each feature contributed to each prediction (TreeSHAP for tree models by default),
   averaged in absolute value per feature: the importance vector.
2. **Two vectors.** The source vector is built out of fold, so no discharge is explained by a model trained on it;
   the target vector comes from one model fitted on the whole source, as a deployed predictor would be.
3. **QFC** is Spearman's rank correlation of the two, averaged over 10 fold orders; 1 means the same order of
   importance.
4. **No-shift floor.** The same measurement between two random halves of the source discharges, 30 times.
5. **z** = (QFC − floor mean) / floor sd, against 2.079, the 5% cut for 30 draws (exact for independent, normal
   draws).
6. **Verdict.** COLLAPSE if z ≤ −cut; ROBUST if z ≥ cut and the floor's mean is not below that of a floor built
   with shuffled labels; otherwise INDISTINGUISHABLE. Every result also carries its resolution, the smallest change
   in QFC the floor can detect.

These are the paper's rules; for random forests with TreeSHAP a test requires exact equality with a copy of the
paper's reference implementation. Every term, rule and step in full: [docs/METHOD.md](docs/METHOD.md).

## What this does not do

- It is not a disruption predictor, and it does not say whether yours is good, only whether your model relies on
  the same features in both environments.
- A high QFC is not a safety argument: a stable explanation needs a preserved importance order, and safe
  deployment needs far more.
- It uses ranks, not sizes.
- Every row counts once, so long shots and densely sampled phases weigh more; resample to a common time base if
  that matters.
- A floor belongs to one source pool, model and set of settings; it does not transfer to another machine.
- Small pools give coarse floors, and few features a coarse rank correlation; below three features it is refused.
- Each verdict is a 5% test of one comparison; screening many needs a correction the package does not make.
- A COLLAPSE is not by itself a fault. It says the model now relies on different features, not that the new
  reliance is wrong.
- It screens a model specification, not a model you have already fitted, because the out-of-fold
  vector and the floor need refits.
- It is not the only detector of changed explanations: in the paper's benchmark the best one depends on the model
  class, and Explanation Shift leads on gradient boosting.

Each limit in full, with the benchmark's counts: [docs/LIMITS.md](docs/LIMITS.md).

## The paper

QFC, its no-shift floor and its verdicts are defined in the author's paper (A. Turayev, not yet published; "the
paper" in these documents). The paper also proves exactly when the importance ranking of a Bayes-optimal
classifier changes under a shift, under the assumptions it states, and the core of that proof is machine-checked in
Lean 4. Random forests, including the one in the C-Mod example, are outside the proof: for them the change is
measured, not proved. The paper and its Lean code are not public yet, so the paper's figures quoted in these
documents cannot be checked from this repository; every figure the package itself computed can be (see
[docs/TESTING.md](docs/TESTING.md)).

## Documentation

| file | what it holds |
|---|---|
| [docs/WORKED_EXAMPLE.md](docs/WORKED_EXAMPLE.md) | the C-Mod run in full: reproducing it, the data, the result, which signals moved, the class-balance controls, and why the paper's floor is higher |
| [docs/METHOD.md](docs/METHOD.md) | the terms, the verdict rules, the procedure step by step, and its match with the paper's reference code |
| [docs/API.md](docs/API.md) | your own frames, the result's fields, your own attribution method, options, cost, determinism, exported names |
| [docs/LIMITS.md](docs/LIMITS.md) | each limit in full |
| [docs/TESTING.md](docs/TESTING.md) | the test groups and what CI runs |
| [docs/ATTACKS.md](docs/ATTACKS.md) | the strongest objections, each with the test or section that answers it |
| [docs/worked_example.json](docs/worked_example.json) | the record of every number the package computed, with the machine, library versions and code hash |
| [CHANGELOG.md](CHANGELOG.md) | what changed between versions |

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

To cite this package: [CITATION.cff](CITATION.cff), together with the dataset you screened. MIT licence
([LICENSE](LICENSE)).
