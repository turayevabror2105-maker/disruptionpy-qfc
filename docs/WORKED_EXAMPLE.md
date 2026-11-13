# Worked example: Alcator C-Mod, the 2000, 2001 and 2003 campaigns to 2005 and 2009

The [README](../README.md) gives the short version. This page gives the run in full: how to reproduce it, the data,
the result, which signals moved, two controls for the change in class balance, and why the paper's figures for this
comparison differ.

## Reproducing it

The README's [three commands](../README.md#run-the-c-mod-example-in-three-commands) install the package, fetch the
data and run the demo:

```bash
python -m disruptionpy_qfc.demo --data DL_DataFrame.csv
```

The second of the three commands fetches the public release at a fixed commit (24,605,618 bytes, SHA-256
`c296028985f880cfa341030af0b54d7fe7092b639156281dba454b0926385318`); from it the demo builds frames with the record's
row, discharge and label counts. On Windows PowerShell 5.1, `curl` is an alias of `Invoke-WebRequest`, so type
`curl.exe` there. The demo prints the frame sizes and the result's `summary()` (QFC, floor, z, cut, resolution and
verdict), and `--out` writes the full result as JSON: `result.to_dict()` under `"result"` and a note on the left-out
campaigns under `"excluded"`.

**How long it takes.** The recorded run took 531 seconds, about nine minutes, on a 32-core Linux machine
(`docs/worked_example.json`, `default.seconds`). A laptop has not been timed. The run is at most 231 fits of a
500-tree forest plus their TreeSHAP attributions (see [Cost](API.md#cost)), and the forest fits use every
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

## Data

The public Open Density Limit Database (Maris, Rea, Trevisan and the Alcator C-Mod Team, MIT PSFC, 2025,
CC BY; `github.com/MIT-PSFC/open_density_limit_database`). Six signals, each turned into eight statistics over the
10 preceding time points of its shot (48 features). The source is 3,363 rows from 40 discharges, positive rate
0.508; the target is 2,504 rows from 38 discharges, positive rate 0.721. Discharges from 2014–2016 are left out
because the public release has them labelled stable throughout (no time point in a density-limit phase).

## Result

| quantity | value |
|---|---|
| target AUC | 0.939 |
| QFC (mean of 10 fold orders) | 0.9607 (orders range 0.9572 to 0.9644) |
| null floor | 0.9805 ± 0.0069, 30 draws |
| z, against the cut of 2.079 | −2.87 |
| resolution | 0.0143 |
| verdict | **COLLAPSE** (10 of the 10 fold orders collapse on their own) |

The terms are defined in [METHOD.md](METHOD.md#what-the-words-mean).

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

## The record

Every number the package computed on this page is in [`docs/worked_example.json`](worked_example.json), written by
`scripts/regenerate_worked_example.py` with the machine, library versions and a hash of the package code it
records; the paper's figures are quoted from its census receipts, which are not yet published.
