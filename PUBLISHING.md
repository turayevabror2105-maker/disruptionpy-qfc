# Publishing notes

This file exists because the information below is needed by whoever pushes this repository, and it would
otherwise live only in a private log.

## The local history here is unrelated to the remote

`pyproject.toml` names `https://github.com/turayevabror2105-maker/disruptionpy-qfc`. That repository
**exists and already has history**: one commit (`f9abdd4`, 2026-08-13, *"Add QFC module for
DisruptionPy — pre-deployment attribution stability screening"*) holding the **version 1 single-file
module** — a 432-line `disruptionpy_qfc.py` with no README, no LICENCE, no tests, no CI and no packaging.

What is in this directory is **version 2.0.0**: a packaged, tested rewrite. Its git history was
initialised here on 2026-09-26 and has **no common ancestor** with that commit. So publishing is not a
fast-forward. Three options, in the order they should be considered:

1. **Fetch and rebase onto the existing commit.** Preserves the v1 commit as the repository's origin
   story, which is honest about the package's history and costs one `git rebase --onto`.
2. **Add v2 as a new commit on top**, treating v1 as a superseded state. Simplest, and keeps the history
   linear and truthful.
3. **Force-push.** Discards the v1 commit. Only if the v1 module is not worth keeping in the record; note
   that the v2 test suite refers to v1's numbers (`V1_QFC = 0.9426`, `V1_AUC = 0.9375`) as a genuine
   cross-version check, so there is some value in v1 remaining visible.

**Nothing has been pushed from here.** The account, and the choice among the three, belong to the owner.

## What version 2 changed, and why it matters for the history

Version 1 had **no null floor at all** — it compared QFC against a fixed threshold of 0.391 borrowed from
an unrelated domain, which the C-Mod era shift passed comfortably. Version 2 builds the floor from a
no-shift split of the source pool and scores against the exact Student-t cut. The same data then reads the
opposite way: **COLLAPSE, z = −2.69**. That is not a bug fix, it is the reason the package exists, and
`tests/test_regression_cmod.py` records in its own docstring how the earlier expectation came to be wrong.

## Before the first push

* `pytest -m "not slow"` — 53 tests, no dataset needed.
* `pytest` — 61, the extra four recomputing the worked example from the Open Density Limit Database.
* `.github/workflows/ci.yml` has **never run**, because the repository has had no CI-triggering push. Its
  command is `pytest -m "not slow" -q` on Python 3.10/3.11/3.12 under `ubuntu-latest`. Two of the three are
  verified locally: **3.10.9**, and **3.12.10** on 2026-09-26 — 48 passed, 4 skipped (the `integration`
  group, which needs DisruptionPy), 0 failed, 58 s — with **pandas 3.0.6, shap 0.52.0 and
  scikit-learn 1.5.1**, a newer major pandas and shap than the 3.10 environment carries, so the declared
  dependency ranges are exercised forward as well. **3.11.9 is now verified too**, on 2026-09-29 —
  48 passed, 4 skipped, 0 failed in 37.8 s — in a throwaway virtual environment with **numpy 2.4.6,
  pandas 3.0.6, scikit-learn 1.9.1, scipy 1.17.1 and shap 0.51.0**, a markedly newer stack than either
  of the other two, so the declared dependency ranges are exercised well forward of their floors. All three interpreter runs above predate the five tests added on 2026-09-30 — the reproducible stand-in frame and the module that imports the `integration` group's assertion bodies and runs them against it — so their 48 is the default-group count as of those dates, not a figure that disagrees with the 53 above. So all
  three interpreters in the CI matrix have now been run locally, and the only thing `ci.yml` itself adds
  is `ubuntu-latest` rather than Windows. Nothing was installed into the 3.10 environment, which the
  author's campaigns and audit daemon run under; each check used its own virtual environment and the
  environments were discarded afterwards.
* The wheel in `dist/` is gitignored and is not part of the history.
