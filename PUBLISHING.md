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

* `pytest -m "not slow"` — 48 tests, no dataset needed.
* `pytest` — 56, the extra four recomputing the worked example from the Open Density Limit Database.
* `.github/workflows/ci.yml` has **never run**, because the repository has had no CI-triggering push. Its
  command is `pytest -m "not slow" -q` on Python 3.10/3.11/3.12 under `ubuntu-latest`, which is what was
  verified locally on 3.10.9; the other two interpreters are untested anywhere.
* The wheel in `dist/` is gitignored and is not part of the history.
