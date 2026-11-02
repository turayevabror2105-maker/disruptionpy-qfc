"""Regenerate ``docs/worked_example.json``, the record of every number the README states about the C-Mod example.

    python scripts/regenerate_worked_example.py --data DL_DataFrame.csv --out-dir PARTS PART
    python scripts/regenerate_worked_example.py --out-dir PARTS assemble

PART is one of the six below; each writes ``PARTS/<part>.json`` and can run in its own process. ``assemble``
reads all six and writes ``docs/worked_example.json``.

  default        screen() at the package defaults: the worked example.
  depth20        the same procedure with the paper's C-Mod tree depth (max_depth=20).
  paper_null     the paper's C-Mod null construction, rebuilt on the package's frames: one forest fitted on half 1,
                 its IN-SAMPLE attributions on half 1 against half 2 (source_pool_null_floor_check.py in the
                 paper's deposit), at depth 20, cap 500, seeds 0-29. It explains why the paper's floor for
                 this condition is higher than the package's.
  prior_gap      the within-null label-prior control: each no-shift draw of ``default`` paired with its own
                 label-prior gap (pre-registered as TASK7_PRIOR_GAP_VS_NULL_QFC_PREREGISTRATION.md).
  prior_matched  the explained-rows label-prior control: the explained 500 target rows stratified to the target's
                 prior and to the source's, everything else fixed, ten paired seeds (pre-registered as
                 PLASMA_LABEL_PRIOR_PREREGISTRATION.md). Uses the default model and the ``default`` floor.
  floor_seed0    the default floor rebuilt from another 30 splits (random_state=0: splits and model seeds 0-29, the
                 split seeds the paper's census used for this row), scored against the ``default`` real value.
                 How much the verdict depends on which 30 splits were drawn.

Expensive: each part is hundreds of 500-tree forests plus TreeSHAP. Run it on a machine with cores to spare.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import disruptionpy_qfc as dq  # noqa: E402
from disruptionpy_qfc.core import _arrays, _real_arm, _treeshap, _explain  # noqa: E402
from disruptionpy_qfc.demo import build_demo_frames  # noqa: E402

PARTS = ("default", "depth20", "paper_null", "prior_gap", "prior_matched", "floor_seed0")


def _env() -> dict:
    import hashlib
    import os

    import pandas, scipy, shap, sklearn
    src = hashlib.sha256()
    for f in sorted((ROOT / "src" / "disruptionpy_qfc").glob("*.py")):   # the code that ran, line endings normalised
        src.update(f.name.encode() + f.read_bytes().replace(b"\r\n", b"\n"))
    return {"python": platform.python_version(), "platform": platform.platform(), "cpus": os.cpu_count(),
            "numpy": np.__version__, "pandas": pandas.__version__, "scipy": scipy.__version__,
            "scikit-learn": sklearn.__version__, "shap": shap.__version__, "package": dq.__version__,
            "src_sha256": src.hexdigest()}


def _run(source, target, model=None):
    t0 = time.time()
    floor = dq.compute_null_floor(source, model=model)
    r = dq.compute_qfc(source, target, null_floor=floor, model=model)
    if r.verdict_unadjudicated == "ROBUST":
        from disruptionpy_qfc.core import _score
        _score(r, floor, dq.compute_chance_floor(source, target, model=model))
    return {"result": r.to_dict(), "null_floor": floor.to_dict(), "seconds": round(time.time() - t0, 1)}


def part_paper_null(source, target):
    """The paper's construction for this row, read from its source_pool_null_floor_check.py: per seed s,
    GroupShuffleSplit(0.5, random_state=s); one RF(500, depth 20, balanced, random_state=s) fitted on half 1;
    500 rows of half 1 and 500 of half 2 drawn by one default_rng(s), in that order; TreeSHAP; Spearman."""
    cols = dq.detect_feature_cols(source)
    X, y, g = _arrays(source, cols, "label", "shot")
    draws = []
    for s in range(30):
        tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=s).split(X, y, g))
        r = np.random.default_rng(s)
        m = dq.default_model(s, max_depth=20).fit(X[tr], y[tr])
        a = tr if len(tr) <= 500 else r.choice(tr, size=500, replace=False)
        b = te if len(te) <= 500 else r.choice(te, size=500, replace=False)
        ia = np.abs(_treeshap(m, X[a], None, s)).mean(axis=0)
        ib = np.abs(_treeshap(m, X[b], None, s)).mean(axis=0)
        draws.append(dq.qfc_statistic(ia, ib))
    d = np.asarray(draws)
    # the paper's real value at seed 42 on its cap-matched random rule: one fold order, depth 20, the same
    # cap rule as the package (verify_shap_cap_headvsrandom.py, mode 'random')
    Xt = target[cols].to_numpy(float)
    s42 = dq.qfc_statistic(*_real_arm(X, y, g, Xt, dq.default_model(42, max_depth=20), 5, 500, 42, _treeshap)[:2])
    return {"construction": "single forest on half 1, in-sample half 1 vs out-of-sample half 2 (the paper's census "
                            "null for this row)", "draws": draws, "mean": float(d.mean()), "sd": float(d.std(ddof=1)),
            "real_depth20_seed42_order0": s42}


def compare_frames(source, d):
    """Are the package's frames the paper's arrays? Rows in the same order, labels and shot partition equal, and
    every paper column equal to one package column. Column ORDER is reported separately: a forest's column order
    changes which features each split samples, so equal columns in another order give a different forest."""
    cols = dq.detect_feature_cols(source)
    X, y, g = _arrays(source, cols, "label", "shot")
    Xp, yp, gp = (np.load(d / n) for n in ("X_source.npy", "y_source.npy", "groups_source.npy"))
    same_part = len(set(zip(g.tolist(), gp.tolist()))) == len(np.unique(g)) == len(np.unique(gp))
    match = [next((i for i in range(X.shape[1]) if np.array_equal(Xp[:, j], X[:, i])), None)
             for j in range(Xp.shape[1])]
    unmatched = {cols[j]: float(np.abs(Xp[:, j] - X[:, j]).max()) for j, m in enumerate(match) if m is None}
    return {"shape_equal": list(Xp.shape) == list(X.shape), "labels_equal": bool(np.array_equal(yp, y)),
            "shot_partition_equal": bool(same_part), "paper_column_to_package_column": match,
            "n_columns_bit_identical": sum(m is not None for m in match),
            "unmatched_max_abs_diff_same_position": unmatched,
            "all_columns_matched": None not in match, "same_column_order": match == list(range(X.shape[1]))}


def part_prior_gap(source, default_part):
    floor = default_part["null_floor"]
    cols = dq.detect_feature_cols(source)
    X, y, g = _arrays(source, cols, "label", "shot")
    gaps, used_seeds = [], []
    for k in range(floor["n_seeds_requested"]):
        a, b = next(GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42 + k).split(X, y, g))
        if len(np.unique(y[a])) < 2 or len(np.unique(g[a])) < 5:
            continue
        gaps.append(abs(float(y[a].mean()) - float(y[b].mean())))
        used_seeds.append(k)
    q = np.asarray(floor["per_seed"], float)
    if len(gaps) != len(q):
        raise RuntimeError("the floor skipped a draw this control did not; pair them by seed before reading")
    gaps = np.asarray(gaps)
    res = default_part["result"]
    real_gap = abs(float(target_prior(default_part) - source_prior(default_part)))
    drop = floor["mean"] - res["qfc"]
    sp = stats.spearmanr(gaps, q)
    lr = stats.linregress(gaps, q)
    n = len(q)
    tcrit = float(stats.t.ppf(0.975, n - 2))
    cut = tcrit / np.sqrt(tcrit ** 2 + n - 2)
    frac = (-lr.slope * real_gap) / drop
    frac_hi = (-(lr.slope - tcrit * lr.stderr) * real_gap) / drop
    big = [{"seed": int(s), "prior_gap": float(gp), "qfc": float(v), "z_vs_floor": float((v - floor["mean"]) / floor["sd"])}
           for s, gp, v in zip(used_seeds, gaps, q) if gp > real_gap]
    rho_sig = abs(sp.statistic) >= cut
    outcome = "P-DRIVEN" if (rho_sig and sp.statistic < 0 and frac >= 0.5) else "P-FLAT"
    return {"n_usable": n, "real_prior_gap": real_gap, "observed_drop": drop, "prior_gap_mean": float(gaps.mean()),
            "prior_gap_max": float(gaps.max()), "spearman_rho": float(sp.statistic), "spearman_p": float(sp.pvalue),
            "spearman_cut_5pct": float(cut), "ols_slope": float(lr.slope), "ols_slope_se": float(lr.stderr),
            "fraction_of_observed_drop": float(frac), "fraction_at_95pct_steep_end": float(frac_hi),
            "draws_with_gap_above_real": big, "outcome": outcome,
            "rule": "P-DRIVEN if rho is beyond its 5% cut, negative, and the slope explains at least half the drop; "
                    "otherwise P-FLAT (fixed before the v2 run; applied unchanged)"}


def source_prior(p):
    return p["priors"]["source"]


def target_prior(p):
    return p["priors"]["target"]


def part_prior_matched(source, target, default_part):
    cols = dq.detect_feature_cols(source)
    X, y, g = _arrays(source, cols, "label", "shot")
    Xt, yt = target[cols].to_numpy(float), target["label"].to_numpy().astype(int)
    p_src, p_tgt = float(y.mean()), float(yt.mean())

    def strat(rng, rate):
        n_pos = int(round(500 * rate))
        pos, neg = np.flatnonzero(yt == 1), np.flatnonzero(yt == 0)
        return np.concatenate([rng.choice(pos, n_pos, replace=False), rng.choice(neg, 500 - n_pos, replace=False)])

    T, M = [], []
    for seed in range(10):
        # the source vector and the deployed model of the real arm (fold order 0); the target arm is replaced below
        src, _, deployed, _ = _real_arm(X, y, g, Xt[:1], dq.default_model(seed), 5, 500, seed, _treeshap)
        rng = np.random.default_rng(seed)
        iT, iM = strat(rng, p_tgt), strat(rng, p_src)
        T.append(dq.qfc_statistic(src, _explain(deployed, Xt[iT], X, None, seed, _treeshap)))
        M.append(dq.qfc_statistic(src, _explain(deployed, Xt[iM], X, None, seed, _treeshap)))
    T, M = np.asarray(T), np.asarray(M)
    f = default_part["null_floor"]
    dif = M - T
    res_sd = float(dif.std(ddof=1))
    gap = abs(default_part["result"]["qfc"] - f["mean"])
    zT, zM = (T.mean() - f["mean"]) / f["sd"], (M.mean() - f["mean"]) / f["sd"]
    if res_sd > gap:
        outcome = "P-UNDERPOWERED"
    elif abs(dif.mean()) < res_sd:
        outcome = "IMMATERIAL"
    elif zM <= -f["exact_cut"]:
        outcome = "P-HOLDS"
    else:
        outcome = "P-VANISHES"
    return {"arm_T": {"stratified_to": p_tgt, "composition": [int(round(500 * p_tgt)), 500 - int(round(500 * p_tgt))],
                      "per_seed": T.tolist(), "mean": float(T.mean()), "sd": float(T.std(ddof=1)), "z": float(zT)},
            "arm_M": {"stratified_to": p_src, "composition": [int(round(500 * p_src)), 500 - int(round(500 * p_src))],
                      "per_seed": M.tolist(), "mean": float(M.mean()), "sd": float(M.std(ddof=1)), "z": float(zM)},
            "paired_delta_M_minus_T": float(dif.mean()), "paired_delta_sd": res_sd, "underpowered_above": gap,
            "outcome": outcome, "fold_order": 0,
            "rule": "P-UNDERPOWERED if the paired sd exceeds |real - floor|; IMMATERIAL if |delta| < paired sd; "
                    "P-HOLDS if the matched arm still collapses; otherwise P-VANISHES (fixed before the v2 run)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=PARTS + ("assemble",))
    ap.add_argument("--data")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--paper-arrays", help="paper_null only: folder with the paper's X_source.npy, y_source.npy and "
                                           "groups_source.npy, to check that the frames are the paper's")
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if a.part == "assemble":
        rec = {p: json.loads((out / (p + ".json")).read_text(encoding="utf-8")) for p in PARTS}
        # How much the verdict depends on which 30 splits the null drew: both 30-split floors, and all 60 pooled.
        both = np.asarray(rec["default"]["null_floor"]["per_seed"] + rec["floor_seed0"]["null_floor"]["per_seed"])
        real = rec["default"]["result"]["qfc"]
        z60 = (real - both.mean()) / both.std(ddof=1)
        rec["split_sensitivity"] = {
            "real": real,
            "floor_seeds_42_71": {"mean": rec["default"]["null_floor"]["mean"], "sd": rec["default"]["null_floor"]["sd"],
                                  "z": rec["default"]["result"]["z"], "verdict": rec["default"]["result"]["verdict"]},
            "floor_seeds_0_29": {"mean": rec["floor_seed0"]["null_floor"]["mean"],
                                 "sd": rec["floor_seed0"]["null_floor"]["sd"], "z": rec["floor_seed0"]["z"],
                                 "verdict": rec["floor_seed0"]["verdict"]},
            "pooled_60": {"mean": float(both.mean()), "sd": float(both.std(ddof=1)), "n": int(both.size),
                          "cut": dq.exact_cut(int(both.size)), "z": float(z60),
                          "verdict": dq.verdict_label(float(z60), dq.exact_cut(int(both.size)))}}
        doc = {"what": "every number the README states about the C-Mod worked example, from one regeneration",
               "generator": "scripts/regenerate_worked_example.py", "package_version": dq.__version__,
               "dataset": "The Open Density Limit Database (Maris, Rea, Trevisan & the Alcator C-Mod Team, MIT PSFC, "
                          "2025), DL_DataFrame.csv", **rec}
        from disruptionpy_qfc.core import _jsonable
        (ROOT / "docs" / "worked_example.json").write_text(json.dumps(_jsonable(doc), indent=1, allow_nan=False) + "\n",
                                                          encoding="utf-8")
        print("written docs/worked_example.json")
        return
    source, target = build_demo_frames(a.data)
    t0 = time.time()
    meta = {"part": a.part, "env": _env(), "started": time.strftime("%Y-%m-%d %H:%M:%S"),
            "frames": {"source_rows": len(source), "source_shots": int(source.shot.nunique()),
                       "target_rows": len(target), "target_shots": int(target.shot.nunique()),
                       "features": len(dq.detect_feature_cols(source))},
            "priors": {"source": float(source.label.mean()), "target": float(target.label.mean())}}
    if a.part == "default":
        meta.update(_run(source, target))
    elif a.part == "depth20":
        meta.update(_run(source, target, model=dq.default_model(max_depth=20)))
    elif a.part == "paper_null":
        meta.update(part_paper_null(source, target))
        if a.paper_arrays:
            meta["frames_vs_paper"] = compare_frames(source, Path(a.paper_arrays))
    elif a.part == "floor_seed0":
        dpart = json.loads((out / "default.json").read_text(encoding="utf-8"))
        floor = dq.compute_null_floor(source, random_state=0)
        z, v = floor.verdict_for(dpart["result"]["qfc"])
        meta.update({"null_floor": floor.to_dict(), "real_from_default": dpart["result"]["qfc"], "z": z, "verdict": v})
    else:
        dpart = json.loads((out / "default.json").read_text(encoding="utf-8"))
        meta.update(part_prior_gap(source, dpart) if a.part == "prior_gap" else
                    part_prior_matched(source, target, dpart))
    meta["seconds_total"] = round(time.time() - t0, 1)
    (out / (a.part + ".json")).write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    print("written", out / (a.part + ".json"), meta["seconds_total"], "s")


if __name__ == "__main__":
    main()
