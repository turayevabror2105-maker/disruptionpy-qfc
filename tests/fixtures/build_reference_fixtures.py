"""Build the two fixtures the identity tests read, from the paper's own archived artefacts.

    python tests/fixtures/build_reference_fixtures.py --archive DIR [DIR ...] --census DIR

``--archive`` names the metric head-to-head archive folders (one JSON per condition, holding the stored
importance vectors of the real comparison and of all 30 null draws, plus the QFC, null mean, null sd
and z the paper's pipeline computed from them). ``--census`` names the folder holding
``census_a_rebuilt.json`` and ``census_a_final_123.json``, the paper's census receipts.

Writes:

* ``reference_vectors.json`` -- seven conditions' stored vectors and stored statistics, so that
  ``test_reference_identity.py`` can recompute QFC, the null moments, z and the verdict from the
  vectors and compare them with what the paper stored, with no dataset and no model fitting.
* ``census_rule_rows.json`` -- the census rows whose verdict the chance-floor rule changed, with the
  numbers the rule read, so the package's rule can be checked against the paper's decision on each.

Each record carries the sha256 of the file it came from.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PICK = {
    "metric_head_to_head_f12_30": ["plasma_temporal_2000_to_2005", "radiation_isotope_population",
                                   "fp_compression_q20", "pcg_A_to_B"],
    "metric_head_to_head_gbm_f12_30": ["plasma_temporal_2000_to_2005", "ecg_ptbxl_to_code15", "ham_noise_gaussian"],
}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", nargs="+", required=True)
    ap.add_argument("--census", required=True)
    a = ap.parse_args()

    rows = []
    for d in map(Path, a.archive):
        for name in PICK[d.name]:
            p = d / (name + ".json")
            r = json.load(io.open(p, encoding="utf-8"))
            m = r["metrics"]["qfc_spearman"]
            rows.append({"arm": d.name, "condition": name, "source_sha256": sha(p),
                         "real_src": r["real_importance_src"], "real_tgt": r["real_importance_tgt"],
                         "null_src": r["null_importance_src_per_seed"], "null_tgt": r["null_importance_tgt_per_seed"],
                         "stored": {"real": m["real"], "null_mean": m["null_mean"], "null_std": m["null_std"],
                                    "z": m["z"]}})
    (HERE / "reference_vectors.json").write_text(json.dumps({"conditions": rows}) + "\n", encoding="utf-8")

    c = Path(a.census)
    final, rebuilt = c / "census_a_final_123.json", c / "census_a_rebuilt.json"
    F = json.load(io.open(final, encoding="utf-8"))
    R = {x["key"]: x for x in json.load(io.open(rebuilt, encoding="utf-8"))["rows"]}
    out, other = [], 0
    for adj in F["adjudicated_rows"]:
        if "chance_floor" not in adj:
            other += 1    # rows changed by the census's other rule (a construction-matched null), not this one
            continue
        x = R[adj["row"]]
        # The census scorer lists a row here only when it changed the row's verdict, and for a row listed with a
        # chance floor it sets the verdict to INDISTINGUISHABLE (census_a_final_123.py, the CHANCE branch).
        out.append({"row": adj["row"], "real": x["real"], "null_mean": x["null_mean"], "null_sd": x["null_sd"],
                    "n_null": x["n"], "census_z": x["z"], "standard": x["standard"],
                    "chance_floor": adj["chance_floor"], "census_adjudicated": "INDISTINGUISHABLE"})
    tallies = {"standard_robust": F["standard"]["tally"]["ROBUST"],
               "adjudicated_robust": F["standard_adjudicated"]["tally"]["ROBUST"],
               "changed_by_this_rule": len(out), "changed_by_the_other_rule": other}
    (HERE / "census_rule_rows.json").write_text(json.dumps(
        {"rule": "a ROBUST row whose no-shift floor mean lies below its permuted-label chance floor is read "
                 "INDISTINGUISHABLE",
         "sources": {"census_a_final_123.json": sha(final), "census_a_rebuilt.json": sha(rebuilt)},
         "census_tallies": tallies, "rows": out}, indent=1) + "\n", encoding="utf-8")
    print(len(rows), "conditions;", len(out), "census rule rows")


if __name__ == "__main__":
    main()
