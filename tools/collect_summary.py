"""Collect the headline numbers used in README.md and docs/RESULTS.md into results/ml/summary.json.

Reads only files that ship with this repository (results/ml/...), so the README tables and figures can be checked
against the per-image result files:
- synthetic holdout (film_holdout3), quake line: per-image re-scoring of the row-anchor model, SAM 2.1 and B
  (results/ml/rescoring/verify_*_holdout3.json) and the repair experiment's own summary (results/ml/repair);
- real paired scans (KIP, MAT): per-line-category metrics from each experiment's real-scan evaluation.

usage: python tools/collect_summary.py
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ML = ROOT / "results" / "ml"
GAINS = (1, 2, 4, 6, 8)


def per_gain(rows, method):
    out = {}
    for g in list(GAINS) + ["all"]:
        rr = [r[method] for r in rows if g == "all" or r["gain"] == g]
        out[str(g)] = {"r": round(float(np.mean([r["r_q"] for r in rr])), 3),
                       "fail": round(float(np.mean([r["fail_q"] for r in rr])), 1),
                       "r_neighbour": round(float(np.mean([r["r_nb"] for r in rr])), 3), "n": len(rr)}
    return out


def r_of(v):
    return v["r_band"] if isinstance(v, dict) else v


def main():
    ra = json.loads((ML / "rescoring" / "verify_ra_holdout3.json").read_text())
    sam = json.loads((ML / "rescoring" / "verify_sam_holdout3.json").read_text())
    rep = json.loads((ML / "repair" / "holdout3_results.json").read_text())["summary"]
    synth = {"row_anchor": per_gain(ra, "RA_viterbi"), "row_anchor_argmax": per_gain(ra, "RA_argmax"),
             "sam21": per_gain(sam, "SAM"), "B": per_gain(ra, "B")}
    for m in ("flat", "oracle", "seq", "seqp", "seq1", "spec"):
        synth["repair_" + m] = {g: {"r": rep[g]["quake"][m]["r"], "fail": round(rep[g]["quake"][m]["fail"], 1)} for g in rep}

    real = {}
    rar = json.loads((ML / "rowanchor" / "real_results.json").read_text())["scans"]
    for scan in ("KIP", "MAT"):
        s_ra = rar[scan]["summary"]
        s_sam = json.loads((ML / "sam21" / f"real_{scan}.json").read_text())["summary"]
        s_rep = json.loads((ML / "repair" / f"real_{scan}.json").read_text())["summary"]
        real[scan] = {}
        for cat in ("quake", "neighbour", "other_active", "quiet"):
            row = {"row_anchor": s_ra[cat]["RA_viterbi"], "row_anchor_argmax": s_ra[cat]["RA_argmax"],
                   "B": s_ra[cat]["B_recomputed"], "DP": s_ra[cat]["DP_published"], "sam21": s_sam[cat]["SAM"],
                   "DP38": s_sam[cat]["DP38_stored"]}
            row = {k: {kk: v[kk] for kk in ("r_band", "in_0.1S_global", "lost_0.5S_global")} for k, v in row.items()}
            for m in ("seq", "spec"):
                if m in s_rep.get(cat, {}):
                    row["repair_" + m] = {"r_band": r_of(s_rep[cat][m])}
            real[scan][cat] = row
    out = {"synthetic_holdout_quake_line": synth, "real_paired": real,
           "notes": {"synthetic": "film_holdout3: 10 unseen earthquakes, 19 records x gains 1,2,4,6,8 = 95 images; "
                                  "each method frozen on validation data before a single test run",
                     "r": "Pearson r between the extracted quake-line path and the ground-truth signal",
                     "fail": "% of drawn columns missing or more than half a line spacing from the true pen position",
                     "real": "r_band = r after a 0.0025-0.08 Hz band-pass against the co-located digital station "
                             "(HGLP), with the scan-to-time alignment fitted once and stored before any method was "
                             "scored"}}
    (ML / "summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v["all"] for k, v in synth.items()}, indent=0))
    for scan in real:
        print(scan, {k: round(v["r_band"], 3) for k, v in real[scan]["quake"].items()})


if __name__ == "__main__":
    main()
