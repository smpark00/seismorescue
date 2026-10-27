"""Development CV of the digitizer on film-geometry renders (docs/RESULTS.md): proposers, judge, detector + proposer C.

usage: python -I film_dev_cv.py <film_bench> <film_cache> <out.json>

Same system as docs/RESULTS.md, re-fitted to film geometry on development records only: the judge v5 features are
retrained per fold on the training groups, and the quiet-line count Q for the single-active regime is chosen per fold
from Q_GRID. Reports, per gain, the whole-image fail and the earthquake-line fail of A, B, judge and system
(GroupKFold by event date, 4 folds x 3 shuffles).
"""
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from active_guard import detect_active, track_active_clean
from arbiter import assemble, cache_proposals
from cv_judges import fit_lr, load_items
from track_joint import score

FILM_CONDS = [("film", 1.0), ("film", 1.5), ("film", 2.0)]
Q_GRID = [3, 4, 5, 6]


def run_system(it, model, q, bench):
    meta = json.loads((bench / it["record"] / it["mode"] / "meta.json").read_text())
    p = model(np.array([r["x"] for r in it["rows"]])) if it["rows"] else np.array([])
    judge = assemble(it["A"], it["B"], it["rows"], ["A" if v > 0.5 else "B" for v in p])
    system = judge.copy()
    k, nq = it["det"]
    if k is not None and nq >= q:
        system[k, it["x0"]:it["x1"]] = it["C"][it["x0"]:it["x1"]]
    return judge, system, meta


def fails(it, P):
    N = it["pen"].shape[0]
    return (score(P, it["pen"], range(N), it["x0"], it["x1"], it["S"])[0],
            score(P, it["pen"], [it["active"]], it["x0"], it["x1"], it["S"])[0])


if __name__ == "__main__":
    bench, cache, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    cache_proposals(bench, cache, FILM_CONDS)
    items = load_items(bench, cache, FILM_CONDS)
    for it in items:                                    # detector + proposer C (GT-free; start = layout start)
        meta = json.loads((bench / it["record"] / it["mode"] / "meta.json").read_text())
        ink = np.load(cache / f"{it['record']}_{it['mode']}_g{it['gain']:g}.npz")["ink"]
        it["det"] = detect_active(it["B"], ink, meta["margin_y"], it["S"], it["x0"], it["x1"])
        k = it["det"][0]
        it["C"] = track_active_clean(it["B"], ink, it["pen"][k, it["x0"]], meta["margin_y"], it["S"], it["x0"], it["x1"]) if k is not None else None
    print(f"items {len(items)}; detector right on {np.mean([it['det'][0] == it['active'] for it in items]):.0%}; "
          f"quiet lines median {np.median([it['det'][1] for it in items]):.0f}", flush=True)
    groups = sorted({it["group"] for it in items})
    res = []
    for rep in range(3):
        gs = groups[:]
        random.Random(rep).shuffle(gs)
        for fold in [gs[i::4] for i in range(4)]:
            tr = [it for it in items if it["group"] not in fold]
            rows = [r for it in tr for r in it["rows"] if r["label"] >= 0]
            model = fit_lr(np.array([r["x"] for r in rows]), np.array([r["label"] for r in rows], float),
                           np.array([r["stake"] for r in rows], float))
            sysf = {q: np.mean([sum(fails(it, run_system(it, model, q, bench)[1])) for it in tr]) for q in Q_GRID}
            q = min(Q_GRID, key=sysf.get)
            for it in items:
                if it["group"] in fold:
                    judge, system, _ = run_system(it, model, q, bench)
                    row = {"rep": rep, "record": it["record"], "group": it["group"], "gain": it["gain"], "q": q,
                           "fired": it["det"][0] is not None and it["det"][1] >= q}
                    for name, P in (("A", it["A"]), ("B", it["B"]), ("judge", judge), ("system", system)):
                        row[name], row[name + "_active"] = fails(it, P)
                    res.append(row)
    for g in (1.0, 1.5, 2.0):
        rr = [r for r in res if r["gain"] == g]
        print(f"film g{g}: whole  " + "  ".join(f"{k} {np.mean([r[k] for r in rr]):5.1f}" for k in ("A", "B", "judge", "system"))
              + f"  | quake line  " + "  ".join(f"{k} {np.mean([r[k + '_active'] for r in rr]):5.1f}" for k in ("A", "B", "judge", "system"))
              + f"  | regime fired {np.mean([r['fired'] for r in rr]):.0%}")
    qs = {}
    for r in res:
        qs[r["q"]] = qs.get(r["q"], 0) + 1
    print("Q chosen:", qs)
    out.write_text(json.dumps(res, indent=1))
