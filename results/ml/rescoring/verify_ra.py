"""Re-score the frozen row-anchor model (docs/RESULTS.md) on film_holdout3 with r (same definition as docs/RESULTS.md) and
fail; compare fail per image with the agent's holdout3_results.json. Frozen config only; nothing is tuned."""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[3]   # repository root
sys.path.insert(0, str(ROOT / "scripts"))
import ml_rowanchor as ra
from track_joint import score
fz = json.loads((ra.OUT / "frozen.json").read_text())
model, ck = ra.load_model("best")
logT = ra.robust(np.load(ra.OUT / "logT.npy"), fz.get("eps", 0.0))
agent = {(r["name"], r["gain"]): r for r in json.loads((ra.OUT / "holdout3_results.json").read_text())["rows"]}
cache = ROOT / "out/cache_film_holdout3"
def metr(P, pen, sig, base, k, x0, x1, S):
    a = (base[k] - P[k, x0:x1]) / S
    apen, asig = (base[k] - pen[k, x0:x1]) / S, (base[k] - sig[k, x0:x1]) / S
    ok = np.isfinite(apen) & np.isfinite(asig)
    an = np.nan_to_num(a)[ok]; t = asig[ok]
    r = np.corrcoef(an, t)[0, 1] if an.std() > 0 else 0.0
    e = np.abs(np.nan_to_num(a, nan=1e9)[ok] - t)
    return r, float(np.mean(e < 0.1)), float(np.mean(e > 0.5))
out = []
for rec in sorted(p for p in ra.HOLD_DIR.iterdir() if p.is_dir()):
    for g in ra.G_HOLD:
        it = ra.load_item(rec, g)
        sig = np.load(rec / f"film/gt_g{g}.npz")["sig"]
        y0s = it["pen"][:, it["x0"]]
        lp = ra.infer(model, it["D"], it["fy"], it["fx"], it["bases"], y0s, it["x0"], it["x1"])
        u0 = [ra.to_bin(y, b, float(it["x0"])) for y, b in zip(y0s, it["bases"])]
        P = {"RA_viterbi": ra.decode(lp, it["bases"], it["x0"], it["x1"], it["W"], u0, logT, fz["alpha"]),
             "RA_argmax": ra.decode(lp, it["bases"], it["x0"], it["x1"], it["W"], u0),
             "B": np.load(cache / f"{rec.name}_film_g{g}.npz")["B"]}
        q, n, x0, x1, S = it["act"], len(it["bases"]), it["x0"], it["x1"], it["S"]
        row = {"name": rec.name, "gain": g}
        for m, est in P.items():
            fq = score(est, it["pen"], [q], x0, x1, S)[0]
            fw = score(est, it["pen"], range(n), x0, x1, S)[0]
            rq, inq, lq = metr(est, it["pen"], sig, it["bases"], q, x0, x1, S)
            nb = [k for k in (q - 1, q + 1) if 0 <= k < n]
            rn = float(np.mean([metr(est, it["pen"], sig, it["bases"], k, x0, x1, S)[0] for k in nb]))
            rw = float(np.mean([metr(est, it["pen"], sig, it["bases"], k, x0, x1, S)[0] for k in range(n)]))
            row[m] = {"fail_q": fq, "fail_w": fw, "r_q": rq, "in01_q": inq, "lost05_q": lq, "r_nb": rn, "r_whole": rw}
            if m in agent[(rec.name, g)]:
                row[m]["agent_fail_q"] = agent[(rec.name, g)][m]["quake"]
        out.append(row)
        print(rec.name, g, {m: round(row[m]["r_q"], 3) for m in P}, flush=True)
(Path(__file__).parent / "verify_ra_holdout3.json").write_text(json.dumps(out, indent=1))
for g in list(ra.G_HOLD) + ["all"]:
    rr = [r for r in out if g == "all" or r["gain"] == g]
    print(f"g{g} n={len(rr)} " + " | ".join(
        f"{m}: r_q {np.mean([r[m]['r_q'] for r in rr]):+.3f} fail_q {np.mean([r[m]['fail_q'] for r in rr]):5.1f} "
        f"in01 {np.mean([r[m]['in01_q'] for r in rr]):.2f} lost05 {np.mean([r[m]['lost05_q'] for r in rr]):.2f} "
        f"r_nb {np.mean([r[m]['r_nb'] for r in rr]):+.3f} r_whole {np.mean([r[m]['r_whole'] for r in rr]):+.3f}"
        for m in ("RA_viterbi", "RA_argmax", "B")))
mism = [(r["name"], r["gain"], m, r[m]["fail_q"], r[m]["agent_fail_q"]) for r in out for m in ("RA_viterbi", "RA_argmax", "B")
        if "agent_fail_q" in r[m] and abs(r[m]["fail_q"] - r[m]["agent_fail_q"]) > 0.11]
print("per-image quake-fail mismatches vs agent:", len(mism), mism[:5])
