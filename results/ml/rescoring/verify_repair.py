"""Independent re-computation of docs/RESULTS.md holdout3 quake-line numbers for B, flat and the GT oracle bound."""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[3]   # repository root
sys.path.insert(0, str(ROOT / "scripts"))
from track_joint import score
bench, cache = ROOT / "out/film_holdout3", ROOT / "out/cache_film_holdout3"
res = {}
for rec in sorted(p for p in bench.iterdir() if p.is_dir()):
    meta = json.loads((rec / "film/meta.json").read_text())
    S, x0, q = float(meta["spacing"]), meta["margin_x"], meta["active_line"]
    for g in (1, 2, 4, 6, 8):
        gt = np.load(rec / f"film/gt_g{g}.npz"); B = np.load(cache / f"{rec.name}_film_g{g}.npz")["B"]
        pen, sig = gt["pen"], gt["sig"]; x1 = pen.shape[1] - x0
        n = pen.shape[0]; base = meta["margin_y"] + S * np.arange(n)
        cols = slice(x0, x1)
        ok = np.isfinite(pen[q, cols]) & np.isfinite(sig[q, cols])
        tsig = (base[q] - sig[q, cols])[ok] / S
        bq = np.nan_to_num((base[q] - B[q, cols]) / S)[ok]
        rB = np.corrcoef(bq, tsig)[0, 1]
        failB = score(B, pen, [q], x0, x1, S)[0]
        failF = np.mean(np.abs(base[q] - pen[q, cols][ok]) > S / 2) * 100
        # oracle: nearest B line (shifted into line q's frame) within S/2 of the true pen, else baseline
        posB = (base[:, None] - B[:, cols]) / S - (np.arange(n) - q)[:, None]   # each B line rel. to line q baseline
        tp = (base[q] - pen[q, cols]) / S
        d = np.abs(np.nan_to_num(posB, nan=1e9) - tp[None])
        j = d.argmin(0); dm = d[j, np.arange(d.shape[1])]
        orc = np.where(dm <= 0.5, np.nan_to_num(posB, nan=0)[j, np.arange(d.shape[1])], 0.0)[ok]
        rO = np.corrcoef(orc, tsig)[0, 1]
        res.setdefault(g, []).append((rB, failB, failF, rO))
s = json.loads((ROOT / "out/ml/repair/holdout3_results.json").read_text())["summary"]
allv = []
for g, v in res.items():
    v = np.array(v); allv.append(v)
    m = s[str(g)]["quake"]
    print(f"g{g} n={len(v)} mine: rB {v[:,0].mean():+.3f} failB {v[:,1].mean():5.1f} failFlat {v[:,2].mean():5.1f} rOracle {v[:,3].mean():+.3f} | "
          f"agent: rB {m['B']['r']:+.3f} failB {m['B']['fail']:5.1f} failFlat {m['flat']['fail']:5.1f} rOracle {m['oracle']['r']:+.3f}")
v = np.concatenate(allv); m = s["all"]["quake"]
print(f"all n={len(v)} mine: rB {v[:,0].mean():+.3f} failB {v[:,1].mean():5.1f} failFlat {v[:,2].mean():5.1f} rOracle {v[:,3].mean():+.3f} | "
      f"agent: rB {m['B']['r']:+.3f} failB {m['B']['fail']:5.1f} failFlat {m['flat']['fail']:5.1f} rOracle {m['oracle']['r']:+.3f}")
print("agent models (all):", {k: (m[k]['r'], m[k]['fail']) for k in m})
