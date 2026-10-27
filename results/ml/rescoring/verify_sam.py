"""Re-score the saved SAM 2.1 holdout3 paths (docs/RESULTS.md) with fail (track_joint.score) and r vs GT sig (docs/RESULTS.md
definition); compare quake fail per image with the agent's eval_holdout3.json."""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[3]   # repository root
sys.path.insert(0, str(ROOT / "scripts"))
from track_joint import score
agent = {(r["record"], int(r["gain"])): r for r in json.loads((ROOT / "out/ml/sam/eval_holdout3.json").read_text())["rows"]}
def r_of(P, pen, sig, base, k, x0, x1, S):
    a = (base[k] - P[k, x0:x1]) / S
    apen, asig = (base[k] - pen[k, x0:x1]) / S, (base[k] - sig[k, x0:x1]) / S
    ok = np.isfinite(apen) & np.isfinite(asig)
    an = np.nan_to_num(a)[ok]
    e = np.abs(np.nan_to_num(a, nan=1e9)[ok] - asig[ok])
    return (np.corrcoef(an, asig[ok])[0, 1] if an.std() > 0 else 0.0), float(np.mean(e < 0.1)), float(np.mean(e > 0.5))
rows, mism, out = [], [], []
for rec in sorted(p for p in (ROOT / "out/film_holdout3").iterdir() if p.is_dir()):
    meta = json.loads((rec / "film/meta.json").read_text())
    S, x0, q, n = float(meta["spacing"]), meta["margin_x"], meta["active_line"], meta["n_lines"]
    base = meta["margin_y"] + S * np.arange(n)
    for g in (1, 2, 4, 6, 8):
        gt = np.load(rec / f"film/gt_g{g}.npz"); pen, sig = gt["pen"], gt["sig"]; x1 = pen.shape[1] - x0
        P = np.load(ROOT / f"out/ml/sam/paths/holdout3/{rec.name}_g{g}.npz")["SAM"].astype(float)
        assert P.shape == pen.shape, (P.shape, pen.shape)
        fq, fw = score(P, pen, [q], x0, x1, S)[0], score(P, pen, range(n), x0, x1, S)[0]
        rq, inq, lq = r_of(P, pen, sig, base, q, x0, x1, S)
        rn = float(np.mean([r_of(P, pen, sig, base, k, x0, x1, S)[0] for k in (q - 1, q + 1) if 0 <= k < n]))
        rows.append((g, rq, fq, fw, inq, lq, rn))
        out.append({"name": rec.name, "gain": g, "SAM": {"r_q": rq, "fail_q": fq, "fail_w": fw, "in01_q": inq, "lost05_q": lq, "r_nb": rn}})
        a = agent[(rec.name, g)]
        if abs(fq - a["SAM_quake"]) > 0.11 or abs(fw - a["SAM"]) > 0.11:
            mism.append((rec.name, g, fq, a["SAM_quake"], fw, a["SAM"]))
R = np.array(rows)
for g in (1, 2, 4, 6, 8, None):
    m = R if g is None else R[R[:, 0] == g]
    print(f"g{g or 'all'} n={len(m)} SAM: r_q {m[:,1].mean():+.3f} fail_q {m[:,2].mean():5.1f} fail_w {m[:,3].mean():4.1f} "
          f"in01 {m[:,4].mean():.2f} lost05 {m[:,5].mean():.2f} r_nb {m[:,6].mean():+.3f} | min r_q {m[:,1].min():+.2f}")
print("mismatches vs agent (quake or whole fail):", len(mism), mism[:5])
(Path(__file__).parent / "verify_sam_holdout3.json").write_text(json.dumps(out, indent=1))
