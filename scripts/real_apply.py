"""Apply the frozen digitizer system (docs/RESULTS.md) to the real WWSSN film chip and assess it without ground truth.

usage: python -I real_apply.py <image> <out_dir> <dev_bench1> <dev_cache1> <dev_bench2> <dev_cache2> <syn_bench> <syn_cache>

Layout agent (classical): invert the negative, estimate the film skew from the row-profile shift of the quiet
lines across column windows and shear it out, then measure each line's baseline (the active line, which has no
sharp profile peak, is placed in the wider-than-usual gap). Then the system exactly as validated: proposers A/B,
judge v5 trained on the development records, active-line detector and proposer C with the measured baselines.
No GT exists, so paths are scored with label-free proxies (ink support, duplication of another line, centring on
the own baseline, jumps), and the same proxies are computed on synthetic held-out records for the true path and
for each method, as a reference for what correct paths look like. The scan's licence is unclear: everything
written here stays in <out_dir> (gitignored); only numbers go into the docs.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import find_peaks

sys.path.insert(0, str(Path(__file__).parent))
from active_guard import detect_active, track_active_clean
from arbiter import assemble, event_table, proposals
from cv_judges import CONDS, ctx_features, fit_lr, load_items
from track_joint import ink_mask

img_path, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
Q, ROWS, X0, X1 = 5, (650, 1060), 200, 3550


# ---------- layout agent ----------
def row_profile(im, c0, c1):
    return (im[ROWS[0]:ROWS[1], c0:c1] < 120).mean(1)


inv = 255 - cv2.imread(str(img_path), 0)
ref = row_profile(inv, 1000, 1300)
xs, shifts = [], []
for c0 in range(X0, X1 - 300, 150):
    prof = row_profile(inv, c0, c0 + 300)
    cc = [np.dot(np.roll(prof, s) - prof.mean(), ref - ref.mean()) for s in range(-30, 31)]
    xs.append(c0 + 150); shifts.append(np.arange(-30, 31)[int(np.argmax(cc))])
slope, icpt = np.polyfit(xs, shifts, 1)                        # rows to roll so the window matches the reference
H, W = inv.shape
map_x, map_y = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
map_y -= (slope * map_x + icpt).astype(np.float32)             # shear: content at row y+shift(x) moves to row y
deskew = cv2.remap(inv, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
prof = np.mean([row_profile(deskew, c, c + 300) for c in range(X0, X1 - 300, 300)], axis=0)
pk, _ = find_peaks(prof, distance=25, prominence=0.03)
rows = [int(p + ROWS[0]) for p in pk]
gaps = np.diff(rows)
S = float(np.median(gaps))
bases = []
for a, gap in zip(rows, np.r_[gaps, 0]):
    bases.append(a)
    if gap > 1.6 * S:                                          # the active line has no sharp peak: fill the gap
        bases.append(a + gap / 2)
bases = np.array(bases, float)
print(f"skew {slope * 1000:+.2f} px per 1000 px; spacing {S:.1f} px; baselines {np.round(bases).astype(int).tolist()}")

# crop to 7 lines centred on the line whose ink spreads most away from its baseline (layout-level guess, used only to
# choose the crop; the system's own detector decides the earthquake line)
dark = deskew[:, X0:X1] < 120
spread = []
for bb in bases:
    band = dark[int(bb - S / 2):int(bb + S / 2)]
    rr = np.nonzero(band)[0] + int(bb - S / 2)
    spread.append(float(np.mean(np.abs(rr - bb) > 4)) if rr.size else 0.0)
ka_guess = int(np.argmax(spread))
print("ink spread per baseline:", [round(v, 2) for v in spread])
sel = list(range(max(0, ka_guess - 3), min(len(bases), ka_guess + 4)))
r0, r1 = int(bases[sel[0]] - 30), int(bases[sel[-1]] + 30)
crop = deskew[r0:r1].copy()
b = bases[sel] - r0
cv2.imwrite(str(out / "deskewed_crop.png"), crop)
print(f"crop rows {r0}-{r1}, lines {len(sel)}, layout-guess active index {sel.index(ka_guess)}")

# ---------- the frozen system ----------
y0 = b.copy()                                                  # start positions = layout baselines at X0
A, B, ink = proposals(crop, y0, X0, X1, S)
rows_ev = event_table(A, B, ink, X0, X1, 6.0)
dev = load_items(Path(sys.argv[3]), Path(sys.argv[4])) + load_items(Path(sys.argv[5]), Path(sys.argv[6]))
tr = [r for it in dev for r in it["rows"] if r["label"] >= 0]
model = fit_lr(np.array([r["x"] for r in tr]), np.array([r["label"] for r in tr], float), np.array([r["stake"] for r in tr], float))
item = {"A": A, "B": B, "S": S, "x0": X0, "x1": X1}
p = model(np.array([ctx_features(item, r) for r in rows_ev])) if rows_ev else np.array([])
judge = assemble(A, B, rows_ev, ["A" if v > 0.5 else "B" for v in p])
k, nq = detect_active(B, ink, 0, S, X0, X1, bases=b)
system = judge.copy()
C = None
if k is not None and nq >= Q:
    C = track_active_clean(B, ink, b[k], 0, S, X0, X1, bases=b)
    system[k, X0:X1] = C[X0:X1]
print(f"events {len(rows_ev)}; detector -> line {k} (layout guess {sel.index(ka_guess)}), quiet lines {nq}/{len(b)}, regime fired {C is not None}")


# ---------- label-free proxies ----------
def proxies(path, k, others, ink, base, S, x0, x1):
    y = path[k, x0:x1]
    cols = np.arange(x0, x1)
    yi = np.clip(np.round(np.nan_to_num(y, nan=base)).astype(int), 1, ink.shape[0] - 2)
    on_ink = float(np.mean(ink[yi - 1, cols] | ink[yi, cols] | ink[yi + 1, cols]))
    oth = np.delete(others, k, 0)[:, x0:x1]
    dup = float(np.mean(np.min(np.abs(oth - y[None]), axis=0) < 2.0))
    rm = median_filter(np.nan_to_num(y - base), size=401, mode="nearest")
    centred = float(np.mean(np.abs(rm) < S / 4))
    jumps = float(np.sum(np.abs(np.diff(y)) > S / 2) / (x1 - x0) * 1000)
    return {"on_ink": round(on_ink, 3), "dup": round(dup, 3), "centred": round(centred, 3), "jumps_per_1000": round(jumps, 2)}


real = {}
for name, P in (("A", A), ("B", B), ("judge", judge), ("system", system)):
    if k is not None:
        real[name] = proxies(P, k, system if name == "system" else P, ink, b[k], S, X0, X1)
print("\nreal film chip, detected earthquake line:")
for name, v in real.items():
    print(f"  {name:7s} {v}")

# reference: the same proxies on synthetic held-out active records (true path vs methods), detector-chosen line
syn = load_items(Path(sys.argv[7]), Path(sys.argv[8]))
ref = {m: [] for m in ("truth", "A", "B", "judge", "system")}
for it in syn:
    if it["mode"] != "active":
        continue
    meta = json.loads((Path(sys.argv[7]) / it["record"] / it["mode"] / "meta.json").read_text())
    ink_s = np.load(Path(sys.argv[8]) / f"{it['record']}_{it['mode']}_g{it['gain']:g}.npz")["ink"]
    pp = model(np.array([r["x"] for r in it["rows"]])) if it["rows"] else np.array([])
    jd = assemble(it["A"], it["B"], it["rows"], ["A" if v > 0.5 else "B" for v in pp])
    kk, nqq = detect_active(it["B"], ink_s, meta["margin_y"], it["S"], it["x0"], it["x1"])
    if kk is None:
        continue
    sy = jd.copy()
    if nqq >= Q:
        sy[kk, it["x0"]:it["x1"]] = track_active_clean(it["B"], ink_s, it["pen"][kk, it["x0"]], meta["margin_y"], it["S"], it["x0"], it["x1"])[it["x0"]:it["x1"]]
    base = meta["margin_y"] + kk * it["S"]
    for name, P in (("truth", it["pen"]), ("A", it["A"]), ("B", it["B"]), ("judge", jd), ("system", sy)):
        ref[name].append(proxies(P, kk, sy if name in ("system", "truth") else P, ink_s, base, it["S"], it["x0"], it["x1"]))
print("\nsynthetic held-out active records (n=%d), detector-chosen line, mean proxies:" % len(ref["truth"]))
for name, v in ref.items():
    print(f"  {name:7s} " + "  ".join(f"{key} {np.mean([x[key] for x in v]):.3f}" for key in v[0]))

# overlays (local only)
vis = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
for j in range(system.shape[0]):
    col = (0, 0, 255) if j == k else (200, 120, 0)
    pts = np.array([(x, system[j, x]) for x in range(X0, X1) if not np.isnan(system[j, x])], np.int32)
    cv2.polylines(vis, [pts], False, col, 1, cv2.LINE_AA)
for i, c0 in enumerate(range(X0, X1, 1100)):
    cv2.imwrite(str(out / f"overlay_system_{i}.png"), vis[:, c0:c0 + 1100])
if k is not None:
    vis2 = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    for P, col in ((A, (0, 160, 0)), (system, (0, 0, 255))):
        pts = np.array([(x, P[k, x]) for x in range(X0, X1) if not np.isnan(P[k, x])], np.int32)
        cv2.polylines(vis2, [pts], False, col, 1, cv2.LINE_AA)
    for i, c0 in enumerate(range(X0, X1, 1100)):
        cv2.imwrite(str(out / f"overlay_active_A_vs_system_{i}.png"), vis2[:, c0:c0 + 1100])
(out / "real_apply_results.json").write_text(json.dumps({"skew_px_per_1000": slope * 1000, "spacing": S, "bases": b.tolist(),
                                                          "detected": k, "n_quiet": nq, "real": real,
                                                          "synthetic_reference": {m: {key: float(np.mean([x[key] for x in v])) for key in v[0]} for m, v in ref.items()}}, indent=1))
