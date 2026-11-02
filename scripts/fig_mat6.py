"""Draw docs/img/mat_line6.png: MAT line 6, film crop, then SAM 3 accepted samples vs the digital record.

Reads only. Needs the digital record HG.MAT (see data/README.md, it is not shipped) and the stored
real-film evaluation under out/. Usage, from the repository root: python scripts/fig_mat6.py out/mat_line6.png
"""
import os, sys
os.environ.setdefault("SEISMO_FLAT_RULE", "fb")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, "scripts")
import numpy as np, cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import flat_baseline as flb
from pathlib import Path

f = flb.real_frame("MAT", "real_{}_paths.npz")
rpe, k = f["rpe"], 6
idx, y, m = flb.line_samples(f, k)
ref = f["ref"]
dev = -(y - f["bases"][k]) * f["al"]["sign"]
db, rb = rpe.bandpass(dev), rpe.bandpass(ref[idx])
r = rpe.pearson(db[m], rb[m])
print("r accepted", round(float(r), 3), "n", idx.size, "acc", round(float(m.mean()), 3), "a_glob", f["a_glob"], "sign", f["al"]["sign"])
S = f["S"]
t_h = (f["t_ref"][idx] + 20 * 60) / 3600.0        # t_ref is seconds after the film's ON time, 00:20:00
film = np.where(m, db / S, np.nan)
digi = f["a_glob"] * rb / S
BLUE, ORANGE, INK, MUTED, GRID, SURF = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
plt.rcParams.update({"font.size": 15, "axes.edgecolor": GRID, "text.color": INK, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})

# film crop in the evaluation frame
img = cv2.imread(str(flb.ROOT / "out/real_paired/MAT/MAT_rescaled.png"), 0)
x0, x1, base = f["x0"], f["x1"], float(f["bases"][k])
P = f["P1"][k]
pad = int(2.6 * S)
top, bot = int(base - pad), int(base + pad)
crop = img[top:bot, x0:x1]

fig, (a0, a1) = plt.subplots(2, 1, figsize=(18.4, 9.2), dpi=100, gridspec_kw={"height_ratios": [1, 2.1], "hspace": 0.12}, facecolor=SURF)
a0.imshow(crop, cmap="gray", aspect="auto", extent=[x0, x1, bot, top], vmin=0, vmax=255)
xs = np.arange(x0, x1)
acc = f["acc"][k]
pa = np.where(acc, P[x0:x1], np.nan)
pf = np.where(~acc, P[x0:x1], np.nan)
a0.plot(xs, pa, color="#1baf7a", lw=1.1)
a0.plot(xs, pf, color="#d03b3b", lw=1.1)
a0.set_ylim(bot, top); a0.set_xlim(x0, x1); a0.set_xticks([]); a0.set_yticks([])
for sp in a0.spines.values():
    sp.set_visible(False)
a0.text(0.0, 1.03, "The film, line 6. Green: the two SAM 3 passes agree. Red: they disagree", transform=a0.transAxes, color=MUTED, fontsize=15)

a1.plot(t_h, digi, color=ORANGE, lw=2, solid_capstyle="round", label="Digital record")
a1.plot(t_h, film, color=BLUE, lw=2, solid_capstyle="round", label="Film, SAM 3, accepted samples")
a1.grid(axis="y", color=GRID, lw=1); a1.set_axisbelow(True)
for s in ("top", "right", "left"):
    a1.spines[s].set_visible(False)
a1.tick_params(length=0)
a1.set_facecolor(SURF)
a1.set_ylabel("Line spacings")
tk = np.arange(np.ceil(t_h[0] * 6) / 6, t_h[-1], 1 / 6)
a1.set_xticks(tk); a1.set_xticklabels([f"{int(t):02d}:{int(round((t % 1) * 60)) % 60:02d}" for t in tk])
a1.set_xlim(t_h[0], t_h[-1])
a1.set_xlabel("UTC, 1973-12-28")
p_h = 5 + 41 / 60 + 35 / 3600
a1.axvline(p_h, color=MUTED, lw=1, ls=(0, (2, 3)))
yl = a1.get_ylim()
a1.text(p_h, yl[1], "  P arrival (catalogue)", color=MUTED, va="top", fontsize=14)
a1.legend(handles=[Line2D([], [], color=BLUE, lw=3, label="Film, SAM 3, accepted samples"), Line2D([], [], color=ORANGE, lw=3, label="Digital record")], loc="upper right", frameon=False, ncol=2)
fig.suptitle("A smaller quake on real film: M6.4, Matsushiro, 1973", x=0.125, y=0.975, ha="left", fontsize=24, fontweight="bold")
fig.text(0.125, 0.905, f"Correlation {r:+.2f} on accepted samples, 1/400–0.08 Hz. One amplitude fit for the whole record. Line picked after the run; development chip.", fontsize=16, color=MUTED)
fig.text(0.125, 0.012, "Digital: HG.MAT LHZ, doi:10.7914/SN/HG, converted to the WWSSN long-period response. Data were accessed from the NSF NGF data archive operated by EarthScope Consortium\n(NSF award 2435260). Film: USGS WWSSN film-chip scan, doi:10.5066/P13PHZMA, CC0.", fontsize=11.5, color=MUTED)
fig.subplots_adjust(top=0.85, bottom=0.15, left=0.125, right=0.975)
out = Path(sys.argv[1]); out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, facecolor=SURF)
print("saved", out)
