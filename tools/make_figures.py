"""Draw the README result charts (light and dark variants) from results/ml/summary.json.

usage: python tools/collect_summary.py && python tools/make_figures.py
Writes docs/img/chart_quake_r_{light,dark}.png and docs/img/chart_amplitude_{light,dark}.png.
Palette: categorical slots 1-2 (blue, orange) for the two learned models, a neutral gray for baselines;
validated for colour-vision deficiency in both modes.
"""
import json
from decimal import ROUND_HALF_EVEN, Decimal
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
IMG = ROOT / "docs" / "img"
THEMES = {
    "light": {"surface": "#fcfcfb", "text": "#0b0b0b", "text2": "#52514e", "muted": "#898781", "grid": "#e1e0d9",
              "axis": "#c3c2b7", "ra": "#2a78d6", "sam": "#eb6834", "glim": "#1baf7a", "band": "#ecebe6"},
    "dark": {"surface": "#1a1a19", "text": "#f0efec", "text2": "#c3c2b7", "muted": "#898781", "grid": "#2c2c2a",
             "axis": "#383835", "ra": "#3987e5", "sam": "#d95926", "glim": "#199e70", "band": "#262624"},
}
METHODS = [("row_anchor", "Row-anchor + Viterbi (ours)", "ra"), ("sam21", "SAM 2.1 fine-tuned (Meta)", "sam"),
           ("repair_seq", "Pointer Transformer (repair)", "muted"), ("repair_spec", "Spectrogram U-Net (repair)", "muted"),
           ("B", "Classical joint tracker (B)", "muted")]


def fmt2(v):
    """two decimals, rounding half to even on the stored 3-decimal value (0.025 -> +0.02, as in the README tables)"""
    q = Decimal(str(v)).quantize(Decimal("0.01"), rounding=ROUND_HALF_EVEN)
    return f"{q:+.2f}" if q != 0 else ("-0.00" if v < 0 else "+0.00")


def style(ax, t):
    ax.set_facecolor(t["surface"])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(t["axis"])
    ax.tick_params(colors=t["text2"], labelsize=9.5, length=0)
    ax.grid(axis="x", color=t["grid"], lw=0.8)
    ax.set_axisbelow(True)


def chart_quake_r(S, mode):
    t = THEMES[mode]
    syn, real = S["synthetic_holdout_quake_line"], S["real_paired"]
    panels = [("Synthetic holdout\n95 images, 10 unseen earthquakes", {m: syn[m]["all"]["r"] for m, _, _ in METHODS}, None,
               syn["repair_oracle"]["all"]["r"]),
              ("Real scan KIP (1975 M7.6)\nquake line vs co-located digital", {m: real["KIP"]["quake"][m]["r_band"] for m, _, _ in METHODS}, 0.066, None),
              ("Real scan MAT (1973 M7.5)\nquake line vs co-located digital", {m: real["MAT"]["quake"][m]["r_band"] for m, _, _ in METHODS}, 0.061, None)]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9), sharey=True, facecolor=t["surface"])
    ys = list(range(len(METHODS)))[::-1]
    for i, (ax, (title, vals, chance, oracle)) in enumerate(zip(axes, panels)):
        style(ax, t)
        if chance:
            ax.axvspan(-chance, chance, color=t["band"], zorder=0, lw=0)
            ax.text(0, -0.95, "chance level", ha="center", va="center", fontsize=8.5, color=t["text2"])
        if oracle:
            ax.axvline(oracle, color=t["text2"], lw=1, ls=(0, (3, 3)))
            ax.text(oracle - 0.02, -0.95, "upper bound for\nrepairing B (oracle)", ha="right", va="center", fontsize=8.5, color=t["text2"])
        for y, (m, label, c) in zip(ys, METHODS):
            v = vals[m]
            ax.barh(y, v, height=0.62, color=t[c], lw=0)
            ax.text(v + (0.02 if v >= 0 else -0.02), y, fmt2(v), va="center", ha="left" if v >= 0 else "right",
                    fontsize=9.5, color=t["text"], fontweight="bold" if c != "muted" else "normal")
        ax.axvline(0, color=t["axis"], lw=1)
        ax.set_xlim(-0.25, 1.05)
        ax.set_ylim(-1.4, len(METHODS) - 0.4)
        ax.set_title(title, fontsize=10.5, color=t["text"], loc="left", pad=8)
        ax.set_xlabel("correlation r with the true quake-line waveform", fontsize=9, color=t["text2"])
        if i == 0:
            ax.set_yticks(ys, [label for _, label, _ in METHODS])
            for tl, (_, _, c) in zip(ax.get_yticklabels(), METHODS):
                tl.set_color(t["text"] if c != "muted" else t["text2"])
                tl.set_fontsize(10)
    fig.suptitle("Recovering the earthquake line: learned extraction vs classical tracking", fontsize=13, color=t["text"],
                 x=0.01, ha="left", y=1.02, fontweight="bold")
    fig.tight_layout()
    fig.savefig(IMG / f"chart_quake_r_{mode}.png", dpi=150, bbox_inches="tight", facecolor=t["surface"])
    plt.close(fig)


def chart_amplitude(S, mode):
    t = THEMES[mode]
    syn = S["synthetic_holdout_quake_line"]
    gains = [1, 2, 4, 6, 8]
    series = [("row_anchor", "Row-anchor + Viterbi", "ra", "o"), ("sam21", "SAM 2.1 fine-tuned", "sam", "s"),
              ("B", "Classical joint tracker (B)", "muted", "D")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.9), facecolor=t["surface"])
    for ax, key, ylab, lim in ((axes[0], "r", "quake-line correlation r", (-0.05, 1.05)),
                               (axes[1], "fail", "quake-line failure (% of columns)", (-3, 80))):
        style(ax, t)
        ax.grid(axis="y", color=t["grid"], lw=0.8)
        ax.grid(axis="x", visible=False)
        for m, label, c, mk in series:
            v = [syn[m][str(g)][key] for g in gains]
            ax.plot(gains, v, color=t[c], lw=2, marker=mk, ms=7, mec=t["surface"], mew=1.5, label=label, zorder=3)
            ax.annotate(f"{v[-1]:.2f}" if key == "r" else f"{v[-1]:.1f}%", (gains[-1], v[-1]), xytext=(8, 0),
                        textcoords="offset points", va="center", fontsize=9.5, color=t["text"])
        ax.set_xticks(gains, [f"{g}" for g in gains])
        ax.set_xlim(0.6, 9.2)
        ax.set_ylim(*lim)
        ax.set_xlabel("earthquake amplitude (peak deflection, in line spacings)", fontsize=9, color=t["text2"])
        ax.set_title(ylab, fontsize=10.5, color=t["text"], loc="left", pad=8)
    leg = axes[0].legend(loc="center left", fontsize=9.5, frameon=False)
    for txt in leg.get_texts():
        txt.set_color(t["text"])
    fig.suptitle("Synthetic holdout by earthquake amplitude (19 records per amplitude)", fontsize=13, color=t["text"],
                 x=0.01, ha="left", y=1.02, fontweight="bold")
    fig.tight_layout()
    fig.savefig(IMG / f"chart_amplitude_{mode}.png", dpi=150, bbox_inches="tight", facecolor=t["surface"])
    plt.close(fig)


BUDGET = [("SAM 3.1 LoRA fine-tunes x3 + evaluation", "A100 80 GB, ~9 GPU-h", 27, "sam"),
          ("Muse Glimmer QLoRA: identity verifier + film-chip reader", "H100 80 GB, ~11 GPU-h", 48, "glim"),
          ("Ablations and final evaluation", "A100 80 GB, ~8 GPU-h", 24, "muted"),
          ("Row-anchor sim-to-real sweeps", "our local GPUs", 0, "muted")]


def chart_budget(mode):
    """planned use of the $100 Modal credit (estimates from Modal list prices incl. CPU and memory)"""
    t = THEMES[mode]
    fig, ax = plt.subplots(figsize=(11, 3.3), facecolor=t["surface"])
    style(ax, t)
    ys = list(range(len(BUDGET)))[::-1]
    for y, (name, gpu, cost, c) in zip(ys, BUDGET):
        ax.barh(y, cost, height=0.6, color=t[c], lw=0)
        ax.text(cost + 1, y, f"${cost}" if cost else "$0 (runs locally)", va="center", ha="left", fontsize=10.5,
                color=t["text"], fontweight="bold")
        ax.text(-1.5, y + 0.13, name, va="center", ha="right", fontsize=10, color=t["text"])
        ax.text(-1.5, y - 0.2, gpu, va="center", ha="right", fontsize=8.5, color=t["text2"])
    ax.axvline(0, color=t["axis"], lw=1)
    ax.set_xlim(0, 100)
    ax.set_yticks([])
    ax.set_xticks([0, 25, 50, 75, 100], ["$0", "$25", "$50", "$75", "$100"])
    total = sum(b[2] for b in BUDGET)
    ax.set_title(f"Planned use of a \\$100 Modal credit (about \\${total}): fine-tuning Meta's SAM 3.1 and Muse Glimmer", fontsize=12.5,
                 color=t["text"], loc="left", pad=10, fontweight="bold", x=-0.62)
    fig.tight_layout()
    fig.savefig(IMG / f"chart_modal_budget_{mode}.png", dpi=150, bbox_inches="tight", facecolor=t["surface"])
    plt.close(fig)


if __name__ == "__main__":
    IMG.mkdir(parents=True, exist_ok=True)
    S = json.loads((ROOT / "results" / "ml" / "summary.json").read_text())
    for mode in THEMES:
        chart_quake_r(S, mode)
        chart_amplitude(S, mode)
        chart_budget(mode)
    print("wrote", sorted(p.name for p in IMG.glob("chart_*.png")))
