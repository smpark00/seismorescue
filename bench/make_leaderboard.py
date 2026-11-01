"""Write LEADERBOARD.md from the result files in bench/results/<split>/*.json (made by score.py) and results/both_directions/real_results.json.

usage: python -I bench/make_leaderboard.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
NOTES = {"sam3_both_directions": "SAM 3 fine-tuned by us, traced from both ends, accepted where the passes agree (τ 0.25 S, at least 20 columns)",
         "sam3_forward": "The same SAM 3, left to right only, every column accepted",
         "flat_line": "No skill: the baseline as the answer",
         "classical_tracker": "Our classical tracker, no learning, started at the true pen position. Built for an earlier film geometry and not tuned again",
         "sam3_plus_offline_network": "The earlier rule, SAM 3 checked by a small network of ours, with the SAM 3 paths started at the true pen position. Retired: every model that runs in our product must be a Meta model"}
ORDER = ["sam3_both_directions", "sam3_plus_offline_network", "sam3_forward", "classical_tracker", "flat_line"]
SPLITS = {"holdout7": "**Main table.** 10 earthquakes (M7.5–8.8), 95 images. It was our test split: rule and code were frozen before it ran, once. Its truth is public from now on.",
          "holdout6": "10 other earthquakes, 90 images. An earlier test split, so the both-directions rows here are a comparison, not a test.",
          "holdout4": "Development set: 10 other earthquakes, 100 images. The rule was chosen here, so these rows are development numbers."}


def fmt(v, d=1):
    return "–" if v is None or v != v else f"{v:.{d}f}"


def interval(ci):
    return f"[{ci[0]:.1f}, {ci[1]:.1f}]" if ci and None not in ci else ""


def table(split):
    rows = {}
    for f in sorted((HERE / "results" / split).glob("*.json")):
        rows[f.stem] = json.loads(f.read_text())
    lines = ["| Baseline | Earthquake line r | Accepted, earthquake line | Right, accepted | Flat line, same columns | Pen moved: right, all columns | Pen moved: right, accepted | Accepted, all lines | Right, accepted | Flat line, same columns |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name in [n for n in ORDER if n in rows] + [n for n in rows if n not in ORDER]:
        r = rows[name]
        q, w, ci = r["with_marks"]["quake"]["rates"], r["with_marks"]["all"]["rates"], r["with_marks"]["quake"]["ci95"]
        lines.append(f"| `{name}` | {r['r_quake_line']:.3f} | {fmt(q['accepted_share'])}% | {fmt(q['accepted_right'])}% {interval(ci['accepted_right'])} | {fmt(q['accepted_flat_line'])}% | "
                     f"{fmt(q['moved_right'])}% | {fmt(q['moved_accepted_right'])}% | {fmt(w['accepted_share'])}% | {fmt(w['accepted_right'])}% | {fmt(w['accepted_flat_line'])}% |")
    return "\n".join(lines), [f"- `{n}`: {NOTES[n]}" for n in ORDER if n in rows]


def real_table():
    d = json.loads((REPO / "results" / "both_directions" / "real_results.json").read_text())["scans"]
    fl = json.loads((REPO / "results" / "flat" / "flat_real.json").read_text())
    out = ["| Chip, lines | Accepted share | r, accepted | r, flagged | Within 0.1 S, accepted: SAM 3 | Flat line |", "|---|---:|---:|---:|---:|---:|"]
    names = {"quake": "earthquake line", "neighbour": "lines next to it", "other_active": "other active lines", "quiet": "quiet lines"}
    for chip in ("MAT", "KIP", "ALQ"):
        for cat, label in names.items():
            f = fl[chip][cat]
            a, b = f.get("accepted") or {}, f.get("disputed") or {}
            ra, rb = a.get("sam3_r"), b.get("sam3_r")
            ina, fla = a.get("sam3_in_0.1S"), a.get("flat_in_0.1S")
            out.append(f"| {chip}, {label} | {100 * f['accept_frac']:.1f}% | {'–' if ra is None else f'{ra:+.2f}'} | {'–' if rb is None else f'{rb:+.2f}'} | "
                       f"{'–' if ina is None else f'{100 * ina:.0f}%'} | {'–' if fla is None else f'{100 * fla:.0f}%'} |")
    return "\n".join(out)


def main():
    doc = ["# SeismoBench: tracing lines on seismogram film", "",
           "A benchmark for one task: follow every pen line on a scanned analog seismogram, including where the lines cross. We know of no shared benchmark for this task (two searches, not exhaustive).", "",
           "**Read this first.**", "",
           "- **The truth of every split is public, so scores rest on trust.**",
           "- **There are no outside participants. Every row is one of our own baselines.**",
           "- The film is synthetic: real broadband records rendered through our calibrated film renderer, so the true pen position is exact. A high score here does not mean the same score on real film, which is why the real chips have their own table below.", "",
           "How to score your own tracer: [bench/README.md](bench/README.md). Definitions are at the top of [bench/score.py](bench/score.py). Rows are in a fixed order, not a ranking. \"Earthquake line r\" scores the whole line and ignores `accept`, so it is the same for the forward and the both-directions rows; `r_accepted` is in the result files. S is one line spacing (38 px). \"Right\" means within S/4. A flat line gets 0% where the pen moved by definition.", ""]
    for split, note in SPLITS.items():
        if not (HERE / "results" / split).exists():
            continue
        t, notes = table(split)
        doc += [f"## {split}", "", note, "", t, ""] + notes + [""]
    doc += ["## Real film (no ranking)", "",
            "Three chips with a digital record at the same site, frozen both-directions rule, against the digital record. Far too few chips to rank anything. Scores come after a low-pass filter, one global amplitude fit and a per-line offset and trend fit. Time alignment and sign were fitted to the same digital record with a classical tracker. A path on the wrong line can still score well in the 0.1 S column. KIP and ALQ images and the digital records are not in this repository, so this table cannot be recomputed from it alone. ALQ is by line type only. Accepted and flagged r are means over lines.", "",
            real_table(), "",
            "Source: [results/both_directions/real_results.json](results/both_directions/real_results.json), [results/flat/flat_real.json](results/flat/flat_real.json).", ""]
    (REPO / "LEADERBOARD.md").write_text("\n".join(doc))
    print("wrote LEADERBOARD.md")


if __name__ == "__main__":
    main()
