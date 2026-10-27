"""Muse Spark (contributor tier) on the synthetic helicorder: what can a VLM read, and what not?

usage: python -I exp_spark.py <scripts_dir> <synth_dir>
T1 handwritten-style label transcription
T2 structure: number of lines, minute marks per line
T3 P-wave arrival line + time estimate (GT from TauP iasp91)
T4 sample-level amplitudes at 10 positions on one line (vs GT, in units of line spacing)
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[1])
import cv2
import numpy as np
from metaapi import spark_ask

d = Path(sys.argv[2])
meta = json.loads((d / "meta.json").read_text())
S, MY, MX, PPS = meta["spacing"], meta["margin_y"], meta["margin_x"], meta["px_per_sec"]
img = cv2.imread(str(d / "heli_g0.8.png"), 0)
gt = np.load(d / "gt_g0.8.npz")
LABEL = "IU ANMO  LP-Z  11 MAR 2011  START 05:40 GMT  15 MIN/LINE"
lab = img.copy()
cv2.putText(lab, LABEL, (60, 38), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.0, 40, 2, cv2.LINE_AA)
cv2.imwrite(str(d / "spark_input_labeled.png"), lab)
rgb = cv2.cvtColor(lab, cv2.COLOR_GRAY2RGB)
res = {}


def ask(name, images, q):
    r = spark_ask(images, q)
    res[name] = {"answer": r["text"], "usage": r["usage"], "secs": round(r["secs"], 1)}
    print(f"\n### {name} ({res[name]['secs']}s, usage={r['usage'] and {k: r['usage'].get(k) for k in ('input_tokens','output_tokens')}})\n{r['text'][:1500]}")
    return r["text"]


# T1 label
t1 = ask("T1_label", [rgb[:60, :1600]], "Transcribe the handwritten label in this image exactly, character for character. Output only the transcription.")
norm = lambda s: re.sub(r"\s+", " ", s.strip().upper())
a, b = norm(t1), norm(LABEL)
import difflib
res["T1_label"]["char_similarity"] = round(difflib.SequenceMatcher(None, a, b).ratio(), 3)

# T2 structure
ask("T2_structure", [rgb],
    "This is a scanned analog seismogram (helicorder). (a) How many separate horizontal trace lines are there? "
    "(b) Small step-like offsets on each trace are minute time marks. How many minute marks are on one full line? "
    "Answer as JSON: {\"lines\": int, \"minute_marks_per_line\": int}.")

# T3 P arrival
ask("T3_p_arrival", [rgb],
    "This analog seismogram starts at 05:40:00 GMT at the left end of the TOP line; each line is 15 minutes, read left to right, "
    "lines top to bottom; minute marks are the small step offsets. A large distant earthquake is recorded. "
    "Which line (1 = top) shows the FIRST arrival (P-wave onset), and what is its arrival time to the nearest 10 seconds? "
    "Answer as JSON: {\"line\": int, \"time_gmt\": \"HH:MM:SS\", \"confidence\": \"low|medium|high\", \"reason\": str}.")

# T4 sample-level amplitudes on line 5 (index 4) over a 300 s window
i = 4
c0, c1 = 1900, 3100
r0, r1 = int(MY + i * S - 1.5 * S), int(MY + i * S + 1.5 * S)
crop = lab[r0:r1, c0:c1]
cv2.imwrite(str(d / "spark_input_line5.png"), crop)
xs = np.linspace(60, (c1 - c0) - 60, 10).astype(int)
gt_dev = [(MY + i * S - gt["pen"][i][c0 + x]) / S for x in xs]  # +up, in line-spacing units
crop_rgb = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
q = ("This crop shows ONE seismogram trace whose zero baseline is the horizontal centre row of the image (row "
     f"{(r1 - r0) // 2} of {r1 - r0}). Image width is {c1 - c0} px. For x = {list(map(int, xs))} px, estimate the trace's vertical "
     f"deflection from the baseline in units of {S} px (positive = above baseline). Answer only as JSON list of 10 numbers.")
t4 = ask("T4_amplitudes", [crop_rgb], q)
try:
    vals = json.loads(re.search(r"\[.*?\]", t4, re.S).group(0))
    vals = np.array(vals[:10], float)
    g = np.array(gt_dev, float)
    res["T4_amplitudes"]["gt"] = [round(v, 2) for v in g]
    res["T4_amplitudes"]["pred"] = [round(float(v), 2) for v in vals]
    res["T4_amplitudes"]["mae_line_units"] = round(float(np.mean(np.abs(vals - g))), 3)
    res["T4_amplitudes"]["mae_px"] = round(float(np.mean(np.abs(vals - g)) * S), 1)
    res["T4_amplitudes"]["corr"] = round(float(np.corrcoef(vals, g)[0, 1]), 3)
    res["T4_amplitudes"]["sign_agreement"] = round(float(np.mean(np.sign(vals) == np.sign(g))), 2)
except Exception as e:
    res["T4_amplitudes"]["parse_error"] = str(e)

res["GT"] = {"label": LABEL, "lines": meta["n_lines"], "minute_marks_per_line": 15,
             "p_arrival_utc": meta["p_arrival_utc"], "p_arrival_line": meta["p_arrival_line"]}
print("\n=== SUMMARY ===")
print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk not in ("answer", "usage")} for k, v in res.items()}, indent=1))
(d / "spark_results.json").write_text(json.dumps(res, indent=1))
