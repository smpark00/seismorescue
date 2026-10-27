"""Spike S3: why does SAM 3.1 find nothing on the real film chip's earthquake section?

usage: python -I exp_real_sam_zoom.py <scripts_dir> <image_path> <out_dir>

The quake trace sits near row 874 between quiet neighbours at rows 838 and 911 (spacing ~37 px). exp_real.py used one
200x1200 crop of 5 lines and got 0 instances. Here: narrower bands (3 lines / active line only), shorter column
chunks, 1-3x zoom, polarity, a median filter against the fine microseism wiggle, and two other prompts.
band3 at 2x is 240x1200, the same size as the synthetic crops where SAM did find the active line.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[1])
import cv2
import numpy as np
from metaapi import sam_image

img = cv2.imread(sys.argv[2], 0)
out = Path(sys.argv[3]); out.mkdir(parents=True, exist_ok=True)
BANDS = {"band3": (815, 935), "band1": (840, 910)}
CHUNKS = {"A": (150, 750), "B": (750, 1350), "C": (450, 750)}
# (band, chunk, zoom, polarity, prompt, median-filter)
CONFIGS = [("band3", "A", 1, "inv", "wavy line", 0), ("band3", "B", 1, "inv", "wavy line", 0),
           ("band3", "A", 2, "inv", "wavy line", 0), ("band3", "B", 2, "inv", "wavy line", 0),
           ("band1", "A", 2, "inv", "wavy line", 0), ("band1", "B", 2, "inv", "wavy line", 0),
           ("band1", "C", 3, "inv", "wavy line", 0), ("band3", "C", 3, "inv", "wavy line", 0),
           ("band3", "A", 2, "neg", "wavy line", 0), ("band3", "A", 2, "inv", "wavy line", 5),
           ("band3", "C", 3, "inv", "squiggly line", 0), ("band3", "C", 3, "inv", "line", 0)]
pal = [(0, 0, 255), (0, 170, 0), (255, 0, 0), (0, 140, 255), (200, 0, 200), (0, 200, 200)]
res = []
for band, chunk, zoom, pol, prompt, med in CONFIGS:
    (r0, r1), (c0, c1) = BANDS[band], CHUNKS[chunk]
    im = img[r0:r1, c0:c1]
    im = 255 - im if pol == "inv" else im
    if med:
        im = cv2.medianBlur(im, med)
    if zoom > 1:
        im = cv2.resize(im, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_CUBIC)
    r = sam_image(cv2.cvtColor(im, cv2.COLOR_GRAY2RGB), prompt)
    objs = [o for o in r["objects"] if o["mask"] is not None]
    name = f"{band}_{chunk}_z{zoom}_{pol}_{prompt.replace(' ', '-')}" + (f"_med{med}" if med else "")
    vis = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    for k, o in enumerate(objs):
        vis[o["mask"]] = (0.3 * vis[o["mask"]] + 0.7 * np.array(pal[k % len(pal)])).astype(np.uint8)
    cv2.imwrite(str(out / f"real_zoom_{name}.png"), vis)
    row = {"name": name, "size": list(im.shape), "n_objects": len(objs),
           "conf": [round(o["conf"], 2) for o in objs if o["conf"] is not None],
           "box_heights_px": [int(o["box"][3] - o["box"][1]) for o in objs if o["box"]],
           "box_widths_px": [int(o["box"][2] - o["box"][0]) for o in objs if o["box"]]}
    res.append(row)
    print(json.dumps(row))
(out / "real_zoom_results.json").write_text(json.dumps(res, indent=1))
