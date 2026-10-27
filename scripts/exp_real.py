"""Real WWSSN film-chip scan (no ground truth): SAM 3.1 instances on quiet vs earthquake crops,
original (white-on-black negative) vs inverted; classical tracker overlay; Spark reading of metadata.

usage: python -I exp_real.py <scripts_dir> <image_path> <out_dir>
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[1])
import cv2
import numpy as np
from metaapi import sam_image, spark_ask

img = cv2.imread(sys.argv[2], 0)
out = Path(sys.argv[3]); out.mkdir(parents=True, exist_ok=True)
crops = {"quiet": (450, 750, 1500, 2700), "quake": (760, 960, 150, 1350)}
pal = [(0, 0, 255), (0, 170, 0), (255, 0, 0), (0, 140, 255), (200, 0, 200), (0, 200, 200), (120, 60, 0), (60, 0, 120), (0, 90, 200), (200, 120, 0)]
res = {}
for name, (r0, r1, c0, c1) in crops.items():
    crop = img[r0:r1, c0:c1]
    for variant, im in [("negative", crop), ("inverted", 255 - crop)]:
        for prompt in ["wavy line", "line"]:
            r = sam_image(cv2.cvtColor(im, cv2.COLOR_GRAY2RGB), prompt)
            objs = [o for o in r["objects"] if o["mask"] is not None]
            heights = [int(o["box"][3] - o["box"][1]) for o in objs if o["box"]]
            res[f"{name}/{variant}/{prompt}"] = {"n_objects": len(objs),
                                                "conf": [round(o["conf"], 2) for o in objs if o["conf"] is not None][:12],
                                                "box_heights_px": heights[:12]}
            if prompt == "wavy line":
                vis = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
                for k, o in enumerate(objs):
                    m = o["mask"]
                    vis[m] = (0.3 * vis[m] + 0.7 * np.array(pal[k % len(pal)])).astype(np.uint8)
                cv2.imwrite(str(out / f"real_sam_{name}_{variant}.png"), vis)
            print(name, variant, prompt, json.dumps(res[f"{name}/{variant}/{prompt}"]))

# Spark: metadata + structure on a downscaled full image (public archival scan; contributor tier)
small = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2), interpolation=cv2.INTER_AREA)
top = img[0:200, 0:2400]
q1 = ("This is the header strip of a scanned analog seismogram film chip. Transcribe exactly (a) the large digit "
      "groups shown under the barcode-like marks, left to right, and (b) the printed institution label text. "
      "Answer as JSON {\"digits\": str, \"label\": str}.")
q2 = ("This is a scanned WWSSN analog seismogram film chip (negative: traces are white). Answer as JSON with keys: "
      "\"n_trace_lines\" (int, count of horizontal trace lines), \"earthquake_lines\" (list of line numbers, 1 = top, that "
      "contain clear earthquake signal), \"calibration_pulses\" (short description of where they are), "
      "\"digitization_hazards\" (list of concrete problems an automatic digitizer would face on this image).")
for key, ims, q in [("spark_header", [cv2.cvtColor(top, cv2.COLOR_GRAY2RGB)], q1),
                    ("spark_structure", [cv2.cvtColor(small, cv2.COLOR_GRAY2RGB)], q2)]:
    r = spark_ask(ims, q)
    res[key] = {"answer": r["text"], "secs": round(r["secs"], 1),
                "tokens": r["usage"] and {k: r["usage"].get(k) for k in ("input_tokens", "output_tokens")}}
    print(f"\n### {key} ({res[key]['secs']}s, {res[key]['tokens']})\n{r['text'][:1800]}")
(out / "real_results.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
