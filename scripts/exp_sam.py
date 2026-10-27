"""SAM 3.1 on synthetic helicorder crops with ground truth.

usage: python -I exp_sam.py <scripts_dir> <synth_dir>

Tests per gain (0.8, 2, 5):
  A) concept prompts -> how many instances, per-line best IoU, merged-line masks
  B) box prompt around one target line -> does the mask stay on that line through crossings?
  C) box + positive points on unambiguous spots of the target line (+ negatives on neighbours)
IoU uses a 1-px-dilated GT trace (pen strokes are ~1.6 px wide).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[1])
import cv2
import numpy as np
import meta_sam_parser as msp
from metaapi import sam_image

d = Path(sys.argv[2])
meta = json.loads((d / "meta.json").read_text())
S, MY = meta["spacing"], meta["margin_y"]
R0, R1 = int(MY + 1.5 * S), int(MY + 6.5 * S)      # crop rows: lines 3..7 (idx 2..6)
C0, C1 = 1900, 3100                                 # 300 s window during surface waves
TARGET = 3                                          # line index 3 (4th line)
CONCEPTS = ["line", "wavy line", "curve", "seismogram trace"]
k3 = np.ones((3, 3), np.uint8)
out = []


def score(mask, gts, tgt=None):
    res = {}
    if mask is None or mask.sum() == 0:
        return {"px": 0}
    ious = [float((mask & g).sum() / (mask | g).sum()) for g in gts]
    covers = [float((mask & g).sum() / max(1, g.sum())) for g in gts]
    res["px"] = int(mask.sum())
    res["best_iou"] = round(max(ious), 3)
    res["lines_covered_gt30pct"] = int(sum(c > 0.3 for c in covers))
    if tgt is not None:
        others = np.any([g for j, g in enumerate(gts) if j != tgt], axis=0)
        res["iou_target"] = round(ious[tgt], 3)
        res["recall_target"] = round(covers[tgt], 3)
        res["leak_to_other_lines"] = round(float((mask & others & ~gts[tgt]).sum() / mask.sum()), 3)
    return res


for g in [0.8, 2.0, 5.0]:
    tag = f"g{g:g}"
    img = cv2.imread(str(d / f"heli_{tag}.png"), 0)[R0:R1, C0:C1]
    gt = np.load(d / f"gt_{tag}.npz")
    inst = gt["inst"][:, R0:R1, C0:C1]
    pen = gt["pen"][:, C0:C1] - R0
    keep = [i for i in range(len(inst)) if inst[i].sum() > 200]
    gts = [cv2.dilate(inst[i].astype(np.uint8), k3).astype(bool) for i in keep]
    rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    h, w = img.shape
    # A) concept prompts
    for c in CONCEPTS:
        r = sam_image(rgb, c)
        objs = r["objects"]
        per = [score(o["mask"], gts) for o in objs]
        union = np.any([o["mask"] for o in objs if o["mask"] is not None], axis=0) if objs else np.zeros_like(gts[0])
        line_best = [round(max([float((o["mask"] & gg).sum() / (o["mask"] | gg).sum()) for o in objs if o["mask"] is not None] or [0]), 3) for gg in gts]
        out.append({"gain": g, "test": "A_concept", "prompt": c, "n_objects": len(objs),
                    "conf": [round(o["conf"], 2) for o in objs if o["conf"] is not None][:8],
                    "per_gt_line_best_iou": line_best,
                    "masks_spanning_2plus_lines": int(sum(p.get("lines_covered_gt30pct", 0) >= 2 for p in per)),
                    "union_recall_all_ink": round(float((union & np.any(gts, 0)).sum() / np.any(gts, 0).sum()), 3)})
        if c == "line":
            vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            rng = np.random.default_rng(1)
            for o in objs:
                if o["mask"] is not None:
                    col = rng.integers(40, 255, 3).tolist()
                    vis[o["mask"]] = (0.4 * vis[o["mask"]] + 0.6 * np.array(col)).astype(np.uint8)
            cv2.imwrite(str(d / f"sam_concept_line_{tag}.png"), vis)
    # B) box prompt around the target line
    ti = keep.index(TARGET)
    yv = pen[TARGET][~np.isnan(pen[TARGET])]
    top, bot = int(max(0, yv.min() - 3)), int(min(h - 1, yv.max() + 3))
    obj = msp.PromptObject(id=1, prompts=[msp.FramePrompt(frame=0, box=msp.PromptBox(left=0, top=top, right=w - 1, bottom=bot))])
    r = sam_image(rgb, msp.build_object_prompt(size=(w, h), objects=[obj]))
    m = r["objects"][0]["mask"] if r["objects"] else None
    out.append({"gain": g, "test": "B_box_target_line", **score(m, gts, ti)})
    if m is not None:
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        vis[m] = (0.3 * vis[m] + 0.7 * np.array([0, 0, 255])).astype(np.uint8)
        cv2.imwrite(str(d / f"sam_box_target_{tag}.png"), vis)
    # C) box + positive points on unambiguous spots + negatives on neighbour lines
    others = [pen[j] for j in keep if j != TARGET]
    xs = np.linspace(20, w - 20, 40).astype(int)
    pos, neg = [], []
    for x in xs:
        y = pen[TARGET][x]
        if np.isnan(y):
            continue
        dmin = min(abs(o[x] - y) for o in others if not np.isnan(o[x]))
        if dmin > 8 and len(pos) < 8:
            pos.append(msp.PromptPoint(x=int(x), y=int(round(y)), label="positive"))
    for j in keep:
        if j == TARGET:
            continue
        for x in xs[::10]:
            y = pen[j][x]
            if not np.isnan(y) and abs(y - pen[TARGET][x]) > 8 and 0 <= y < h and len(neg) < 6:
                neg.append(msp.PromptPoint(x=int(x), y=int(round(y)), label="negative"))
    obj = msp.PromptObject(id=1, prompts=[msp.FramePrompt(frame=0, box=msp.PromptBox(left=0, top=top, right=w - 1, bottom=bot), points=pos + neg)])
    r = sam_image(rgb, msp.build_object_prompt(size=(w, h), objects=[obj]))
    m = r["objects"][0]["mask"] if r["objects"] else None
    out.append({"gain": g, "test": "C_box+points", "n_pos": len(pos), "n_neg": len(neg), **score(m, gts, ti)})
    if m is not None:
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        vis[m] = (0.3 * vis[m] + 0.7 * np.array([0, 0, 255])).astype(np.uint8)
        cv2.imwrite(str(d / f"sam_points_target_{tag}.png"), vis)

for o in out:
    print(json.dumps(o))
(d / "sam_results.json").write_text(json.dumps(out, indent=1))
