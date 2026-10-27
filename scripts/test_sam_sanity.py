"""Sanity check: SAM 3.1 on simple shapes, to validate request + mask decoding + coordinates."""
import sys
sys.path.insert(0, sys.argv[1])  # scripts dir (explicit, since we run with -I)

import numpy as np
import cv2
from metaapi import sam_image

h, w = 300, 480
img = np.full((h, w, 3), 255, np.uint8)
gt = []
for (cx, cy, r) in [(110, 150, 60), (330, 120, 45)]:
    cv2.circle(img, (cx, cy), r, (200, 30, 30), -1)
    m = np.zeros((h, w), bool)
    cv2.circle(m.view(np.uint8), (cx, cy), r, 1, -1)
    gt.append(m)

res = sam_image(img, "red circle")
print(f"secs={res['secs']:.1f} objects={len(res['objects'])} diagnostics={res['diagnostics']}")
for o in res["objects"]:
    m = o["mask"]
    best = max((int((m & g).sum()) / int((m | g).sum()), i) for i, g in enumerate(gt)) if m is not None else (0, -1)
    print(f"id={o['id']} box={tuple(round(v) for v in o['box']) if o['box'] else None} "
          f"conf={o['conf']} mask_px={int(m.sum()) if m is not None else None} best_IoU={best[0]:.3f} (gt#{best[1]})")
