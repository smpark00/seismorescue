"""Glimmer line-identity verifier task (research/38): the shared contract between the data builder
(glimmer_verifier_data.py, local CPU) and the Modal pilot (ml_glimmer_modal.py).

One example = a film2 crop around a place where the quake line crosses or touches another line, plus two candidate
paths that start on the quake line at the left edge. One stays on the quake line; the other switches to the other
line at the crossing (the typical digitizer error). Glimmer must say which one stays: "A" or "B".

Data files (written by glimmer_verifier_data.py, read by load_split):
  out/glimmer_verifier/<split>.npz    crops  uint8   [N, H, W]     film crop, full render height (H = 406), W = WIN
                                      cands  float32 [N, 2, W]     candidate A, B: y (crop rows) per column, NaN = not drawn
                                      labels uint8   [N]           0: A is the true path, 1: B is
  out/glimmer_verifier/<split>.jsonl  one JSON object per example, same order (id, record, gain, P, type, ...)
Only synthetic film2 training renders (out/film2_train) are used: never holdouts, real scans or data_local/.
"""
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

VAL_GROUPS = ("20090103", "20110311", "20120831", "20230510")   # = SAM 3 film2 validation dates (research/31, 37)
WIN = 192                      # crop width in columns (1 px/s on film2: 192 s)
SCALE = 2                      # crop is upscaled 2x so 1-2 px strokes cover more of a 14 px vision patch
HEADER, GAP = 28, 12           # label strip height, gap between panels (px, after scaling): 1176 x 840 = 84 x 60 14-px patches
ANSWERS = ("A", "B")
SYSTEM = "You check digitized traces on scanned analog seismograms. Reasoning strength: low."
PROMPT = ("The image has three panels showing the same crop of a scanned seismogram film. Each dark wavy line is the "
          "pen trace of one hour of recording, and traces can touch and cross. The panel 'film' is the raw scan. "
          "The panels 'A' and 'B' show the same crop with one red candidate path each. Both candidates start on the "
          "same trace at the left edge and differ after a crossing. Which candidate stays on that same pen trace all "
          "the way to the right edge, instead of switching to another trace where the traces meet? "
          "Answer with one letter: A or B.")


def _panel(crop, cand=None):
    g = cv2.resize(crop, (crop.shape[1] * SCALE, crop.shape[0] * SCALE), interpolation=cv2.INTER_LINEAR)
    rgb = np.repeat(g[:, :, None], 3, axis=2)
    if cand is None:
        return rgb
    over = rgb.copy()
    x = np.arange(len(cand), dtype=np.float32)
    ok = np.isfinite(cand) & (cand >= 0) & (cand <= crop.shape[0] - 1)
    run = []                                             # draw each finite run as one polyline (NaN breaks the path)
    for i in range(len(cand) + 1):
        if i < len(cand) and ok[i]:
            run.append(i)
            continue
        if len(run) >= 2:
            pts = np.stack([(x[run] + 0.5) * SCALE - 0.5, (cand[run] + 0.5) * SCALE - 0.5], 1)
            cv2.polylines(over, [np.round(pts * 16).astype(np.int32)], False, (220, 20, 20), 2, cv2.LINE_AA, shift=4)
        run = []
    return cv2.addWeighted(over, 0.7, rgb, 0.3, 0)       # semi-transparent: the ink under the path stays visible


def compose(crop, cand_a, cand_b, ink=True):
    """RGB PIL image: [film | A | B] side by side under a white label strip. ink=False is the no-ink control: every
    panel shows a flat background at the crop's median grey instead of the film; size, red paths and labels stay"""
    if not ink:
        crop = np.full_like(crop, int(np.median(crop)))
    panels = [_panel(crop), _panel(crop, cand_a), _panel(crop, cand_b)]
    h, w = panels[0].shape[:2]
    out = np.full((HEADER + h, 3 * w + 2 * GAP, 3), 255, np.uint8)
    for i, (p, name) in enumerate(zip(panels, ("film", "A", "B"))):
        x0 = i * (w + GAP)
        out[HEADER:, x0:x0 + w] = p
        (tw, th), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.putText(out, name, (x0 + (w - tw) // 2, (HEADER + th) // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2,
                    cv2.LINE_AA)
    return Image.fromarray(out)


def messages(answer=None):
    """chat messages for the Glimmer processor; the image goes in through processor(images=[...])"""
    m = [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
    if answer is not None:
        m.append({"role": "assistant", "content": answer})   # renders "<|start|>assistant to=user<|message|>A<|eot|>"
    return m


def load_split(root, split):
    """-> list of dicts: crop, cands, label (0 = A true), meta (the jsonl row)"""
    root = Path(root)
    with np.load(root / f"{split}.npz") as z:             # each z[key] decompresses the whole member: read each once
        crops, cands, labels = z["crops"], z["cands"], z["labels"]
    rows = [json.loads(s) for s in (root / f"{split}.jsonl").read_text().splitlines() if s.strip()]
    assert len(rows) == len(labels) == len(crops) == len(cands), (split, len(rows), len(labels))
    return [{"crop": crops[i], "cands": cands[i], "label": int(labels[i]), "meta": rows[i]} for i in range(len(rows))]
