"""Minimal Meta Model API helpers for the seismogram experiments.

- SAM 3.1 image segmentation (concept / box prompts) -> per-object masks in
  full-image coordinates, decoded with the official meta_sam_parser.
- Muse Spark (contributor tier) image+text questions.

The API key is read from MODEL_API_KEY and is never printed.
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import time

import numpy as np
import requests
from PIL import Image

import meta_sam_parser as msp

BASE_URL = "https://api.meta.ai/v1/responses"
SPARK_MODEL = "muse-spark-1.3-contributor"


def _headers(stream: bool) -> dict:
    key = os.environ.get("MODEL_API_KEY")
    if not key:
        raise RuntimeError("MODEL_API_KEY is not set")
    h = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    if stream:
        h["Accept"] = "text/event-stream"
    return h


def png_data_url(img: np.ndarray | Image.Image) -> str:
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _sse_events(payload: dict) -> list[dict]:
    """POST a streaming request and return the decoded SSE event dicts."""
    events: list[dict] = []
    with requests.post(BASE_URL, headers=_headers(True), json=payload, stream=True, timeout=300) as r:
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:500]}")
        for raw in r.iter_lines(decode_unicode=True):
            if not raw or not raw.startswith("data:"):
                continue
            data = raw[len("data:"):].strip()
            if data == "[DONE]":
                break
            try:
                events.append(json.loads(data))
            except json.JSONDecodeError:
                continue
    return events


async def _parse(events: list[dict]):
    async def gen():
        for e in events:
            yield e

    parsed = msp.parse_responses_stream(gen(), msp.image_segmentation_format())
    async for _ in parsed:
        pass
    return await parsed.final_result()


def sam_image(img: np.ndarray | Image.Image, text: str, *, encoding: str = "one_bit",
              confidence: bool = True) -> dict:
    """Run SAM 3.1 on one image. `text` is a noun phrase or a box/point token prompt.

    Returns {"objects": [{"id", "box", "conf", "mask"(bool HxW full image)}], "raw", "secs"}.
    """
    if isinstance(img, np.ndarray):
        h, w = img.shape[:2]
    else:
        w, h = img.size
    payload = {
        "model": "sam-3.1",
        "input": [{
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_text", "text": text},
                {"type": "input_image", "image_url": png_data_url(img)},
            ],
        }],
        "stream": True,
        "metadata": {"include_confidence": "true" if confidence else "false",
                     "mask_encoding": encoding},
    }
    t0 = time.time()
    events = _sse_events(payload)
    result = asyncio.run(_parse(events))
    secs = time.time() - t0

    boxes: dict[str, dict] = {}
    masks: dict[str, np.ndarray] = {}
    for rec in result.records:
        if isinstance(rec, msp.SegmentationBoxRecord):
            boxes[rec.object_id] = {"box": (rec.left, rec.top, rec.right, rec.bottom),
                                    "conf": rec.confidence}
        elif isinstance(rec, msp.SegmentationMaskRecord):
            m = rec.mask
            raster = np.frombuffer(msp.decode_mask_to_raster(m), dtype=np.uint8)
            raster = raster.reshape(m.height, m.width)
            # Place the (possibly low-res) mask raster into the box bounds of the full image.
            b = rec.bounds
            full = np.zeros((h, w), dtype=bool)
            x0, y0 = int(round(b.left)), int(round(b.top))
            x1, y1 = int(round(b.right)), int(round(b.bottom))
            bw, bh = max(1, x1 - x0), max(1, y1 - y0)
            if (m.width, m.height) != (bw, bh):
                rimg = Image.fromarray(raster * 255).resize((bw, bh), Image.NEAREST)
                raster = (np.asarray(rimg) > 127).astype(np.uint8)
            ys, xs = slice(max(0, y0), min(h, y0 + bh)), slice(max(0, x0), min(w, x0 + bw))
            full[ys, xs] = raster[: ys.stop - ys.start, : xs.stop - xs.start].astype(bool)
            masks[rec.object_id] = full
    objects = []
    for oid in sorted(set(boxes) | set(masks), key=lambda s: (len(s), s)):
        objects.append({"id": oid, "box": boxes.get(oid, {}).get("box"),
                        "conf": boxes.get(oid, {}).get("conf"), "mask": masks.get(oid)})
    return {"objects": objects, "raw": result.raw_output, "secs": secs,
            "diagnostics": [d.message for d in result.diagnostics]}


def spark_ask(images: list, question: str, *, model: str = SPARK_MODEL,
              effort: str | None = None) -> dict:
    """Ask Muse Spark (contributor tier by default) about one or more images."""
    content = [{"type": "input_text", "text": question}]
    for im in images:
        content.append({"type": "input_image", "image_url": png_data_url(im)})
    payload = {"model": model,
               "input": [{"type": "message", "role": "user", "content": content}]}
    if effort:
        payload["reasoning"] = {"effort": effort}
    t0 = time.time()
    r = requests.post(BASE_URL, headers=_headers(False), json=payload, timeout=600)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:500]}")
    d = r.json()
    text = " ".join(c.get("text", "") for item in d.get("output", [])
                    for c in (item.get("content") or []) if c.get("type") in ("output_text", "text"))
    return {"text": text, "usage": d.get("usage"), "secs": time.time() - t0}
