"""E9 (research/48): Muse Glimmer (open weights, served locally by llama.cpp) reads the same public film chips with the
same question and images as Spark did in E4 (research/42); scored like E4 and through the L1 border table (research/44).

usage: python -I scripts/e9_glimmer_labels.py dev | run | score   [--url http://127.0.0.1:8792]
- dev: the two E4 development chips (MAT 1973 LHZ, ALQ 1975 SHZ) -> out/e9/dev/
- run: once (start marker first), the 30 chips of data/e4_chips, one at a time -> out/e9/responses/<chip>.json, then score
- score: re-scores the saved responses -> out/e9/e9_results.json (E4's Spark scores side by side)
Nothing leaves this machine: the server is llama-server on localhost. Server settings are recorded from /props.
"""
import argparse
import base64
import json
import sys
import time
from pathlib import Path

import cv2
import requests

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from e4_spark_labels import CHIPS, DEV, QUESTION, images, parse, score_one, truth, wilson   # noqa: E402
from l1_context import CHIP_FIX, FIELDS, decode_border, got_fields, score_rows              # noqa: E402

OUT = ROOT / "out" / "e9"
SETTINGS = {"max_tokens": 4096, "temperature": 0.0, "chat_template_kwargs": {"reasoning_strength": "medium"}}  # fixed in dev


def png_url(img):
    return "data:image/png;base64," + base64.b64encode(cv2.imencode(".png", img)[1].tobytes()).decode()


def ask(url, path):
    """one question, asked again only after an HTTP error or unparsable JSON (E4's rule)"""
    rec = {"chip": path.stem}
    content = [{"type": "text", "text": QUESTION}] + [{"type": "image_url", "image_url": {"url": png_url(im)}} for im in images(path)]
    for attempt in (1, 2):
        t = time.time()
        try:
            r = requests.post(f"{url}/v1/chat/completions", json={"messages": [{"role": "user", "content": content}], **SETTINGS},
                              timeout=3600)
            r.raise_for_status()
            d = r.json()
            m = d["choices"][0]["message"]
            rec.update({"attempt": attempt, "text": m.get("content") or "", "reasoning_chars": len(m.get("reasoning_content") or ""),
                        "parsed": parse(m.get("content") or ""), "usage": d.get("usage"), "timings": d.get("timings"),
                        "secs": round(time.time() - t, 1), "error": None})
            if rec["parsed"] is not None:
                break
        except Exception as e:                                    # noqa: BLE001 (recorded, retried once)
            rec.update({"attempt": attempt, "parsed": None, "error": f"{type(e).__name__}: {str(e)[:300]}", "secs": round(time.time() - t, 1)})
    return rec


def server_props(url):
    try:
        p = requests.get(f"{url}/props", timeout=30).json()
        return {k: p.get(k) for k in ("build_info", "model_path", "n_ctx", "modalities", "chat_template_kwargs")} | {"raw_keys": sorted(p)[:40]}
    except Exception as e:                                        # noqa: BLE001
        return {"error": str(e)[:200]}


def run(url, chips, out, retry_errors=False):
    """a reply is kept and not asked again; with retry_errors (dev runs) a failed one is asked again"""
    out.mkdir(parents=True, exist_ok=True)
    for p in chips:
        f = out / f"{p.stem}.json"
        if f.exists() and not (retry_errors and json.loads(f.read_text()).get("error")):
            continue
        if not p.exists():
            print(f"[{p.stem}] not downloaded, skipped (see data/README.md)", flush=True)
            continue
        rec = ask(url, p)
        f.write_text(json.dumps(rec, indent=1))
        rd = rec.get("parsed") or {}
        print(f"[{p.stem}] {rec['secs']} s | border {rd.get('border_digits')} | station {rd.get('station')} | "
              f"{rd.get('on_date')} {rd.get('on_time')} | error {rec.get('error')}", flush=True)


def score():
    e4 = json.loads((ROOT / "out" / "e4" / "e4_results.json").read_text())
    rows = []
    for f in sorted((OUT / "responses").glob("*.json")):
        rec = json.loads(f.read_text())
        read = rec.get("parsed") or {}
        label = score_one(read, truth(rec["chip"]))               # E4's scoring of the label fields
        dec = decode_border(read["border_digits"]) if isinstance(read.get("border_digits"), str) else None
        sta = dec["station"]["code"] if dec and dec["station"] else read.get("station")
        got = got_fields(sta, dec["on_utc"] if dec else None, dec["component"] if dec else None)
        tr = {k: truth(rec["chip"])[k] for k in FIELDS}
        digits = "".join(c for c in str(read.get("border_digits") or "") if c.isdigit())
        rows.append({"chip": rec["chip"], "secs": rec.get("secs"), "error": rec.get("error"), "label": label,
                     "table": score_rows(got, tr), "table_chip": score_rows(got, tr | CHIP_FIX.get(rec["chip"], {})),
                     "border_16_digits": len(digits) == 16})
    n = len(rows)
    tal = lambda key, f: (lambda k: {"correct": k, "n": n, "wilson95": wilson(k, n)})(sum(bool(r[key][f]) for r in rows))   # noqa: E731
    res = {"what": "research/48 E9: Muse Glimmer (local llama.cpp) on the 30 E4 chips, E4 question and images", "n": n,
           "label_fields": {f: tal("label", f) for f in ("station", "on_date", "on_time", "component", "all4")},
           "table_path": {f: tal("table", f) for f in FIELDS + ("all4",)},
           "table_path_chip_truth": {f: tal("table_chip", f) for f in FIELDS + ("all4",)},
           "border_16_digits": sum(r["border_16_digits"] for r in rows),
           "secs_median": sorted(r["secs"] or 0 for r in rows)[n // 2] if n else None,
           "errors": sum(r["error"] is not None for r in rows),
           "spark_e4": {f: e4["summary"][f] for f in ("station", "on_date", "on_time", "component", "all4")},
           "server": json.loads((OUT / "server.json").read_text()) if (OUT / "server.json").exists() else None, "rows": rows}
    (OUT / "e9_results.json").write_text(json.dumps(res, indent=1))
    print({k: f"{v['correct']}/{n}" for k, v in res["label_fields"].items()}, "| table", {k: f"{v['correct']}/{n}" for k, v in res["table_path"].items()},
          "| 16 digits", res["border_16_digits"], "| median s", res["secs_median"], "| errors", res["errors"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["dev", "run", "score"])
    ap.add_argument("--url", default="http://127.0.0.1:8792")
    a = ap.parse_args()
    if a.cmd == "dev":
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "dev_server.json").write_text(json.dumps(server_props(a.url), indent=1))
        run(a.url, DEV, OUT / "dev", retry_errors=True)
    elif a.cmd == "run":
        marker = OUT / "run_started.json"
        if marker.exists():
            raise SystemExit("E9 already started: one run only (research/48)")
        chips = sorted(CHIPS.glob("*.TIFF"))
        if len(chips) != 30:
            raise SystemExit(f"E9 runs on the 30 E4 chips, found {len(chips)} under {CHIPS}: fetch them first (data/README.md)")
        OUT.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps({"started": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "settings": SETTINGS}, indent=1))
        (OUT / "server.json").write_text(json.dumps(server_props(a.url), indent=1))
        run(a.url, chips, OUT / "responses")
        score()
    else:
        score()
