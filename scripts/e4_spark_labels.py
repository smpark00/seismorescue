"""E4 (research/42): Muse Spark reads the label of real public film chips; truth = the EarthScope file names.

usage: MODEL_API_KEY=... python -I scripts/e4_spark_labels.py dev | run | score

- dev: the two development chips (MAT 1973 LHZ, ALQ 1975 SHZ) -> out/e4/dev/; checks the path and parsing only.
- run: once (start marker first), the 30 chips of data/e4_chips, 4 requests at a time -> out/e4/responses/<chip>.json,
  then score.
- score: re-scores the saved responses -> out/e4/e4_results.json.
Only public CC0 chips are sent (research/36 appendix B 3). The key is read from MODEL_API_KEY and never printed.
"""
import json
import math
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import metaapi  # noqa: E402

OUT = ROOT / "out" / "e4"
CHIPS = ROOT / "data" / "e4_chips"
DEV = [ROOT / "data" / "real_scans" / "MAT_73_12_28_0020_LHZ.TIFF", ROOT / "data" / "real_scans" / "ALQ_75_10_27_1551_SHZ.TIFF"]
PRICE = {"input": 0.10e-6, "output": 0.20e-6}      # contributor tier, $ per token (research/15)
QUESTION = """These two images are scans of one WWSSN seismogram film chip (a photographic seismogram record).
Image 1 is the whole chip, image 2 is its top part at higher resolution: the film border with printed digit groups
and the station label form. Read the record's metadata from what is printed or written on the chip itself.
The first two printed digit groups on the film border usually give the ON date as MMDDYY and the ON time as HHMM.
Return only a JSON object with these keys:
"station" (the station code, usually 3-4 capital letters), "component" ("Z", "N" or "E"),
"period" ("LP" for long period or "SP" for short period), "on_date" ("YYYY-MM-DD"), "on_time" ("HHMM", the ON time),
"off_date" ("YYYY-MM-DD"), "off_time" ("HHMM"), "magnification" (a number),
"border_digits" (the printed digit groups on the film border, as one string), "notes" (one short sentence).
Use null for anything you cannot read."""


def images(path):
    """the whole chip at 2048 px wide and its top 35 % at 3072 px wide (never upscaled)"""
    img = cv2.imread(str(path), 0)
    if img is None:
        raise ValueError(f"cannot read {path}: damaged file or not an image")
    h, w = img.shape
    out = []
    for part, width in ((img, 2048), (img[:int(0.35 * h)], 3072)):
        width = min(width, w)
        out.append(cv2.resize(part, (width, round(part.shape[0] * width / w)), interpolation=cv2.INTER_AREA))
    return out


def parse(text):
    a, b = text.find("{"), text.rfind("}")
    if a < 0 or b <= a:
        return None
    try:
        return json.loads(text[a:b + 1])
    except json.JSONDecodeError:
        return None


def ask(path):
    """one question; asked again only after an HTTP error or unparsable JSON"""
    rec = {"chip": path.stem}
    for attempt in (1, 2):
        try:
            r = metaapi.spark_ask(images(path), QUESTION)
            rec.update({"attempt": attempt, "text": r["text"], "parsed": parse(r["text"]), "usage": r["usage"],
                        "secs": round(r["secs"], 1), "error": None})
            if rec["parsed"] is not None:
                break
        except Exception as e:                                     # noqa: BLE001 (recorded, retried once)
            rec.update({"attempt": attempt, "parsed": None, "error": f"{type(e).__name__}: {str(e)[:300]}"})
    return rec


def truth(stem):
    sta, yy, mo, dd, hhmm, cha = re.match(r"([A-Z0-9]+)_(\d\d)_(\d\d)_(\d\d)_(\d{4})_([A-Z]{3})", stem).groups()
    return {"station": sta, "on_date": f"19{yy}-{mo}-{dd}", "on_time": hhmm, "component": cha[2],
            "period": "LP" if cha[0] == "L" else "SP"}


def minutes(hhmm):
    return int(hhmm[:2]) * 60 + int(hhmm[2:])


def score_one(js, t):
    s = lambda k: str((js or {}).get(k) or "").strip().upper()  # noqa: E731
    tm = re.sub(r"\D", "", s("on_time"))
    ok_tm = len(tm) in (3, 4) and int(tm.zfill(4)[:2]) < 24 and int(tm.zfill(4)[2:]) < 60
    row = {"station": s("station") == t["station"], "on_date": s("on_date") == t["on_date"],
           "on_time": ok_tm and tm.zfill(4) == t["on_time"],
           "on_time_1min": ok_tm and abs(minutes(tm.zfill(4)) - minutes(t["on_time"])) <= 1,
           "component": s("component") == t["component"], "period": s("period") == t["period"]}
    row["all4"] = row["station"] and row["on_date"] and row["on_time"] and row["component"]
    return row


def wilson(k, n, z=1.96):
    if n == 0:
        return [float("nan")] * 2
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 3), round(c + h, 3)]


def score(resp_dir=OUT / "responses", dst=OUT / "e4_results.json"):
    rows = []
    for f in sorted(resp_dir.glob("*.json")):
        r = json.loads(f.read_text())
        t = truth(r["chip"])
        rows.append({"chip": r["chip"], "truth": t, "read": r.get("parsed"), "error": r.get("error"),
                     "attempt": r.get("attempt"), "secs": r.get("secs"), "usage": r.get("usage"), **score_one(r.get("parsed"), t)})
    n = len(rows)
    keys = ("all4", "station", "on_date", "on_time", "on_time_1min", "component", "period")
    summ = {k: {"correct": sum(r[k] for r in rows), "n": n, "rate": round(sum(r[k] for r in rows) / n, 3) if n else None,
                "wilson95": wilson(sum(r[k] for r in rows), n)} for k in keys}
    tok_in = sum((r["usage"] or {}).get("input_tokens", 0) for r in rows)
    tok_out = sum((r["usage"] or {}).get("output_tokens", 0) for r in rows)
    summ["json_failures"] = sum(r["read"] is None for r in rows)
    summ["tokens"] = {"input": tok_in, "output": tok_out, "cost_usd": round(tok_in * PRICE["input"] + tok_out * PRICE["output"], 4)}
    summ["secs_median"] = sorted(r["secs"] or 0 for r in rows)[n // 2] if n else None
    dst.write_text(json.dumps({"summary": summ, "rows": rows}, indent=1))
    for k in keys:
        print(f"{k:13s} {summ[k]['correct']:2d}/{n}  {summ[k]['rate']}  95% {summ[k]['wilson95']}")
    print("json failures", summ["json_failures"], "| tokens", summ["tokens"], "| median secs", summ["secs_median"])
    return summ


def run(paths, resp_dir):
    resp_dir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(4) as pool:
        for rec in pool.map(ask, paths):
            (resp_dir / f"{rec['chip']}.json").write_text(json.dumps(rec, indent=1))
            print(rec["chip"], rec.get("attempt"), rec.get("secs"), json.dumps(rec.get("parsed"))[:300] if rec.get("parsed") else rec.get("error"), flush=True)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    cmd = sys.argv[1]
    if cmd == "dev":
        run(DEV, OUT / "dev")
        score(OUT / "dev", OUT / "dev_results.json")
    elif cmd == "run":
        chips = sorted(CHIPS.glob("*.TIFF"))
        assert len(chips) == 30, len(chips)
        with open(OUT / "run_started.json", "x") as f:
            f.write(json.dumps({"started_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "n_chips": len(chips),
                                "model": metaapi.SPARK_MODEL}, indent=1))
        run(chips, OUT / "responses")
        score()
    elif cmd == "score":
        score()
