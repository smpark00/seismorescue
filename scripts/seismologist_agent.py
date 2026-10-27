"""Seismologist agent (Muse Spark + tools) on digitized analog records.

usage: python -I seismologist_agent.py <scripts_dir> <series_dir> <catalog_dir> <out_dir> [run_tag] [workers] [limit]

For each record/condition the agent gets the station and start time (the metadata agent's label reading) and must
return JSON: the event id (largest catalogue event whose P arrives inside the record), its P arrival and the time of
its surface-wave maximum (seconds since record start), and spans it believes the digitization got wrong.
Tools measure, Spark decides (docs/RESULTS.md P1). One helicorder line = 900 s: a trace that jumped to the neighbouring
line shows up as the same waveform 900 s early/late, which the `line_shift_similarity` tool exposes.
Transcripts and answers go to <out_dir>/<run_tag>/.
"""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, sys.argv[1])
import numpy as np
import requests
from metaapi import BASE_URL, SPARK_MODEL, _headers, png_data_url
from seismo_tools import aic_pick, plot_png, sta_lta

series_dir, cat_dir, out_root = Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
RUN = sys.argv[5] if len(sys.argv) > 5 else "run1"
WORKERS = int(sys.argv[6]) if len(sys.argv) > 6 else 16
LIMIT = int(sys.argv[7]) if len(sys.argv) > 7 else 0
out = out_root / RUN
out.mkdir(parents=True, exist_ok=True)
MAX_TURNS = 14

INSTRUCTIONS = """You are a careful seismologist analysing a DIGITIZED analog long-period vertical seismogram.
The analog sheet was a helicorder: 8 lines of 900 s each; it was digitized automatically into one series
(1 sample/s, amplitude in units of line spacing). Automatic digitization can make errors: at crossings a line can
jump onto its neighbour's trace, so part of a big waveform may appear exactly 900 s too early or too late, or be
duplicated on two lines. Spans marked low-confidence by the digitizer are more likely wrong.

Tasks:
1. event_id: the largest-magnitude catalogue event whose P arrival falls inside the record (0-7200 s).
2. p_arrival_s: that event's P arrival, measured with the tools (seconds since record start).
3. surface_peak_s: time of that event's surface-wave maximum (seconds since record start).
4. suspect_spans: list of [start_s, end_s] spans you believe the digitization got wrong (may be empty).

Rules: never read numbers off a picture; use the measuring tools (aic_pick, find_peak, onsets) for every number.
Use pictures (plot) only to judge shape and plausibility. Check physical consistency: P before S before surface
waves, predicted times from the catalogue tool, and the 900 s line-jump signature.
When done, reply with ONLY a JSON object:
{"event_id": str, "p_arrival_s": number, "surface_peak_s": number, "suspect_spans": [[a, b], ...], "notes": str}"""

TOOLS = [
    {"type": "function", "name": "catalog", "description": "Catalogue events (M>=5.5) near the record time with predicted P, S and surface-wave (R1) arrival times at this station in seconds since record start.",
     "parameters": {"type": "object", "properties": {}, "required": []}},
    {"type": "function", "name": "onsets", "description": "STA/LTA onset detector on the digitized series. Returns onset times (s) with peak STA/LTA ratio.",
     "parameters": {"type": "object", "properties": {"sta_s": {"type": "integer"}, "lta_s": {"type": "integer"}, "threshold": {"type": "number"}}, "required": ["sta_s", "lta_s", "threshold"]}},
    {"type": "function", "name": "aic_pick", "description": "Precise onset time with the AIC picker inside [t_approx-window, t_approx+window]. Returns pick_s and an after/before amplitude ratio.",
     "parameters": {"type": "object", "properties": {"t_approx_s": {"type": "number"}, "window_s": {"type": "number"}}, "required": ["t_approx_s", "window_s"]}},
    {"type": "function", "name": "find_peak", "description": "Time and value of the maximum absolute amplitude between start_s and end_s.",
     "parameters": {"type": "object", "properties": {"start_s": {"type": "number"}, "end_s": {"type": "number"}}, "required": ["start_s", "end_s"]}},
    {"type": "function", "name": "low_confidence", "description": "Spans the digitizer marked low-confidence between start_s and end_s, and their fraction.",
     "parameters": {"type": "object", "properties": {"start_s": {"type": "number"}, "end_s": {"type": "number"}}, "required": ["start_s", "end_s"]}},
    {"type": "function", "name": "line_shift_similarity", "description": "Correlation between the series in [start_s, end_s] and the same span shifted by -900 s and +900 s (one helicorder line). A high correlation means a duplicated or line-jumped waveform.",
     "parameters": {"type": "object", "properties": {"start_s": {"type": "number"}, "end_s": {"type": "number"}}, "required": ["start_s", "end_s"]}},
    {"type": "function", "name": "plot", "description": "Picture of the digitized series between start_s and end_s (low-confidence spans shaded). For judging shapes only.",
     "parameters": {"type": "object", "properties": {"start_s": {"type": "number"}, "end_s": {"type": "number"}}, "required": ["start_s", "end_s"]}},
]


def run_tool(name, args, z, cat):
    a, low = z["digi"], z["lowconf"]
    if name == "catalog":
        return json.dumps([{k: e[k] for k in ("id", "time", "mag", "dist_deg", "depth_km", "P_s", "S_s", "R1_s")} for e in cat["events"]]), None
    if name == "onsets":
        return json.dumps(sta_lta(a, int(args.get("sta_s", 10)), int(args.get("lta_s", 120)), float(args.get("threshold", 3)))[:40]), None
    if name == "aic_pick":
        return json.dumps(aic_pick(a, float(args["t_approx_s"]), float(args.get("window_s", 40)))), None
    if name == "find_peak":
        s, e = max(0, int(args["start_s"])), min(len(a), int(args["end_s"]))
        if e <= s:
            return json.dumps({"error": "empty span"}), None
        i = s + int(np.nanargmax(np.abs(a[s:e])))
        return json.dumps({"peak_s": i, "amplitude": round(float(a[i]), 3)}), None
    if name == "low_confidence":
        s, e = max(0, int(args["start_s"])), min(len(a), int(args["end_s"]))
        m = low[s:e]
        edges = np.flatnonzero(np.diff(np.r_[0, m.astype(int), 0]))
        return json.dumps({"fraction": round(float(m.mean()) if len(m) else 0, 3),
                           "spans": [[int(s + p), int(s + q)] for p, q in zip(edges[::2], edges[1::2])][:30]}), None
    if name == "line_shift_similarity":
        s, e = max(0, int(args["start_s"])), min(len(a), int(args["end_s"]))
        res = {}
        for sh in (-900, 900):
            s2, e2 = s + sh, e + sh
            if s2 < 0 or e2 > len(a) or e - s < 20:
                res[str(sh)] = None
                continue
            x, y = a[s:e], a[s2:e2]
            res[str(sh)] = round(float(np.corrcoef(x, y)[0, 1]), 3) if x.std() > 0 and y.std() > 0 else None
        return json.dumps(res), None
    if name == "plot":
        img = plot_png(a, low, float(args["start_s"]), float(args["end_s"]), f"digitized series {int(args['start_s'])}-{int(args['end_s'])} s")
        return "picture attached in the next message", img
    return json.dumps({"error": f"unknown tool {name}"}), None


def post(payload):
    for attempt in range(6):
        try:
            r = requests.post(BASE_URL, headers=_headers(False), json=payload, timeout=600)
            if r.status_code == 200:
                return r.json()
            err = f"HTTP {r.status_code}: {r.text[:300]}"
        except Exception as ex:
            err = str(ex)[:300]
        time.sleep(6 * (attempt + 1))
    raise RuntimeError(err)


def run_agent(item):
    f, rec = item
    done = out / f"{f.stem}.json"
    if done.exists():
        return json.loads(done.read_text())
    z = np.load(f)
    cat = json.loads((cat_dir / f"{rec}.json").read_text())
    user = (f"Station {cat['station']} (lat {cat['sta_lat']:.2f}, lon {cat['sta_lon']:.2f}). Record starts "
            f"{cat['start_utc']} UTC and lasts 7200 s. Analyse it with the tools and answer.")
    base = {"model": SPARK_MODEL, "instructions": INSTRUCTIONS, "tools": TOOLS, "reasoning": {"effort": "medium"}}
    # full history every turn (previous_response_id lookups failed intermittently with "response not found")
    history = [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": user}]}]
    resp = post({**base, "input": history})
    log, usage, answer = [], [resp.get("usage")], None
    for turn in range(MAX_TURNS):
        calls = [o for o in resp.get("output", []) if o.get("type") == "function_call"]
        history += [{"type": "function_call", "call_id": c["call_id"], "name": c["name"], "arguments": c.get("arguments") or "{}"} for c in calls]
        if not calls:
            text = " ".join(c.get("text", "") for o in resp.get("output", []) for c in (o.get("content") or []) if c.get("type") in ("output_text", "text"))
            log.append({"final": text})
            try:
                answer = json.loads(text[text.index("{"):text.rindex("}") + 1])
            except Exception:
                answer = None
            break
        inp, imgs = [], []
        for c in calls:
            try:
                args = json.loads(c.get("arguments") or "{}")
                txt, img = run_tool(c["name"], args, z, cat)
            except Exception as ex:
                args, txt, img = {}, json.dumps({"error": str(ex)[:200]}), None
            log.append({"tool": c["name"], "args": args, "result": txt[:2000]})
            inp.append({"type": "function_call_output", "call_id": c["call_id"], "output": txt})
            if img is not None:
                imgs.append(img)
        if imgs:
            inp.append({"type": "message", "role": "user", "content": [{"type": "input_image", "image_url": png_data_url(i)} for i in imgs]})
        if turn == MAX_TURNS - 2:
            inp.append({"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Tool budget reached: give your final JSON answer now."}]})
        history += inp
        resp = post({**base, "input": history})
        usage.append(resp.get("usage"))
    res = {"item": f.stem, "record": rec, "answer": answer, "turns": len([l for l in log if "tool" in l]),
           "out_tokens": sum((u or {}).get("output_tokens", 0) for u in usage),
           "in_tokens": sum((u or {}).get("input_tokens", 0) for u in usage), "log": log}
    done.write_text(json.dumps(res, indent=1))
    return res


items = [(f, f.stem.rsplit("_", 2)[0]) for f in sorted(series_dir.glob("*.npz"))]
if LIMIT:
    items = items[:LIMIT]
t0 = time.time()
with ThreadPoolExecutor(max_workers=WORKERS) as ex:
    results = list(ex.map(lambda it: (lambda: run_agent(it))() if True else None, items))
print(f"{len(results)} runs in {time.time() - t0:.0f} s; answers {sum(r['answer'] is not None for r in results)}; "
      f"out tokens {sum(r['out_tokens'] for r in results):,}; in tokens {sum(r['in_tokens'] for r in results):,}")
