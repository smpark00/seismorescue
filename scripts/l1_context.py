"""L1 context stage (research/44): Muse Spark reads a public WWSSN film chip's border and label and calls four
deterministic tools; the tools decode the border with the WWSSN station-number table, run L0 with the ON and OFF read
from the chip, and place the catalogue quakes on the record lines with TauP and L0's time model. Tools measure, Spark reads and chooses (research/14 P1): every number in
record.json comes from a tool; Spark's text only fills the label fields, the main-event choice and the notes.

usage: MODEL_API_KEY=... python -I scripts/l1_context.py run <chip> [--out DIR]   # L1 (with L0 inside) -> record.json
       python -I scripts/l1_context.py replay          # research/44 §4: E4's stored border reads through the table
       MODEL_API_KEY=... python -I scripts/l1_context.py test     # research/44 §6 (E7), once; then `score`
       python -I scripts/l1_context.py score           # re-scores the saved E7 records
Only public CC0 chips are sent (research/36 appendix B 3): a chip needs a sidecar <stem>.json recording a CC0 licence
and an EarthScope source (as fetch_e4_chips.py writes); data_local/, out/ and KIP / calibration names are refused.
The key is read from MODEL_API_KEY and never printed. Spark never sees the chip's file name; it is compared afterwards.
"""
import argparse
import datetime
import hashlib
import json
import math
import os
import re
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import l0_geometry as l0                                         # noqa: E402
import metaapi                                                   # noqa: E402
from e4_spark_labels import PRICE, images, parse, wilson         # noqa: E402
from obspy import UTCDateTime                                    # noqa: E402

TABLE_FILE = ROOT / "data" / "wwssn_stations" / "wwssn_stations.json"
TABLE = json.loads(TABLE_FILE.read_text())["stations"]
COMPONENTS = {"01": "SPZ", "04": "LPZ", "05": "LPN", "06": "LPE"}   # 04-06: 2,095 IRIS seismogram links (research/44
                                                                   # §2); 01: one chip, E4 dev ALQ_75 SHZ (research/42)
OUT = ROOT / "out" / "l1"
CACHE = OUT / "catalog_cache"
E7_CHIPS = ROOT / "data" / "e7_chips"
DOC44 = ROOT.parent / "44-l1-context.md"                          # internal notes, not published; hashed only if present
MAX_REQUESTS, MAX_TOOL_CALLS, MAX_REPAIRS = 6, 6, 1               # call budget per chip (research/44 §3)
MIN_MAG, SURFACE_KMS, KM_PER_DEG = 5.5, (4.0, 3.0), 111.19
FIELDS = ("station", "on_date", "on_time", "component")
CHIP_FIX = {"WIN_71_01_29_0124_LHZ": {"on_time": "0623"}, "LON_68_03_22_1616_LHZ": {"on_date": "1968-03-21"}}  # 42 §6.3

INSTRUCTIONS = """You are the context agent of SeismoRescue, which digitizes WWSSN analog seismogram film chips.
You get two images of one film chip: image 1 is the whole chip, image 2 its top part at higher resolution with the film
border and the station label form.
1. Read the printed digit groups on the film border. There are four groups, 'MMDDYY HHMM NNNN CC': the ON date, the ON
   time (UTC), the WWSSN station number and the component code. They are the primary record metadata. Call
   decode_border with exactly the digits you read; never guess an unreadable digit.
2. Read the handwritten or typed label form (station code, component, magnification, OFF date and time). It is only a
   cross-check. If decode_border cannot identify the station, call station_by_code with the label's station code.
3. Call measure_record with the OFF date and time read from the label form (null if unreadable). It finds the record
   lines and measures the time marks.
4. Call catalog to get the earthquakes this record should show, with the record lines where their P waves and surface
   waves are predicted from the measured time model.
5. Choose the main event: the catalogue event this record most likely shows best, usually the one with the largest
   expected surface-wave amplitude inside the record.
Never compute dates, times, distances or line numbers yourself; the tools do that. When done, reply with ONLY this JSON
object (null for anything unreadable):
{"border_digits": str, "label_station": str, "label_component": "Z" | "N" | "E", "magnification": number,
 "off_date": "YYYY-MM-DD", "off_time": "HHMM", "main_event_id": str, "notes": str}"""

TOOLS = [
    {"type": "function", "name": "decode_border",
     "description": "Decode the printed film-border digit groups 'MMDDYY HHMM NNNN CC' (ON date, ON time UTC, WWSSN station "
                    "number, component code) with the WWSSN station-number table. Returns the ON time, the station (code, "
                    "coordinates) and the component, and lists any problem.",
     "parameters": {"type": "object", "properties": {"border_digits": {"type": "string", "description": "the digit groups as printed, e.g. '122873 0020 0064 04'"}},
                    "required": ["border_digits"]}},
    {"type": "function", "name": "station_by_code",
     "description": "Look up a WWSSN station by its code (for example read from the label form): number, coordinates, place.",
     "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}},
    {"type": "function", "name": "measure_record",
     "description": "Run the geometry and time module (L0) on this chip with the decoded ON time and the OFF date and time "
                    "read from the label form: it finds the record lines (their number follows from ON and OFF), measures "
                    "the line period and the minute marks, and returns the line count, the line period and the record's "
                    "start and end (UTC). Call it after decode_border.",
     "parameters": {"type": "object", "properties": {"off_date": {"type": "string", "description": "YYYY-MM-DD, or null"},
                                                     "off_time": {"type": "string", "description": "HHMM, or null"}},
                    "required": ["off_date", "off_time"]}},
    {"type": "function", "name": "catalog",
     "description": "USGS catalogue earthquakes (M >= min_magnitude, default 5.5) that this record should show, for the "
                    "decoded station and ON time: distance, TauP P arrival and surface-wave window (group velocity 4.0-3.0 "
                    "km/s), each placed on the record lines (line 0 = first record line) by the measured time model, and "
                    "the rank of the expected surface-wave amplitude.",
     "parameters": {"type": "object", "properties": {"min_magnitude": {"type": "number"}}, "required": []}},
]


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def guard(chip):
    """research/36 appendix B 3: KIP, its S3 copy, the calibration scans and paired waveforms never go to an API.
    Allow-list: the chip must carry a sidecar <stem>.json with a CC0 licence and an EarthScope source URL; local-only
    folders, derived images under out/ and KIP / calibration names are refused outright (research/44 §7.6)"""
    chip = Path(chip).resolve()
    low = chip.name.lower()
    if (ROOT / "data_local") in chip.parents or (ROOT / "out") in chip.parents or any(k in low for k in ("kip", "cal_", "calib")):
        raise SystemExit(f"{chip.name}: local-only or derived scan, not sent to any API")
    side = chip.with_suffix(".json")
    info = json.loads(side.read_text()) if side.exists() else {}
    if "CC0" not in json.dumps(info.get("licence", "")) or not str(info.get("source_url", "")).startswith("https://data.earthscope.org/"):
        raise SystemExit(f"{chip.name}: no sidecar recording a public CC0 EarthScope source; not sent to any API")
    return chip


# ---------- deterministic tools ----------
def decode_border(digits):
    """the four printed border groups 'MMDDYY HHMM NNNN CC' (research/42 §6.3; the IRIS file names of research/44 §2)
    -> ON (UTC), station number -> WWSSN table, component code -> COMPONENTS; problems are listed, nothing is guessed"""
    groups = re.findall(r"\d+", digits or "")
    if [len(x) for x in groups[:2]] == [6, 4]:
        g = groups[:4]
    else:                                                         # run together or split oddly: cut 6 + 4 + 4 + 2
        s = "".join(groups)
        g = [s[a:b] for a, b in ((0, 6), (6, 10), (10, 14), (14, 16)) if len(s) >= b]
    out = {"groups": g, "on_utc": None, "station_number": None, "station": None, "component_code": None,
           "component": None, "problems": []}
    if len(g) < 2:
        out["problems"].append("no 'MMDDYY HHMM' at the start of the border digits")
        return out
    try:
        on = datetime.datetime(1900 + int(g[0][4:]), int(g[0][:2]), int(g[0][2:4]), int(g[1][:2]), int(g[1][2:]))
        out["on_utc"] = on.strftime("%Y-%m-%dT%H:%M:00")
    except ValueError as e:
        out["problems"].append(f"ON date/time {g[0]} {g[1]} is not valid ({e})")
    if len(g) > 2 and len(g[2]) == 4:
        out["station_number"] = g[2]
        s = TABLE.get(g[2])
        if s is None:
            out["problems"].append(f"station number {g[2]} is not in the WWSSN table")
        else:
            out["station"] = {"code": s["code"], "number": g[2], "lat": s["lat"], "lon": s["lon"], "elev_m": s["elev_m"],
                              "place": s["place"]}
    else:
        out["problems"].append("no 4-digit station number (group 3)")
    if len(g) > 3 and len(g[3]) == 2:
        out["component_code"] = g[3]
        out["component"] = COMPONENTS.get(g[3])
        if out["component"] is None:
            out["problems"].append(f"component code {g[3]} is not known (known: {COMPONENTS})")
    else:
        out["problems"].append("no 2-digit component code (group 4)")
    return out


def station_by_code(code):
    c = str(code or "").strip().upper()
    for num, s in sorted(TABLE.items()):
        if s["code"] == c:
            return {"found": True, "code": c, "number": num, "lat": s["lat"], "lon": s["lon"], "elev_m": s["elev_m"],
                    "place": s["place"]}
    return {"found": False, "code": c, "problem": f"station code {c} is not in the WWSSN table"}


class Clock:
    """L0's time model t(x, k) = ON + T0 + P k + (x - xa) / pps_r (research/40) with the ON read from the border.
    Without a time model (L0 failed, unsupported line format) only the nominal 24 h span is known and no line."""

    def __init__(self, geo, on_utc):
        tm = geo.get("time") or {}
        self.on = UTCDateTime(on_utc)
        self.ok = tm.get("T0") is not None
        self.reason = None if self.ok else (geo.get("failed") or tm.get("reason") or "no line-0 trace start")
        self.ambiguous = bool(tm.get("minute_ambiguous"))
        if self.ok:
            self.T0, self.P, self.pps = tm["T0"], tm["P"], tm["pps_r"]
            self.xa, self.x0, self.x1, self.n = geo["xa"], geo["x0"], geo["x1"], geo["n_lines"]
            xe = tm.get("last_line_trace_end_x")
            self.start, self.end = self.utc(0, tm["x_on"]), self.utc(self.n - 1, self.x1 if xe is None else xe)
        else:
            self.start, self.end = self.on, self.on + 24 * 3600

    def utc(self, k, x):
        return self.on + self.T0 + self.P * k + (x - self.xa) / self.pps

    def locate(self, t):
        """(line, column) of UTC t: line k holds [x0, x0 + P pps_r) from its own start; None without a time model"""
        if not self.ok:
            return None, None
        s = float(UTCDateTime(t) - self.on) - self.T0
        k = math.floor((s - (self.x0 - self.xa) / self.pps) / self.P)
        return k, self.xa + (s - self.P * k) * self.pps

    def window(self, k0, k1):
        """lines k0..k1 widened by one line each way, kept inside the record (the '+-1' of research/36 §3)"""
        if k0 is None:
            return None
        return [max(0, min(k0, k1) - 1), min(self.n - 1, max(k0, k1) + 1)]


def events_near(t0, t1, min_mag):
    """USGS FDSN events in [t0, t1] with M >= min_mag, cached as JSON (the demo replays the cache)"""
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"usgs_{t0.strftime('%Y%m%dT%H%M%S')}_{t1.strftime('%Y%m%dT%H%M%S')}_M{min_mag:g}.json"
    if f.exists():
        return json.loads(f.read_text())
    from obspy.clients.fdsn import Client
    from obspy.clients.fdsn.header import FDSNNoDataException
    try:
        cat = Client("USGS").get_events(starttime=t0, endtime=t1, minmagnitude=min_mag, orderby="time-asc")
    except FDSNNoDataException:
        cat = []
    rows = []
    for ev in cat:
        o = ev.preferred_origin() or ev.origins[0]
        m = ev.preferred_magnitude() or (ev.magnitudes[0] if ev.magnitudes else None)
        eid = re.search(r"eventid=([^&]+)", str(ev.resource_id))
        rows.append({"id": eid.group(1) if eid else str(ev.resource_id), "origin_utc": str(o.time)[:22],
                     "lat": o.latitude, "lon": o.longitude, "depth_km": round((o.depth or 0) / 1000, 1),
                     "mag": None if m is None else round(m.mag, 1), "mag_type": None if m is None else m.magnitude_type,
                     "place": ev.event_descriptions[0].text if ev.event_descriptions else None})
    f.write_text(json.dumps(rows, indent=1))
    return rows


_TAUP = []


def place_events(station, clock, min_mag=MIN_MAG):
    """catalogue events whose P or surface waves fall inside the record, placed on its lines, ranked by the expected
    surface-wave amplitude M - 1.66 log10(distance) (the Ms distance term)"""
    from obspy.geodetics import locations2degrees
    if not _TAUP:
        from obspy.taup import TauPyModel
        _TAUP.append(TauPyModel("iasp91"))
    rows = []
    for e in events_near(clock.start - 2 * 3600, clock.end, min_mag):
        dist = locations2degrees(e["lat"], e["lon"], station["lat"], station["lon"])
        arr = _TAUP[0].get_travel_times(max(0.0, e["depth_km"]), dist, phase_list=["P", "Pdiff", "PKIKP"])
        ot = UTCDateTime(e["origin_utc"])
        tp = ot + arr[0].time if arr else None
        tr = [ot + dist * KM_PER_DEG / v for v in SURFACE_KMS]
        p_in = tp is not None and clock.start <= tp <= clock.end
        if not (p_in or (tr[1] >= clock.start and tr[0] <= clock.end)):
            continue
        kp, xp = clock.locate(tp) if tp is not None else (None, None)
        kr = [clock.locate(t)[0] for t in tr]
        rows.append({**{k: e[k] for k in ("id", "origin_utc", "mag", "mag_type", "place", "depth_km")},
                     "dist_deg": round(float(dist), 1), "P_phase": arr[0].name if arr else None,
                     "P_utc": str(tp)[:19] if tp is not None else None, "P_inside_record": p_in,
                     "P_line": kp, "P_column": None if xp is None else int(round(xp)), "P_lines_pm1": clock.window(kp, kp),
                     "surface_utc": [str(t)[:19] for t in tr], "surface_lines": kr,
                     "surface_lines_pm1": clock.window(*kr),
                     "amp_score": None if e["mag"] is None else round(e["mag"] - 1.66 * math.log10(max(dist, 1.0)), 2)})
    rows.sort(key=lambda r: -(r["amp_score"] if r["amp_score"] is not None else -99))
    for i, r in enumerate(rows):
        r["amp_rank"] = i + 1
    return rows


def off_utc(on, off_date, off_time):
    """the label OFF as UTC when it is 12-36 h after ON (one daily record), else None and why"""
    if not (isinstance(off_date, str) and isinstance(off_time, str) and re.fullmatch(r"\d{4}-\d\d-\d\d", off_date)
            and re.fullmatch(r"\d{4}", off_time)):
        return None, "OFF not read: L0 counted the lines without it"
    try:
        off = UTCDateTime(f"{off_date}T{off_time[:2]}:{off_time[2:]}:00")
    except Exception:                                             # noqa: BLE001 (e.g. hour 25)
        return None, f"OFF {off_date} {off_time} is not a valid time: L0 counted the lines without it"
    hours = (off - UTCDateTime(on)) / 3600
    if not 12 <= hours <= 36:
        return None, f"OFF is {hours:.1f} h after ON (12-36 h expected): L0 counted the lines without it"
    return str(off)[:19], None


class Geometry:
    """L0 runs on one chip, cached by (ON, OFF): with both, L0 keeps ceil((OFF - ON) / 1 h) + 1 lines (research/25
    rule); without OFF it keeps every line with any trace and may count label text as lines (MAT: 27 for 25)"""

    def __init__(self, chip, out):
        self.chip, self.out, self.runs = chip, Path(out), {}

    def get(self, on, off):
        if (on, off) not in self.runs:
            d = self.out / f"l0_{len(self.runs)}"
            geo, _ = l0.run(self.chip, on, off, out=d)
            self.runs[(on, off)] = geo | {"l0_dir": d.name}
        return self.runs[(on, off)]


def measure_summary(geo, on, off, note):
    if geo.get("failed"):
        return {"time_model": f"none: L0 failed ({geo['failed']})", "off_used": off, "off_note": note}
    clock, tm = Clock(geo, on), geo.get("time") or {}
    out = {"lines": geo["n_lines"], "off_used": off, "off_note": note,
           "time_model": "measured" if clock.ok else f"none ({clock.reason})"}
    if clock.ok:
        out.update({"line_period_s": round(clock.P, 1), "record_utc": [str(clock.start)[:19], str(clock.end)[:19]],
                    "on_minute_ambiguous": clock.ambiguous})
    if tm.get("P_from_off") is not None:
        out["line_period_implied_by_off_s"] = round(tm["P_from_off"], 1)
    return out


class Tools:
    """the agent's tools; the record itself is rebuilt from the final answer by finalize()"""

    def __init__(self, chip, out):
        self.geom, self.station, self.on, self.geo, self.measured_off = Geometry(chip, out), None, None, None, ()

    def run(self, name, args):
        if name == "decode_border":
            d = decode_border(str(args.get("border_digits", "")))
            self.on = d["on_utc"] or self.on
            self.station = d["station"] or self.station
            return d
        if name == "station_by_code":
            s = station_by_code(args.get("code"))
            if s["found"]:
                self.station = s
            return s
        if name == "measure_record":
            if self.on is None:
                return {"error": "the ON time is not known yet: call decode_border first"}
            off, note = off_utc(self.on, args.get("off_date"), args.get("off_time"))
            self.geo, self.measured_off = self.geom.get(self.on, off), (off,)
            return measure_summary(self.geo, self.on, off, note)
        if name == "catalog":
            if self.station is None or self.on is None:
                return {"error": "the station and ON time are not known yet: call decode_border (or station_by_code) first"}
            if self.geo is None:                                  # measure_record skipped: L0 without OFF
                self.geo = self.geom.get(self.on, None)
            clock = Clock(self.geo, self.on)
            ev = place_events(self.station, clock, float(args.get("min_magnitude") or MIN_MAG))
            return {"station": self.station["code"], "record_utc": [str(clock.start)[:19], str(clock.end)[:19]],
                    "time_model": "measured" if clock.ok else f"none ({clock.reason}); no line positions",
                    "lines": clock.n if clock.ok else None, "line_period_s": round(clock.P, 1) if clock.ok else None,
                    "on_minute_ambiguous": clock.ambiguous, "events": ev[:12]}
        return {"error": f"unknown tool {name}"}


# ---------- the agent ----------
def post(payload):
    for attempt in range(4):
        try:
            r = requests.post(metaapi.BASE_URL, headers=metaapi._headers(False), json=payload, timeout=600)
            if r.status_code == 200:
                return r.json()
            err = f"HTTP {r.status_code}: {r.text[:300]}"
        except Exception as ex:                                   # noqa: BLE001 (retried, then raised)
            err = f"{type(ex).__name__}: {str(ex)[:300]}"
        time.sleep(6 * (attempt + 1))
    raise RuntimeError(err)


USER = "Public scan of a WWSSN film chip. Read the record metadata and use the tools as instructed."


def finalize(ans, geom):
    """the record's numbers, rebuilt deterministically from the final answer (border first, label code as fallback)"""
    ans = ans if isinstance(ans, dict) else {}
    dec = decode_border(ans["border_digits"]) if isinstance(ans.get("border_digits"), str) else None
    station, src = (dec["station"], "film border number + WWSSN table") if dec and dec["station"] else (None, None)
    if station is None and isinstance(ans.get("label_station"), str):
        s = station_by_code(ans["label_station"])
        if s["found"]:
            station, src = s, "label form code (Spark read) + WWSSN table"
    on = dec["on_utc"] if dec else None
    off, off_note = off_utc(on, ans.get("off_date"), ans.get("off_time")) if on else (None, None)
    geo = geom.get(on, off) if on else None
    clock = Clock(geo, on) if on else None
    events = place_events(station, clock) if station and clock else []
    return {"dec": dec, "station": station, "station_source": src, "on": on, "off": off, "off_note": off_note,
            "geo": geo, "clock": clock, "events": events}


def validate(ans, fin):
    if not isinstance(ans, dict):
        return ["the reply is not one JSON object"]
    errs = [f"missing key {k}" for k in ("border_digits", "label_station", "label_component", "magnification",
                                          "off_date", "off_time", "main_event_id", "notes") if k not in ans]
    bd = ans.get("border_digits")
    if bd is not None and (not isinstance(bd, str) or fin["on"] is None):
        errs.append("border_digits must be the printed digits as a string that decodes to an ON date and time, or null")
    if ans.get("label_component") not in (None, "Z", "N", "E"):
        errs.append("label_component must be Z, N, E or null")
    if ans.get("magnification") is not None and not isinstance(ans["magnification"], (int, float)):
        errs.append("magnification must be a number or null")
    if ans.get("off_date") is not None and not re.fullmatch(r"\d{4}-\d\d-\d\d", str(ans["off_date"])):
        errs.append("off_date must be YYYY-MM-DD or null")
    if ans.get("off_time") is not None and not re.fullmatch(r"\d{4}", str(ans["off_time"])):
        errs.append("off_time must be HHMM or null")
    ids = [e["id"] for e in fin["events"]]
    if ans.get("main_event_id") is not None and ans["main_event_id"] not in ids:
        errs.append(f"main_event_id must be one of the catalog ids for the final station and ON time: {ids[:12]}")
    if ans.get("main_event_id") is None and ids:
        errs.append("main_event_id is null although the catalog lists events inside the record")
    return errs


def run_agent(chip, out, log):
    tools = Tools(chip, out)
    content = [{"type": "input_text", "text": USER}]
    content += [{"type": "input_image", "image_url": metaapi.png_data_url(im)} for im in images(chip)]
    history = [{"type": "message", "role": "user", "content": content}]
    base = {"model": metaapi.SPARK_MODEL, "instructions": INSTRUCTIONS, "tools": TOOLS}
    st = {"requests": 0, "tool_calls": 0, "repairs": 0, "usage": [], "answer": None, "errors": ["no final answer"],
          "tools": tools}
    for req in range(MAX_REQUESTS):
        t = time.time()
        resp = post({**base, "input": history})
        st["requests"] += 1
        st["usage"].append(resp.get("usage") or {})
        calls = [o for o in resp.get("output", []) if o.get("type") == "function_call"]
        log.append({"request": req, "secs": round(time.time() - t, 1), "function_calls": len(calls)})
        if calls:
            if req == MAX_REQUESTS - 1:
                st["errors"] = ["no final answer within the call budget"]
                break
            for c in calls:
                history.append({"type": "function_call", "call_id": c["call_id"], "name": c["name"],
                                "arguments": c.get("arguments") or "{}"})
                try:
                    args = json.loads(c.get("arguments") or "{}")
                    res = tools.run(c["name"], args) if st["tool_calls"] < MAX_TOOL_CALLS else {"error": "tool budget used up"}
                except Exception as ex:                           # noqa: BLE001 (only the type goes back to Spark:
                    args, res = {}, {"error": f"tool {c['name']} failed ({type(ex).__name__})"}   # messages can hold paths)
                    log.append({"local_only_error": f"{type(ex).__name__}: {str(ex)[:300]}"})
                st["tool_calls"] += 1
                log.append({"tool": c["name"], "args": args, "result": res})
                history.append({"type": "function_call_output", "call_id": c["call_id"], "output": json.dumps(res)})
            if st["tool_calls"] >= MAX_TOOL_CALLS or req == MAX_REQUESTS - 2:
                history.append({"type": "message", "role": "user", "content": [
                    {"type": "input_text", "text": "Tool budget reached: reply now with the final JSON object only."}]})
            continue
        text = " ".join(c.get("text", "") for o in resp.get("output", []) for c in (o.get("content") or [])
                        if c.get("type") in ("output_text", "text"))
        ans = parse(text)
        fin = finalize(ans, tools.geom)
        errs = validate(ans, fin)
        log.append({"final": text, "validation_errors": errs})
        st.update({"answer": ans, "errors": errs, "final": fin})
        if not errs or st["repairs"] >= MAX_REPAIRS or req == MAX_REQUESTS - 1:
            break
        st["repairs"] += 1
        history += [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]},
                    {"type": "message", "role": "user", "content": [{"type": "input_text", "text":
                     "Your JSON failed these checks: " + "; ".join(errs) + ". Reply with the corrected JSON object only."}]}]
    return st


def record(chip, st, secs):
    ans = st["answer"] if isinstance(st["answer"], dict) else {}
    fin = st.get("final") or finalize(ans, st["tools"].geom if st.get("tools") else Geometry(chip, OUT / "tmp"))
    dec, station, clock, geo = fin["dec"], fin["station"], fin["clock"], fin["geo"] or {}
    main = next((e for e in fin["events"] if e["id"] == ans.get("main_event_id")), None)
    flags = list(dec["problems"]) if dec else ["border digits not read"]
    if fin["station_source"] and fin["station_source"].startswith("label"):
        flags.append("station from the label form: the border station number was unreadable or not in the table")
    if station and ans.get("label_station") and str(ans["label_station"]).upper() != station["code"]:
        flags.append(f"label form reads station {ans['label_station']}, the border number gives {station['code']}")
    name = l0.parse_name(chip.stem)
    if name:
        got = {"station": station["code"] if station else None, "on": fin["on"][:16] if fin["on"] else None,
               "channel": (dec or {}).get("component")}
        want = {"station": name["station"], "on": name["on"][:16],
                "channel": {"LHZ": "LPZ", "LHN": "LPN", "LHE": "LPE", "SHZ": "SPZ"}.get(name["channel"], name["channel"])}
        flags += [f"file name says {k} {want[k]}, the chip gives {got[k]}" for k in want if got[k] != want[k]]
    if fin["off_note"]:
        flags.append(fin["off_note"])
    measured = st["tools"].measured_off if st.get("tools") else ()
    if measured and measured[0] != fin["off"]:
        flags.append(f"Spark measured the record with OFF {measured[0]} but answered OFF {fin['off']}; the record uses the "
                     "answer, so its lines may differ from those Spark saw when it chose the main event")
    tm = geo.get("time") or {}
    if tm.get("P_from_off") is not None and abs(tm["P_from_off"] - tm["P"]) > 5:
        flags.append(f"OFF and the last line's trace end imply a line period of {tm['P_from_off']:.1f} s, the minute "
                     f"marks {tm['P']:.1f} s: line count or OFF may be wrong")
    if clock is not None and not clock.ok:
        flags.append(f"no time model: {clock.reason}; events listed without lines")
    if clock is not None and clock.ambiguous:
        flags.append("ON minute ambiguous (+-1 min, research/40)")
    u = st["usage"]
    tin, tout = sum(x.get("input_tokens", 0) for x in u), sum(x.get("output_tokens", 0) for x in u)
    return {
        "chip": chip.name, "status": "ok" if not st["errors"] else "agent_invalid",
        "station": None if station is None else {**{k: station.get(k) for k in ("code", "number", "lat", "lon", "elev_m", "place")},
                                                 "source": fin["station_source"]},
        "component": None if not dec else {"code": dec["component_code"], "name": dec["component"], "source": "film border"},
        "on_utc": fin["on"], "on_source": "film border (read by Muse Spark, decoded by tool)" if fin["on"] else None,
        "off": {"date": ans.get("off_date"), "time": ans.get("off_time"), "source": "label form (read by Muse Spark)",
                "used_by_L0": fin["off"]},
        "magnification": ans.get("magnification"), "border_digits": ans.get("border_digits"),
        "label": {"station": ans.get("label_station"), "component": ans.get("label_component")},
        "record_utc": None if clock is None else [str(clock.start)[:19], str(clock.end)[:19]],
        "time_model": {k: tm.get(k) for k in ("T0", "P", "pps_r", "x_on", "minute_ambiguous", "reason", "P_from_off")}
                      | {k: geo.get(k) for k in ("n_lines", "xa", "x0", "x1", "failed", "l0_dir")},
        "main_event": None if main is None else {**main, "chosen_by": "Muse Spark (from the catalog tool's list)"},
        "events": fin["events"][:10], "flags": flags, "notes": ans.get("notes"),
        "l1": {"model": metaapi.SPARK_MODEL, "requests": st["requests"], "tool_calls": st["tool_calls"],
               "repairs": st["repairs"], "validation_errors": st["errors"], "input_tokens": tin, "output_tokens": tout,
               "cost_usd": round(tin * PRICE["input"] + tout * PRICE["output"], 5), "seconds": round(secs, 1),
               "table_sha256": sha_file(TABLE_FILE), "code_sha256": sha_file(__file__)},
    }


def l1(chip, out):
    """the L1 stage for one chip (L0 runs inside, with the ON and OFF read) -> out/record.json, l1_transcript.json"""
    chip = guard(chip)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    t, log = time.time(), []
    try:
        st = run_agent(chip, out, log)
    except Exception as ex:                                       # noqa: BLE001 (API down: recorded, not hidden)
        st = {"requests": 0, "tool_calls": 0, "repairs": 0, "usage": [], "answer": None,
              "errors": [f"agent error {type(ex).__name__}: {str(ex)[:300]}"]}
    rec = record(chip, st, time.time() - t)
    (out / "l1_transcript.json").write_text(json.dumps({"instructions": INSTRUCTIONS, "tools": TOOLS,
                                                        "user": USER, "log": log}, indent=1, default=str))
    tmp = out / "record.json.tmp"                                 # last and atomic: a record.json is always whole
    tmp.write_text(json.dumps(rec, indent=1, default=str))
    os.replace(tmp, out / "record.json")
    me = rec["main_event"] or {}
    print(f"[{chip.stem}] {rec['status']} | border {rec['border_digits']} -> {(rec['station'] or {}).get('code')} "
          f"{rec['on_utc']} {(rec['component'] or {}).get('name')} | main {me.get('id')} M{me.get('mag')} "
          f"{me.get('dist_deg')} deg P line {me.get('P_line')} surface lines {me.get('surface_lines')} | "
          f"{rec['l1']['requests']} requests {rec['l1']['tool_calls']} tools ${rec['l1']['cost_usd']} "
          f"{rec['l1']['seconds']} s | flags {len(rec['flags'])}", flush=True)
    return rec


def cmd_run(chip, out=None):
    chip = guard(chip)
    return l1(chip, Path(out or OUT / "run" / chip.stem))


# ---------- evaluation (research/44 §4, §6) ----------
def got_fields(station_code, on, component):
    return {"station": station_code.upper() if isinstance(station_code, str) else None, "on_date": on[:10] if on else None,
            "on_time": on[11:16].replace(":", "") if on else None, "component": component[2] if component else None}


def tally(rows, key):
    out = {}
    for f in FIELDS + ("all4",):
        k = sum(r[key][f] for r in rows)
        out[f] = {"correct": k, "n": len(rows), "rate": round(k / len(rows), 3), "wilson95": wilson(k, len(rows))}
    return out


def score_rows(got, truth):
    ok = {f: got[f] == truth[f] for f in FIELDS}
    ok["all4"] = all(ok.values())
    return ok


def cmd_replay():
    """research/44 §4: E4's 30 stored border reads through decode_border + table, no new question"""
    e4 = json.loads((ROOT / "out" / "e4" / "e4_results.json").read_text())
    rows = []
    for r in e4["rows"]:
        read = r.get("read") or {}
        dec = decode_border(read["border_digits"]) if isinstance(read.get("border_digits"), str) else None
        if dec and dec["station"]:
            sta, src = dec["station"]["code"], "border number + table"
        else:
            sta, src = read.get("station"), "label read (E4)"
        got = got_fields(sta, dec["on_utc"] if dec else None, dec["component"] if dec else None)
        truth = {f: r["truth"][f] for f in FIELDS}
        rows.append({"chip": r["chip"], "border_digits": read.get("border_digits"), "station_source": src,
                     "problems": dec["problems"] if dec else ["no border digits"], "got": got, "truth": truth,
                     "ok": score_rows(got, truth), "ok_chip": score_rows(got, truth | CHIP_FIX.get(r["chip"], {}))})
    res = {"what": "research/44 §4 replay: E4 stored border reads -> decode_border + WWSSN table (no API)",
           "table_sha256": sha_file(TABLE_FILE), "code_sha256": sha_file(__file__),
           "filename_truth": tally(rows, "ok"), "chip_truth": tally(rows, "ok_chip"),
           "e4_spark_alone": {f: e4["summary"][f] for f in ("station", "on_date", "on_time", "component", "all4")},
           "rows": rows}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "replay_e4.json").write_text(json.dumps(res, indent=1))
    for key in ("filename_truth", "chip_truth"):
        print(key, {f: f"{v['correct']}/{v['n']}" for f, v in res[key].items()})
    print("E4 Spark alone", {f: f"{v['correct']}/{v['n']}" for f, v in res["e4_spark_alone"].items()})
    for r in rows:
        if not r["ok"]["all4"]:
            print(" ", r["chip"], r["border_digits"], r["station_source"], r["got"], r["problems"])


def widest_line(geo):
    """the full line (not line 0, not the last) with the largest L0 trace spread (research/40 trace_paths), no ML"""
    s = {int(k): v for k, v in (geo.get("line_spread") or {}).items() if 0 < int(k) < geo["n_lines"] - 1}
    return max(s, key=s.get) if s else None


def cmd_test():
    """research/44 §6 (E7): the frozen L0 + L1 on the new public chips, once"""
    out = OUT / "e7"
    marker = out / "started.json"
    if marker.exists():
        raise SystemExit(f"E7 already started at {json.loads(marker.read_text())['started']}: one run only (research/44 §6)")
    chips = sorted(E7_CHIPS.glob("*.TIFF"))
    out.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({"started": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "chips": [c.name for c in chips],
                                  "code_sha256": sha_file(__file__), "table_sha256": sha_file(TABLE_FILE),
                                  "doc_sha256": sha_file(DOC44) if DOC44.exists() else None}, indent=1))
    with ThreadPoolExecutor(4) as ex:                             # L0 runs inside each agent's measure_record call
        list(ex.map(lambda c: l1(c, out / c.stem), chips))
    cmd_score()


def cmd_score():
    out = OUT / "e7"
    rows = []
    for d in sorted(p for p in out.iterdir() if (p / "record.json").exists()):
        rec = json.loads((d / "record.json").read_text())
        geo = json.loads((d / rec["time_model"]["l0_dir"] / "geometry.json").read_text()) if rec["time_model"].get("l0_dir") else {}
        name = l0.parse_name(d.name)
        truth = {"station": name["station"], "on_date": name["on"][:10], "on_time": name["on"][11:13] + name["on"][14:16],
                 "component": name["channel"][2]}
        got = got_fields((rec["station"] or {}).get("code"), rec["on_utc"], (rec["component"] or {}).get("name"))
        me = rec["main_event"]
        sec = None
        if me and me.get("surface_lines_pm1") and (geo.get("time") or {}).get("T0") is not None and not geo.get("failed"):
            w = widest_line(geo)
            lo, hi = me["surface_lines_pm1"]
            sec = {"widest_line": w, "surface_lines_pm1": [lo, hi], "inside": w is not None and lo <= w <= hi,
                   "P_line": me.get("P_line"), "main_event": me["id"], "mag": me.get("mag"), "dist_deg": me.get("dist_deg")}
        rows.append({"chip": d.name, "status": rec["status"], "got": got, "truth": truth, "ok": score_rows(got, truth),
                     "time_model": (geo.get("time") or {}).get("T0") is not None, "l0_failed": geo.get("failed"),
                     "secondary": sec, "flags": rec["flags"], "l1": rec["l1"]})
    n = len(rows)
    sec = [r["secondary"] for r in rows if r["secondary"]]
    secs = sorted(r["l1"]["seconds"] for r in rows)
    res = {"what": "research/44 §6 E7: frozen L0 + L1 on new public LHZ chips; truth = EarthScope file names",
           "n": n, "fields": tally(rows, "ok"),
           "agent_valid": {"k": sum(r["status"] == "ok" for r in rows), "n": n,
                           "wilson95": wilson(sum(r["status"] == "ok" for r in rows), n)},
           "within_budget": sum(r["l1"]["requests"] <= MAX_REQUESTS and r["l1"]["tool_calls"] <= MAX_TOOL_CALLS for r in rows),
           "cost_usd": round(sum(r["l1"]["cost_usd"] for r in rows), 4),
           "seconds_median": statistics.median(secs) if n else None,
           "time_model_chips": sum(r["time_model"] for r in rows),
           "secondary_widest_line_in_window": {"k": sum(s["inside"] for s in sec), "n": len(sec)},
           "rows": rows}
    (out / "e7_results.json").write_text(json.dumps(res, indent=1, default=str))
    print({f: f"{v['correct']}/{v['n']}" for f, v in res["fields"].items()}, "| valid", res["agent_valid"]["k"], "/", n,
          "| secondary", res["secondary_widest_line_in_window"], "| $", res["cost_usd"], "| median s", res["seconds_median"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "replay", "test", "score"])
    ap.add_argument("chip", nargs="?")
    ap.add_argument("--out")
    a = ap.parse_args()
    {"run": lambda: cmd_run(a.chip, a.out), "replay": cmd_replay, "test": cmd_test, "score": cmd_score}[a.cmd]()
