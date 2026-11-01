"""A1: the frozen L0 + L1 on about 300 new public LHZ chips, in draw order, unattended. Stored results: results/a1/.
The chips are not in this repository; `fetch_e4_chips.py data/a1_chips --seed 20261101 --n 300 --per-station 4 --max-gb 14` with
--exclude-index for the E4 and E7 lists draws them (results/a1/README.md).

usage: MODEL_API_KEY=... python -I scripts/a1_batch.py run [--workers 4]   # data/a1_chips -> out/a1/<chip>/
       python -I scripts/a1_batch.py score                                 # out/a1/a1_results.json (no API)
       MODEL_API_KEY=... python -I scripts/a1_batch.py reread              # second L1 read of every mismatch: out/a1/reread/

One run: the first start writes out/a1/started.json with the code, table and document hashes; a later start refuses if any
hash changed and otherwise only finishes what is missing, so a power cut does not spoil the sample. A chip whose agent
never produced an answer because the API or network failed ("agent error"), or whose record.json cannot be read, is run
again and the retry is reported; any other outcome is final. Ten agent errors in a row stop the run (network or API down) so
the remaining chips are not spent. A run that ends with chips left says which and exits 1; rerun it.
Only chips with a public CC0 EarthScope sidecar reach the API (l1_context.guard).
"""
import argparse
import json
import re
import shutil
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import l0_geometry as l0                                          # noqa: E402
import l1_context as l1c                                          # noqa: E402
from l1_context import wilson                                     # noqa: E402

CHIPS = ROOT / "data" / "a1_chips"
OUT = ROOT / "out" / "a1"
DOC = ROOT.parent / "53-l0-polarity-a1.md"                         # internal notes, not published; hashed only if present
ERR_STOP = 10
PERIOD_FLAG = "imply a line period"


def hashes():
    sha = l1c.sha_file
    return {"l0_geometry": sha(HERE / "l0_geometry.py"), "l1_context": sha(HERE / "l1_context.py"),
            "real_paired_eval": sha(HERE / "real_paired_eval.py"), "table": sha(l1c.TABLE_FILE),
            "index": sha(CHIPS / "index.json"), "a1_batch": sha(Path(__file__)), "e4_spark_labels": sha(HERE / "e4_spark_labels.py"),
            "metaapi": sha(HERE / "metaapi.py"), "track_joint": sha(HERE / "track_joint.py")}


def agent_error(rec):
    return any(str(e).startswith("agent error") for e in (rec["l1"].get("validation_errors") or []))


def read_record(p):
    """None if the file is missing or cannot be read"""
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return None


def done(chip):
    rec = read_record(OUT / chip.stem / "record.json")
    return rec is not None and not agent_error(rec)


def cmd_run(workers):
    index = json.loads((CHIPS / "index.json").read_text())
    chips = [CHIPS / p["file"] for p in index["picks"]]            # draw order
    OUT.mkdir(parents=True, exist_ok=True)
    marker = OUT / "started.json"
    if marker.exists():
        first = json.loads(marker.read_text())
        if first["hashes"] != hashes():
            raise SystemExit(f"code changed since the start: {first['hashes']} vs {hashes()}")
    else:
        if DOC.exists():
            shutil.copy(DOC, OUT / "doc53_at_freeze.md")
        marker.write_text(json.dumps({"started": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "n": len(chips), "workers": workers,
                                      "hashes": hashes(), "doc_sha256": l1c.sha_file(DOC) if DOC.exists() else None}, indent=1))
    retries = OUT / "retries.jsonl"
    todo = [c for c in chips if not done(c)]
    print(f"{len(chips) - len(todo)} done, {len(todo)} to run", flush=True)
    streak = 0
    with ThreadPoolExecutor(workers) as ex:
        pending = iter(todo)
        running = {}

        def submit():
            c = next(pending, None)
            if c is None:
                return False
            old = OUT / c.stem / "record.json"
            if old.exists():                                       # an earlier agent error: keep its message, run again
                prev = read_record(old)
                with retries.open("a") as f:
                    f.write(json.dumps({"chip": c.stem, "at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
                                        "error": prev["l1"]["validation_errors"] if prev else "record.json could not be read"}) + "\n")
                old.unlink()
            running[ex.submit(l1c.l1, c, OUT / c.stem)] = c
            return True
        for _ in range(workers):
            submit()
        while running:
            for f in as_completed(list(running)):
                c = running.pop(f)
                try:
                    rec = f.result()
                    streak = streak + 1 if agent_error(rec) else 0
                except Exception as e:                             # noqa: BLE001 (a crash in record(): logged, retried on resume)
                    print("CRASH", c.stem, repr(e)[:300], flush=True)
                    streak += 1
                if streak >= ERR_STOP:
                    print(f"{ERR_STOP} agent errors in a row: stopping, resume later", flush=True)
                    pending = iter(())
                else:
                    submit()
                break
    left = [c.stem for c in chips if not done(c)]
    if left:
        print(f"{len(left)} chips have no final record (rerun `run`): {left[:20]}", flush=True)
        raise SystemExit(1)
    cmd_score()


def cmd_score():
    index = json.loads((CHIPS / "index.json").read_text())
    rows = []
    for p in index["picks"]:
        d = OUT / p["file"][:-5]
        if not (d / "record.json").exists():
            continue
        rec = json.loads((d / "record.json").read_text())
        l0_dir = rec["time_model"].get("l0_dir")
        geo = json.loads((d / l0_dir / "geometry.json").read_text()) if l0_dir and (d / l0_dir / "geometry.json").exists() else {}
        name = l0.parse_name(d.name)
        truth = {"station": name["station"], "on_date": name["on"][:10], "on_time": name["on"][11:13] + name["on"][14:16],
                 "component": name["channel"][2]}
        got = l1c.got_fields((rec["station"] or {}).get("code"), rec["on_utc"], (rec["component"] or {}).get("name"))
        ok = l1c.score_rows(got, truth)
        m, t = geo.get("marks") or {}, geo.get("time") or {}
        me = rec["main_event"]
        sec = None
        if me and me.get("surface_lines_pm1") and t.get("T0") is not None and not geo.get("failed"):
            w = l1c.widest_line(geo)
            lo, hi = me["surface_lines_pm1"]
            sec = {"widest_line": w, "window": [lo, hi], "inside": w is not None and lo <= w <= hi, "P_line": me.get("P_line")}
        rows.append({"rank": p["rank"], "chip": d.name, "status": rec["status"], "got": got, "truth": truth, "ok": ok,
                     "border_digits": rec["border_digits"], "label": rec["label"], "off": rec["off"],
                     "l0": {"ran": bool(geo), "failed": geo.get("failed"), "negative": geo.get("negative"), "n_lines": geo.get("n_lines"),
                            "n_expect": geo.get("n_expect"), "marks_found": m.get("found"), "minutes_per_line": m.get("minutes_per_line"),
                            "format_ok": m.get("format_ok"), "P": m.get("P"), "P_from_off": t.get("P_from_off"),
                            "time_model": t.get("T0") is not None, "reason": t.get("reason")},
                     "period_flag": any(PERIOD_FLAG in f for f in rec["flags"]), "secondary": sec, "flags": rec["flags"],
                     "requests": rec["l1"]["requests"], "tool_calls": rec["l1"]["tool_calls"], "repairs": rec["l1"]["repairs"],
                     "validation_errors": rec["l1"]["validation_errors"], "cost_usd": rec["l1"]["cost_usd"], "seconds": rec["l1"]["seconds"]})
    n = len(rows)
    if not n:
        raise SystemExit("no records")

    def share(k, of):
        return {"k": k, "n": of, "rate": round(k / of, 4) if of else None, "wilson95": wilson(k, of) if of else None}
    valid = [r for r in rows if r["status"] == "ok"]

    def conflict(r):
        """L1 gave a value and it differs from the file name. A field L1 could not give (a station not in the table) is no conflict"""
        return any(r["got"][f] is not None and not r["ok"][f] for f in l1c.FIELDS)
    mismatch = [r for r in valid if conflict(r)]
    unread = [r for r in valid if not conflict(r) and not r["ok"]["all4"]]
    ran = [r for r in rows if r["l0"]["ran"]]
    failed = [r for r in ran if r["l0"]["failed"]]
    no_marks = [r for r in ran if not r["l0"]["failed"] and not r["l0"]["marks_found"]]
    bad_format = [r for r in ran if r["l0"]["marks_found"] and r["l0"]["format_ok"] is False]
    fmt_ok = [r for r in ran if r["l0"]["marks_found"] and r["l0"]["format_ok"]]
    tm = [r for r in ran if r["l0"]["time_model"]]
    p_near = [r for r in fmt_ok if r["l0"]["P"] is not None and abs(r["l0"]["P"] - 3588) <= 1]
    checkable = [r for r in tm if r["l0"]["P_from_off"] is not None]
    sec = [r["secondary"] for r in rows if r["secondary"]]
    retried = (OUT / "retries.jsonl").read_text().count("\n") if (OUT / "retries.jsonl").exists() else 0
    secs = [r["seconds"] for r in rows]
    res = {"what": "A1: frozen L0 + L1 on new public LHZ chips; development numbers do not carry over",
           "n_scored": n, "n_drawn": len(index["picks"]), "stations": len({r["chip"][:3] for r in rows}),
           "S1_unattended_valid_answer": share(len(valid), n),
           "S1_within_budget": sum(r["requests"] <= l1c.MAX_REQUESTS and r["tool_calls"] <= l1c.MAX_TOOL_CALLS for r in rows),
           "S1_api_retries": retried,
           "P1_file_name_mismatch_among_valid": share(len(mismatch), len(valid)),
           "P1_field_not_given_among_valid": share(len(unread), len(valid)),
           "P1_field_not_given_chips": [{"chip": r["chip"], "fields": [f for f in l1c.FIELDS if r["got"][f] is None]} for r in unread],
           "fields_vs_file_name_all_rows": l1c.tally(rows, "ok"),
           "S2_l0_ran": len(ran), "S2_layout_failed": share(len(failed), len(ran)), "S2_marks_not_found": share(len(no_marks), len(ran)),
           "S2_unsupported_format": share(len(bad_format), len(ran)), "S2_format_ok": share(len(fmt_ok), len(ran)),
           "S2_time_model": share(len(tm), len(ran)), "S2_P_3588_pm1_among_format_ok": share(len(p_near), len(fmt_ok)),
           "S2_negative_read": sum(bool(r["l0"]["negative"]) for r in ran),
           "S2_negative_chips": [r["chip"] for r in ran if r["l0"]["negative"]],
           "S3_period_flag_among_time_model": share(sum(r["period_flag"] for r in tm), len(tm)),
           "S3_period_flag_among_checkable": share(sum(r["period_flag"] for r in checkable), len(checkable)),
           "S4_widest_line_in_window": share(sum(s["inside"] for s in sec), len(sec)),
           "cost_usd": round(sum(r["cost_usd"] for r in rows), 3), "seconds_median": statistics.median(secs),
           "mismatches": [{k: r[k] for k in ("chip", "got", "truth", "border_digits", "label", "flags")} for r in mismatch],
           "rows": rows}
    (OUT / "a1_results.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({k: v for k, v in res.items() if k not in ("rows", "mismatches", "fields_vs_file_name_all_rows")}, indent=1, default=str))
    print("mismatches:", [(r["chip"], {k: r["got"][k] for k in l1c.FIELDS if not r["ok"][k]}) for r in mismatch])


def cmd_reread():
    res = json.loads((OUT / "a1_results.json").read_text())

    def second(chip):
        return read_record(OUT / "reread" / chip / "record.json")
    todo = [r["chip"] for r in res["mismatches"] if second(r["chip"]) is None or agent_error(second(r["chip"]))]
    with ThreadPoolExecutor(4) as ex:
        futs = {ex.submit(l1c.l1, CHIPS / f"{c}.TIFF", OUT / "reread" / c): c for c in todo}
        for f in as_completed(futs):
            try:
                f.result()
            except Exception as e:                                 # noqa: BLE001 (reported below as no second read)
                print("CRASH", futs[f], repr(e)[:300], flush=True)
    rows = []
    for r in res["mismatches"]:
        rec = second(r["chip"])
        if rec is None:
            rows.append({"chip": r["chip"], "first": r["got"], "second": None, "truth": r["truth"], "status": "no second read"})
            continue
        got = l1c.got_fields((rec["station"] or {}).get("code"), rec["on_utc"], (rec["component"] or {}).get("name"))
        rows.append({"chip": r["chip"], "first": r["got"], "second": got, "truth": r["truth"], "reads_agree": got == r["got"],
                     "second_matches_name": got == r["truth"], "border_first": r["border_digits"], "border_second": rec["border_digits"],
                     "label_second": rec["label"], "status": rec["status"], "cost_usd": rec["l1"]["cost_usd"]})
    (OUT / "a1_reread.json").write_text(json.dumps(rows, indent=1, default=str))
    print(len(rows), "mismatches; reads agree", sum(r.get("reads_agree", False) for r in rows), "; second read matches the name",
          sum(r.get("second_matches_name", False) for r in rows), "; no second read", sum(r["second"] is None for r in rows),
          "; $", round(sum(r.get("cost_usd", 0) for r in rows), 3))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "score", "reread"])
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    {"run": lambda: cmd_run(a.workers), "score": cmd_score, "reread": cmd_reread}[a.cmd]()
