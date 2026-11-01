"""A1 name checks: verdicts on the mismatch chips from two blind transcriptions (readers saw neither the file name nor Spark's reads).

usage: python -I scripts/a1_name_checks.py <reader A json> <reader B json> <id -> chip map json>     # writes out/a1/name_checks.json

For each file-name field the prints on the chip are compared with the name. A source counts only where both readers agree.
  date:      the border digits (month first), the label's ON date, the earliest date stamp on the film
  time:      the border digits, the label's ON time
  component: the border code through the table (04 Z, 05 N, 06 E), the label
  station:   the label, the border number through the station table; also "same number as other chips filed under this station"
Field status: ok (a source supports the name, none contradicts), differs (a source contradicts, none supports, no reader saw support),
conflict (the chip's own prints disagree), unread (no agreed source). Border digits that match the name when read day first are
ambiguous and count only if another source settles the order.
Verdict: name_wrong = some field differs. l1_wrong = no field differs and every field where Spark's first read contradicted the name is ok.
unclear = the rest. The judges are Claude, not independent people, and name_wrong means the name differs from the prints, not which is right.
"""
import collections
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TABLE = json.loads((ROOT / "data" / "wwssn_stations" / "wwssn_stations.json").read_text())["stations"]
OUT = ROOT / "out" / "a1"
COMP = {"04": "Z", "05": "N", "06": "E", "01": "Z"}
MONTHS = {m: i + 1 for i, m in enumerate("JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC".split())}


def digits(s, n):
    s = re.sub(r"\D", "", str(s)) if s is not None else ""
    return s if len(s) == n else None


def mmddyy(s):
    """(month first, day first) from a written date; a date with a month name has one reading"""
    if not s or "?" in str(s):
        return None, None
    t = str(s).upper()
    m = re.search(r"(\d{1,2})\W*(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\w*\W*(\d{2,4})", t)
    if m:
        v = f"{MONTHS[m.group(2)]:02d}{int(m.group(1)):02d}{m.group(3)[-2:]}"
        return v, v
    m = re.search(r"(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\w*\W*(\d{1,2})\W+(\d{2,4})", t)
    if m:
        v = f"{MONTHS[m.group(1)]:02d}{int(m.group(2)):02d}{m.group(3)[-2:]}"
        return v, v
    m = re.search(r"(\d{1,2})\s*[-/.,\s]\s*(\d{1,2})\s*[-/.,\s]\s*(\d{2,4})", t)
    if m:
        a, b, y = int(m.group(1)), int(m.group(2)), m.group(3)[-2:]
        if a > 12 >= b:                                            # 28-06-68: only one reading
            return f"{b:02d}{a:02d}{y}", f"{b:02d}{a:02d}{y}"
        return f"{a:02d}{b:02d}{y}", f"{b:02d}{a:02d}{y}"
    return None, None


def earliest_stamp(stamps):
    vals = [mmddyy(s)[0] for s in (stamps or [])]
    vals = [v for v in vals if v and v.isdigit() and 1 <= int(v[:2]) <= 12]
    return min(vals, key=lambda v: (v[4:], v[:4])) if vals else None


def sources(reader, field, sib):
    """{source: value} for one reader"""
    b, lab = reader.get("border") or {}, reader.get("label") or {}
    if field == "date":
        return {"border": digits(b.get("date"), 6), "label": mmddyy(lab.get("on_date"))[0], "stamp": earliest_stamp(reader.get("date_stamps"))}
    if field == "time":
        return {"border": digits(b.get("time"), 4), "label": digits(lab.get("on_time"), 4)}
    if field == "component":
        c = digits(b.get("component_code"), 2)
        v = str(lab.get("component") or "").strip().upper()
        return {"border": COMP.get(c), "label": v[-1] if v else None}
    n = digits(b.get("station_number"), 4)
    v = str(lab.get("station") or "").strip().upper()
    return {"label": None if "?" in v else v or None, "number": (TABLE.get(n) or {}).get("code") if n else None, "same_number": sib(n)}


def judge(a, b, name, field, sib):
    sa, sb = sources(a, field, sib), sources(b, field, sib)
    want = name[field]
    agreed = {k: sa[k] for k in sa if sa[k] is not None and sa[k] == sb[k]}
    ambiguous = []
    if field == "date":
        for k in ("border", "label"):
            r = agreed.get(k)
            alt = r[2:4] + r[:2] + r[4:] if r else None
            if r and r != want and alt == want:
                ambiguous.append(k)
                del agreed[k]
    if field == "station" and agreed.get("same_number") != want:
        agreed.pop("same_number", None)                           # only ever supports the name, never contradicts it
    sup = sorted(k for k, v in agreed.items() if v == want)
    con = sorted(k for k, v in agreed.items() if v != want)
    seen = any(v == want for v in list(sa.values()) + list(sb.values()) if v is not None)
    if sup:
        st = "conflict" if con else "ok"
    elif con:
        st = "conflict" if seen else "differs"                    # a date stamp with a month name settles an ambiguous border
    else:
        st = "ambiguous" if ambiguous else "unread"
    return {"status": st, "name": want, "agreed": agreed, "supports_name": sup, "contradicts_name": con, "ambiguous": ambiguous, "reader_a": sa, "reader_b": sb}


def main(pa, pb, pm):
    ra = {r["id"]: r for r in json.loads(Path(pa).read_text())}
    rb = {r["id"]: r for r in json.loads(Path(pb).read_text())}
    idmap = json.loads(Path(pm).read_text())
    res = json.loads((OUT / "a1_results.json").read_text())
    sample = set(json.loads((OUT / "check_sample.json").read_text())["sample30"])
    first = {m["chip"]: m for m in res["mismatches"]}
    votes = collections.defaultdict(collections.Counter)            # station in the file name -> printed numbers, from Spark's reads of all chips
    for x in res["rows"]:
        g = (x["border_digits"] or "").split()
        if len(g) == 4 and len(g[2]) == 4:
            votes[x["chip"][:3]][g[2]] += 1
    rows = []
    for nn, chip in sorted(idmap.items()):
        p = chip.split("_")
        name = {"station": p[0], "date": p[2] + p[3] + p[1], "time": p[4], "component": "Z"}

        def sib(n):
            others = votes[p[0]].copy()
            g = (next(x for x in res["rows"] if x["chip"] == chip)["border_digits"] or "").split()
            if len(g) == 4:
                others[g[2]] -= 1                                  # not counting this chip itself
            return p[0] if n and others[n] > 0 else None
        fields = {f: judge(ra[nn], rb[nn], name, f, sib) for f in name}
        m = first[chip]
        tr = {"station": "station", "date": "on_date", "time": "on_time", "component": "component"}
        flagged = [f for f in fields if m["got"][tr[f]] is not None and m["got"][tr[f]] != m["truth"][tr[f]]]
        differs = [f for f in fields if fields[f]["status"] == "differs"]
        verdict = "name_wrong" if differs else "l1_wrong" if all(fields[f]["status"] == "ok" for f in flagged) else "unclear"
        rows.append({"chip": chip, "id": nn, "in_sample30": chip in sample, "verdict": verdict, "differs": differs, "spark_flagged": flagged,
                     "spark_first": m["got"], "fields": fields, "notes": [ra[nn].get("notes"), rb[nn].get("notes")]})
    tally = collections.Counter(r["verdict"] for r in rows)
    s30 = collections.Counter(r["verdict"] for r in rows if r["in_sample30"])
    out = {"what": "A1 visual check of the mismatch chips: two blind readers, Claude, not independent people", "n_checked": len(rows),
           "all": dict(tally), "sample30": dict(s30), "estimate_name_wrong": round(s30["name_wrong"] / 30 * len(rows), 1), "rows": rows}
    (OUT / "name_checks.json").write_text(json.dumps(out, indent=1))
    print(dict(tally), "sample30", dict(s30), "estimate", out["estimate_name_wrong"])
    for r in rows:
        print(r["chip"].ljust(23), r["verdict"].ljust(10), "S30" if r["in_sample30"] else "   ", {f: r["fields"][f]["status"] for f in r["fields"]})


if __name__ == "__main__":
    main(*sys.argv[1:4])
