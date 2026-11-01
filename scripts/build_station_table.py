"""WWSSN station-number table for L1 (research/44): border digit group 3 (station number) -> station code and coordinates.

usage: python -I build_station_table.py [out_dir]          (default out_dir: ../data/wwssn_stations)

Source: the IRIS Seismo-Archives pages "Seismograms of the <year> <place> Earthquake" (WWSSN chips scanned in 2004 by the
USGS/ICSU scanning project). Each page lists its stations as 'NNNN=CODE [lat, lon, elev m]: place' and links one PDF per
seismogram named MMDDYY_HHMM_NNNN_CC.pdf with the link text CODE_LPZ / CODE_LPN / CODE_LPE, i.e. the four printed border
groups of the film chip (research/42 §6.3). The pages are gone from ds.iris.edu; the Wayback Machine copies listed in
PAGES (1962-1972, the WWSSN era) are fetched once into <out_dir>/raw/ (not redistributed) and parsed with regular
expressions only. Writes <out_dir>/wwssn_stations.json: per number the code, coordinates, place and the pages that list
it; component codes with their link counts; every disagreement found (number<->code, link text vs header).
"""
import json
import re
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from html import unescape
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "data" / "wwssn_stations"
UA = {"User-Agent": "build_station_table/1.0 (research benchmark)"}
PAGES = [                                                         # (page, Wayback timestamp) from the CDX index, 10-29
    ("1962qazvin", "20060908043227"), ("1963skopje", "20060908042402"), ("1964alaska", "20060908075529"),
    ("1964niigata", "20060908042415"), ("1965puget_sound", "20060908043233"), ("1966gisborne", "20060908073833"),
    ("1966parkfield", "20060908043854"), ("1966uganda-zaire", "20060908073813"), ("1966varto", "20060908073709"),
    ("1967caracas", "20060908042405"), ("1967huila", "20060908043214"), ("1967koyna", "20060908042322"),
    ("1968dasht-e-bayaz", "20050729090343"), ("1968inangahua", "20060908054710"), ("1968sanmartin", "20060908044239"),
    ("1968tokachi-oki", "20060908042213"), ("1968valle_del_belice", "20060908042353"), ("1969sardo", "20060908073843"),
    ("1970dagestan", "20060908073658"), ("1970peru-ecudador", "20060908043221"), ("1970peru", "20060908053954"),
    ("1971sanfernando", "20060908042346"), ("1972managua", "20060908074017"),
]
ORIG = "http://www.iris.edu/seismo/quakes/{}/pdf/"
WAYBACK = "https://web.archive.org/web/{}id_/{}"                  # id_: the archived bytes, no Wayback banner
HEADER = re.compile(r"(\d{4})\s*=\s*([A-Z0-9]{2,5})\s*,?\s*\[\s*([0-9.]+)\s*([NS])\s*,\s*([0-9.]+)\s*([EW])\s*,"
                    r"\s*(-?[0-9.]+)\s*m\s*\]\s*:?\s*([^\n\r]{0,60})")
LINK = re.compile(r'href="(\d{6})_(\d{4})_(\d{4})_(\d\d)\.pdf"\s*>\s*([A-Z0-9]{2,5})\s*_\s*(LP[ZNE]|SP[ZNE])\s*<')


def fetch(name, ts):
    raw = OUT / "raw" / f"{name}.html"
    if raw.exists():
        return raw.read_text(encoding="utf-8", errors="replace")
    url = WAYBACK.format(ts, ORIG.format(name))
    for wait in (0, 30, 90, 180):
        time.sleep(wait or 3)
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                data = r.read()
            raw.parent.mkdir(parents=True, exist_ok=True)
            raw.write_bytes(data)
            return data.decode("utf-8", errors="replace")
        except Exception as e:                                    # the archive refuses bursts: back off and retry
            err = f"{type(e).__name__}: {e}"
            print(f"  {name}: {err} (retry)", flush=True)
    raise RuntimeError(f"{name}: {err}")


def text_of(page):
    page = re.sub(r"<script.*?</script>", "", page, flags=re.S)
    return re.sub(r"[ \t\xa0]+", " ", unescape(re.sub(r"<[^>]+>", " ", page)))


def main():
    rows, links, pages = defaultdict(list), [], []
    for name, ts in PAGES:
        page = fetch(name, ts)
        hdr = [m.groups() for m in HEADER.finditer(text_of(page))]
        lk = [m.groups() for m in LINK.finditer(page)]
        for num, code, la, ns, lo, ew, el, place in hdr:
            rows[num].append({"code": code, "lat": round(float(la) * (1 if ns == "N" else -1), 2),
                              "lon": round(float(lo) * (1 if ew == "E" else -1), 2), "elev_m": float(el),
                              "place": place.strip(), "page": name})
        links += [(name,) + g for g in lk]
        pages.append({"page": name, "url": ORIG.format(name), "wayback": WAYBACK.format(ts, ORIG.format(name)),
                      "stations": len(hdr), "seismogram_links": len(lk),
                      "unparsed_headers": len(re.findall(r"\b\d{4}\s*=\s*[A-Z]", text_of(page))) - len(hdr)})
        print(f"{name}: {len(hdr)} stations, {len(lk)} links", flush=True)
    stations, problems = {}, []
    for num in sorted(rows):
        r = rows[num]
        codes = Counter(x["code"] for x in r)
        coords = Counter((x["lat"], x["lon"], x["elev_m"]) for x in r)
        code, (lat, lon, el) = codes.most_common(1)[0][0], coords.most_common(1)[0][0]
        if len(codes) > 1:
            problems.append({"kind": "number with two codes", "number": num, "codes": dict(codes)})
        stations[num] = {"code": code, "lat": lat, "lon": lon, "elev_m": el,
                         "place": Counter(x["place"] for x in r).most_common(1)[0][0],
                         "pages": sorted({x["page"] for x in r}),
                         "coord_variants": [list(c) for c in coords] if len(coords) > 1 else []}
    by_code = defaultdict(set)
    for num, s in stations.items():
        by_code[s["code"]].add(num)
    problems += [{"kind": "code with two numbers", "code": c, "numbers": sorted(n)} for c, n in by_code.items() if len(n) > 1]
    comp = defaultdict(Counter)
    for page, mmddyy, hhmm, num, cc, code, cha in links:
        comp[cc][cha] += 1
        if num in stations and stations[num]["code"] != code:
            problems.append({"kind": "link text vs header", "page": page, "number": num, "link_code": code,
                             "header_code": stations[num]["code"]})
        if num not in stations:
            problems.append({"kind": "link number without header", "page": page, "number": num, "link_code": code})
    out = {"what": "WWSSN station number (film border group 3) -> station code, coordinates; component code (group 4)",
           "source": "IRIS Seismo-Archives earthquake seismogram pages (USGS/ICSU WWSSN scanning project, 2004), "
                     "Wayback Machine copies; parsed by scripts/build_station_table.py",
           "made": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "pages": pages,
           "component_codes": {cc: dict(c) for cc, c in sorted(comp.items())},
           "stations": stations, "problems": problems}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "wwssn_stations.json").write_text(json.dumps(out, indent=1))
    print(f"{len(stations)} station numbers from {len(pages)} pages, {len(links)} seismogram links; component codes "
          f"{ {cc: dict(c) for cc, c in comp.items()} }; problems {len(problems)} -> {OUT / 'wwssn_stations.json'}")


if __name__ == "__main__":
    main()
