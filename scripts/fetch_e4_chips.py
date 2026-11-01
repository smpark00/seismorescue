"""Draw and fetch the E4 benchmark sample: 30 random WWSSN long-period vertical (LHZ) film chips from EarthScope.

usage: python -I fetch_e4_chips.py [out_dir]          (default out_dir: ../data/e4_chips)
       python -I fetch_e4_chips.py ../data/e7_chips --seed 20261030 --n 20 --per-station 1 \
              --exclude-index ../data/e4_chips/index.json --purpose "..."     (E7, research/44 §6)
The defaults are the E4 draw (research/42); the options only serve later draws.

Enumerates every *_LHZ.TIFF in the EarthScope Film Chip HTML directory listings (data/ -> data/<STA>/), drops
MAT_73_12_28_0020_LHZ and station KIP, sorts, shuffles with numpy.random.default_rng(20261029) and walks the shuffled
list, taking a file unless its station already has 2 picks, until 30. A file over 80 MB, or one that fails twice, is
skipped and the walk goes on. Per pick it writes <FILE>.TIFF, <FILE>.earthscope.json (EarthScope metadata) and a
<FILE>.json sidecar (provenance + filename_truth), then index.json and lhz_names.txt (the enumerated names, for audit).
No image is opened or decoded: only the byte count (vs Content-Length) and the TIFF magic bytes are checked.
Re-runnable: complete picks are not downloaded again and skip decisions recorded in index.json are kept; if the live
listing no longer matches the recorded one it stops instead of redrawing.
"""
import argparse
import datetime
import hashlib
import json
import re
import sys
import time
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("out_dir", nargs="?", type=Path, default=HERE.parent / "data" / "e4_chips")
ap.add_argument("--seed", type=int, default=20261029)
ap.add_argument("--n", type=int, default=30)
ap.add_argument("--per-station", type=int, default=2)
ap.add_argument("--exclude-index", type=Path, action="append", default=[], help="index.json of an earlier draw: its picks are left out")
ap.add_argument("--max-gb", type=float, default=1.5, help="download budget in decimal GB")
ap.add_argument("--purpose", default="E4 benchmark: 30 random WWSSN long-period vertical (LHZ) film-chip scans; "
                                     "filenames are the ground truth")
ARGS = ap.parse_args()
OUT = ARGS.out_dir
MAT_SIDECAR = HERE.parent / "data" / "real_scans" / "MAT_73_12_28_0020_LHZ.json"      # source of citation + licence block
BASE = "https://data.earthscope.org/archive/seismology/products/film_chip/data/"
SEED, N_PICKS, PER_STATION = ARGS.seed, ARGS.n, ARGS.per_station
MAX_FILE, MAX_TOTAL = 80 * 10**6, ARGS.max_gb * 10**9                                 # bytes (decimal MB / GB)
EXCLUDE_FILES, EXCLUDE_STATIONS = {"MAT_73_12_28_0020_LHZ"}, {"KIP"}
EXCLUDE_FILES |= {p["file"][:-5] for f in ARGS.exclude_index for p in json.loads(f.read_text())["picks"]}
NAME = re.compile(r"^([A-Z]{3})_(\d\d)_(\d\d)_(\d\d)_(\d{4})_([A-Z]{3})\.TIFF$")       # STA_YY_MM_DD_HHMM_CHA.TIFF
MAGIC = (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+")                              # TIFF / BigTIFF, either byte order
UA = {"User-Agent": "fetch_e4_chips/1.0 (research benchmark)"}
REF = json.loads(MAT_SIDECAR.read_text())                                             # product_citation + licence, copied as is


class Skip(Exception):
    """Candidate not taken (over the size limit, or failed twice); the walk goes on with the next one."""


def get(url, check=lambda body: None, tries=4):
    """Whole small body (listing page, metadata JSON): checked against Content-Length and `check`, retried."""
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=90) as r:
                body, want = r.read(), r.headers.get("Content-Length")
            if want is not None and len(body) != int(want):
                raise OSError(f"{len(body)} of {want} bytes")
            check(body)
            return body
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(2 * (k + 1))


def is_html(body):
    if not body.rstrip().endswith(b"</html>"):
        raise ValueError("incomplete listing page")


def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def enumerate_tiffs():
    """(station codes, {station: sorted *.TIFF names of every channel}) from the HTML directory listings."""
    top = get(BASE, is_html).decode()
    stations = sorted(set(re.findall(r'<a href="' + re.escape(BASE) + r'([^/"?]+)/">', top)))

    def tiffs(sta):
        page = get(f"{BASE}{sta}/", is_html).decode()
        return sorted(set(re.findall(r'<a href="' + re.escape(f"{BASE}{sta}/") + r'([^"?/]+\.TIFF)\?type=file">\1</a>', page)))

    with ThreadPoolExecutor(4) as ex:
        return stations, dict(zip(stations, ex.map(tiffs, stations)))


def have(name):
    """Sidecar of an already complete pick (all three files present, TIFF bytes and sha256 as recorded), else None."""
    stem = name[:-5]
    tif, side = OUT / name, OUT / f"{stem}.json"
    if not (tif.exists() and side.exists() and (OUT / f"{stem}.earthscope.json").exists()):
        return None
    info = json.loads(side.read_text())
    return info if tif.stat().st_size == info["bytes"] and sha256_file(tif) == info["sha256"] else None


def fetch(name):
    """Download TIFF + metadata JSON of `name` into OUT (one retry); write the sidecar and return it. Raises Skip."""
    stem, sta = name[:-5], name[:3]
    url = f"{BASE}{sta}/{stem}"
    part = OUT / f"{name}.part"
    for attempt in (1, 2):
        try:
            meta = get(url + ".json", json.loads)
            with urllib.request.urlopen(urllib.request.Request(url + ".TIFF", headers=UA), timeout=90) as r:
                want = int(r.headers["Content-Length"])
                if want > MAX_FILE:
                    raise Skip(f"over {MAX_FILE // 10**6} MB ({want} bytes)")
                h, got = hashlib.sha256(), 0
                with part.open("wb") as f:
                    while chunk := r.read(1 << 20):
                        f.write(chunk)
                        h.update(chunk)
                        got += len(chunk)
            with part.open("rb") as f:
                magic = f.read(4)
            if got != want or magic not in MAGIC:
                raise OSError(f"got {got} of {want} bytes, header {magic!r}")
            break
        except Skip:
            part.unlink(missing_ok=True)
            raise
        except Exception as e:
            part.unlink(missing_ok=True)
            if attempt == 2:
                raise Skip(f"download failed twice: {e!r}")
            print("retry", name, repr(e), flush=True)
            time.sleep(3)
    part.replace(OUT / name)
    (OUT / f"{stem}.earthscope.json").write_bytes(meta)
    _, yy, mo, dd, hm, cha = NAME.match(name).groups()
    side = {
        "file": name,
        "downloaded": datetime.date.today().isoformat(),
        "source_url": url + ".TIFF",
        "source_metadata_url": url + ".json",
        "product_citation": REF["product_citation"],
        "licence": REF["licence"],
        "sha256": h.hexdigest(),
        "bytes": got,
        "filename_truth": {"station": sta, "date": f"19{yy}-{mo}-{dd}", "on_time_hhmm": hm, "channel": cha},
    }
    (OUT / f"{stem}.json").write_text(json.dumps(side, indent=2) + "\n")
    return side


# 1. enumerate every *_LHZ.TIFF
stations, tiffs = enumerate_tiffs()
lhz = {s: [n for n in tiffs[s] if n.endswith("_LHZ.TIFF")] for s in stations}
all_lhz = sorted(n for s in stations for n in lhz[s])
bad = [n for s in stations for n in lhz[s] if not NAME.match(n) or n[:3] != s]
if bad:
    sys.exit(f"LHZ names not in STA_YY_MM_DD_HHMM_LHZ.TIFF form or outside their station directory: {bad[:5]}")
names_text = "\n".join(all_lhz) + "\n"
names_sha = hashlib.sha256(names_text.encode()).hexdigest()
fetched_utc = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
print(f"{len(all_lhz)} LHZ files in {sum(1 for s in stations if lhz[s])} of {len(stations)} station directories", flush=True)

OUT.mkdir(parents=True, exist_ok=True)
prior = json.loads((OUT / "index.json").read_text()) if (OUT / "index.json").exists() else {}
if prior and prior["enumeration"]["lhz_names_sha256"] != names_sha:
    sys.exit("the EarthScope listing no longer matches the recorded run (lhz_names_sha256); not redrawing, use a new out_dir")
decided = {s["file"]: s for s in prior.get("skipped", [])}                            # skip decisions of the recorded run

# 2. exclude, 3. sort + shuffle + walk
pool = [n for n in all_lhz if n[:-5] not in EXCLUDE_FILES and n[:3] not in EXCLUDE_STATIONS]
order = np.random.default_rng(SEED).permutation(len(pool))
picks, skipped, taken, total = [], [], Counter(), 0
for pos, i in enumerate(order):
    name, sta = pool[i], pool[i][:3]
    if taken[sta] >= PER_STATION:
        continue
    if name in decided:
        skipped.append(decided[name])
        continue
    info = have(name)
    if info is None:
        if total > MAX_TOTAL:
            sys.exit(f"download budget of {MAX_TOTAL / 1e9} GB used up after {len(picks)} picks")
        try:
            info = fetch(name)
        except Skip as e:
            skipped.append({"file": name, "shuffled_index": int(pos), "reason": str(e)})
            print("skip", name, e, flush=True)
            continue
    hm = info["filename_truth"]["on_time_hhmm"]
    picks.append({
        "rank": len(picks) + 1, "file": name, "shuffled_index": int(pos), "pool_index": int(i),
        "bytes": info["bytes"], "sha256": info["sha256"], "downloaded": info["downloaded"],
        "hhmm_valid": int(hm[:2]) < 24 and int(hm[2:]) < 60,
    })
    taken[sta] += 1
    total += info["bytes"]
    print(f"{len(picks):2d} {name[:-5]}  {info['bytes'] / 1e6:5.1f} MB  {info['sha256'][:12]}", flush=True)
    if len(picks) == N_PICKS:
        break
if len(picks) < N_PICKS:
    sys.exit(f"pool exhausted after {len(picks)} picks")

# index.json + the enumerated names
(OUT / "lhz_names.txt").write_text(names_text)
index = {
    "purpose": ARGS.purpose,
    "script": str(Path(__file__).resolve()),
    "enumeration": {
        "endpoints": [BASE, BASE + "<STA>/"],
        "method": "HTML directory listings (served as plain HTML to non-browser clients); no index or JSON listing found",
        "fetched_utc": fetched_utc,
        "n_station_dirs": len(stations),
        "n_stations_with_lhz": sum(1 for s in stations if lhz[s]),
        "total_lhz": len(all_lhz),
        "per_station_lhz": {s: len(lhz[s]) for s in stations},
        "all_channel_tiff_total": sum(len(v) for v in tiffs.values()),
        "lhz_names_file": "lhz_names.txt",
        "lhz_names_sha256": names_sha,
    },
    "exclusions": {
        "files": sorted(EXCLUDE_FILES),
        "stations": {s: len(lhz.get(s, [])) for s in sorted(EXCLUDE_STATIONS)},
        "n_removed": len(all_lhz) - len(pool),
        "n_pool": len(pool),
        "n_pool_stations": len({n[:3] for n in pool}),
    },
    "seed": SEED,
    "rule": (
        "pool = sorted(LHZ names minus exclusions); order = numpy.random.default_rng(seed).permutation(len(pool)); "
        f"walk pool[order[k]] for k = 0, 1, ...: take unless its station already has {PER_STATION} picks; skip (and go on) if "
        f"Content-Length > 80 MB or the download fails twice (one retry); stop at {N_PICKS} picks"
    ),
    "per_station_cap": PER_STATION,
    "max_file_bytes": MAX_FILE,
    "numpy_version": np.__version__,
    "n_picks": len(picks),
    "total_bytes": total,
    "picks": picks,
    "skipped": skipped,
}
(OUT / "index.json").write_text(json.dumps(index, indent=2) + "\n")
print(f"{len(picks)} picks, {total / 1e6:.1f} MB total, {len(skipped)} skipped; wrote {OUT / 'index.json'}")
