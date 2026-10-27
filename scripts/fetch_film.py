"""Refetch long (8 h) windows for existing benchmark records, for film-geometry renders (render_helicorder.py film).

usage: python -I fetch_film.py <src_dir_with_sidecars> <out_dir> [hours]

For each <tag>.json sidecar from fetch_benchmark.py (same event, same station), downloads LHZ for `hours` hours
starting a whole hour before the P arrival minus an offset of 1-4 h chosen per record from a hash of its tag, so the
quake lands on helicorder line 2-5 (1 h per line) as on real 24 h film. Writes <tag>.mseed and an updated sidecar.
"""
import hashlib
import json
import sys
from pathlib import Path

from obspy import UTCDateTime
from obspy.clients.fdsn import Client

src, out = Path(sys.argv[1]), Path(sys.argv[2])
HOURS = float(sys.argv[3]) if len(sys.argv) > 3 else 8
out.mkdir(parents=True, exist_ok=True)
wave = None
for name in ("EARTHSCOPE", "IRIS"):
    try:
        wave = Client(name)
        break
    except Exception:
        continue
for side in sorted(src.glob("*.json")):
    info = json.loads(side.read_text())
    if (out / f"{side.stem}.mseed").exists():
        continue
    p = UTCDateTime(info["p_arrival_utc"])
    offset_h = 1 + int(hashlib.md5(side.stem.encode()).hexdigest(), 16) % 4          # 1..4 h before P
    t0 = UTCDateTime(int(p.timestamp // 3600) * 3600 - offset_h * 3600)
    try:
        st = wave.get_waveforms(info["net"], info["sta"], info["loc"], "LHZ", t0, t0 + HOURS * 3600)
        st.merge(fill_value="interpolate")
        if st[0].stats.npts < 0.99 * HOURS * 3600:
            print("short", side.stem, st[0].stats.npts)
            continue
    except Exception as e:
        print("failed", side.stem, str(e)[:120])
        continue
    st.write(str(out / f"{side.stem}.mseed"), format="MSEED")
    (out / f"{side.stem}.json").write_text(json.dumps({**info, "start_utc": str(t0), "hours": HOURS}, indent=1))
    print(side.stem, "start", t0, flush=True)
