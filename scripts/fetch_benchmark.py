"""Download a multi-event, multi-station set of long-period waveforms for cross-validated benchmarks.

usage: python -I fetch_benchmark.py <out_dir> [n_events] [stations_per_event] [seed] [exclude_dir]
exclude_dir: comma-separated dirs; skip every event on a date already used there (held-out sets share no event day)
default: 12 events (M>=7.5, 2005-2025, USGS catalogue, random with seed 0), 2 stations each from IU/II LHZ at
30-90 degrees. Each record starts 15 min before the theoretical P arrival and lasts 2 h (8 helicorder lines of 15 min),
like the Tohoku/ANMO record. Writes <event>_<net>.<sta>.mseed plus a sidecar .json (event, station, distance, P time).
"""
import json
import random
import sys
from pathlib import Path

from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from obspy.geodetics import locations2degrees
from obspy.taup import TauPyModel

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
N_EV = int(sys.argv[2]) if len(sys.argv) > 2 else 12
N_ST = int(sys.argv[3]) if len(sys.argv) > 3 else 2
rng = random.Random(int(sys.argv[4]) if len(sys.argv) > 4 else 0)
# exclude_dir may list several directories separated by commas
used_days = {p.name[:8] for d in (sys.argv[5].split(",") if len(sys.argv) > 5 else []) for p in Path(d).glob("*.mseed")}
taup = TauPyModel("iasp91")
usgs = Client("USGS")
wave = None
for name in ("EARTHSCOPE", "IRIS"):
    try:
        wave = Client(name)
        break
    except Exception:
        continue

cat = usgs.get_events(starttime=UTCDateTime("2005-01-01"), endtime=UTCDateTime("2025-12-31"), minmagnitude=7.5,
                      orderby="time")
events = list(cat)
rng.shuffle(events)
got = 0
for ev in events:
    if got >= N_EV:
        break
    o = ev.preferred_origin() or ev.origins[0]
    mag = (ev.preferred_magnitude() or ev.magnitudes[0]).mag
    if o.time.strftime('%Y%m%d') in used_days:
        continue
    try:
        inv = wave.get_stations(network="IU,II", channel="LHZ", starttime=o.time, endtime=o.time + 3 * 3600, level="channel")
    except Exception as e:
        print("stations failed", o.time, e)
        continue
    cands = []
    for net in inv:
        for sta in net:
            dist = locations2degrees(o.latitude, o.longitude, sta.latitude, sta.longitude)
            if 30 <= dist <= 90:
                locs = sorted({ch.location_code for ch in sta.channels if ch.code == "LHZ"})
                cands.append((net.code, sta.code, locs[0] if locs else "", sta.latitude, sta.longitude, dist))
    rng.shuffle(cands)
    saved = 0
    for net, sta, loc, slat, slon, dist in cands:
        if saved >= N_ST:
            break
        p = o.time + taup.get_travel_times(max(0.0, o.depth / 1000), dist, phase_list=["P"])[0].time
        t0 = UTCDateTime(int(p.timestamp // 60) * 60 - 15 * 60)       # whole minute, 15 min before P
        try:
            st = wave.get_waveforms(net, sta, loc, "LHZ", t0, t0 + 2 * 3600)
            st.merge(fill_value="interpolate")
            tr = st[0]
            if tr.stats.npts < 0.99 * 7200:
                continue
        except Exception:
            continue
        tag = f"{o.time.strftime('%Y%m%d')}_M{mag:.1f}_{net}.{sta}"
        st.write(str(out / f"{tag}.mseed"), format="MSEED")
        (out / f"{tag}.json").write_text(json.dumps({
            "event_time": str(o.time), "event_lat": o.latitude, "event_lon": o.longitude,
            "event_depth_km": o.depth / 1000, "mag": mag, "net": net, "sta": sta, "loc": loc,
            "sta_lat": slat, "sta_lon": slon, "dist_deg": dist, "p_arrival_utc": str(p), "start_utc": str(t0)}, indent=1))
        print(tag, f"dist={dist:.1f}", flush=True)
        saved += 1
    got += saved > 0
print("events:", got)
