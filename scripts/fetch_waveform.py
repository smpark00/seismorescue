"""Download the ground-truth waveform used by the synthetic helicorder experiments.

usage: python -I fetch_waveform.py <out.mseed> [NET STA LOC CHA START_UTC HOURS]
default: 2011 Tohoku M9.0 recorded at IU.ANMO.00.LHZ, 05:40-07:40 UTC (1 Hz, 7200 samples, ~30 KB)
"""
import sys
from pathlib import Path

from obspy import UTCDateTime
from obspy.clients.fdsn import Client

out = Path(sys.argv[1])
net, sta, loc, cha, start, hours = (sys.argv[2:8] if len(sys.argv) >= 8 else
                                    ["IU", "ANMO", "00", "LHZ", "2011-03-11T05:40:00", "2"])
t0 = UTCDateTime(start)
t1 = t0 + float(hours) * 3600
client = None
for name in ("EARTHSCOPE", "IRIS"):          # IRIS was renamed EarthScope; keep a fallback
    try:
        client = Client(name)
        break
    except Exception:
        continue
st = client.get_waveforms(net, sta, loc, cha, t0, t1)
st.merge(fill_value="interpolate")
out.parent.mkdir(parents=True, exist_ok=True)
st.write(str(out), format="MSEED")
print(st[0], "->", out, out.stat().st_size, "bytes")
