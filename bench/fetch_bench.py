"""Download the SeismoBench splits and baseline submissions from this repository's GitHub Release, check their hashes, unpack them under bench_data/.

usage: python -I bench/fetch_bench.py [holdout7 holdout6 holdout4 baselines]     (default: holdout7 and baselines)
SEISMOBENCH_URL points it at a mirror of the release. The hashes are checked either way.
"""
import hashlib
import os
import sys
import tarfile
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
URL = os.environ.get("SEISMOBENCH_URL", "https://github.com/smpark00/seismorescue/releases/download/bench-v1/").rstrip("/") + "/"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main(names):
    known = dict(reversed(line.split()) for line in (HERE / "SHA256SUMS").read_text().splitlines() if line.strip())
    out = HERE.parent / "bench_data"
    out.mkdir(exist_ok=True)
    for name in names:
        tar = f"{name}.tar"
        if tar not in known:
            sys.exit(f"unknown: {name}. Known: {sorted(k[:-4] for k in known)}")
        dst = out / tar
        if not dst.exists() or sha256(dst) != known[tar]:
            print(f"downloading {tar}", flush=True)
            try:
                urllib.request.urlretrieve(URL + tar, dst)
            except OSError as e:
                sys.exit(f"{tar}: could not download it from {URL}: {e}")
        if sha256(dst) != known[tar]:
            sys.exit(f"{tar}: wrong hash, not unpacking")
        with tarfile.open(dst) as t:
            t.extractall(out, filter="data")
        print(f"{name}: ok, under {out}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ["holdout7", "baselines"])
