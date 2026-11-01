"""L4 human review (research/45): a local page for one rescue output folder (scripts/rescue.py). It lists the spans the
agreement rule left for review, shows them zoomed with the two SAM 3 passes, and when a person clicks on the right
stroke, SAM 3 film2 (frozen checkpoint, lambda and decoding, research/37) re-tracks the line from that point. The person
then accepts the re-tracked span or holds it.

usage: python -I scripts/review_server.py out/rescue/<chip> [--port 8791] [--host 127.0.0.1] [--allow-host HOST:PORT]
                                                             [--record | --replay | --record-spans [--lines 6,23]]
--record        also store every re-track in <folder>/review/recorded_retracks.json
--record-spans  no page: place one scripted click per review window on the backward pass, store the re-tracks, stop
--replay        no model, no GPU, no Hugging Face account: a click shows the stored re-track of that window, labelled as
                recorded. It is not SAM 3 reacting to your click; run without --replay for that.

The product stays files (research/36 §2): every decision is appended to <folder>/review/decisions.json (bound to the
sha256 of paths.npz), and <folder>/review/paths_reviewed.npz is rewritten (the SAM 3 path with the accepted re-tracks,
a per-column status: 0 no record, 1 auto-accepted, 2 disputed, 3 accepted after a human-guided SAM 3 re-track,
4 held). Starting again replays decisions.json without the model. An accepted re-track only replaces disputed columns;
auto-accepted columns keep the forward path, and the accepted path is the server's own re-track (the page only names
it). Served to this machine only by default (Host and Origin are checked, POST bodies must be JSON); the model runs
offline and nothing is sent to any API.
"""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")                     # never resolve model revisions online (research/45 §5)
import argparse                                                  # noqa: E402
import base64                                                    # noqa: E402
import hashlib                                                   # noqa: E402
import itertools                                                 # noqa: E402
import json                                                      # noqa: E402
import sys                                                       # noqa: E402
import threading                                                 # noqa: E402
import time                                                      # noqa: E402
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer   # noqa: E402
from pathlib import Path                                         # noqa: E402
from urllib.parse import parse_qs, urlparse                      # noqa: E402

import cv2                                                       # noqa: E402
import numpy as np                                               # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from obspy import UTCDateTime                                    # noqa: E402

SAM3_CKPT = ROOT / "out" / "ml" / "sam3_film2_modal" / "ckpt_sam3_film2_best.pt"
SAM3_FROZEN = ROOT / "out" / "ml" / "sam3_film2" / "frozen_method.json"
PAGE = HERE / "review_page.html"
RECORDED = "recorded_retracks.json"
S = 38                                                            # line spacing of the rescaled frame (l0.S_TARGET)
VIEW_W, MARGIN, SNAP, GAP, MIN_SPAN, KEEP_CANDS = 480, 128, 5, 30, 15, 50
NONE, AUTO, DISPUTED, HUMAN, HELD = 0, 1, 2, 3, 4
BGR = {AUTO: (74, 191, 47), DISPUTED: (77, 72, 229), HUMAN: (0, 183, 245), HELD: (150, 150, 150)}


def sam3():
    """the SAM 3 module, imported on first use: export_mseed.py reuses Review without torch"""
    import ml_sam3
    return ml_sam3


def runs(mask):
    e = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(np.int8), [0]))))
    return list(zip(e[::2].tolist(), e[1::2].tolist()))


class Review:
    def __init__(self, folder, replay_recorded=False, record=False):
        self.dir = Path(folder).resolve()
        z = np.load(self.dir / "paths.npz")
        self.paths_sha = hashlib.sha256((self.dir / "paths.npz").read_bytes()).hexdigest()
        self.sam3, self.back = z["SAM3"].astype(float), z["SAM3_back"].astype(float)
        self.bases, self.x0, self.x1 = z["bases"].astype(float), int(z["x0"]), int(z["x1"])
        self.geo = json.loads((self.dir / "l0" / "geometry.json").read_text())
        self.img = cv2.imread(str(self.dir / "l0" / f"{self.geo['name']}_rescaled.png"), 0)
        self.prov = json.loads((self.dir / "provenance.json").read_text())
        rec = self.dir / "l1" / "record.json"
        self.record = json.loads(rec.read_text()) if rec.exists() else None
        tm = self.geo["time"]
        valid = np.zeros(self.sam3.shape, bool)
        valid[:, self.x0:self.x1] = True
        if tm.get("line0_trace_start_x") is not None:
            valid[0, :int(tm["line0_trace_start_x"])] = False
        if tm.get("last_line_trace_end_x") is not None:
            valid[-1, int(tm["last_line_trace_end_x"]) + 1:] = False
        self.status0 = np.where(valid, np.where(z["accepted"], AUTO, DISPUTED), NONE).astype(np.int8)
        self.on = UTCDateTime(self.geo["on"]) if self.geo.get("on") and tm.get("T0") is not None else None
        self.tm = tm
        self.out = self.dir / "review"
        self.out.mkdir(exist_ok=True)
        self.dec_file = self.out / "decisions.json"
        saved = json.loads(self.dec_file.read_text()) if self.dec_file.exists() else {"paths_sha256": self.paths_sha, "decisions": []}
        if not isinstance(saved, dict) or saved.get("paths_sha256") != self.paths_sha:
            raise SystemExit(f"{self.dec_file} belongs to another paths.npz (was rescue.py run again?): "
                             "move it away to start a new review")
        self.decisions = saved["decisions"]
        self.lock, self.model, self.fz = threading.Lock(), None, json.loads(SAM3_FROZEN.read_text())
        self.cands, self._cid, self.version, self._overview = {}, itertools.count(1), 0, None
        self.record_retracks, self.recorded = record, None
        if replay_recorded:
            self.recorded = self.load_recorded()
        self.replay()

    # ---------- state ----------
    def replay(self):
        self.path, self.status = self.sam3.copy(), self.status0.copy()
        for d in self.decisions:
            self.apply(d)
        self.version += 1

    def apply(self, d):
        k, a, b = d["line"], d["x"][0], d["x"][1]
        m = self.status[k, a:b] == DISPUTED                       # views: assignments write through
        if d["action"] == "accept":
            self.path[k, a:b][m] = np.asarray(d["path"], float)[m]
            self.status[k, a:b][m] = HUMAN
        elif d["action"] == "hold":
            self.status[k, a:b][m] = HELD

    def save(self):
        """temp file + os.replace, so a crash never leaves a half-written decisions.json"""
        tmp = self.out / "decisions.json.tmp"
        tmp.write_text(json.dumps({"paths_sha256": self.paths_sha, "decisions": self.decisions}, indent=1))
        os.replace(tmp, self.dec_file)
        tmpz = self.out / "paths_reviewed.tmp.npz"
        np.savez_compressed(tmpz, SAM3_reviewed=self.path.astype(np.float32), status=self.status, bases=self.bases,
                            x0=self.x0, x1=self.x1, paths_sha256=self.paths_sha,
                            status_codes=json.dumps({"0": "no record", "1": "auto-accepted (both SAM 3 passes agree)", "2": "disputed",
                                                     "3": "accepted after a human-guided SAM 3 re-track", "4": "held"}))
        os.replace(tmpz, self.out / "paths_reviewed.npz")
        self.version += 1

    def utc(self, k, x):
        if self.on is None:
            return None
        return str(self.on + self.tm["T0"] + self.tm["P"] * k + (x - self.geo["xa"]) / self.tm["pps_r"])[:19]

    def locate(self, t):
        if self.on is None:
            return None, None
        s = float(UTCDateTime(t) - self.on) - self.tm["T0"]
        k = int(np.floor((s - (self.x0 - self.geo["xa"]) / self.tm["pps_r"]) / self.tm["P"]))
        return k, self.geo["xa"] + (s - self.tm["P"] * k) * self.tm["pps_r"]

    def spans(self, k):
        out, cur = [], None
        for a, b in runs(self.status[k] == DISPUTED):
            if cur and a - cur[1] <= GAP:
                cur[1] = b
            else:
                if cur:
                    out.append(cur)
                cur = [a, b]
        if cur:
            out.append(cur)
        return [{"x": [a, b], "utc": [self.utc(k, a), self.utc(k, b)]} for a, b in out if b - a >= MIN_SPAN]

    def state(self):
        lines = []
        for k in range(len(self.bases)):
            st = self.status[k]
            n = int((st != NONE).sum())
            frac = {name: round(float((st == c).sum()) / n, 3) if n else None
                    for name, c in (("auto", AUTO), ("disputed", DISPUTED), ("human", HUMAN), ("held", HELD))}
            lines.append({"line": k, "record_cols": n, **frac, "spans": self.spans(k), "utc": [self.utc(k, self.x0), self.utc(k, self.x1)]})
        rec, me = self.record or {}, (self.record or {}).get("main_event") or {}
        kp, xp = self.locate(me["P_utc"]) if me.get("P_utc") else (None, None)
        allv = self.status != NONE
        return {
            "chip": self.geo["name"], "width": int(self.img.shape[1]), "height": int(self.img.shape[0]), "S": S,
            "x0": self.x0, "x1": self.x1, "view_w": VIEW_W, "margin": MARGIN, "bases": [round(float(b), 1) for b in self.bases],
            "station": rec.get("station"), "component": rec.get("component"), "on_utc": rec.get("on_utc") or self.geo.get("on"),
            "main_event": None if not me else {**{k: me.get(k) for k in ("id", "origin_utc", "mag", "mag_type", "place",
                                                                         "dist_deg", "P_utc", "surface_utc", "surface_lines")},
                                              "P_line": kp, "P_x": None if xp is None else round(xp, 1)},
            "totals": {name: round(float((self.status[allv] == c).mean()), 4)
                       for name, c in (("auto", AUTO), ("disputed", DISPUTED), ("human", HUMAN), ("held", HELD))},
            "decisions": len(self.decisions), "lines": lines,
            "model": {"ckpt_sha256": self.fz["ckpt_sha256"][:12], "lam": self.fz["lam_final"], "ready": self.model is not None,
                      "recorded_run": None if self.recorded is None else
                      {k: self.recorded.get(k) for k in ("made", "device", "clicked_by")} | {"n": len(self.recorded["clicks"])}},
        }

    # ---------- pictures ----------
    def overview_jpg(self, width=1600):
        cached = self._overview
        if cached is not None and cached[0] == self.version:
            return cached[1]
        v = self.version
        f = width / self.img.shape[1]
        vis = cv2.cvtColor(cv2.resize(self.img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA), cv2.COLOR_GRAY2BGR)
        vis = (0.45 * vis + 0.55 * 255).astype(np.uint8)
        for k in range(len(self.bases)):
            for code, col in BGR.items():
                for a, b in runs(self.status[k] == code):
                    xs = np.arange(a, b)
                    pts = np.stack([xs * f, self.path[k, a:b] * f], 1)
                    ok = np.isfinite(pts[:, 1])
                    if ok.sum() > 1:
                        cv2.polylines(vis, [np.round(pts[ok] * 4).astype(np.int32)], False, col, 1, cv2.LINE_AA, shift=2)
        jpg = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()
        if v == self.version:                                     # a decision during the render: do not cache it
            self._overview = (v, jpg)
        return jpg

    def view(self, k, a, b):
        if not (0 <= k < len(self.bases)):
            raise ValueError("no such line")
        a, b = max(0, int(a)), min(self.img.shape[1], int(b))
        if b - a < 2:
            raise ValueError("empty window")
        r0 = int(max(0, round(self.bases[k] - 3.5 * S)))
        r1 = int(min(self.img.shape[0], round(self.bases[k] + 3.5 * S)))
        png = cv2.imencode(".png", self.img[r0:r1, a:b])[1].tobytes()
        rnd = lambda v: [None if not np.isfinite(y) else round(float(y), 2) for y in v]   # noqa: E731
        return {"line": k, "x": [a, b], "rows": [r0, r1], "png": base64.b64encode(png).decode(),
                "sam3": rnd(self.path[k, a:b]), "back": rnd(self.back[k, a:b]), "status": self.status[k, a:b].tolist(),
                "utc": [self.utc(k, a), self.utc(k, b)], "base": round(float(self.bases[k]), 1)}

    # ---------- SAM 3 ----------
    def load_model(self):
        m3 = sam3()
        assert m3.sha256(SAM3_CKPT) == self.fz["ckpt_sha256"], "SAM 3 checkpoint is not the frozen file"
        self.dev = m3.get_device("auto")
        self.model, self.ck = m3.load_finetuned(SAM3_CKPT, self.dev)

    # ---------- stored re-tracks ----------
    def load_recorded(self):
        f = self.out / RECORDED
        if not f.exists():
            raise SystemExit(f"{f} not found: --replay needs the stored re-tracks (examples/ has them for the MAT chip)")
        rec = json.loads(f.read_text())
        if rec.get("paths_sha256") != self.paths_sha:
            raise SystemExit(f"{f} belongs to another paths.npz: the stored re-tracks cannot be replayed on this one")
        return rec

    def store(self, r, clicked_by):
        """append one live re-track to the recording, with what produced it"""
        f, m3 = self.out / RECORDED, sam3()
        rec = json.loads(f.read_text()) if f.exists() else {
            "what": "stored SAM 3 film2 re-tracks for review_server.py --replay", "paths_sha256": self.paths_sha,
            "model": {"name": "SAM 3 film2", "ckpt_sha256": self.fz["ckpt_sha256"], "lam": self.fz["lam_final"],
                      "base_weights": self.fz["hf_id"], "revision": self.fz["revision"]},
            "device": str(self.dev), "gpu": m3.torch.cuda.get_device_name(0) if m3.torch.cuda.is_available() else None,
            "torch": m3.torch.__version__, "made": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "clicked_by": clicked_by, "clicks": []}
        rec["clicks"].append({**{k: r[k] for k in ("line", "x", "x_end", "y_click", "y_snapped", "secs", "path",
                                                   "max_dev_from_accepted_S")}, "clicked_by": clicked_by})
        tmp = f.with_suffix(".tmp")
        tmp.write_text(json.dumps(rec))
        os.replace(tmp, f)

    def recorded_retrack(self, k, x, x_end):
        """the stored re-track of the window the click fell in (the page sends x_end = window end + MARGIN)"""
        hi = int(x_end) - MARGIN
        near = [c for c in self.recorded["clicks"] if c["line"] == k and hi - VIEW_W <= c["x"] < hi]
        if not near:
            lines = sorted({c["line"] for c in self.recorded["clicks"]})
            raise ValueError(f"recorded run: no stored re-track in this window (stored on lines {lines}). Press N for the "
                             "next span. SAM 3 is not running here; see the README to run it live.")
        c = min(near, key=lambda c: abs(c["x"] - x))
        cid = f"c{next(self._cid)}"
        self.cands[cid] = {"line": k, "x": c["x"], "x_end": c["x_end"], "y_click": c["y_click"], "y_snapped": c["y_snapped"],
                           "path": np.array([np.nan if v is None else v for v in c["path"]], float)}
        while len(self.cands) > KEEP_CANDS:
            self.cands.pop(next(iter(self.cands)))
        return {**{f: c[f] for f in ("line", "x", "x_end", "y_click", "y_snapped", "secs", "path", "max_dev_from_accepted_S")},
                "cand": cid, "recorded": True}

    def record_spans(self, lines=None):
        """one scripted click per review window: at the left of each window, on the backward pass (the path the reviewer
        would be comparing against). These clicks are not a person's and nobody checked they sit on the right stroke"""
        who = "script: one click per review window, on the backward SAM 3 pass; not a person, not checked by eye"
        m3, n, skipped = sam3(), 0, 0
        for k in (range(len(self.bases)) if lines is None else lines):
            for sp in self.spans(k):
                for x in range(sp["x"][0], sp["x"][1], VIEW_W - 40):
                    y = self.back[k, x] if np.isfinite(self.back[k, x]) else self.sam3[k, x]
                    if not np.isfinite(y) or self.x1 - x < m3.S.WIN_W:
                        skipped += 1
                        continue
                    self.store(self.retrack(k, x, float(y), x + VIEW_W - 40 + MARGIN), who)
                    n += 1
        print(f"[review] stored {n} re-tracks in {self.out / RECORDED} (skipped {skipped})", flush=True)

    def retrack(self, k, x, y, x_end):
        """SAM 3 film2 from the clicked point (snapped to the darkest row within +-5) to x_end: the click is the first
        window's point prompt, exactly as a line start is in research/37 (segment_image + decode_paths). The result is
        kept on the server under an id; an accept names that id, so the stored path is what gets accepted"""
        if not (0 <= k < len(self.bases)):
            raise ValueError("no such line")
        if self.recorded is not None:
            return self.recorded_retrack(k, x, x_end)
        m3 = sam3()
        if not np.isfinite(x):
            raise ValueError("click outside the image")
        x, H, W = int(round(x)), self.img.shape[0], self.img.shape[1]
        x_end = int(min(self.x1, max(int(x_end), x + m3.S.WIN_W)))
        if x < self.x0 or x_end - x < m3.S.WIN_W:
            raise ValueError(f"too close to the end of the line for one SAM 3 window ({m3.S.WIN_W} columns): click further left")
        if not (np.isfinite(x) and np.isfinite(y)):
            raise ValueError("click outside the image")
        yc = int(round(y))
        if not (0 <= yc < H):
            raise ValueError("click outside the image")
        lo, hi = max(0, yc - SNAP), min(H, yc + SNAP + 1)
        col = self.img[lo:hi, max(0, x - 1):min(W, x + 2)].astype(float).mean(1)
        ys = float(lo + int(np.argmin(col)))
        top = int(np.clip(round(self.bases[k]) - m3.S.WIN_H // 2, 0, H - m3.S.WIN_H))
        t = time.time()
        with self.lock, m3.amp_ctx(self.dev, bool(self.fz["amp_bf16"])):
            probb, _ = m3.S.segment_image(self.model, self.img, np.array([ys]), x, x_end, tops=np.array([top]), dev=self.dev)
            P = m3.S.decode_paths(probb, np.array([ys]), x, x_end, float(self.fz["lam_final"]), self.dev, tops=np.array([top]))
            path = P[0, x:x_end].copy()
            old, st = self.path[k, x:x_end], self.status[k, x:x_end]
            agree = np.isin(st, (AUTO, HUMAN)) & np.isfinite(path) & np.isfinite(old)
            dev_auto = float(np.max(np.abs(path[agree] - old[agree])) / S) if agree.any() else None
            cid = f"c{next(self._cid)}"
            self.cands[cid] = {"line": k, "x": x, "x_end": x_end, "path": path, "y_click": float(y), "y_snapped": ys}
            while len(self.cands) > KEEP_CANDS:
                self.cands.pop(next(iter(self.cands)))
        res = {"cand": cid, "line": k, "x": x, "x_end": x_end, "y_click": float(y), "y_snapped": ys,
               "secs": round(time.time() - t, 2), "path": [None if not np.isfinite(v) else round(float(v), 2) for v in path],
               "max_dev_from_accepted_S": None if dev_auto is None else round(dev_auto, 3)}
        if self.record_retracks:
            self.store(res, "person, in the review page")
        return res

    def decide(self, d):
        k, a, b, action = int(d["line"]), int(d["x"][0]), int(d["x"][1]), d["action"]
        if action not in ("accept", "hold") or not (0 <= k < len(self.bases)) or not (self.x0 <= a < b <= self.x1):
            raise ValueError("bad decision")
        with self.lock:
            rec = {"id": len(self.decisions) + 1, "time": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "line": k, "x": [a, b],
                   "utc": [self.utc(k, a), self.utc(k, b)], "action": action,
                   "disputed_cols": int((self.status[k, a:b] == DISPUTED).sum())}
            if rec["disputed_cols"] == 0:
                raise ValueError("nothing to review in this range")
            if action == "accept":
                c = self.cands.get(d.get("cand"))
                if c is None or c["line"] != k or a != c["x"] or b > c["x_end"]:
                    raise ValueError("this re-track does not belong to this window: click again")
                path = c["path"][:b - a]
                if len(path) != b - a or not np.isfinite(path).all() or path.min() < 0 or path.max() > self.img.shape[0] - 1:
                    raise ValueError("the re-tracked path is not finite inside the image")
                rec.update({"path": [round(float(v), 2) for v in path],
                            "click": {"x": c["x"], "y": c["y_click"], "y_snapped": c["y_snapped"]},
                            "model": {"name": "SAM 3 film2 (research/37)", "ckpt_sha256": self.fz["ckpt_sha256"],
                                      "lam": self.fz["lam_final"]}})
                if self.recorded is not None:
                    rec["recorded_run"] = "stored re-track (review_server.py --replay), SAM 3 did not run here"
            self.decisions.append(rec)
            self.apply(rec)
            self.save()
        return rec

    def undo(self):
        with self.lock:
            if self.decisions:
                self.decisions.pop()
                self.replay()
                self.save()


class Handler(BaseHTTPRequestHandler):
    review, hosts = None, set()

    def send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def trusted(self, post=False):
        """only this machine's page: the Host must be ours (stops DNS rebinding), a cross-site Origin is refused, and
        POST bodies must be JSON (a cross-site JSON POST needs a CORS preflight, which this server never grants)"""
        host = self.headers.get("Host", "")
        origin = self.headers.get("Origin")
        if host not in self.hosts or (origin is not None and origin != f"http://{host}"):
            return False
        return not post or self.headers.get("Content-Type", "").split(";")[0].strip() == "application/json"

    def log_message(self, fmt, *args):                            # quiet: one line per API call only
        if "/api/" in self.path and "/api/view" not in self.path:
            sys.stderr.write(f"[review] {self.command} {self.path.split('?')[0]} {args[1] if len(args) > 1 else ''}\n")

    def do_GET(self):
        if not self.trusted():
            return self.send(403, {"error": "forbidden"})
        u, rv = urlparse(self.path), self.review
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path in ("/", "/review.html"):
                return self.send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            if u.path == "/favicon.ico":
                return self.send(204, b"", "image/x-icon")
            if u.path == "/api/state":
                return self.send(200, rv.state())
            if u.path == "/api/overview.jpg":
                return self.send(200, rv.overview_jpg(), "image/jpeg")
            if u.path == "/api/view":
                return self.send(200, rv.view(int(q["line"]), int(q["x0"]), int(q["x1"])))
            return self.send(404, {"error": "not found"})
        except Exception as e:                                    # noqa: BLE001 (reported to the page)
            return self.send(400, {"error": f"{type(e).__name__}: {e}"})

    def do_POST(self):
        if not self.trusted(post=True):
            return self.send(403, {"error": "forbidden"})
        rv = self.review
        try:
            d = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            if self.path == "/api/retrack":
                return self.send(200, rv.retrack(int(d["line"]), float(d["x"]), float(d["y"]), float(d["x_end"])))
            if self.path == "/api/decide":
                return self.send(200, rv.decide(d))
            if self.path == "/api/undo":
                rv.undo()
                return self.send(200, {"ok": True})
            return self.send(404, {"error": "not found"})
        except Exception as e:                                    # noqa: BLE001 (reported to the page)
            return self.send(400, {"error": f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--port", type=int, default=8791)            # 8765 is taken on the workstation (research/45 §4)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--allow-host", action="append", default=[], help="extra Host header to accept, e.g. 192.168.0.12:8791")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--record", action="store_true", help="also store every re-track for --replay")
    mode.add_argument("--replay", action="store_true", help="no model: show the stored re-tracks, labelled as recorded")
    mode.add_argument("--record-spans", action="store_true", help="no page: one scripted click per review window, then stop")
    ap.add_argument("--lines", help="with --record-spans: only these lines, e.g. 6,23")
    a = ap.parse_args()
    Handler.review = Review(a.folder, replay_recorded=a.replay, record=a.record)
    Handler.hosts = {f"127.0.0.1:{a.port}", f"localhost:{a.port}", *a.allow_host}
    t = time.time()
    if a.replay:
        print(f"[review] {Handler.review.geo['name']}: RECORDED RUN, SAM 3 is not loaded; {len(Handler.review.recorded['clicks'])} stored "
              f"re-tracks; {len(Handler.review.decisions)} earlier decisions replayed; http://127.0.0.1:{a.port}/", flush=True)
    else:
        Handler.review.load_model()
        if a.record_spans:
            Handler.review.record_spans([int(v) for v in a.lines.split(",")] if a.lines else None)
            raise SystemExit(0)
        print(f"[review] {Handler.review.geo['name']}: SAM 3 film2 ready in {time.time() - t:.0f} s; "
              f"{len(Handler.review.decisions)} earlier decisions replayed; http://127.0.0.1:{a.port}/", flush=True)
    ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()
