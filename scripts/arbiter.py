"""Arbiter for SeismoRescue (docs/RESULTS.md-multiagent-design.md): two proposer trackers, events where they disagree,
rule features per event, and assembly of the final paths.

Proposer A = greedy single-line tracker (good on the large-amplitude line), proposer B = joint physics tracker
(good in tangles and on quiet lines). Outside events the two agree within TAU px, so an event is a self-contained
choice "A's segment or B's segment" for one line.
"""
import cv2
import numpy as np

from track_joint import ink_mask, track_joint, tracker_greedy


def fill_nan(y):
    ok = ~np.isnan(y)
    if ok.sum() < 2:
        return y
    out = y.copy()
    idx = np.arange(len(y))
    out[~ok] = np.interp(idx[~ok], idx[ok], y[ok])
    return out


def proposals(img, y0, x0, x1, S):
    _, binv = cv2.threshold(cv2.GaussianBlur(img, (0, 0), 0.8), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    A = np.array([tracker_greedy(binv > 0, y, x0, x1, S) for y in y0])
    ink = ink_mask(img)
    B = track_joint(ink, y0, x0, x1, S)
    for k in range(len(y0)):                       # greedy leaves NaN where it found no ink: bridge inside [x0, x1)
        A[k, x0:x1] = fill_nan(A[k, x0:x1])
    return A, B, ink


def cache_proposals(bench, cache, conds):
    """proposals A, B and the ink mask for every rendered record x condition (skips cached ones)"""
    import json
    cache.mkdir(parents=True, exist_ok=True)
    for rec in sorted(p for p in bench.iterdir() if p.is_dir()):
        for mode, g in conds:
            cf = cache / f"{rec.name}_{mode}_g{g:g}.npz"
            if cf.exists():
                continue
            d = rec / mode
            meta = json.loads((d / "meta.json").read_text())
            img = cv2.imread(str(d / f"heli_g{g:g}.png"), 0)
            pen = np.load(d / f"gt_g{g:g}.npz")["pen"]
            x0 = meta["margin_x"]
            A, B, ink = proposals(img, pen[:, x0], x0, img.shape[1] - x0, meta["spacing"])
            np.savez_compressed(cf, A=A, B=B, ink=ink)


def events(A, B, x0, x1, tau, close=10, min_len=3):
    """per line: column spans [s, e) where |A - B| > tau (gaps shorter than `close` merged)"""
    ev = []
    for k in range(A.shape[0]):
        d = np.zeros(A.shape[1], bool)
        d[x0:x1] = np.nan_to_num(np.abs(A[k, x0:x1] - B[k, x0:x1]), nan=0) > tau
        d = cv2.morphologyEx(d.astype(np.uint8)[None], cv2.MORPH_CLOSE, np.ones((1, close), np.uint8))[0] > 0
        edges = np.flatnonzero(np.diff(np.r_[0, d.astype(int), 0]))
        for s, e in zip(edges[::2], edges[1::2]):
            if e - s >= min_len:
                ev.append((k, int(s), int(e)))
    return ev


def features(seg, k, s, e, ink, others):
    """rule features of one candidate segment: roughness, ink support, duplication of another line"""
    rough = float(np.mean(np.abs(np.diff(seg, 2)))) if len(seg) > 2 else 0.0
    H = ink.shape[0]
    yi = np.clip(np.round(seg).astype(int), 1, H - 2)
    cols = np.arange(s, e)
    on_ink = float(np.mean(ink[yi - 1, cols] | ink[yi, cols] | ink[yi + 1, cols]))
    dup = float(np.mean(np.min(np.abs(others[:, s:e] - seg[None]), axis=0) < 2.0)) if len(others) else 0.0
    return np.array([rough, 1 - on_ink, dup])


def event_table(A, B, ink, x0, x1, tau):
    """for every event: features of A's and B's segment (others = B's other lines)"""
    rows = []
    for k, s, e in events(A, B, x0, x1, tau):
        others = np.delete(B, k, 0)
        rows.append({"line": k, "s": s, "e": e,
                     "fA": features(A[k, s:e], k, s, e, ink, others),
                     "fB": features(B[k, s:e], k, s, e, ink, others),
                     # self-duplication: greedy lines have no exclusivity, so A_k sitting on another A line means A_k
                     # jumped traces; measuring duplication only against B wrongly blames A when B swapped two lines
                     "dupAA": float(np.mean(np.min(np.abs(np.delete(A, k, 0)[:, s:e] - A[k, s:e][None]), axis=0) < 2.0)),
                     "dupBB": float(np.mean(np.min(np.abs(others[:, s:e] - B[k, s:e][None]), axis=0) < 2.0))})
    return rows


def decide(rows, w):
    """rule judge: pick the cheaper segment; margin = cost difference"""
    out = []
    for r in rows:
        cA, cB = float(w @ r["fA"]), float(w @ r["fB"])
        out.append(("A" if cA < cB else "B", abs(cA - cB)))
    return out


def assemble(A, B, rows, choice):
    est = B.copy()
    for r, c in zip(rows, choice):
        if c == "A":
            est[r["line"], r["s"]:r["e"]] = A[r["line"], r["s"]:r["e"]]
    return est


def oracle_choice(A, B, rows, pen, S):
    """upper bound of this architecture: per event, the segment with fewer failed columns vs GT"""
    ch = []
    for r in rows:
        k, s, e = r["line"], r["s"], r["e"]
        g = pen[k, s:e]
        fa = np.nansum(np.abs(A[k, s:e] - g) > S / 2)
        fb = np.nansum(np.abs(B[k, s:e] - g) > S / 2)
        ch.append("A" if fa < fb else "B")
    return ch
