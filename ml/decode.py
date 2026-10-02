"""Python mirror of the device decoder: ps_centernet_decode(), ps_nms(), ps_iou()
and ps_estimate_distances() from core/src/ps_detect.c.

Arithmetic is done in float32 in the same order as the C code so the parity test
(ml/tests/test_decode_parity.py) can compare boxes to ~1e-6.
"""
from __future__ import annotations

import math

import numpy as np

f32 = np.float32
W, H = 160, 120
MAX_DET = 16           # PS_MAX_DETECTIONS
NUM_CLASSES = 2        # 0 = fire, 1 = person
DEVICE_SCORE_THRESHOLD = 0.35  # ps_config_default(): score_threshold
HFOV_DEG = 57.0                 # ps_config_default(): hfov_deg
PERSON_HEIGHT_M = 1.7           # ps_config_default(): person_height_m


def _sigmoid32(v):
    return f32(1.0) / (f32(1.0) + np.exp(-f32(v)))


def iou(a, b):
    """ps_iou(): boxes are dicts with x, y, w, h (top-left + size)."""
    x0 = max(f32(a["x"]), f32(b["x"]))
    y0 = max(f32(a["y"]), f32(b["y"]))
    x1 = min(f32(a["x"]) + f32(a["w"]), f32(b["x"]) + f32(b["w"]))
    y1 = min(f32(a["y"]) + f32(a["h"]), f32(b["y"]) + f32(b["h"]))
    iw, ih = f32(x1 - x0), f32(y1 - y0)
    if iw <= 0 or ih <= 0:
        return f32(0.0)
    inter = f32(iw * ih)
    return f32(inter / (f32(a["w"]) * f32(a["h"]) + f32(b["w"]) * f32(b["h"]) - inter))


def nms(dets, iou_threshold=0.5):
    """ps_nms(): stable insertion sort by score (descending), then greedy per-class NMS."""
    d = list(dets)
    for i in range(1, len(d)):
        key = d[i]
        j = i - 1
        while j >= 0 and d[j]["score"] < key["score"]:
            d[j + 1] = d[j]
            j -= 1
        d[j + 1] = key
    keep = [True] * len(d)
    for i in range(len(d)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(d)):
            if keep[j] and d[j]["cls"] == d[i]["cls"] and iou(d[i], d[j]) > f32(iou_threshold):
                keep[j] = False
    return [x for x, k in zip(d, keep) if k]


def centernet_decode(heat, wh, off, stride=4, thr=DEVICE_SCORE_THRESHOLD, max_det=MAX_DET):
    """Mirror of ps_centernet_decode(). heat/wh/off: arrays [2, gh, gw].

    Returns a list of dicts {cls, score, x, y, w, h} (x, y = top-left, input pixels).
    `max_det` defaults to the device cap; eval may raise it for mAP but the
    device behaviour is max_det=16.
    """
    heat = np.asarray(heat, dtype=np.float32)
    wh = np.asarray(wh, dtype=np.float32)
    off = np.asarray(off, dtype=np.float32)
    nc, gh, gw = heat.shape
    thr_logit = f32(np.log(f32(thr) / (f32(1.0) - f32(thr))))
    out = []
    for c in range(min(nc, NUM_CLASSES)):
        hm = heat[c]
        # 3x3 local maximum (a neighbour strictly greater disqualifies), edges ignored.
        pad = np.pad(hm, 1, mode="constant", constant_values=-np.inf)
        nb = np.max(np.stack([pad[1 + dy:1 + dy + gh, 1 + dx:1 + dx + gw]
                              for dy in (-1, 0, 1) for dx in (-1, 0, 1)]), axis=0)
        cand = (hm >= thr_logit) & (hm >= nb)
        for gy, gx in zip(*np.nonzero(cand)):  # row-major order == C loop order
            v = hm[gy, gx]
            cx = f32(f32(gx) + off[0, gy, gx]) * f32(stride)
            cy = f32(f32(gy) + off[1, gy, gx]) * f32(stride)
            w, h = wh[0, gy, gx], wh[1, gy, gx]
            if w < 1.0 or h < 1.0:
                continue
            d = dict(cls=c, score=float(_sigmoid32(v)),
                     x=float(f32(cx - w * f32(0.5))), y=float(f32(cy - h * f32(0.5))),
                     w=float(w), h=float(h))
            if len(out) < max_det:
                out.append(d)
            else:
                weakest = 0
                for i in range(1, len(out)):
                    if out[i]["score"] < out[weakest]["score"]:
                        weakest = i
                if out[weakest]["score"] < d["score"]:
                    out[weakest] = d
    return nms(out, 0.5)


def focal_px(hfov_deg=HFOV_DEG):
    return float(f32(W * 0.5) / f32(math.tan(float(f32(hfov_deg * 0.5 * 3.14159265 / 180.0)))))


def person_distance_from_box(w, h, hfov_deg=HFOV_DEG, person_height_m=PERSON_HEIGHT_M):
    """ps_estimate_distances(): est = f_px * person_height / max(w, h)."""
    px = max(float(w), float(h), 1.0)
    return float(f32(focal_px(hfov_deg)) * f32(person_height_m) / f32(px))


def estimate_distances(dets, hfov_deg=HFOV_DEG, person_height_m=PERSON_HEIGHT_M):
    """Adds dist_m / dist_min_m / dist_max_m / truncated to each detection (in place)."""
    for d in dets:
        d["truncated"] = bool(d["x"] <= 0.5 or d["y"] <= 0.5 or
                              d["x"] + d["w"] >= W - 0.5 or d["y"] + d["h"] >= H - 0.5)
        if d["cls"] != 1:
            d["dist_m"] = d["dist_min_m"] = d["dist_max_m"] = 0.0
            continue
        est = person_distance_from_box(d["w"], d["h"], hfov_deg, person_height_m)
        d["dist_m"] = est
        d["dist_min_m"] = est * 0.55
        d["dist_max_m"] = est * 1.15
        if d["truncated"]:
            d["dist_min_m"] = 0.3
    return dets
