"""Reference decoder for FireDoorNet outputs (mirrors decode.js exactly).

Inputs are the three ONNX outputs for ONE image, NCHW with the batch dim dropped:
  heat [2, GH, GW]  sigmoid probabilities (class 0 fire, 1 door)
  wh   [2, GH, GW]  width, height in model-input pixels (IN_W x IN_H = 320 x 256)
  off  [2, GH, GW]  centre offset x, y within the cell, [0, 1)
Returns a list of dicts {cls: 'fire'|'door', score, x, y, w, h} with x, y = top-left corner and
w, h the size, all normalised to [0, 1] of the input frame (multiply by video width/height).
"""
from __future__ import annotations

import numpy as np

IN_W, IN_H, STRIDE = 320, 256, 8
CLASS_NAMES = ('fire', 'door')
# operating thresholds chosen on validation data (see MODEL.md)
THRESHOLDS = {'fire': 0.50, 'door': 0.35}


def iou(a, b):
    x0, y0 = max(a['x'], b['x']), max(a['y'], b['y'])
    x1, y1 = min(a['x'] + a['w'], b['x'] + b['w']), min(a['y'] + a['h'], b['y'] + b['h'])
    iw, ih = x1 - x0, y1 - y0
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    return inter / (a['w'] * a['h'] + b['w'] * b['h'] - inter)


def decode(heat, wh, off, thresholds=None, max_det=50, nms_iou=0.45, min_score=None):
    """thresholds: dict per class name (defaults to THRESHOLDS); min_score overrides all (for AP)."""
    thr = dict(THRESHOLDS if thresholds is None else thresholds)
    heat = np.asarray(heat, np.float32)
    C, GH, GW = heat.shape
    dets = []
    for c in range(C):
        t = min_score if min_score is not None else thr[CLASS_NAMES[c]]
        hm = heat[c]
        pad = np.pad(hm, 1, mode='constant', constant_values=-1.0)
        nb = np.max(np.stack([pad[1 + dy:1 + dy + GH, 1 + dx:1 + dx + GW]
                              for dy in (-1, 0, 1) for dx in (-1, 0, 1)]), axis=0)
        ys, xs = np.nonzero((hm >= t) & (hm >= nb))
        for gy, gx in zip(ys, xs):
            w, h = float(wh[0, gy, gx]), float(wh[1, gy, gx])
            if w < 1.0 or h < 1.0:
                continue
            cx = (gx + float(off[0, gy, gx])) * STRIDE
            cy = (gy + float(off[1, gy, gx])) * STRIDE
            x0, y0 = max(0.0, cx - w / 2), max(0.0, cy - h / 2)
            x1, y1 = min(float(IN_W), cx + w / 2), min(float(IN_H), cy + h / 2)
            dets.append(dict(cls=CLASS_NAMES[c], score=float(hm[gy, gx]),
                             x=float(x0 / IN_W), y=float(y0 / IN_H), w=float((x1 - x0) / IN_W), h=float((y1 - y0) / IN_H)))
    dets.sort(key=lambda d: -d['score'])           # stable: ties keep (class, row, col) order
    keep = []
    for d in dets:
        if all(k['cls'] != d['cls'] or iou(k, d) <= nms_iou for k in keep):
            keep.append(d)
            if len(keep) >= max_det:
                break
    return keep
