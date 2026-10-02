"""Datasets, on-disk format, augmentation and CenterNet target encoding.

On-disk format (also written by the C simulator: build/sim/ps_sim --dump-dataset DIR):
    DIR/images/NNNNNN.bin   int16 little-endian deci-C, 160*120, row-major, RAW frame
                            (before the device's denoise stage)
    DIR/labels/NNNNNN.txt   one line per object: `cls x y w h dist_m`
                            cls 0 = fire, 1 = person; x, y = top-left, pixels;
                            dist_m = 0 when unknown / not a person
    DIR/meta.csv            optional, header `id,smoke`; id is the integer NNNNNN

Network input = ps_thermal_model_code(median3x3(raw)) / 255 to match the device,
which feeds its denoised frame to the model. Pass denoise=False for frames that
were recorded after the device's denoise stage.
"""
from __future__ import annotations

import csv
import math
import os
from multiprocessing import Pool

import numpy as np
import torch
from torch.utils.data import Dataset

from . import synth
from .thermal import H, W, median3x3, model_code

STRIDE = 4
GH, GW = H // STRIDE, W // STRIDE
NUM_CLASSES = 2


# ------------------------------------------------------------------ disk I/O
def write_sample(root, idx, dc, boxes, cls, dist, smoke=None, meta_fh=None):
    os.makedirs(os.path.join(root, "images"), exist_ok=True)
    os.makedirs(os.path.join(root, "labels"), exist_ok=True)
    np.asarray(dc, dtype="<i2").reshape(H, W).tofile(os.path.join(root, "images", f"{idx:06d}.bin"))
    with open(os.path.join(root, "labels", f"{idx:06d}.txt"), "w") as f:
        for c, b, d in zip(cls, boxes, dist):
            f.write(f"{int(c)} {b[0]:.1f} {b[1]:.1f} {b[2]:.1f} {b[3]:.1f} {float(d):.2f}\n")
    if meta_fh is not None and smoke is not None:
        meta_fh.write(f"{idx},{smoke:.2f}\n")


def read_sample(root, idx):
    dc = np.fromfile(os.path.join(root, "images", f"{idx:06d}.bin"), dtype="<i2").reshape(H, W)
    rows = []
    lp = os.path.join(root, "labels", f"{idx:06d}.txt")
    if os.path.exists(lp):
        with open(lp) as f:
            for line in f:
                p = line.split()
                if len(p) >= 5:
                    rows.append([float(v) for v in p[:6]] + [0.0] * (6 - len(p[:6])))
    a = np.array(rows, np.float32).reshape(-1, 6)
    return dc.astype(np.int16), a[:, 1:5].copy(), a[:, 0].astype(np.int64), a[:, 5].copy()


def read_meta(root):
    p = os.path.join(root, "meta.csv")
    out = {}
    if os.path.exists(p):
        with open(p) as f:
            for r in csv.DictReader(f):
                out[int(r["id"])] = float(r["smoke"])
    return out


def _gen_one(args):
    root, i, seed = args
    s = synth.generate(seed)
    write_sample(root, i, s.dc, s.boxes, s.cls, s.dist)
    return i, s.smoke


def generate_dir(root, n, seed0=0, workers=4):
    """Render n synthetic frames into the on-disk format (deterministic in seed0)."""
    os.makedirs(root, exist_ok=True)
    jobs = [(root, i, seed0 + i) for i in range(n)]
    with Pool(workers) as pool:
        res = pool.map(_gen_one, jobs, chunksize=32)
    with open(os.path.join(root, "meta.csv"), "w") as f:
        f.write("id,smoke\n")
        for i, sm in sorted(res):
            f.write(f"{i},{sm:.2f}\n")
    with open(os.path.join(root, "COMPLETE"), "w") as f:
        f.write(f"{n} {seed0}\n")


# ------------------------------------------------------------------ targets
def gaussian_radius(h, w, min_overlap=0.7):
    """CenterNet gaussian radius (in grid cells) for a box of h x w cells."""
    a1, b1, c1 = 1, (h + w), w * h * (1 - min_overlap) / (1 + min_overlap)
    r1 = (b1 + math.sqrt(b1 ** 2 - 4 * a1 * c1)) / 2
    a2, b2, c2 = 4, 2 * (h + w), (1 - min_overlap) * w * h
    r2 = (b2 + math.sqrt(b2 ** 2 - 4 * a2 * c2)) / 2
    a3, b3, c3 = 4 * min_overlap, -2 * min_overlap * (h + w), (min_overlap - 1) * w * h
    r3 = (b3 + math.sqrt(b3 ** 2 - 4 * a3 * c3)) / 2
    return min(r1, r2, r3)


def encode_targets(boxes, cls):
    """boxes (N,4) xywh px -> heat [2,GH,GW] in [0,1], wh [2,GH,GW], off [2,GH,GW], mask [GH,GW].

    Centre cell gx = floor(cx / 4); off = cx / 4 - gx, matching ps_centernet_decode():
    cx = (gx + off) * stride.
    """
    heat = np.zeros((NUM_CLASSES, GH, GW), np.float32)
    wh = np.zeros((2, GH, GW), np.float32)
    off = np.zeros((2, GH, GW), np.float32)
    mask = np.zeros((GH, GW), np.float32)
    ys, xs = np.mgrid[0:GH, 0:GW]
    # Larger boxes first so small objects win shared centre cells.
    order = np.argsort([-b[2] * b[3] for b in boxes]) if len(boxes) else []
    for k in order:
        x, y, w, h = boxes[k]
        if w < 1 or h < 1:
            continue
        cx, cy = (x + w / 2) / STRIDE, (y + h / 2) / STRIDE
        gx, gy = min(max(int(cx), 0), GW - 1), min(max(int(cy), 0), GH - 1)
        r = max(0.0, gaussian_radius(h / STRIDE, w / STRIDE))
        sigma = (2 * int(r) + 1) / 6
        g = np.exp(-((xs - gx) ** 2 + (ys - gy) ** 2) / (2 * sigma * sigma))
        c = int(cls[k])
        np.maximum(heat[c], g, out=heat[c])
        heat[c, gy, gx] = 1.0
        wh[:, gy, gx] = (w, h)
        off[:, gy, gx] = (np.clip(cx - gx, 0, 0.999), np.clip(cy - gy, 0, 0.999))
        mask[gy, gx] = 1.0
    return heat, wh, off, mask


# ------------------------------------------------------------------ augmentation
def augment(dc, boxes, cls, dist, rng):
    """Geometric + radiometric augmentation on the raw deci-C frame."""
    dc = dc.astype(np.float32)
    boxes = boxes.copy(); dist = dist.copy()
    if rng.random() < 0.5:                                   # horizontal flip
        dc = dc[:, ::-1]
        if len(boxes):
            boxes[:, 0] = W - boxes[:, 0] - boxes[:, 2]
    if rng.random() < 0.5:                                   # zoom (in or out) about a random point
        s = rng.uniform(0.75, 1.35)
        cx, cy = rng.uniform(0.3, 0.7) * W, rng.uniform(0.3, 0.7) * H
        src_x = (np.arange(W) - cx) / s + cx
        src_y = (np.arange(H) - cy) / s + cy
        valid_x = (src_x >= 0) & (src_x <= W - 1)
        valid_y = (src_y >= 0) & (src_y <= H - 1)
        xi = np.clip(np.round(src_x).astype(int), 0, W - 1)
        yi = np.clip(np.round(src_y).astype(int), 0, H - 1)
        fill = float(np.median(dc))
        out = dc[yi][:, xi]
        out[~valid_y, :] = fill
        out[:, ~valid_x] = fill
        dc = out
        if len(boxes):
            x0 = (boxes[:, 0] - cx) * s + cx; y0 = (boxes[:, 1] - cy) * s + cy
            x1 = (boxes[:, 0] + boxes[:, 2] - cx) * s + cx; y1 = (boxes[:, 1] + boxes[:, 3] - cy) * s + cy
            x0c, y0c = np.clip(x0, 0, W), np.clip(y0, 0, H)
            x1c, y1c = np.clip(x1, 0, W), np.clip(y1, 0, H)
            keep = ((x1c - x0c) >= 2) & ((y1c - y0c) >= 2) & \
                   ((x1c - x0c) * (y1c - y0c) >= 0.25 * (x1 - x0) * (y1 - y0))
            boxes = np.stack([x0c, y0c, x1c - x0c, y1c - y0c], 1)[keep]
            cls = cls[keep]; dist = dist[keep] / s
    # radiometric: Lepton absolute accuracy is several C; gain error a few %.
    dc = dc * rng.uniform(0.97, 1.03) + rng.normal(0, 15)
    if rng.random() < 0.5:
        dc = dc + rng.normal(0, rng.uniform(1, 4), dc.shape)
    return np.clip(np.round(dc), -400, 6500).astype(np.int16), boxes.astype(np.float32), cls, dist.astype(np.float32)


def to_input(dc, denoise=True):
    """Raw deci-C frame -> float32 network input (1, H, W) = code / 255."""
    if denoise:
        dc = median3x3(dc)
    return (model_code(dc).astype(np.float32) / 255.0)[None]


# ------------------------------------------------------------------ datasets
class _Base(Dataset):
    def __init__(self, augment=False, denoise=True, seed=0):
        self.do_aug = augment
        self.denoise = denoise
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, e):
        self.epoch = e

    def raw(self, i):
        raise NotImplementedError

    def __getitem__(self, i):
        dc, boxes, cls, dist, smoke = self.raw(i)
        if self.do_aug:
            rng = np.random.default_rng((self.seed, self.epoch, i))
            dc, boxes, cls, dist = augment(dc, boxes, cls, dist, rng)
        x = to_input(dc, self.denoise)
        heat, wh, off, mask = encode_targets(boxes, cls)
        return dict(x=torch.from_numpy(x), heat=torch.from_numpy(heat), wh=torch.from_numpy(wh),
                    off=torch.from_numpy(off), mask=torch.from_numpy(mask), idx=i, smoke=float(smoke))


class SynthDataset(_Base):
    """On-the-fly synthetic frames; frame i is synth.generate(seed0 + i)."""

    def __init__(self, n, seed0=0, augment=False, denoise=True):
        super().__init__(augment, denoise, seed0)
        self.n, self.seed0 = n, seed0

    def __len__(self):
        return self.n

    def raw(self, i):
        s = synth.generate(self.seed0 + i)
        return s.dc, s.boxes, s.cls, s.dist, s.smoke


class DiskDataset(_Base):
    """Directory in the on-disk format above (synthetic, simulator dump or real)."""

    def __init__(self, root, augment=False, denoise=True, seed=0):
        super().__init__(augment, denoise, seed)
        self.root = root
        self.ids = sorted(int(f[:-4]) for f in os.listdir(os.path.join(root, "images")) if f.endswith(".bin"))
        self.smoke = read_meta(root)

    def __len__(self):
        return len(self.ids)

    def raw(self, i):
        dc, boxes, cls, dist = read_sample(self.root, self.ids[i])
        return dc, boxes, cls, dist, self.smoke.get(self.ids[i], 0.0)


def collate(batch):
    out = {k: torch.stack([b[k] for b in batch]) for k in ("x", "heat", "wh", "off", "mask")}
    out["idx"] = [b["idx"] for b in batch]
    out["smoke"] = [b["smoke"] for b in batch]
    return out
