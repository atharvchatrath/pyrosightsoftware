"""Training dataset: crops, mosaic, photometric augmentation and CenterNet targets.

Each sample yields
  img   [3, H, W] float32, x = rgb / 127.5 - 1
  heat  [C, h, w] gaussian heatmap target (h = H / 8)
  mask  [C, h, w] 1 where that class is supervised (fully), 0 = ignore
  weak  [h, w]    1 inside regions known to contain fire somewhere (MIL), else 0
  wh    [2, h, w] box size in input pixels at object centres
  off   [2, h, w] centre offset in [0,1) at object centres
  reg   [h, w]    1 at object centres
"""
from __future__ import annotations

import json
import math
import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from model import IN_H, IN_W, NUM_CLASSES, STRIDE

R = os.path.dirname(os.path.abspath(__file__))
cv2.setNumThreads(1)


def load_index(split=None, kinds=None):
    items = json.load(open(os.path.join(R, 'data/index.json')))
    return [i for i in items if (split is None or i['split'] == split) and (kinds is None or i['kind'] in kinds)]


def imread(path):
    im = cv2.imread(os.path.join(R, path), cv2.IMREAD_COLOR)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


# ------------------------------------------------------------------ geometry
def choose_crop(W, H, item, rng):
    """Return crop window (x0, y0, cw, ch) in source pixels."""
    a = math.exp(rng.uniform(math.log(0.75), math.log(1.9)))    # crop aspect w/h (portrait..16:9)
    kind = item['kind']
    focus = item.get('focus') or []
    if focus and rng.random() < 0.55:
        f = rng.choice(focus)
        bx0, by0, bx1, by1 = f[1] * W, f[2] * H, f[3] * W, f[4] * H
        bw, bh = max(bx1 - bx0, 4), max(by1 - by0, 4)
        frac = rng.uniform(0.25, 0.85) if f[0] == 'Human face' else rng.uniform(0.15, 0.7)
        ch = max(bh, bw / a) / frac
        cw = ch * a
        cw, ch = min(cw, W), min(ch, H)
        x0 = rng.uniform(min(bx0, bx1 - cw), max(bx0, bx1 - cw)) if cw >= bw else bx0 + (bw - cw) / 2
        y0 = rng.uniform(min(by0, by1 - ch), max(by0, by1 - ch)) if ch >= bh else by0 + (bh - ch) / 2
        x0 = min(max(0, x0), W - cw)
        y0 = min(max(0, y0), H - ch)
        return x0, y0, cw, ch
    if rng.random() < 0.25:
        return 0.0, 0.0, float(W), float(H)
    # largest window of aspect a inside the image, then scale its area
    if W / H > a:
        mh, mw = H, H * a
    else:
        mw, mh = W, W / a
    lo = 0.6 if kind == 'fire_weak' else 0.3
    s = math.sqrt(rng.uniform(lo, 1.0))
    cw, ch = mw * s, mh * s
    return rng.uniform(0, W - cw), rng.uniform(0, H - ch), cw, ch


def crop_resize(im, item, out_w, out_h, rng, crop=None):
    H, W = im.shape[:2]
    x0, y0, cw, ch = crop or choose_crop(W, H, item, rng)
    sx, sy = out_w / cw, out_h / ch
    M = np.float32([[sx, 0, -x0 * sx], [0, sy, -y0 * sy]])
    interp = cv2.INTER_AREA if sx < 1 and sy < 1 else cv2.INTER_LINEAR
    out = cv2.warpAffine(im, M, (out_w, out_h), flags=interp, borderMode=cv2.BORDER_REFLECT)
    boxes, ignores = [], []
    for b in item['boxes']:
        c = int(b[0])
        bx0, by0, bx1, by1 = b[1] * W, b[2] * H, b[3] * W, b[4] * H
        area = max(1e-6, (bx1 - bx0) * (by1 - by0))
        nx0, ny0 = max(bx0, x0), max(by0, y0)
        nx1, ny1 = min(bx1, x0 + cw), min(by1, y0 + ch)
        if nx1 <= nx0 or ny1 <= ny0:
            continue
        vis = (nx1 - nx0) * (ny1 - ny0) / area
        ob = [(nx0 - x0) * sx, (ny0 - y0) * sy, (nx1 - x0) * sx, (ny1 - y0) * sy]
        if vis >= 0.4 and ob[2] - ob[0] >= 2 and ob[3] - ob[1] >= 2:
            boxes.append([c] + ob)
        else:
            ignores.append([c] + ob)
    return out, boxes, ignores


# ------------------------------------------------------------------ photometric
def photometric(img, rng):
    x = img.astype(np.float32)
    if rng.random() < 0.8:          # brightness / contrast
        c = rng.uniform(0.65, 1.35)
        b = rng.uniform(-30, 30)
        x = (x - 128) * c + 128 + b
    if rng.random() < 0.5:          # saturation
        g = x.mean(axis=2, keepdims=True)
        x = g + (x - g) * rng.uniform(0.6, 1.3)
    if rng.random() < 0.3:          # mild hue shift (keeps fire orange/yellow)
        hsv = cv2.cvtColor(np.clip(x, 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.int16)
        hsv[..., 0] = (hsv[..., 0] + rng.randint(-4, 4)) % 180
        x = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
    if rng.random() < 0.3:          # white balance / colour cast
        x = x * np.float32([rng.uniform(0.85, 1.15), rng.uniform(0.9, 1.1), rng.uniform(0.85, 1.15)])
    if rng.random() < 0.25:         # gamma
        x = 255 * (np.clip(x, 0, 255) / 255) ** rng.uniform(0.7, 1.4)
    if rng.random() < 0.25:         # blur (defocus / motion)
        if rng.random() < 0.5:
            x = cv2.GaussianBlur(x, (0, 0), rng.uniform(0.6, 1.6))
        else:
            k = rng.choice([3, 5, 7])
            ker = np.zeros((k, k), np.float32)
            if rng.random() < 0.5:
                ker[k // 2, :] = 1.0 / k
            else:
                ker[:, k // 2] = 1.0 / k
            x = cv2.filter2D(x, -1, ker)
    if rng.random() < 0.3:          # sensor noise
        x = x + np.random.normal(0, rng.uniform(2, 8), x.shape).astype(np.float32)
    x = np.clip(x, 0, 255).astype(np.uint8)
    if rng.random() < 0.25:         # JPEG artefacts
        ok, enc = cv2.imencode('.jpg', x, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(25, 80)])
        x = cv2.imdecode(enc, cv2.IMREAD_UNCHANGED)
    return x


# ------------------------------------------------------------------ targets
def draw_targets(boxes, ignores, sup_tiles, weak_tiles, H=IN_H, W=IN_W):
    """sup_tiles: list of (x0, y0, x1, y1, sup[C]) in input px; weak_tiles: list of rects."""
    h, w = H // STRIDE, W // STRIDE
    heat = np.zeros((NUM_CLASSES, h, w), np.float32)
    mask = np.zeros((NUM_CLASSES, h, w), np.float32)
    weak = np.zeros((h, w), np.float32)
    wh = np.zeros((2, h, w), np.float32)
    off = np.zeros((2, h, w), np.float32)
    reg = np.zeros((h, w), np.float32)
    for (x0, y0, x1, y1, sup) in sup_tiles:
        gx0, gy0 = int(round(x0 / STRIDE)), int(round(y0 / STRIDE))
        gx1, gy1 = int(round(x1 / STRIDE)), int(round(y1 / STRIDE))
        for c in range(NUM_CLASSES):
            mask[c, gy0:gy1, gx0:gx1] = sup[c]
    for (x0, y0, x1, y1) in weak_tiles:
        weak[int(round(y0 / STRIDE)):int(round(y1 / STRIDE)), int(round(x0 / STRIDE)):int(round(x1 / STRIDE))] = 1
    for b in ignores:      # partially visible objects: don't punish either way
        c = b[0]
        mask[c, int(b[2] // STRIDE):int(math.ceil(b[4] / STRIDE)), int(b[1] // STRIDE):int(math.ceil(b[3] / STRIDE))] = 0
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    # draw big boxes first so small ones win centre-cell collisions
    for b in sorted(boxes, key=lambda b: -(b[3] - b[1]) * (b[4] - b[2])):
        c, x0, y0, x1, y1 = b[0], b[1], b[2], b[3], b[4]
        bw, bh = x1 - x0, y1 - y0
        cx, cy = (x0 + x1) / 2 / STRIDE, (y0 + y1) / 2 / STRIDE
        ix, iy = min(int(cx), w - 1), min(int(cy), h - 1)
        sx = max(0.54 * bw / STRIDE / 6, 0.4)
        sy = max(0.54 * bh / STRIDE / 6, 0.4)
        g = np.exp(-((xs - ix) ** 2 / (2 * sx * sx) + (ys - iy) ** 2 / (2 * sy * sy)))
        heat[c] = np.maximum(heat[c], g)
        heat[c, iy, ix] = 1.0
        mask[c, iy, ix] = 1.0
        wh[:, iy, ix] = (bw, bh)
        off[:, iy, ix] = (cx - ix, cy - iy)
        reg[iy, ix] = 1.0
    return heat, mask, weak, wh, off, reg


class TrainSet(Dataset):
    """Draws `length` samples per epoch with fixed proportions per kind."""

    def __init__(self, length=6000, mix=None, mosaic_p=0.25, seed=0):
        items = load_index('train')
        self.by_kind = {}
        for it in items:
            self.by_kind.setdefault(it['kind'], []).append(it)
        self.mix = mix or {'fire_box': 0.32, 'fire_weak': 0.18, 'door': 0.25, 'neg': 0.25}
        self.kinds = list(self.mix)
        self.cum = np.cumsum([self.mix[k] for k in self.kinds])
        self.length = length
        self.mosaic_p = mosaic_p
        self.seed = seed

    def __len__(self):
        return self.length

    def _pick(self, rng):
        k = self.kinds[int(np.searchsorted(self.cum, rng.random() * self.cum[-1]))]
        return rng.choice(self.by_kind[k])

    def _tile(self, rng, x0, y0, tw, th, canvas, boxes, ignores, sup_tiles, weak_tiles):
        it = self._pick(rng)
        im = imread(it['path'])
        t, b, ig = crop_resize(im, it, tw, th, rng)
        if rng.random() < 0.5:
            t = t[:, ::-1]
            b = [[c, tw - x1, y1_, tw - x0_, y2] for c, x0_, y1_, x1, y2 in b]
            ig = [[c, tw - x1, y1_, tw - x0_, y2] for c, x0_, y1_, x1, y2 in ig]
        canvas[y0:y0 + th, x0:x0 + tw] = t
        boxes += [[c, bx0 + x0, by0 + y0, bx1 + x0, by1 + y0] for c, bx0, by0, bx1, by1 in b]
        ignores += [[c, bx0 + x0, by0 + y0, bx1 + x0, by1 + y0] for c, bx0, by0, bx1, by1 in ig]
        sup_tiles.append((x0, y0, x0 + tw, y0 + th, it['sup']))
        if it['kind'] == 'fire_weak':
            weak_tiles.append((x0, y0, x0 + tw, y0 + th))

    def __getitem__(self, idx):
        rng = random.Random((self.seed * 1000003 + idx * 7919 + random.getrandbits(32)))
        canvas = np.zeros((IN_H, IN_W, 3), np.uint8)
        boxes, ignores, sup_tiles, weak_tiles = [], [], [], []
        if rng.random() < self.mosaic_p:
            cx = int(round(rng.uniform(0.3, 0.7) * IN_W / STRIDE)) * STRIDE
            cy = int(round(rng.uniform(0.3, 0.7) * IN_H / STRIDE)) * STRIDE
            for (x0, y0, tw, th) in [(0, 0, cx, cy), (cx, 0, IN_W - cx, cy), (0, cy, cx, IN_H - cy), (cx, cy, IN_W - cx, IN_H - cy)]:
                self._tile(rng, x0, y0, tw, th, canvas, boxes, ignores, sup_tiles, weak_tiles)
        else:
            self._tile(rng, 0, 0, IN_W, IN_H, canvas, boxes, ignores, sup_tiles, weak_tiles)
        img = photometric(canvas, rng)
        heat, mask, weak, wh, off, reg = draw_targets(boxes, ignores, sup_tiles, weak_tiles)
        x = torch.from_numpy(img.astype(np.float32).transpose(2, 0, 1) / 127.5 - 1.0)
        return x, torch.from_numpy(heat), torch.from_numpy(mask), torch.from_numpy(weak), \
            torch.from_numpy(wh), torch.from_numpy(off), torch.from_numpy(reg)


def preprocess(rgb):
    """Eval/deploy preprocessing: stretch the whole frame to IN_W x IN_H, x = rgb/127.5 - 1, NCHW."""
    im = cv2.resize(rgb, (IN_W, IN_H), interpolation=cv2.INTER_AREA if rgb.shape[1] > IN_W else cv2.INTER_LINEAR)
    return im.astype(np.float32).transpose(2, 0, 1) / 127.5 - 1.0
