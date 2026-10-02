"""Small PIL-only visualisation helpers (no matplotlib dependency)."""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from .thermal import model_code

COLORS = {0: (255, 60, 30), 1: (40, 220, 255)}
NAMES = {0: "fire", 1: "person"}


def _palette():
    # "ironbow"-like: black -> purple -> red -> orange -> yellow -> white
    stops = np.array([[0, 0, 0], [70, 0, 120], [190, 20, 90], [240, 100, 0], [255, 200, 0], [255, 255, 230]], np.float32)
    x = np.linspace(0, len(stops) - 1, 256)
    i = np.clip(x.astype(int), 0, len(stops) - 2)
    f = (x - i)[:, None]
    return (stops[i] * (1 - f) + stops[i + 1] * f).astype(np.uint8)


PAL = _palette()


def render(dc, gt=None, pred=None, scale=3, caption=None):
    """dc: (H, W) deci-C. gt: list of (cls, x, y, w, h). pred: list of dicts from decode.
    Colour uses the model input code, so what you see is what the network sees."""
    code = model_code(dc)
    rgb = PAL[code]
    im = Image.fromarray(rgb).resize((dc.shape[1] * scale, dc.shape[0] * scale), Image.NEAREST)
    dr = ImageDraw.Draw(im)
    for c, x, y, w, h in (gt or []):
        dr.rectangle([x * scale, y * scale, (x + w) * scale - 1, (y + h) * scale - 1], outline=(255, 255, 255), width=1)
    for d in (pred or []):
        col = COLORS[d["cls"]]
        x, y, w, h = d["x"], d["y"], d["w"], d["h"]
        dr.rectangle([x * scale, y * scale, (x + w) * scale - 1, (y + h) * scale - 1], outline=col, width=2)
        dr.text((x * scale + 2, max(0, y * scale - 11)), f"{NAMES[d['cls']]} {d['score']:.2f}", fill=col)
    if caption:
        dr.text((3, 3), caption, fill=(255, 255, 255))
    return im


def grid(images, cols=4):
    w, h = images[0].size
    rows = (len(images) + cols - 1) // cols
    out = Image.new("RGB", (cols * w + (cols - 1) * 4, rows * h + (rows - 1) * 4), (30, 30, 30))
    for i, im in enumerate(images):
        out.paste(im, ((i % cols) * (w + 4), (i // cols) * (h + 4)))
    return out
