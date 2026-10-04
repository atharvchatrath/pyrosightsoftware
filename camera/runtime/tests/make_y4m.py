#!/usr/bin/env python3
"""Write Y4M clips for Chromium's fake camera (--use-file-for-fake-video-capture).
    python3 make_y4m.py OUT.y4m IMAGE [IMAGE ...] [--seconds 3] [--fps 15]
Each image is letterboxed into 640x480 and drifts slowly (so frames differ);
the images are shown one after another."""
import argparse

import numpy as np
from PIL import Image


def rgb_to_i420(rgb):
    r, g, b = [rgb[..., i].astype(np.float32) for i in range(3)]
    y = 0.257 * r + 0.504 * g + 0.098 * b + 16
    u = -0.148 * r - 0.291 * g + 0.439 * b + 128
    v = 0.439 * r - 0.368 * g - 0.071 * b + 128
    u = u.reshape(u.shape[0] // 2, 2, u.shape[1] // 2, 2).mean((1, 3))
    v = v.reshape(v.shape[0] // 2, 2, v.shape[1] // 2, 2).mean((1, 3))
    c = lambda a: np.clip(np.round(a), 0, 255).astype(np.uint8).tobytes()  # noqa: E731
    return c(y) + c(u) + c(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('images', nargs='+')
    ap.add_argument('--seconds', type=float, default=3)
    ap.add_argument('--fps', type=int, default=15)
    a = ap.parse_args()
    W, H = 640, 480
    with open(a.out, 'wb') as f:
        f.write(b'YUV4MPEG2 W%d H%d F%d:1 Ip A1:1 C420jpeg\n' % (W, H, a.fps))
        for path in a.images:
            im = Image.open(path).convert('RGB')
            s = min(W / im.size[0], H / im.size[1]) * 0.95
            im = im.resize((int(im.size[0] * s), int(im.size[1] * s)), Image.BILINEAR)
            n = int(a.seconds * a.fps)
            for k in range(n):
                canvas = Image.new('RGB', (W, H), (40, 40, 40))
                dx = int(8 * np.sin(2 * np.pi * k / n))
                canvas.paste(im, ((W - im.size[0]) // 2 + dx, (H - im.size[1]) // 2))
                f.write(b'FRAME\n' + rgb_to_i420(np.asarray(canvas)))
    print('wrote', a.out)


if __name__ == '__main__':
    main()
