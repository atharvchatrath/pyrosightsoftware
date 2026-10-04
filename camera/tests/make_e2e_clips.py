#!/usr/bin/env python3
"""Fake-camera clips for the end-to-end test (tests/e2e_test.js).

Each clip is a 640x480 I420 .y4m (Chromium --use-file-for-fake-video-capture) built
from held-out test images in camera/testdata: every image is letterboxed (95 %) into
the frame, drifts a few pixels so frames differ, and stays for --seconds. The bottom
8 rows carry a frame-number barcode (12 blocks of 32 px from x = 128: 11 bits of frame
index, LSB first, then an even-parity bit) outside the picture, so the test can tell
which frame each detection ran on and compare it with that image's boxes.

    python3 tests/make_e2e_clips.py --clips-dir DIR      # writes DIR/*.y4m + tests/out/e2e_clips.json

Clips (ground truth boxes normalised to the 640x480 frame):
  faces     8 close-up people photos (Open Images, CC BY 2.0)        persons + faces
  group     8 photos with 3-6 people (Open Images, CC BY 2.0)        persons
  nopeople  8 photos without people (Open Images, CC BY 2.0)         nothing
  fire     12 FireNET test images (MIT)                              fire boxes
  doors    10 door photos (Open Images, CC BY 2.0)                   door boxes
  lights   12 hard negatives: lamps, bulbs, sunsets, oranges ... (Open Images, CC BY 2.0), no fire
  still_face, still_group, still_fire, still_door   one image each, for the key screenshots
  firevideo  testdata/fire/clips/fire_road_burning (FireNET repo video1, CGTN/CCTV footage:
             LOCAL TESTING ONLY, not published), fire in every frame, no boxes
"""
import argparse
import json
import os

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
CAMERA = os.path.dirname(HERE)
TD = os.path.join(CAMERA, 'testdata')
W, H = 640, 480
BAR_X0, BAR_BLOCK, BAR_BITS, BAR_Y0 = 128, 32, 12, 472
EXCLUDE = {'d8d6fdd2bdc2e8e3'}


def rgb_to_i420(rgb):
    r, g, b = [rgb[..., i].astype(np.float32) for i in range(3)]
    y = 0.257 * r + 0.504 * g + 0.098 * b + 16
    u = -0.148 * r - 0.291 * g + 0.439 * b + 128
    v = 0.439 * r - 0.368 * g - 0.071 * b + 128
    u = u.reshape(u.shape[0] // 2, 2, u.shape[1] // 2, 2).mean((1, 3))
    v = v.reshape(v.shape[0] // 2, 2, v.shape[1] // 2, 2).mean((1, 3))
    c = lambda a: np.clip(np.round(a), 0, 255).astype(np.uint8).tobytes()  # noqa: E731
    return c(y) + c(u) + c(v)


def barcode(rgb, idx):
    bits = [(idx >> i) & 1 for i in range(BAR_BITS - 1)]
    bits.append(sum(bits) & 1)
    rgb[BAR_Y0:H] = 0
    for i, b in enumerate(bits):
        x0 = BAR_X0 + i * BAR_BLOCK
        rgb[BAR_Y0:H, x0:x0 + BAR_BLOCK] = 255 if b else 0
    return rgb


def place(path, k, n):
    im = Image.open(path).convert('RGB')
    s = min(W / im.size[0], H / im.size[1]) * 0.95
    nw, nh = int(im.size[0] * s), int(im.size[1] * s)
    im = im.resize((nw, nh), Image.BILINEAR)
    dx = int(round(8 * np.sin(2 * np.pi * k / n)))
    ox, oy = (W - nw) // 2 + dx, (H - nh) // 2
    canvas = np.full((H, W, 3), 40, np.uint8)
    canvas[oy:oy + nh, ox:ox + nw] = np.asarray(im)
    return canvas, (ox, oy, nw, nh)


def tr(box, geo):
    ox, oy, nw, nh = geo
    x, y, w, h = box
    return [round((ox + x * nw) / W, 5), round((oy + y * nh) / H, 5), round(w * nw / W, 5), round(h * nh / H, 5)]


def write_slideshow(out, items, fps, seconds):
    n = int(round(fps * seconds))
    segs, idx = [], 0
    with open(out, 'wb') as f:
        f.write(b'YUV4MPEG2 W%d H%d F%d:1 Ip A1:1 C420jpeg\n' % (W, H, fps))
        for it in items:
            seg = {'frames': [idx, idx + n - 1], 'image': os.path.relpath(it['path'], CAMERA), 'credit': it.get('credit'),
                   'boxes_by_frame0': None}
            geo0 = None
            for k in range(n):
                fr, geo = place(it['path'], k, n)
                if geo0 is None:
                    geo0 = geo
                f.write(b'FRAME\n' + rgb_to_i420(barcode(fr, idx)))
                idx += 1
            # boxes for the undrifted placement (drift is at most 8 px = 1.25 % of the width)
            seg['gt'] = {k: [tr(b, geo0) for b in v] for k, v in it['gt'].items()}
            del seg['boxes_by_frame0']
            segs.append(seg)
    return {'file': os.path.basename(out), 'fps': fps, 'n_frames': idx, 'segments': segs}


def reencode_with_barcode(src, out):
    with open(src, 'rb') as f:
        header = f.readline()
        parts = dict((p[0], p[1:]) for p in header.decode().split()[1:])
        w, h = int(parts['W']), int(parts['H'])
        fps = int(parts['F'].split(':')[0]) // max(1, int(parts['F'].split(':')[1]))
        assert (w, h) == (W, H), 'only 640x480 clips'
        fsz = w * h * 3 // 2
        n = 0
        with open(out, 'wb') as g:
            g.write(header)
            while True:
                line = f.readline()
                if not line:
                    break
                buf = bytearray(f.read(fsz))
                if len(buf) < fsz:
                    break
                y = np.frombuffer(buf, np.uint8, w * h).reshape(h, w).copy()
                rgb = np.repeat(y[..., None], 3, 2)
                bar = barcode(rgb.copy(), n)[BAR_Y0:H, :, 0]
                y[BAR_Y0:H] = np.where(bar > 0, 235, 16)
                buf[:w * h] = y.tobytes()
                cw, ch = w // 2, h // 2
                for off in (w * h, w * h + cw * ch):
                    plane = np.frombuffer(buf, np.uint8, cw * ch, off).reshape(ch, cw).copy()
                    plane[BAR_Y0 // 2:] = 128
                    buf[off:off + cw * ch] = plane.tobytes()
                g.write(b'FRAME\n' + bytes(buf))
                n += 1
    return {'file': os.path.basename(out), 'fps': fps, 'n_frames': n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--clips-dir', default=os.path.join(HERE, 'out', 'clips'))
    ap.add_argument('--fps', type=int, default=10)
    ap.add_argument('--seconds', type=float, default=3.0)
    ap.add_argument('--only', default=None, help='comma list of clips to (re)write; the manifest keeps the others')
    a = ap.parse_args()
    only = set(a.only.split(',')) if a.only else None
    os.makedirs(a.clips_dir, exist_ok=True)
    people = json.load(open(os.path.join(TD, 'people', 'manifest.json')))['items']
    fire = json.load(open(os.path.join(TD, 'fire', 'manifest.json')))['images']
    door = json.load(open(os.path.join(TD, 'door', 'manifest.json')))['images']
    neg = json.load(open(os.path.join(TD, 'negatives', 'manifest.json')))['images']

    def pitem(it):
        return {'path': os.path.join(TD, 'people', it['file']), 'credit': '%s, %s' % (it.get('author'), it.get('licence')),
                'gt': {'person': it['persons'], 'face': it['faces']}}

    def box_item(it, base, cls):
        return {'path': os.path.join(TD, base, it['file']),
                'credit': '%s, %s' % (it.get('author', it.get('source')), it.get('license', 'MIT (FireNET)')),
                'gt': {cls: [[b['x'], b['y'], b['w'], b['h']] for b in it['boxes'] if b['cls'] == cls]}}

    closeup = [i for i in people if i['group'] == 'closeup' and i.get('rotation') in ('', '0.0', None)]
    # d8d6fdd2bdc2e8e3: excluded by hand (a nude figure on a bike; not suitable for screenshots)
    multi = [i for i in people if i['group'] == 'multi' and 3 <= len(i['persons']) <= 6 and i.get('rotation') in ('', '0.0', None)
             and i['id'] not in EXCLUDE]
    multi.sort(key=lambda i: -np.mean([p[3] for p in i['persons']]))   # bigger people first
    nop = [i for i in people if i['group'] == 'negative' and i.get('rotation') in ('', '0.0', None)]
    want_roles = ['neg_lamp', 'neg_light_bulb', 'neg_sunset', 'neg_orange', 'neg_pumpkin', 'neg_street_light',
                  'neg_christmas_tree', 'neg_neon', 'neg_dusk', 'neg_sunlight', 'neg_human_hand', 'neg_computer_monitor']
    lights = []
    for r in want_roles:
        cand = [i for i in neg if i['role'] == r]
        if cand:
            lights.append({'path': os.path.join(TD, 'negatives', cand[0]['file']), 'role': r,
                           'credit': '%s, %s' % (cand[0].get('author'), cand[0].get('license')), 'gt': {}})
    clips = {
        'faces': [pitem(i) for i in closeup[::6][:8]],
        'group': [pitem(i) for i in multi[:8]],
        'nopeople': [pitem(i) for i in nop[::6][:8]],
        'fire': [box_item(i, 'fire', 'fire') for i in fire[::7][:12]],
        'doors': [box_item(i, 'door', 'door') for i in door[::6][:10]],
        'lights': lights,
    }
    man_path = os.path.join(HERE, 'out', 'e2e_clips.json')
    old = json.load(open(man_path)) if only and os.path.exists(man_path) else {'clips': {}}
    # single-image clips for the key screenshots (tests/e2e_shots.js): no cuts, so boxes, tracking and
    # the picture on screen always belong together
    byid = {os.path.basename(i['file']).split('.')[0].replace('oi_', ''): i for i in people + door + fire}
    clips['still_face'] = [pitem(byid['d96e2962d8db7d41'])]
    clips['still_group'] = [pitem(byid['f1d99418db05e6a1'])]
    clips['still_fire'] = [box_item(byid['firenet_pic_14'], 'fire', 'fire')]
    clips['still_door'] = [box_item(byid['fbd2f16697562dd1'], 'door', 'door')]
    manifest = {'note': 'generated by tests/make_e2e_clips.py; barcode = frame index (see docstring)',
                'barcode': {'x0': BAR_X0, 'block': BAR_BLOCK, 'bits': BAR_BITS, 'y0': BAR_Y0},
                'clips_dir': a.clips_dir, 'clips': old['clips']}
    for name, items in clips.items():
        if only and name not in only:
            continue
        out = os.path.join(a.clips_dir, 'e2e_%s.y4m' % name)
        manifest['clips'][name] = write_slideshow(out, items, a.fps, a.seconds)
        manifest['clips'][name]['dir'] = a.clips_dir
        print('wrote %s: %d images, %d frames' % (out, len(items), manifest['clips'][name]['n_frames']), flush=True)
    src = os.path.join(TD, 'fire', 'clips', 'fire_road_burning_640x480.y4m')
    if os.path.exists(src) and (not only or 'firevideo' in only):
        out = os.path.join(a.clips_dir, 'e2e_firevideo.y4m')
        c = reencode_with_barcode(src, out)
        c['dir'] = a.clips_dir
        c['segments'] = [{'frames': [0, c['n_frames'] - 1], 'image': 'testdata/fire/clips/fire_road_burning_640x480.y4m',
                          'credit': 'FireNET repo video1.mp4 (CGTN/CCTV footage): local testing only',
                          'gt': {'fire_present': True}}]
        manifest['clips']['firevideo'] = c
        print('wrote %s: %d frames' % (out, c['n_frames']))
    with open(man_path, 'w') as f:
        json.dump(manifest, f, indent=1)


if __name__ == '__main__':
    main()
