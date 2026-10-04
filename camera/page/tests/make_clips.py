#!/usr/bin/env python3
"""Test clips for the PyroSight Camera page (fake webcam input for headless Chromium).

    python3 tests/make_clips.py            # writes tests/out/*.y4m (+ pan_grey128.bin, clips.json)

Clips (640x480, 15 fps, Y4M 4:2:0; every frame carries its index as a 10-bit
barcode in the bottom-left 120x6 px so a test can tell which frame is shown):

  pan      camera ROTATING over a 360-degree cylindrical panorama made of four
           Open Images photos (CC BY 2.0): still, pan right to 120 deg, hold,
           pan back to 0, hold. True yaw per frame in clips.json. Also written
           as 128x128 grey float32 frames (pan_grey128.bin) for the node test.
  people   Open Images close-up / small-group photos (CC BY 2.0), 2 s each,
           slowly drifting.
  mixed    a person photo and a FireNET fire photo (MIT) side by side, then a
           person and a door photo: one clip that can show white, purple and
           green boxes together (+ mixed_640x480.webm for the file-upload test).
  still_fire, still_door  the two halves of "mixed", 10 s each, almost still
           (for screenshots: detection takes 1-3 s per frame here), and the
           same pictures as person_fire.jpg / person_door.jpg.
"""
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CAM = os.path.abspath(os.path.join(HERE, '..', '..'))
TD = os.path.join(CAM, 'testdata')
OUT = os.path.join(HERE, 'out')
W, H, FPS = 640, 480, 15
FOV = 65.0
F = (max(W, H) / 2) / np.tan(np.radians(FOV / 2))


def imread_rgb(p):
    im = cv2.imread(p, cv2.IMREAD_COLOR)
    if im is None:
        sys.exit('cannot read ' + p)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def fit_cover(rgb, w, h):
    ih, iw = rgb.shape[:2]
    s = max(w / iw, h / ih)
    r = cv2.resize(rgb, (int(round(iw * s)), int(round(ih * s))), interpolation=cv2.INTER_AREA)
    y0, x0 = (r.shape[0] - h) // 2, (r.shape[1] - w) // 2
    return r[y0:y0 + h, x0:x0 + w]


def fit_contain(rgb, w, h, bg=0):
    ih, iw = rgb.shape[:2]
    s = min(w / iw, h / ih)
    r = cv2.resize(rgb, (int(round(iw * s)), int(round(ih * s))), interpolation=cv2.INTER_AREA)
    out = np.full((h, w, 3), bg, np.uint8)
    y0, x0 = (h - r.shape[0]) // 2, (w - r.shape[1]) // 2
    out[y0:y0 + r.shape[0], x0:x0 + r.shape[1]] = r
    return out


def barcode(frame, idx):
    for b in range(10):
        v = 255 if (idx >> b) & 1 else 0
        frame[H - 6:H, b * 12:(b + 1) * 12] = v
    return frame


class Y4M:
    def __init__(self, path):
        self.f = open(path, 'wb')
        self.f.write(b'YUV4MPEG2 W%d H%d F%d:1 Ip A1:1 C420jpeg\n' % (W, H, FPS))
        self.n = 0

    def write(self, rgb):
        assert rgb.shape == (H, W, 3) and rgb.dtype == np.uint8
        rgb = barcode(rgb.copy(), self.n)
        self.f.write(b'FRAME\n')
        self.f.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2YUV_I420).tobytes())
        self.n += 1
        return rgb

    def close(self):
        self.f.close()


def pan_clip(meta):
    files = ['negatives/images/oi_000e4e7ed48c932d.jpg', 'door/images/oi_73e6ceb9800a3ece.jpg',
             'negatives/images/oi_00ec4ba83d648c33.jpg', 'door/images/oi_0224eb6948c6627e.jpg']
    tiles = [fit_cover(imread_rgb(os.path.join(TD, p)), 640, 480) for p in files]
    pano = np.concatenate(tiles, axis=1)            # 2560 x 480 = 360 degrees around a cylinder
    PW, PH = pano.shape[1], pano.shape[0]
    rcyl = PW / (2 * np.pi)                          # cylinder radius in px
    yaws = [0.0] * 20
    while yaws[-1] < 120:
        yaws.append(min(120.0, yaws[-1] + 3.0))
    yaws += [120.0] * 10
    while yaws[-1] > 0:
        yaws.append(max(0.0, yaws[-1] - 4.0))
    yaws += [0.0] * 15
    u = np.arange(W) - W / 2 + 0.5
    v = np.arange(H) - H / 2 + 0.5
    U, V = np.meshgrid(u, v)
    ang = np.arctan2(U, F)                             # ray azimuth relative to the optical axis
    hcyl = V / np.sqrt(U ** 2 + F ** 2) * rcyl         # ray height where it hits the cylinder
    rng = np.random.default_rng(1)
    y4m = Y4M(os.path.join(OUT, 'pan_640x480.y4m'))
    grey = []
    for i, yaw in enumerate(yaws):
        mapx = ((np.radians(yaw) + ang) * rcyl) % PW
        mapy = hcyl + PH / 2
        fr = cv2.remap(pano, mapx.astype(np.float32), mapy.astype(np.float32), cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REPLICATE)
        gain = 1.0 + 0.05 * np.sin(i / 7.0)            # slight exposure flicker
        fr = np.clip(fr.astype(np.float32) * gain + rng.normal(0, 3.0, fr.shape), 0, 255).astype(np.uint8)
        fr = y4m.write(fr)
        g = cv2.cvtColor(fr, cv2.COLOR_RGB2GRAY)
        grey.append(cv2.resize(g, (128, 128), interpolation=cv2.INTER_AREA).astype(np.float32))
    y4m.close()
    np.stack(grey).tofile(os.path.join(OUT, 'pan_grey128.bin'))
    meta['pan'] = {'file': 'pan_640x480.y4m', 'frames': len(yaws), 'fps': FPS, 'fovDeg': FOV, 'yaw': yaws,
                   'sources': files, 'note': 'cylindrical panorama, pure rotation, Gaussian noise sigma 3'}


def slideshow(name, frames_per, items, meta, drift=1.0):
    y4m = Y4M(os.path.join(OUT, name + '_640x480.y4m'))
    segs = []
    for k, img in enumerate(items):
        segs.append({'frames': [y4m.n, y4m.n + frames_per - 1], 'what': img['what']})
        for i in range(frames_per):
            t = i / max(1, frames_per - 1)
            # slow drift and zoom so consecutive frames differ like a hand-held camera
            s = 1.0 + 0.04 * t * drift
            M = cv2.getRotationMatrix2D((W / 2, H / 2), 0, s)
            M[0, 2] += 10 * drift * np.sin(2 * np.pi * t)
            M[1, 2] += 5 * drift * np.cos(2 * np.pi * t)
            fr = cv2.warpAffine(img['rgb'], M, (W, H), borderMode=cv2.BORDER_REPLICATE)
            y4m.write(fr)
    y4m.close()
    meta[name] = {'file': name + '_640x480.y4m', 'frames': y4m.n, 'fps': FPS, 'segments': segs}


def main():
    os.makedirs(OUT, exist_ok=True)
    meta = {}
    pan_clip(meta)
    man = json.load(open(os.path.join(TD, 'people/manifest.json')))
    by = {it['id']: it for it in man['items']}
    pick = ['d96e2962d8db7d41', '7a23330c4180f7ea', '56fd536256dae19b']
    multi = [it['id'] for it in man['items'] if it['group'] == 'multi'][:1]
    general = [it['id'] for it in man['items'] if it['group'] == 'general'][:2]
    items = []
    for pid in pick + general + multi:
        it = by[pid]
        items.append({'rgb': fit_contain(imread_rgb(os.path.join(TD, 'people', it['file'])), W, H, 20),
                      'what': '%s %s (%s, CC BY 2.0)' % (it['group'], pid, it['author'])})
    slideshow('people', 30, items, meta)

    person = imread_rgb(os.path.join(TD, 'people', by['7a23330c4180f7ea']['file']))
    person2 = imread_rgb(os.path.join(TD, 'people', by['d96e2962d8db7d41']['file']))
    fire = imread_rgb(os.path.join(TD, 'fire/images/firenet_img_10.jpg'))
    door = imread_rgb(os.path.join(TD, 'door/images/oi_4e14c94a0dda414e.jpg'))  # a door the early fire/door model finds
    a = np.concatenate([fit_contain(person, 320, 480, 20), fit_cover(fire, 320, 480)], axis=1)
    b = np.concatenate([fit_contain(person2, 320, 480, 20), fit_cover(door, 320, 480)], axis=1)
    slideshow('mixed', 45, [{'rgb': a, 'what': 'person | fire'}, {'rgb': b, 'what': 'person | door'}], meta)
    # near-still versions for screenshots (detections take 1-3 s here, so the scene must not change meanwhile)
    slideshow('still_fire', 150, [{'rgb': a, 'what': 'person | fire'}], meta, drift=0.1)
    slideshow('still_door', 150, [{'rgb': b, 'what': 'person | door'}], meta, drift=0.1)
    cv2.imwrite(os.path.join(OUT, 'person_fire.jpg'), cv2.cvtColor(a, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(OUT, 'person_door.jpg'), cv2.cvtColor(b, cv2.COLOR_RGB2BGR))
    json.dump(meta, open(os.path.join(OUT, 'clips.json'), 'w'), indent=1)
    # a WebM copy of the mixed clip for the "open a video file" test (Chromium plays VP8 without proprietary codecs)
    import subprocess
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', os.path.join(OUT, 'mixed_640x480.y4m'), '-c:v', 'libvpx',
                    '-b:v', '1M', '-threads', '2', os.path.join(OUT, 'mixed_640x480.webm')], check=True)
    for k, v in meta.items():
        print(k, v['file'], v['frames'], 'frames')


if __name__ == '__main__':
    main()
