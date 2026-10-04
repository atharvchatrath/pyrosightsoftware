"""Build held-out test media for end-to-end page tests in camera/testdata/{fire,door,negatives}.

Everything here comes from splits the model never trained on (FireNET official validation split,
Open Images test/validation images in our 'test' split, FireNET video1.mp4 which is not in training).
Each folder gets manifest.json. Boxes are normalised [0,1] {x, y, w, h} (top-left + size) relative to
the file they describe (image file, or the 640x480 frame of a .y4m / .mp4).

.y4m files (640x480, 15 fps, I420, 'C420jpeg') are for Chromium's fake camera:
  --use-fake-device-for-media-stream --use-file-for-fake-video-capture=/abs/path/file.y4m
"""
from __future__ import annotations

import csv
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from data import load_index  # noqa: E402

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
A = os.path.join(R, '..', 'assets')
T = os.path.join(R, '..', 'testdata')
VIDEO = os.path.join(A, 'firenet/repo/video1.mp4')
FW, FH, FPS = 640, 480, 15


def oi_meta():
    m = {}
    for f in glob.glob(os.path.join(R, 'data/oi/*-with-rotation.csv')):
        for r in csv.DictReader(open(f)):
            m[r['ImageID']] = r
    return m


def write_y4m(path, frames_bgr, fps=FPS):
    h, w = frames_bgr[0].shape[:2]
    with open(path, 'wb') as f:
        f.write(f'YUV4MPEG2 W{w} H{h} F{fps}:1 Ip A1:1 C420jpeg\n'.encode())
        for fr in frames_bgr:
            f.write(b'FRAME\n')
            f.write(cv2.cvtColor(fr, cv2.COLOR_BGR2YUV_I420).tobytes())


def write_mp4(path, frames_bgr, fps=FPS):
    h, w = frames_bgr[0].shape[:2]
    p = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{w}x{h}',
                          '-r', str(fps), '-i', '-', '-c:v', 'libx264', '-threads', '2', '-pix_fmt', 'yuv420p', '-crf', '23',
                          '-movflags', '+faststart', path], stdin=subprocess.PIPE)
    for fr in frames_bgr:
        p.stdin.write(fr.tobytes())
    p.stdin.close()
    p.wait()


def letterbox(img, boxes):
    """Fit img into FWxFH (black bars); boxes [[x0,y0,x1,y1] normalised] -> {x,y,w,h} in frame."""
    h, w = img.shape[:2]
    s = min(FW / w, FH / h)
    nw, nh = int(round(w * s)), int(round(h * s))
    ox, oy = (FW - nw) // 2, (FH - nh) // 2
    fr = np.zeros((FH, FW, 3), np.uint8)
    fr[oy:oy + nh, ox:ox + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    nb = [dict(x=(ox + b[0] * nw) / FW, y=(oy + b[1] * nh) / FH, w=(b[2] - b[0]) * nw / FW, h=(b[3] - b[1]) * nh / FH)
          for b in boxes]
    return fr, nb


def video_segment(a, b, step=2):
    """Frames a..b of video1 (30 fps) at 15 fps, centre-cropped 16:9 -> 4:3 and resized to 640x480."""
    c = cv2.VideoCapture(VIDEO)
    c.set(cv2.CAP_PROP_POS_FRAMES, a)
    out = []
    for i in range(a, b):
        ok, f = c.read()
        if not ok:
            break
        if (i - a) % step:
            continue
        H, W = f.shape[:2]
        cw = H * 4 // 3
        x0 = (W - cw) // 2
        out.append(cv2.resize(f[:, x0:x0 + cw], (FW, FH), interpolation=cv2.INTER_AREA))
    return out


def stable_pick(items, n, key):
    return sorted(items, key=lambda it: hashlib.md5(key(it).encode()).hexdigest())[:n]


def main():
    meta = oi_meta()
    rotated = {k for k, v in meta.items() if v.get('Rotation') not in ('', '0.0', None)}
    for d in ['fire', 'door', 'negatives']:
        os.makedirs(os.path.join(T, d, 'images'), exist_ok=True)
        os.makedirs(os.path.join(T, d, 'clips'), exist_ok=True)

    # ---------------------------------------------------------------- fire
    fire = load_index('test', ['fire_box'])
    man = dict(description='Held-out fire media (never trained on). Boxes normalised {x,y,w,h} top-left+size.',
               images=[], clips=[])
    for it in fire:
        src = os.path.join(R, it['path'])
        dst = 'images/firenet_' + os.path.basename(src).replace(' ', '_').replace('(', '').replace(')', '')
        shutil.copy(src, os.path.join(T, 'fire', dst))
        man['images'].append(dict(file=dst, expect='fire', source='FireNET validation split (MIT)',
                                  boxes=[dict(cls='fire', x=b[1], y=b[2], w=b[3] - b[1], h=b[4] - b[2]) for b in it['boxes']]))
    # slideshow of 8 fire images (8 frames each) -> per-frame GT
    pick = stable_pick([i for i in fire if max((b[3] - b[1]) * (b[4] - b[2]) for b in i['boxes']) > 0.03], 8, lambda i: i['path'])
    frames, per = [], []
    for it in pick:
        fr, nb = letterbox(cv2.imread(os.path.join(R, it['path'])), [b[1:5] for b in it['boxes']])
        for _ in range(8):
            frames.append(fr)
        per.append(dict(frames=[len(frames) - 8, len(frames) - 1], source=os.path.basename(it['path']),
                        boxes=[dict(cls='fire', **b) for b in nb]))
    write_y4m(os.path.join(T, 'fire/clips/fire_slideshow_640x480.y4m'), frames)
    write_mp4(os.path.join(T, 'fire/clips/fire_slideshow_640x480.mp4'), frames)
    man['clips'].append(dict(file='clips/fire_slideshow_640x480.y4m', mp4='clips/fire_slideshow_640x480.mp4', fps=FPS,
                             n_frames=len(frames), expect='fire', segments=per,
                             note='8 FireNET validation images, letterboxed, 8 frames each; boxes per segment.'))
    for name, (a, b) in {'fire_road_ignition': (150, 330), 'fire_road_burning': (600, 780)}.items():
        fr = video_segment(a, b)
        write_y4m(os.path.join(T, f'fire/clips/{name}_640x480.y4m'), fr)
        write_mp4(os.path.join(T, f'fire/clips/{name}_640x480.mp4'), fr)
        man['clips'].append(dict(file=f'clips/{name}_640x480.y4m', mp4=f'clips/{name}_640x480.mp4', fps=FPS, n_frames=len(fr),
                                 expect='fire', source=f'FireNET repo video1.mp4 frames {a}-{b} (every 2nd), centre-cropped to 4:3',
                                 note='CCTV news footage: a small burning scooter mid-road, flames visible in every frame '
                                      '(~3-8% of frame width), smoke. No per-frame boxes; use as fire-present clip.'))
    json.dump(man, open(os.path.join(T, 'fire/manifest.json'), 'w'), indent=1)

    # ---------------------------------------------------------------- door
    door = [i for i in load_index('test', ['door']) if i['oi_id'] not in rotated]
    door = stable_pick(door, 60, lambda i: i['oi_id'])
    man = dict(description='Held-out Open Images (test split here) images with Door boxes (group-of boxes flagged).',
               images=[], clips=[])
    for it in door:
        dst = f"images/oi_{it['oi_id']}.jpg"
        shutil.copy(os.path.join(R, it['path']), os.path.join(T, 'door', dst))
        mm = meta.get(it['oi_id'], {})
        man['images'].append(dict(file=dst, expect='door', source='Open Images V5 ' + it['src'].split('/')[-1],
                                  license=mm.get('License'), author=mm.get('Author'), url=mm.get('OriginalLandingURL'),
                                  boxes=[dict(cls='door', x=b[1], y=b[2], w=b[3] - b[1], h=b[4] - b[2], group_of=bool(b[5]))
                                         for b in it['boxes'] if int(b[0]) == 1]))
    big = [i for i in door if max((b[3] - b[1]) * (b[4] - b[2]) for b in i['boxes'] if int(b[0]) == 1) > 0.08][:8]
    frames, per = [], []
    for it in big:
        fr, nb = letterbox(cv2.imread(os.path.join(R, it['path'])), [b[1:5] for b in it['boxes'] if int(b[0]) == 1])
        frames += [fr] * 8
        per.append(dict(frames=[len(frames) - 8, len(frames) - 1], source=f"oi_{it['oi_id']}.jpg", boxes=[dict(cls='door', **b) for b in nb]))
    write_y4m(os.path.join(T, 'door/clips/door_slideshow_640x480.y4m'), frames)
    write_mp4(os.path.join(T, 'door/clips/door_slideshow_640x480.mp4'), frames)
    man['clips'].append(dict(file='clips/door_slideshow_640x480.y4m', mp4='clips/door_slideshow_640x480.mp4', fps=FPS,
                             n_frames=len(frames), expect='door', segments=per))
    json.dump(man, open(os.path.join(T, 'door/manifest.json'), 'w'), indent=1)

    # ---------------------------------------------------------------- negatives
    neg = [i for i in load_index('test', ['neg']) if i['oi_id'] not in rotated]
    by_role = {}
    for it in neg:
        by_role.setdefault(it['role'], []).append(it)
    man = dict(description='Hard negatives (no fire): lamps, bulbs, sunsets, faces, orange things, screens, indoor. '
                           'Held out from training. Door boxes listed where present (door is allowed there).',
               images=[], clips=[], external=[])
    for role, its in sorted(by_role.items()):
        for it in stable_pick(its, 6 if role != 'neg_human_face' else 12, lambda i: i['oi_id']):
            dst = f"images/oi_{it['oi_id']}.jpg"
            shutil.copy(os.path.join(R, it['path']), os.path.join(T, 'negatives', dst))
            mm = meta.get(it['oi_id'], {})
            man['images'].append(dict(file=dst, expect='no_fire', role=role, license=mm.get('License'), author=mm.get('Author'),
                                      url=mm.get('OriginalLandingURL'),
                                      door_boxes=[dict(cls='door', x=b[1], y=b[2], w=b[3] - b[1], h=b[4] - b[2]) for b in it['boxes'] if int(b[0]) == 1]))
    # close-up face slideshow (simulates a webcam user): face fills ~45% of frame height
    faces = []
    for it in stable_pick(by_role.get('neg_human_face', []) + by_role.get('neg_human_hand', []), 400, lambda i: i['oi_id']):
        im = cv2.imread(os.path.join(R, it['path']))
        H, W = im.shape[:2]
        for f in it['focus']:
            if f[0] != 'Human face':
                continue
            fw, fh = (f[3] - f[1]) * W, (f[4] - f[2]) * H
            if fh < 90:
                continue
            ch = fh / 0.45
            cw = ch * 4 / 3
            cx, cy = (f[1] + f[3]) / 2 * W, (f[2] + f[4]) / 2 * H
            x0, y0 = int(cx - cw / 2), int(cy - ch * 0.45)
            if x0 < 0 or y0 < 0 or x0 + cw > W or y0 + ch > H:
                continue
            faces.append(cv2.resize(im[y0:int(y0 + ch), x0:int(x0 + cw)], (FW, FH), interpolation=cv2.INTER_AREA))
            break
        if len(faces) >= 10:
            break
    frames = [f for f in faces for _ in range(6)]
    write_y4m(os.path.join(T, 'negatives/clips/faces_closeup_640x480.y4m'), frames)
    write_mp4(os.path.join(T, 'negatives/clips/faces_closeup_640x480.mp4'), frames)
    man['clips'].append(dict(file='clips/faces_closeup_640x480.y4m', mp4='clips/faces_closeup_640x480.mp4', fps=FPS,
                             n_frames=len(frames), expect='no_fire', note=f'{len(faces)} close-up faces (Open Images test, CC BY 2.0), 6 frames each'))
    # lamps / lights / sunsets slideshow
    lights = []
    for role in ['neg_lamp', 'neg_light_bulb', 'neg_traffic_light', 'neg_sunset', 'neg_street_light', 'neg_pumpkin',
                 'neg_orange', 'neg_incandescent_light_bulb', 'neg_neon', 'neg_television']:
        for it in stable_pick(by_role.get(role, []), 1, lambda i: i['oi_id']):
            fr, _ = letterbox(cv2.imread(os.path.join(R, it['path'])), [])
            lights.append((role, fr))
    frames = [f for _, f in lights for _ in range(6)]
    write_y4m(os.path.join(T, 'negatives/clips/lights_and_orange_640x480.y4m'), frames)
    write_mp4(os.path.join(T, 'negatives/clips/lights_and_orange_640x480.mp4'), frames)
    man['clips'].append(dict(file='clips/lights_and_orange_640x480.y4m', mp4='clips/lights_and_orange_640x480.mp4', fps=FPS,
                             n_frames=len(frames), expect='no_fire', roles=[r for r, _ in lights], note='6 frames per image'))
    fr = video_segment(0, 96)
    write_y4m(os.path.join(T, 'negatives/clips/road_before_fire_640x480.y4m'), fr)
    write_mp4(os.path.join(T, 'negatives/clips/road_before_fire_640x480.mp4'), fr)
    man['clips'].append(dict(file='clips/road_before_fire_640x480.y4m', mp4='clips/road_before_fire_640x480.mp4', fps=FPS,
                             n_frames=len(fr), expect='no_fire', source='FireNET repo video1.mp4 frames 0-95 (before ignition)'))
    for c in ['Neutral', 'Smoke']:
        man['external'].append(dict(dir=os.path.relpath(os.path.join(A, 'deepquest/FIRE-SMOKE-DATASET/Test', c), T),
                                    expect='no_fire' if c == 'Neutral' else 'smoke (may contain small flames)',
                                    note='DeepQuest FIRE-SMOKE-DATASET Test split; no licence stated, so referenced in place, not copied.'))
    json.dump(man, open(os.path.join(T, 'negatives/manifest.json'), 'w'), indent=1)
    print('fire', len(fire), 'door', len(door), 'neg images', len(man['images']), 'faces', len(faces))


if __name__ == '__main__':
    main()
