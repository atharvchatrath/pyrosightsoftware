"""Draw detections of the ONNX model on a grid of held-out images -> runs/vis_<name>.jpg
python3 tools/visualize.py export/firedoor.onnx
"""
import glob
import json
import os
import sys

import cv2
import numpy as np
import onnxruntime as ort

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from data import preprocess  # noqa: E402
from decode import decode, THRESHOLDS  # noqa: E402

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
T = os.path.join(R, '..', 'testdata')
COL = {'fire': (255, 80, 200), 'door': (80, 255, 40)}   # BGR of #C850FF and #28FF50


def draw(sess, path, thr):
    bgr = cv2.imread(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, wh, off = sess.run(None, {'input': preprocess(rgb)[None]})
    dets = decode(h[0], wh[0], off[0], thresholds=thr)
    im = cv2.resize(bgr, (320, 240))
    for d in dets:
        x0, y0 = int(d['x'] * 320), int(d['y'] * 240)
        x1, y1 = int((d['x'] + d['w']) * 320), int((d['y'] + d['h']) * 240)
        cv2.rectangle(im, (x0, y0), (x1, y1), COL[d['cls']], 2)
        cv2.putText(im, f"{d['cls'].upper()} {d['score']:.2f}", (x0 + 2, max(10, y0 + 12)), 0, 0.4, COL[d['cls']], 1)
    return im


def main():
    sess = ort.InferenceSession(sys.argv[1], providers=['CPUExecutionProvider'])
    thr = dict(THRESHOLDS)
    if len(sys.argv) > 2:
        thr = json.loads(sys.argv[2])
    sets = {
        'fire': [os.path.join(T, 'fire', i['file']) for i in json.load(open(os.path.join(T, 'fire/manifest.json')))['images']][::5][:16],
        'door': [os.path.join(T, 'door', i['file']) for i in json.load(open(os.path.join(T, 'door/manifest.json')))['images']][:16],
        'neg': [os.path.join(T, 'negatives', i['file']) for i in json.load(open(os.path.join(T, 'negatives/manifest.json')))['images']][::4][:24],
    }
    for name, paths in sets.items():
        tiles = [draw(sess, p, thr) for p in paths]
        while len(tiles) % 4:
            tiles.append(np.zeros_like(tiles[0]))
        grid = np.concatenate([np.concatenate(tiles[i:i + 4], 1) for i in range(0, len(tiles), 4)], 0)
        cv2.imwrite(os.path.join(R, 'runs', f'vis_{name}.jpg'), grid)


if __name__ == '__main__':
    main()
