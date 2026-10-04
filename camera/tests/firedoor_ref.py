#!/usr/bin/env python3
"""onnxruntime references for the fire/door op-list parity test (firedoor_parity.js).

Runs camera/firedoor/export/firedoor.onnx on held-out test images (camera/testdata
fire/door/negatives + the two export/sample_io inputs), preprocessed exactly as the
deploy path in firedoor/data.py (whole frame stretched to 320x256, INTER_AREA,
x = rgb/127.5 - 1), and stores the inputs (NHWC, the op-list layout) and the
outputs (NCHW, ONNX layout) with runtime/tests/tensorfile.py.

    python3 tests/firedoor_ref.py [--n-fire 12 --n-door 10 --n-neg 8] [-o tests/out/firedoor_ref]
"""
import argparse
import json
import os
import sys

import numpy as np

os.environ.setdefault('OMP_NUM_THREADS', '2')
HERE = os.path.dirname(os.path.abspath(__file__))
CAMERA = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(CAMERA, 'runtime', 'tests'))
sys.path.insert(0, os.path.join(CAMERA, 'firedoor'))
import tensorfile  # noqa: E402
import cv2  # noqa: E402
import onnxruntime as ort  # noqa: E402
from data import preprocess  # noqa: E402  (firedoor deploy preprocessing)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--onnx', default=os.path.join(CAMERA, 'firedoor', 'export', 'firedoor.onnx'))
    ap.add_argument('-o', '--out', default=os.path.join(HERE, 'out', 'firedoor_ref'))
    ap.add_argument('--n-fire', type=int, default=12)
    ap.add_argument('--n-door', type=int, default=10)
    ap.add_argument('--n-neg', type=int, default=8)
    a = ap.parse_args()
    so = ort.SessionOptions()
    so.intra_op_num_threads = 2
    sess = ort.InferenceSession(a.onnx, so, providers=['CPUExecutionProvider'])
    cases, arrays = [], []
    td = os.path.join(CAMERA, 'testdata')
    picks = []
    for kind, n in (('fire', a.n_fire), ('door', a.n_door), ('negatives', a.n_neg)):
        m = json.load(open(os.path.join(td, kind, 'manifest.json')))
        imgs = m['images']
        step = max(1, len(imgs) // max(1, n))
        for it in imgs[::step][:n]:
            picks.append((kind, os.path.join(td, kind, it['file'])))
    for name in ('fire', 'door'):
        raw = np.fromfile(os.path.join(CAMERA, 'firedoor', 'export', 'sample_io', name + '_input_1x3x256x320.f32'), '<f4')
        picks.append(('sample_io', raw.reshape(3, 256, 320)))
    for i, (kind, src) in enumerate(picks):
        if isinstance(src, str):
            bgr = cv2.imread(src, cv2.IMREAD_COLOR)
            x = preprocess(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))       # [3, 256, 320]
            label = '%s/%s' % (kind, os.path.basename(src))
        else:
            x = src
            label = 'sample_io/%d' % i
        x = np.ascontiguousarray(x[None].astype(np.float32))
        heat, wh, off = sess.run(['heat', 'wh', 'off'], {'input': x})
        c = 'c%02d' % i
        cases.append(c)
        arrays.append(('in/' + c, x.transpose(0, 2, 3, 1)))         # NHWC for the op list
        arrays.append(('out/%s/heat' % c, heat))
        arrays.append(('out/%s/wh' % c, wh))
        arrays.append(('out/%s/off' % c, off))
        arrays.append(('label/' + c, np.frombuffer(label.encode().ljust(64)[:64], np.uint8).astype(np.float32)))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    tensorfile.save(a.out, arrays, {'cases': cases, 'onnx': os.path.relpath(a.onnx, CAMERA)})
    print('wrote %s (%d cases)' % (a.out, len(cases)))


if __name__ == '__main__':
    main()
