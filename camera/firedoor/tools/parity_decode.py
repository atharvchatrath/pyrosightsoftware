"""Check decode.js == decode.py on real ONNX outputs (and NHWC layout handling).

Runs the ONNX model on held-out images, writes raw outputs to runs/parity/*.bin, decodes in Python,
then runs node tools/parity_decode.js and compares every box.
"""
import json
import os
import subprocess
import sys

import numpy as np
import onnxruntime as ort

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from data import imread, load_index, preprocess  # noqa: E402
from decode import decode  # noqa: E402

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
P = os.path.join(R, 'runs', 'parity')
os.makedirs(P, exist_ok=True)
sess = ort.InferenceSession(os.path.join(R, 'export/firedoor.onnx'), providers=['CPUExecutionProvider'])
items = load_index('test', ['fire_box'])[:10] + load_index('test', ['door'])[:10] + load_index('test', ['neg'])[:10]
cases = []
for k, it in enumerate(items):
    h, wh, off = sess.run(None, {'input': preprocess(imread(it['path']))[None]})
    np.concatenate([h.ravel(), wh.ravel(), off.ravel()]).astype('<f4').tofile(os.path.join(P, f'{k}.bin'))
    for ms in [None, 0.05]:
        cases.append(dict(k=k, minScore=ms, py=decode(h[0], wh[0], off[0], min_score=ms)))
json.dump(cases, open(os.path.join(P, 'py.json'), 'w'))
out = subprocess.run(['node', os.path.join(R, 'tools/parity_decode.js'), P], capture_output=True, text=True)
print(out.stdout[-2000:], out.stderr[-2000:])
