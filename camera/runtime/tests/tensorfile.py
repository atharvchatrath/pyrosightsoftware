"""Tiny float32 tensor container shared by the Python reference scripts and
the node tests: NAME.json = {"arrays": [{"name", "shape", "offset"}], ...},
NAME.bin = little-endian float32 data."""
import json

import numpy as np


def save(prefix, arrays, extra=None):
    meta = {'arrays': []}
    if extra:
        meta.update(extra)
    blob = bytearray()
    for name, a in arrays:
        a = np.ascontiguousarray(a, dtype='<f4')
        meta['arrays'].append({'name': name, 'shape': list(a.shape), 'offset': len(blob)})
        blob += a.tobytes()
    with open(prefix + '.json', 'w') as f:
        json.dump(meta, f)
    with open(prefix + '.bin', 'wb') as f:
        f.write(blob)


def load(prefix):
    meta = json.load(open(prefix + '.json'))
    blob = open(prefix + '.bin', 'rb').read()
    out = {}
    for a in meta['arrays']:
        n = int(np.prod(a['shape'])) if a['shape'] else 1
        out[a['name']] = np.frombuffer(blob, '<f4', n, a['offset']).reshape(a['shape'])
    return out, meta
