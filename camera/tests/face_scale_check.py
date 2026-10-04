#!/usr/bin/env python3
"""Distance-label sanity check, step 1: decode the 50 close-up people photos (camera/testdata/people,
Open Images, CC BY 2.0) to raw RGB for tests/face_scale_check.js.
    python3 tests/face_scale_check.py OUTDIR"""
import json, os, sys
import numpy as np
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__)); TD = os.path.join(os.path.dirname(HERE), 'testdata', 'people')
out = sys.argv[1]; os.makedirs(out, exist_ok=True)
items = [i for i in json.load(open(os.path.join(TD, 'manifest.json')))['items'] if i['group'] == 'closeup' and i.get('rotation') in ('', '0.0', None)]
meta, blob = [], bytearray()
for it in items:
    a = np.asarray(Image.open(os.path.join(TD, it['file'])).convert('RGB'))
    meta.append({'file': it['file'], 'w': a.shape[1], 'h': a.shape[0], 'offset': len(blob), 'faces': it['faces']})
    blob += a.tobytes()
open(os.path.join(out, 'closeups.bin'), 'wb').write(blob)
json.dump(meta, open(os.path.join(out, 'closeups.json'), 'w'))
print(len(meta), 'images')
