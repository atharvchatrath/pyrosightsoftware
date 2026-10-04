#!/usr/bin/env python3
"""Decode the test corpus to raw RGB uint8 for the node tests (node has no
JPEG decoder). Usage: decode_images.py MANIFEST OUT_PREFIX [--with-mediapipe] [extra files
relative to the manifest]. OUT.json = {"images": [{"file", "width", "height", "offset"}]},
OUT.bin = concatenated HxWx3 uint8. EXIF orientation is applied like a browser."""
import json
import os
import sys

import numpy as np
from PIL import Image, ImageOps


def main(manifest, out, extra=()):
    m = json.load(open(manifest))
    base = os.path.dirname(os.path.abspath(manifest))
    extra = list(extra)
    if '--with-mediapipe' in extra:   # add the hand-labelled MediaPipe portraits
        extra.remove('--with-mediapipe')
        extra += [it['file'] for it in json.load(open(os.path.join(base, 'mediapipe.json')))['items']]
    files = [it['file'] for it in m['items']] + extra
    meta, blob = {'images': []}, bytearray()
    for f in files:
        im = ImageOps.exif_transpose(Image.open(base + '/' + f)).convert('RGB')
        a = np.asarray(im, np.uint8)
        meta['images'].append({'file': f, 'width': a.shape[1], 'height': a.shape[0], 'offset': len(blob)})
        blob += a.tobytes()
    json.dump(meta, open(out + '.json', 'w'))
    open(out + '.bin', 'wb').write(blob)
    print(len(files), 'images,', len(blob), 'bytes')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
