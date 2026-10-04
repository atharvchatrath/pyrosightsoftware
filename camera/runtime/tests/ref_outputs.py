#!/usr/bin/env python3
"""Reference outputs for the parity tests.

Runs a .tflite (ai_edge_litert, no delegate) or .onnx (onnxruntime, CPU)
model on random inputs and on real images and stores inputs (NHWC, the op-list
layout) and outputs (source layout) with tensorfile.save().

    python3 ref_outputs.py MODEL OUT_PREFIX [--random 4] [--images 'glob' --n-images 8]
                           [--range -1 1]
"""
import argparse
import glob
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
import tensorfile  # noqa: E402


def letterbox(path, h, w):
    im = Image.open(path).convert('RGB')
    s = min(w / im.size[0], h / im.size[1])
    nw, nh = max(1, round(im.size[0] * s)), max(1, round(im.size[1] * s))
    im = im.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new('RGB', (w, h))
    canvas.paste(im, ((w - nw) // 2, (h - nh) // 2))
    return np.asarray(canvas, np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('model')
    ap.add_argument('out')
    ap.add_argument('--random', type=int, default=4)
    ap.add_argument('--images', default=None)
    ap.add_argument('--n-images', type=int, default=8)
    ap.add_argument('--range', type=float, nargs=2, default=[-1.0, 1.0],
                    help='value range inputs are mapped to (random and images)')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    if a.model.endswith('.tflite'):
        from ai_edge_litert.interpreter import Interpreter, OpResolverType
        it = Interpreter(model_path=a.model,
                         experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
        it.allocate_tensors()
        ind = it.get_input_details()[0]
        outd = it.get_output_details()
        shape_nhwc = list(ind['shape'])

        def run(x):
            it.set_tensor(ind['index'], x)
            it.invoke()
            return [(o['name'], it.get_tensor(o['index']).copy()) for o in outd]
        in_name = ind['name']
    else:
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = 2
        sess = ort.InferenceSession(a.model, so, providers=['CPUExecutionProvider'])
        inp = sess.get_inputs()[0]
        n, c, h, w = [d if isinstance(d, int) else 1 for d in inp.shape]
        shape_nhwc = [n, h, w, c]
        in_name = inp.name

        def run(x):
            outs = sess.run(None, {in_name: np.transpose(x, (0, 3, 1, 2)).copy()})
            return [(o.name, v) for o, v in zip(sess.get_outputs(), outs)]

    lo, hi = a.range
    rng = np.random.default_rng(a.seed)
    arrays = []
    names = []
    for i in range(a.random):
        x = rng.uniform(lo, hi, shape_nhwc).astype(np.float32)
        names.append('random%d' % i)
        arrays.append(('in/random%d' % i, x))
        for on, v in run(x):
            arrays.append(('out/random%d/%s' % (i, on), v))
    if a.images:
        files = sorted(glob.glob(a.images))[:a.n_images]
        for k, f in enumerate(files):
            img = letterbox(f, shape_nhwc[1], shape_nhwc[2])
            if shape_nhwc[3] == 1:
                img = img.mean(-1, keepdims=True)
            x = (img / 255.0 * (hi - lo) + lo)[None].astype(np.float32)
            nm = 'img%d' % k
            names.append(nm)
            arrays.append(('in/' + nm, x))
            for on, v in run(x):
                arrays.append(('out/%s/%s' % (nm, on), v))
    tensorfile.save(a.out, arrays, {'cases': names, 'input_name': in_name, 'model': os.path.basename(a.model)})
    print('saved %d cases to %s' % (len(names), a.out))


if __name__ == '__main__':
    main()
