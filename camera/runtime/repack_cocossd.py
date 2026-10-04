#!/usr/bin/env python3
"""Re-pack the TF.js COCO-SSD graph model (ssdlite_mobilenet_v2, 18 MB float32)
into a small, self-contained person detector for the PyroSight Camera page.

What it does
  1. Cuts the TensorArray/while-loop preprocessor out of the graph: the new
     input 'image' is the already resized and normalised [1,300,300,3] float
     tensor (people.js does tf.image.resizeBilinear(…, [300,300]) then
     x*2/255-1, exactly what the removed nodes did). The graph then has no
     control flow and runs with model.execute() (sync, tidy-friendly).
  2. Keeps only the requested COCO classes in the six ClassPredictor heads
     (default: person). Scores are per-class sigmoids, so dropping the other
     classes does not change the person score of any anchor.
  3. Stores weights per-output-channel uint8 (default) or float16 in one blob
     using the op-list weight encoding (FORMAT.md), decoded in JS by
     PSOpList.decodeWeights and handed to
     tf.loadGraphModel(tf.io.fromMemory({modelTopology, weightSpecs, weightData})).

    python3 repack_cocossd.py --src ../assets/cocossd --out models/cocossd_person \
        [--keep-classes person] [--weights uint8|float16|float32] [--js]

Writes OUT.json and OUT.weights.bin (and OUT.js with both, base64, for inlining).
"""
import argparse
import base64
import copy
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_oplist import quant_axis  # noqa: E402

# COCO-SSD class table (index = position in the 90-way score vector + 1)
COCO = {1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus', 7: 'train', 8: 'truck',
        9: 'boat', 10: 'traffic light', 11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
        16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear',
        24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase',
        34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat',
        40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle',
        46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple',
        54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut',
        61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table', 70: 'toilet',
        72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave',
        79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock', 86: 'vase',
        87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'}
NAME2ID = {v: k for k, v in COCO.items()}
NUM_LOGITS = 91        # background + 90
SCORES_OUT = 'Postprocessor/Slice'
BOXES_OUT = 'Postprocessor/ExpandDims_1'
NEW_INPUT = 'image'


def load_tfjs(src):
    m = json.load(open(os.path.join(src, 'model.json')))
    wm = m['weightsManifest']
    W = {}
    order = []
    for grp in wm:
        blob = b''.join(open(os.path.join(src, p), 'rb').read() for p in grp['paths'])
        off = 0
        for w in grp['weights']:
            if 'quantization' in w:
                raise ValueError('source model is already quantised')
            n = int(np.prod(w['shape'])) if w['shape'] else 1
            dt = {'float32': np.float32, 'int32': np.int32}[w['dtype']]
            W[w['name']] = np.frombuffer(blob, dt, n, off).reshape(w['shape']).copy()
            order.append(w['name'])
            off += 4 * n
    return m, W, order


def _compact(v):
    if isinstance(v, dict):
        out = {}
        for k, x in v.items():
            x = _compact(x)
            if x == [] or x == {}:
                continue
            out[k] = x
        return out
    if isinstance(v, list):
        return [_compact(x) for x in v]
    return v


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', default=os.path.join(os.path.dirname(__file__), '../assets/cocossd'))
    ap.add_argument('--out', required=True)
    ap.add_argument('--keep-classes', default='person', help="comma list of COCO names, or 'all'")
    ap.add_argument('--weights', default='uint8', choices=['uint8', 'float16', 'float32'])
    ap.add_argument('--min-quant', type=int, default=1024)
    ap.add_argument('--js', action='store_true')
    ap.add_argument('--keep-verbose-topology', action='store_true')
    a = ap.parse_args()

    m, W, order = load_tfjs(a.src)
    topo = copy.deepcopy(m['modelTopology'])
    nodes = topo['node']
    byname = {n['name']: n for n in nodes}

    # ---- 1. new input, drop the preprocessor
    first = [n for n in nodes if 'Preprocessor/sub' in n.get('input', [])]
    assert len(first) == 1 and first[0]['op'] == 'Conv2D', first
    first[0]['input'] = [NEW_INPUT if i == 'Preprocessor/sub' else i for i in first[0]['input']]
    nodes.append({'name': NEW_INPUT, 'op': 'Placeholder', 'input': [],
                  'attr': {'dtype': {'type': 1},
                           'shape': {'shape': {'dim': [{'size': '1'}, {'size': '300'}, {'size': '300'},
                                                       {'size': '3'}]}}}})

    # ---- 2. class pruning
    if a.keep_classes == 'all':
        keep_ids = sorted(COCO)
    else:
        keep_ids = [NAME2ID[c.strip()] for c in a.keep_classes.split(',')]
    cols_per_anchor = [0] + keep_ids              # keep background (dropped later by the Slice)
    K = len(cols_per_anchor)
    for i in range(6):
        wn = 'BoxPredictor_%d/ClassPredictor/weights' % i
        bn = 'BoxPredictor_%d/ClassPredictor/biases' % i
        w, b = W[wn], W[bn]
        A = w.shape[3] // NUM_LOGITS
        cols = [aa * NUM_LOGITS + c for aa in range(A) for c in cols_per_anchor]
        W[wn] = np.ascontiguousarray(w[..., cols])
        W[bn] = np.ascontiguousarray(b[cols])
    assert int(W['BoxPredictor_0/stack_1/2']) == NUM_LOGITS
    W['BoxPredictor_0/stack_1/2'] = np.array(K, np.int32)

    # ---- prune unreachable nodes from the two outputs
    need = set()
    stack = [SCORES_OUT, BOXES_OUT]
    while stack:
        n = stack.pop()
        n = n.split(':')[0].lstrip('^')
        if n in need:
            continue
        need.add(n)
        stack.extend(byname[n]['input'] if n in byname else [])
        if n == NEW_INPUT:
            continue
    byname[NEW_INPUT] = nodes[-1]
    topo['node'] = [n for n in nodes if n['name'] in need]
    consts = [n['name'] for n in topo['node'] if n['op'] == 'Const']
    dropped = [n['name'] for n in nodes if n['name'] not in need]
    pre = [d for d in dropped if not d.startswith('Preprocessor') and d not in ('image_tensor', 'ToFloat')]

    # compact the topology: drop empty attr lists/values (tfjs never needs them)
    if not a.keep_verbose_topology:
        for n in topo['node']:
            n['attr'] = _compact(n.get('attr', {}))
            if n['op'] == 'Const':      # value comes from the weights; keep dtype/shape only
                t = n['attr'].get('value', {}).get('tensor', {})
                n['attr']['value'] = {'tensor': {k: v for k, v in t.items() if k in ('dtype', 'tensorShape')}}

    # which consts are conv / depthwise filters (per-channel quant axis)
    layout = {}
    for n in topo['node']:
        if n['op'] in ('Conv2D',) and len(n['input']) > 1:
            layout[n['input'][1]] = 'hwio'
        if n['op'] == 'DepthwiseConv2dNative':
            layout[n['input'][1]] = 'depthwise'

    # ---- 3. pack weights
    blob = bytearray()
    packed, specs = [], []
    stats = dict(float32=0, float16=0, uint8=0, int32=0)
    errs = {}

    def align():
        while len(blob) % 4:
            blob.append(0)

    for name in consts:
        arr = W[name]
        spec = dict(name=name, shape=list(arr.shape))
        specs.append(dict(name=name, shape=list(arr.shape), dtype='int32' if arr.dtype == np.int32 else 'float32'))
        align()
        if arr.dtype == np.int32:
            spec.update(dtype='int32', offset=len(blob), bytes=arr.nbytes)
            blob += arr.astype('<i4').tobytes()
            stats['int32'] += arr.nbytes
        elif a.weights == 'float32' or arr.size < a.min_quant or name not in layout:
            spec.update(dtype='float32', offset=len(blob), bytes=arr.nbytes)
            blob += arr.astype('<f4').tobytes()
            stats['float32'] += arr.nbytes
        elif a.weights == 'float16':
            h = arr.astype('<f2')
            spec.update(dtype='float16', offset=len(blob), bytes=h.nbytes)
            blob += h.tobytes()
            stats['float16'] += h.nbytes
            errs[name] = float(np.abs(h.astype(np.float32) - arr).max())
        else:
            axis = quant_axis(layout[name], arr)
            ch = int(np.prod(arr.shape[axis:]))
            flat = arr.reshape(-1, ch).astype(np.float64)
            mn, mx = flat.min(0), flat.max(0)
            sc = (mx - mn) / 255.0
            sc[sc == 0] = 1.0
            q = np.clip(np.round((flat - mn) / sc), 0, 255).astype(np.uint8)
            spec.update(dtype='uint8', offset=len(blob), bytes=q.nbytes)
            blob += q.tobytes()
            align()
            spec['quant'] = dict(axis=axis, channels=ch, min_offset=len(blob), scale_offset=len(blob) + 4 * ch)
            blob += mn.astype('<f4').tobytes() + sc.astype('<f4').tobytes()
            stats['uint8'] += q.nbytes + 8 * ch
            deq = (mn.astype(np.float32) + q * sc.astype(np.float32)).reshape(arr.shape)
            errs[name] = float(np.abs(deq - arr).max())
        packed.append(spec)
    align()

    doc = {
        'format': 'pyrosight-tfjs-graph', 'version': 1,
        'modelTopology': topo,
        'weightSpecs': specs,
        'packed': packed,
        'weights_bytes': len(blob),
        'input': {'name': NEW_INPUT, 'shape': [1, 300, 300, 3],
                  'preprocess': 'tf.image.resizeBilinear(rgb, [300,300], false, false) * (2/255) - 1'},
        'outputs': {'scores': SCORES_OUT, 'boxes': BOXES_OUT},
        'classes': [{'index': i, 'coco_id': cid, 'name': COCO[cid]} for i, cid in enumerate(keep_ids)],
        'meta': {
            'source': 'TF.js COCO-SSD lite_mobilenet_v2 (tfjs-models, Apache-2.0)',
            'source_sha256': hashlib.sha256(open(os.path.join(a.src, 'model.json'), 'rb').read()).hexdigest(),
            'weight_storage': a.weights, 'weight_bytes_by_dtype': stats,
            'max_abs_weight_error': max(errs.values()) if errs else 0.0,
            'nodes': len(topo['node']), 'dropped_nodes': len(dropped),
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out + '.json', 'w') as f:
        json.dump(doc, f, separators=(',', ':'))
    with open(a.out + '.weights.bin', 'wb') as f:
        f.write(blob)
    if a.js:
        with open(a.out + '.js', 'w') as f:
            f.write('(globalThis.PS_PEOPLE_ASSETS = globalThis.PS_PEOPLE_ASSETS || {}).person = '
                    '{json: %s, weightsB64: "%s"};\n' % (json.dumps(doc, separators=(',', ':')),
                                                        base64.b64encode(blob).decode()))
    print('kept classes %s; %d nodes (%d dropped, %d non-preprocessor); blob %d bytes %s; '
          'max weight err %.4g' % ([COCO[c] for c in keep_ids], len(topo['node']), len(dropped), len(pre),
                                   len(blob), stats, doc['meta']['max_abs_weight_error']))
    if pre:
        print('  dropped non-preprocessor nodes:', pre[:20])


if __name__ == '__main__':
    main()
