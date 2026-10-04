#!/usr/bin/env python3
"""Python TFLite reference for the BlazeFace pipeline (independent numpy
implementation of MediaPipe's SsdAnchorsCalculator, TensorsToDetections and
weighted NMS for face_detection_short_range).

For each image: letterbox to 128x128 like MediaPipe (cv2.warpAffine, zero
border, see letterbox()), run the TFLite model with
ai_edge_litert, decode, map back to image coordinates. Saves the 128x128
inputs (tensorfile) so face_parity.js can run the op list on identical input,
and the detections as JSON.

    python3 face_ref_tflite.py LIST.txt OUT_PREFIX [--min-score 0.5]
"""
import argparse
import json
import math
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, os.path.dirname(__file__))
import tensorfile  # noqa: E402

MODEL = os.path.join(os.path.dirname(__file__), '../../assets/blazeface/blaze_face_short_range.tflite')


def anchors():
    strides = [8, 16, 16, 16]
    min_scale, max_scale, n_layers = 0.1484375, 0.75, 4
    out = []
    layer = 0
    while layer < n_layers:
        ars, scales = [], []
        last = layer
        while last < n_layers and strides[last] == strides[layer]:
            scale = min_scale + (max_scale - min_scale) * last / (n_layers - 1)
            ars.append(1.0)
            scales.append(scale)
            nxt = 1.0 if last == n_layers - 1 else min_scale + (max_scale - min_scale) * (last + 1) / (n_layers - 1)
            scales.append(math.sqrt(scale * nxt))
            ars.append(1.0)
            last += 1
        fm = int(math.ceil(128 / strides[layer]))
        for y in range(fm):
            for x in range(fm):
                for _ in range(len(ars)):
                    out.append(((x + 0.5) / fm, (y + 0.5) / fm, 1.0, 1.0))   # fixed_anchor_size
        layer = last
    return np.array(out, np.float64)


def js_round(v):
    return int(math.floor(v + 0.5))


def letterbox(rgb):
    """MediaPipe ImageToTensorCalculator (OpenCV path): the whole image, kept
    aspect, centred in a square ROI, cv2.warpAffine bilinear with a zero
    border on uint8, then mapped to [-1, 1]."""
    H, W = rgb.shape[:2]
    S = max(W, H)
    sc = 128.0 / S
    M = np.array([[sc, 0, (128 - W * sc) / 2], [0, sc, (128 - H * sc) / 2]], np.float32)
    w = cv2.warpAffine(rgb, M, (128, 128), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    x = w.astype(np.float32) / 127.5 - 1
    return x[None], (S, (S - W) / 2.0, (S - H) / 2.0)


def decode(reg, cls, anc, min_score):
    reg = reg.reshape(896, 16).astype(np.float64)
    s = 1 / (1 + np.exp(-np.clip(cls.reshape(896).astype(np.float64), -100, 100)))
    dets = []
    for i in np.nonzero(s >= min_score)[0]:
        ax, ay, aw, ah = anc[i]
        cx = reg[i, 0] / 128 * aw + ax
        cy = reg[i, 1] / 128 * ah + ay
        w = reg[i, 2] / 128 * aw
        h = reg[i, 3] / 128 * ah
        kp = [(reg[i, 4 + 2 * k] / 128 * aw + ax, reg[i, 5 + 2 * k] / 128 * ah + ay) for k in range(6)]
        dets.append(dict(score=float(s[i]), box=[cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], kp=kp))
    return weighted_nms(dets, 0.3)


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def weighted_nms(dets, thr):
    rem = sorted(dets, key=lambda d: -d['score'])
    out = []
    while rem:
        d = rem[0]
        cand = [e for e in rem if iou(d['box'], e['box']) > thr]
        rest = [e for e in rem if iou(d['box'], e['box']) <= thr]
        tot = sum(e['score'] for e in cand)
        box = [sum(e['score'] * e['box'][k] for e in cand) / tot for k in range(4)]
        kp = [(sum(e['score'] * e['kp'][k][0] for e in cand) / tot,
               sum(e['score'] * e['kp'][k][1] for e in cand) / tot) for k in range(6)]
        out.append(dict(score=d['score'], box=box, kp=kp))
        rem = rest
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('list')
    ap.add_argument('out')
    ap.add_argument('--min-score', type=float, default=0.5)
    a = ap.parse_args()
    from ai_edge_litert.interpreter import Interpreter, OpResolverType
    it = Interpreter(model_path=MODEL, experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
    it.allocate_tensors()
    ind = it.get_input_details()[0]['index']
    outs = {o['name']: o['index'] for o in it.get_output_details()}
    anc = anchors()
    assert len(anc) == 896
    files = [l.strip() for l in open(a.list) if l.strip()]
    arrays, res = [], []
    for k, f in enumerate(files):
        rgb = np.asarray(ImageOps.exif_transpose(Image.open(f)).convert('RGB'))
        x, (S, px, py) = letterbox(rgb)
        it.set_tensor(ind, x)
        it.invoke()
        reg = it.get_tensor(outs['regressors'])
        cls = it.get_tensor(outs['classificators'])
        dets = decode(reg, cls, anc, a.min_score)
        ims = []
        for d in dets:
            fx = lambda v: (v * S - px) / rgb.shape[1]  # noqa: E731
            fy = lambda v: (v * S - py) / rgb.shape[0]  # noqa: E731
            x0, y0, x1, y1 = d['box']
            ims.append(dict(score=d['score'], lb=d['box'], lb_kp=d['kp'],
                            x=fx(x0), y=fy(y0), w=fx(x1) - fx(x0), h=fy(y1) - fy(y0),
                            kp=[(fx(p[0]), fy(p[1])) for p in d['kp']]))
        arrays.append(('in/img%d' % k, x))
        res.append(dict(file=f, width=rgb.shape[1], height=rgb.shape[0], letterbox=[S, px, py], faces=ims))
    tensorfile.save(a.out + '_inputs', arrays, {'cases': ['img%d' % k for k in range(len(files))]})
    json.dump({'min_score': a.min_score, 'images': res}, open(a.out + '.json', 'w'))
    print(len(files), 'images,', sum(len(r['faces']) for r in res), 'faces')


if __name__ == '__main__':
    main()
