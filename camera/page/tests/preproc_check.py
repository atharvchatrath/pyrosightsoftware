#!/usr/bin/env python3
"""Does the page's fire/door preprocessing (bilinear to 2x, then 2x2 average, as in app.js)
match the training/eval preprocessing (cv2 INTER_AREA) closely enough? Runs the ONNX model
with onnxruntime both ways on the held-out fire/door test images and compares outputs.
    python3 tests/preproc_check.py ../firedoor/export/main_best.onnx
"""
import glob, json, os, sys
import cv2, numpy as np, onnxruntime as ort
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'firedoor'))
import decode as D  # firedoor/decode.py
HERE = os.path.dirname(os.path.abspath(__file__))
TD = os.path.join(HERE, '..', '..', 'testdata')
W, H = 320, 256

def ref(rgb):
    im = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA if rgb.shape[1] > W else cv2.INTER_LINEAR)
    return im.astype(np.float32)

def page(rgb):  # tf.image.resizeBilinear(halfPixelCenters) to 2x, then avgPool 2x2 (app.js)
    big = cv2.resize(rgb.astype(np.float32), (2 * W, 2 * H), interpolation=cv2.INTER_LINEAR)
    return big.reshape(H, 2, W, 2, 3).mean(axis=(1, 3))

sess = ort.InferenceSession(sys.argv[1], providers=['CPUExecutionProvider'])
files = sorted(glob.glob(os.path.join(TD, 'fire/images/*.jpg')))[:30] + sorted(glob.glob(os.path.join(TD, 'door/images/*.jpg')))[:30]
pix, heat, agree, n_ref, n_page = [], [], 0, 0, 0
for f in files:
    rgb = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (640, 480), interpolation=cv2.INTER_AREA)  # camera-sized frame
    a, b = ref(rgb), page(rgb)
    pix.append(np.abs(a - b).mean())
    outs = [sess.run(None, {'input': (x.transpose(2, 0, 1)[None] / 127.5 - 1).astype(np.float32)}) for x in (a, b)]
    heat.append(np.abs(outs[0][0] - outs[1][0]).max())
    da = D.decode(*[o[0] for o in outs[0]]) if hasattr(D, 'decode') else []
    db = D.decode(*[o[0] for o in outs[1]]) if hasattr(D, 'decode') else []
    n_ref += len(da); n_page += len(db)
    agree += sum(1 for d in da if any(e['cls'] == d['cls'] and D.iou(d, e) > 0.5 for e in db)) if hasattr(D, 'iou') else 0
print(json.dumps({'images': len(files), 'mean_abs_pixel_diff': float(np.mean(pix)), 'max_heat_diff': float(np.max(heat)),
                  'median_heat_diff': float(np.median(heat)), 'boxes_ref': n_ref, 'boxes_page': n_page,
                  'ref_boxes_matched_iou50': agree}, indent=1))
