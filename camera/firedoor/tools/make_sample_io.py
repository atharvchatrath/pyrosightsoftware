"""Write export/sample_io/: model inputs and onnxruntime outputs for checking any runtime.

    python3 tools/make_sample_io.py [export/firedoor.onnx]

fire.jpg and door.jpg (inputs kept as they are) and window.jpg (a held-out Open Images test photo,
testdata/window, CC BY 2.0): <name>_input_1x3x256x320.f32 (float32 little-endian, NCHW, x = rgb/127.5 - 1),
<name>_heat_1xCx32x40.f32 (C = 3 classes: fire, door, window), <name>_wh_1x2x32x40.f32,
<name>_off_1x2x32x40.f32, for the window model also <name>_wh_w_1x2x32x40.f32 and <name>_off_w_1x2x32x40.f32
(size and offset of window boxes), and <name>_expected.json (decode.py at the default thresholds).
"""
import json
import os
import shutil
import sys

import cv2
import numpy as np
import onnxruntime as ort

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, R)
from data import preprocess  # noqa: E402
from decode import decode, THRESHOLDS  # noqa: E402

S = os.path.join(R, 'export', 'sample_io')
WINDOW_SRC = os.path.join(R, '..', 'testdata', 'window', 'images', 'oi_4775ce4f3625482e.jpg')   # the still_window photo of tests/make_e2e_clips.py


def main():
    onnx_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(R, 'export', 'firedoor.onnx')
    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    if not os.path.exists(os.path.join(S, 'window.jpg')) and os.path.exists(WINDOW_SRC):
        shutil.copy(WINDOW_SRC, os.path.join(S, 'window.jpg'))
    for name in ('fire', 'door', 'window'):
        jpg = os.path.join(S, name + '.jpg')
        inp = os.path.join(S, name + '_input_1x3x256x320.f32')
        if not os.path.exists(jpg):
            continue
        rgb = cv2.cvtColor(cv2.imread(jpg), cv2.COLOR_BGR2RGB)
        if os.path.exists(inp):
            x = np.fromfile(inp, '<f4').reshape(1, 3, 256, 320)
        else:
            x = preprocess(rgb)[None].astype('<f4')
            x.tofile(inp)
        names = [o.name for o in sess.get_outputs()]
        outs = dict(zip(names, sess.run(names, {'input': x})))
        heat = outs['heat']
        C = heat.shape[1]
        heat.astype('<f4').tofile(os.path.join(S, '%s_heat_1x%dx32x40.f32' % (name, C)))
        for k in names[1:]:
            outs[k].astype('<f4').tofile(os.path.join(S, '%s_%s_1x2x32x40.f32' % (name, k)))
        dets = decode(heat[0], outs['wh'][0], outs['off'][0], wh_w=outs['wh_w'][0] if 'wh_w' in outs else None,
                      off_w=outs['off_w'][0] if 'off_w' in outs else None)
        json.dump(dict(image=name + '.jpg', image_size=list(rgb.shape[:2]), classes=C, thresholds=THRESHOLDS,
                       resize='cv2.resize to 320x256 (INTER_AREA when shrinking), RGB, x/127.5-1, NCHW',
                       detections_default_thresholds=dets), open(os.path.join(S, name + '_expected.json'), 'w'), indent=1)
        print(name, C, 'classes,', len(dets), 'boxes:', [(d['cls'], round(d['score'], 3)) for d in dets])


if __name__ == '__main__':
    main()
