"""Frame-level fire recall / false alarms on FireNET video1.mp4 (all frames, 30 fps), with and without a
simple temporal filter ("show FIRE only if fire was detected in >= k of the last n frames").

python3 tools/video_eval.py export/firedoor.onnx 0.45
"""
import json
import os
import sys

import cv2
import numpy as np
import onnxruntime as ort

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from data import preprocess  # noqa: E402
from decode import decode  # noqa: E402
from eval_full import VIDEO, VIDEO_FIRE, VIDEO_NOFIRE  # noqa: E402

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')


def main():
    so = ort.SessionOptions()
    so.intra_op_num_threads = int(os.environ.get('ORT_THREADS', '2'))
    sess = ort.InferenceSession(sys.argv[1], so, providers=['CPUExecutionProvider'])
    thr = float(sys.argv[2]) if len(sys.argv) > 2 else 0.45
    c = cv2.VideoCapture(VIDEO)
    scores, labels = [], []
    i = 0
    while True:
        ok, f = c.read()
        if not ok:
            break
        h, wh, off = sess.run(None, {'input': preprocess(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))[None]})
        d = decode(h[0], wh[0], off[0], min_score=0.05)
        scores.append(max([x['score'] for x in d if x['cls'] == 'fire'], default=0.0))
        labels.append(1 if any(a <= i <= b for a, b in VIDEO_FIRE) else (0 if any(a <= i <= b for a, b in VIDEO_NOFIRE) else -1))
        i += 1
    s, lab = np.array(scores), np.array(labels)
    res = dict(frames=len(s), fire_frames=int((lab == 1).sum()), nofire_frames=int((lab == 0).sum()), thr=thr, filters={})
    for k, n in [(1, 1), (2, 3), (3, 5)]:
        det = s >= thr
        filt = np.array([det[max(0, j - n + 1):j + 1].sum() >= k for j in range(len(det))])
        res['filters'][f'{k}_of_{n}'] = dict(fire_frame_recall=float(filt[lab == 1].mean()),
                                            nofire_frame_false_alarm=float(filt[lab == 0].mean()))
    # hysteresis: switch FIRE on at score >= thr, keep it on while score >= thr_low
    for low in (0.35, 0.4):
        on, hy = False, []
        for v in s:
            on = v >= thr or (on and v >= low)
            hy.append(on)
        hy = np.array(hy)
        res['filters'][f'hysteresis_on{thr}_keep{low}'] = dict(fire_frame_recall=float(hy[lab == 1].mean()),
                                                              nofire_frame_false_alarm=float(hy[lab == 0].mean()))
    for t in (0.35, 0.4, 0.45, 0.5):
        res['filters'][f'raw_thr{t}'] = dict(fire_frame_recall=float((s >= t)[lab == 1].mean()),
                                             nofire_frame_false_alarm=float((s >= t)[lab == 0].mean()))
    print(json.dumps(res, indent=1))
    json.dump(res, open(os.path.join(R, 'runs', 'video1_eval.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
