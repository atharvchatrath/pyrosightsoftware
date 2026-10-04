"""Full held-out evaluation of the exported ONNX model (or a PyTorch checkpoint).

python3 eval_full.py export/firedoor.onnx  [--split test|val]  -> runs/eval_<split>.json
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import os

import cv2
import numpy as np
import onnxruntime as ort

from data import imread, load_index, preprocess
from decode import decode
from evaluate import ap_and_pr, fp_per_image, image_present_rate

R = os.path.dirname(os.path.abspath(__file__))
A = os.path.join(R, '..', 'assets')
VIDEO = os.path.join(A, 'firenet/repo/video1.mp4')
# frame ranges labelled by eye from runs/video1_contact30.jpg (30 fps source)
VIDEO_FIRE = [(120, 440), (570, 790), (930, 1180)]       # flames clearly visible
VIDEO_NOFIRE = [(0, 100), (1195, 1266)]                   # road before ignition, end card
THRS = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]


def rotated_ids():
    rot = set()
    for f in glob.glob(os.path.join(R, 'data/oi/*-with-rotation.csv')):
        for r in csv.DictReader(open(f)):
            if r.get('Rotation') not in ('', '0.0', None):
                rot.add(r['ImageID'])
    return rot


class Runner:
    def __init__(self, path):
        so = ort.SessionOptions()
        so.intra_op_num_threads = int(os.environ.get("ORT_THREADS", "2"))
        self.s = ort.InferenceSession(path, so, providers=['CPUExecutionProvider'])

    def __call__(self, rgbs):
        out = []
        for r in rgbs:
            h, wh, off = self.s.run(None, {'input': preprocess(r)[None]})
            out.append((h[0], wh[0], off[0]))
        return out


def run_items(run, items):
    outs = []
    for i in range(0, len(items), 32):
        outs += run([imread(it['path']) for it in items[i:i + 32]])
    return outs


def run_paths(run, paths):
    outs = []
    for p in paths:
        im = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
        outs += run([im])
    return outs


def dets_of(outs):
    return [decode(h, w, o, min_score=0.05) for h, w, o in outs]


def video_frames(step=3):
    c = cv2.VideoCapture(VIDEO)
    n = int(c.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for i in range(n):
        ok, f = c.read()
        if not ok:
            break
        if i % step == 0:
            lab = 1 if any(a <= i <= b for a, b in VIDEO_FIRE) else (0 if any(a <= i <= b for a, b in VIDEO_NOFIRE) else -1)
            if lab >= 0:
                frames.append((i, lab, cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('model')
    ap.add_argument('--split', default='test')
    args = ap.parse_args()
    run = Runner(args.model)
    rot = rotated_ids()
    n_rot = {}

    def keep(its, name):
        k = [i for i in its if i.get('oi_id') not in rot]
        n_rot[name] = len(its) - len(k)
        return k
    sp = args.split
    sets = {
        'firenet': load_index(sp, ['fire_box']),
        'oi_door': keep(load_index(sp, ['door']), 'oi_door'),
        'oi_neg': keep(load_index(sp, ['neg']), 'oi_neg'),
        'oi_fire_weak': keep(load_index(sp, ['fire_weak']), 'oi_fire_weak'),
    }
    D = {k: dets_of(run_items(run, v)) for k, v in sets.items()}
    res = dict(split=sp, model=os.path.relpath(args.model, R), counts={k: len(v) for k, v in sets.items()},
               excluded_rotated_oi_images=n_rot)
    if sp == 'test':
        dq = {c: sorted(glob.glob(os.path.join(A, 'deepquest/FIRE-SMOKE-DATASET/Test', c, '*'))) for c in ['Fire', 'Neutral', 'Smoke']}
        for c, ps in dq.items():
            D['dq_' + c] = dets_of(run_paths(run, ps))
            res['counts']['dq_' + c] = len(ps)
        vf = video_frames()
        vo = run([f for _, _, f in vf])
        D['video_fire'] = dets_of([o for o, (_, l, _) in zip(vo, vf) if l == 1])
        D['video_nofire'] = dets_of([o for o, (_, l, _) in zip(vo, vf) if l == 0])
        res['counts']['video_fire_frames'] = len(D['video_fire'])
        res['counts']['video_nofire_frames'] = len(D['video_nofire'])
    # AP
    res['fire_ap50_firenet'] = ap_and_pr(D['firenet'], sets['firenet'], 0, 0.5)['ap']
    door_items = sets['oi_door'] + sets['oi_neg']
    door_dets = D['oi_door'] + D['oi_neg']
    res['door_ap50_oi'] = ap_and_pr(door_dets, door_items, 1, 0.5)['ap']
    res['door_gt_boxes'] = ap_and_pr(door_dets, door_items, 1, 0.5)['n_gt']
    # cleaner numbers: (a) AP on images with a verified Door label only (their doors are exhaustively boxed),
    # (b) door false alarms on images where annotators verified 'no Door'
    res['door_ap50_oi_door_images_only'] = ap_and_pr(D['oi_door'], sets['oi_door'], 1, 0.5)['ap']
    import pickle
    oi = pickle.load(open(os.path.join(R, 'data/oi/oi_cache.pkl'), 'rb'))
    no_door = oi['neg'].get('Door', set())
    vnd = [(d, it) for d, it in zip(D['oi_neg'], sets['oi_neg']) if it.get('oi_id') in no_door]
    res['counts']['oi_neg_verified_no_door'] = len(vnd)
    # threshold sweeps
    fire_sweep, door_sweep = [], []
    for t in THRS:
        f = ap_and_pr(D['firenet'], sets['firenet'], 0, t)
        row = dict(thr=t, firenet_recall=f['recall'], firenet_precision=f['precision'],
                   firenet_image_recall=image_present_rate(D['firenet'], 'fire', t),
                   oi_fire_weak_image_recall=image_present_rate(D['oi_fire_weak'], 'fire', t))
        fp, rate = fp_per_image(D['oi_neg'] + D['oi_door'], 'fire', t)
        row.update(oi_neg_fire_fp_per_img=fp, oi_neg_fire_img_rate=rate)
        if sp == 'test':
            row['dq_fire_image_recall'] = image_present_rate(D['dq_Fire'], 'fire', t)
            row['dq_neutral_fp_per_img'], row['dq_neutral_img_rate'] = fp_per_image(D['dq_Neutral'], 'fire', t)
            row['dq_smoke_fire_rate'] = image_present_rate(D['dq_Smoke'], 'fire', t)
            row['video_fire_frame_recall'] = image_present_rate(D['video_fire'], 'fire', t)
            row['video_nofire_frame_fa'] = image_present_rate(D['video_nofire'], 'fire', t)
        fire_sweep.append(row)
        d = ap_and_pr(door_dets, door_items, 1, t)
        drow = dict(thr=t, door_recall=d['recall'], door_precision=d['precision'])
        nd = [x for x, it in zip(D['oi_neg'], sets['oi_neg']) if not any(int(b[0]) == 1 for b in it['boxes'])]
        drow['door_fp_per_img_on_doorless_oi_neg'], drow['door_img_rate_doorless_oi_neg'] = fp_per_image(nd, 'door', t)
        drow['door_fp_per_img_on_firenet'] = fp_per_image(D['firenet'], 'door', t)[0]
        if vnd:
            drow['door_fp_per_img_verified_no_door'], drow['door_img_rate_verified_no_door'] = fp_per_image([d for d, _ in vnd], 'door', t)
        dd = ap_and_pr(D['oi_door'], sets['oi_door'], 1, t)
        drow['door_recall_door_images'], drow['door_precision_door_images'] = dd['recall'], dd['precision']
        if sp == 'test':
            drow['door_fp_per_img_dq_neutral'] = fp_per_image(D['dq_Neutral'], 'door', t)[0]
        door_sweep.append(drow)
    res['fire_sweep'] = fire_sweep
    res['door_sweep'] = door_sweep
    # per-role hard-negative breakdown at a few thresholds
    by_role = collections.defaultdict(list)
    for d, it in zip(D['oi_neg'], sets['oi_neg']):
        by_role[it['role']].append(d)
    res['oi_neg_fire_img_rate_by_role'] = {r: {str(t): image_present_rate(v, 'fire', t) for t in (0.3, 0.4, 0.5)} | {'n': len(v)}
                                           for r, v in sorted(by_role.items())}
    out = os.path.join(R, 'runs', f'eval_{sp}.json')
    json.dump(res, open(out, 'w'), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ('fire_sweep', 'door_sweep', 'oi_neg_fire_img_rate_by_role')}, indent=1))
    for row in fire_sweep:
        print('FIRE', ' '.join(f'{k}={v:.3f}' if isinstance(v, float) else f'{k}={v}' for k, v in row.items()))
    for row in door_sweep:
        print('DOOR', ' '.join(f'{k}={v:.3f}' if isinstance(v, float) else f'{k}={v}' for k, v in row.items()))
    for r, v in res['oi_neg_fire_img_rate_by_role'].items():
        print('ROLE', r, v)


if __name__ == '__main__':
    main()
