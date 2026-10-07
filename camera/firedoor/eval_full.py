"""Full held-out evaluation of the exported ONNX model (or a PyTorch checkpoint).

python3 eval_full.py export/firedoor.onnx  [--split test|val] [--out runs/x.json]  -> runs/eval_<split>.json

Works for the 2-class model (fire, door) and the 3-class model (fire, door, window). Window numbers
(3-class only) and "door boxes on windows" (both) use the Open Images images where "Window" was
human-verified (sup[2] = 1: their window boxes are exhaustive, or no window is present).
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
from evaluate import ap_and_pr, decode_out, fp_per_image, image_present_rate

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
            outs = self.s.run(None, {'input': preprocess(r)[None]})     # heat, wh, off[, wh_w, off_w]
            out.append(tuple(o[0] for o in outs))
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
    return [decode_out(o, min_score=0.05) for o in outs]


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


def _center_in(d, b, pad=0.0):
    cx, cy = d['x'] + d['w'] / 2, d['y'] + d['h'] / 2
    return b[0] - pad <= cx <= b[2] + pad and b[1] - pad <= cy <= b[3] + pad


def _iou(d, b):
    x0, y0, x1, y1 = max(d['x'], b[0]), max(d['y'], b[1]), min(d['x'] + d['w'], b[2]), min(d['y'] + d['h'], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    i = (x1 - x0) * (y1 - y0)
    return i / (d['w'] * d['h'] + (b[2] - b[0]) * (b[3] - b[1]) - i)


def window_eval(res, D, sets, wthr):
    """Window AP / sweep / false windows by role (3-class model) and door boxes on windows (any model)."""
    pairs = [(d, it) for k in ('oi_door', 'oi_neg', 'oi_window', 'oi_wneg') for d, it in zip(D[k], sets[k]) if it['sup'][2]]
    if not pairs:
        return
    dets = [p[0] for p in pairs]
    items = [p[1] for p in pairs]
    has_window = any(d['cls'] == 'window' for dd in dets for d in dd)
    wins = lambda it, mn=0.0: [b[1:5] for b in it['boxes'] if int(b[0]) == 2 and not b[5] and (b[3] - b[1]) * (b[4] - b[2]) >= mn]
    with_w = [(d, it) for d, it in pairs if wins(it)]
    no_w = [(d, it) for d, it in pairs if not any(int(b[0]) == 2 for b in it['boxes'])]
    res['counts']['window_verified_images'] = len(pairs)
    res['counts']['window_verified_with_windows'] = len(with_w)
    res['counts']['window_verified_without_windows'] = len(no_w)
    res['counts']['window_gt_boxes'] = sum(len(wins(it)) for it in items)
    res['counts']['window_gt_boxes_ge1pct'] = sum(len(wins(it, 0.01)) for it in items)
    if has_window:
        res['window_ap50'] = ap_and_pr(dets, items, 2, wthr)['ap']
        # boxes under 1 % of the picture as "don't care" (group-of): the windows the page is meant for
        big = [dict(it, boxes=[b if int(b[0]) != 2 or (b[3] - b[1]) * (b[4] - b[2]) >= 0.01 else b[:5] + [1] for b in it['boxes']])
               for it in items]
        res['window_ap50_ge1pct'] = ap_and_pr(dets, big, 2, wthr)['ap']
        # large windows only (at least 5 % of the picture: a window in front of you in a room)
        big5 = [dict(it, boxes=[b if int(b[0]) != 2 or (b[3] - b[1]) * (b[4] - b[2]) >= 0.05 else b[:5] + [1] for b in it['boxes']])
                for it in items]
        res['window_ap50_ge5pct'] = ap_and_pr(dets, big5, 2, wthr)['ap']
        res['counts']['window_gt_boxes_ge5pct'] = sum(len(wins(it, 0.05)) for it in items)
        wi = [it for it in items if it['kind'] == 'window']
        res['window_ap50_window_photos_only'] = ap_and_pr([d for d, it in pairs if it['kind'] == 'window'], wi, 2, wthr)['ap']
        sweep = []
        for t in THRS:
            a = ap_and_pr(dets, items, 2, t)
            b = ap_and_pr(dets, big, 2, t)
            b5 = ap_and_pr(dets, big5, 2, t)
            row = dict(thr=t, window_recall=a['recall'], window_precision=a['precision'],
                       window_recall_ge1pct=b['recall'], window_precision_ge1pct=b['precision'],
                       window_recall_ge5pct=b5['recall'],
                       window_image_hit_rate_ge5pct=float(np.mean([any(d['cls'] == 'window' and d['score'] >= t and any(_center_in(d, w) for w in wins(it, 0.05)) for d in dd)
                                                                   for dd, it in with_w if wins(it, 0.05)])),
                       window_image_hit_rate=float(np.mean([any(d['cls'] == 'window' and d['score'] >= t and any(_center_in(d, w) for w in wins(it, 0.01)) for d in dd)
                                                            for dd, it in with_w if wins(it, 0.01)])))
            row['window_fp_per_img_no_window'], row['window_img_rate_no_window'] = fp_per_image([d for d, _ in no_w], 'window', t)
            look = [(d, it) for d, it in zip(D['oi_wneg'], sets['oi_wneg']) if it.get('lookalikes')]
            row['window_on_lookalike_per_img'] = float(np.mean([sum(1 for x in d if x['cls'] == 'window' and x['score'] >= t and
                                                                    any(_center_in(x, b) for b in it['lookalikes'])) for d, it in look])) if look else float('nan')
            row['window_fp_per_img_firenet'] = fp_per_image(D['firenet'], 'window', t)[0]
            if 'dq_Neutral' in D:
                row['window_fp_per_img_dq_neutral'] = fp_per_image(D['dq_Neutral'], 'window', t)[0]
            sweep.append(row)
        res['window_sweep'] = sweep
        by_role = collections.defaultdict(list)
        for d, it in zip(D['oi_wneg'] + D['oi_neg'], sets['oi_wneg'] + sets['oi_neg']):
            if any(int(b[0]) == 2 for b in it['boxes']):
                continue
            by_role[it['role']].append((d, it))
        res['window_fp_by_role'] = {}
        for r, v in sorted(by_role.items()):
            ver = [d for d, it in v if it['sup'][2]]
            on = [sum(1 for x in d if x['cls'] == 'window' and x['score'] >= wthr and any(_center_in(x, b) for b in it.get('lookalikes', [])))
                  for d, it in v if it.get('lookalikes')]
            res['window_fp_by_role'][r] = dict(n=len(v), window_boxes_per_img=fp_per_image([d for d, _ in v], 'window', wthr)[0],
                                               imgs_with_window_box=fp_per_image([d for d, _ in v], 'window', wthr)[1],
                                               n_window_verified_absent=len(ver),
                                               verified_absent_window_boxes_per_img=fp_per_image(ver, 'window', wthr)[0] if ver else None,
                                               n_with_lookalike_box=len(on),
                                               window_boxes_on_the_lookalike_per_img=float(np.mean(on)) if on else None)
    # the other way round: WINDOW boxes on doors (centre inside a door box, not on any window box), all
    # images with door boxes (window-verified or not; a window on them is excluded by the window check)
    if has_window:
        wod = []
        for t in (0.35, wthr, 0.5):
            n_on, n_img, n_doors = 0, 0, 0
            for k in ('oi_door', 'oi_neg', 'oi_window', 'oi_wneg'):
                for dd, it in zip(D[k], sets[k]):
                    doors = [b[1:5] for b in it['boxes'] if int(b[0]) == 1 and not b[5]]
                    if not doors:
                        continue
                    allw = [b[1:5] for b in it['boxes'] if int(b[0]) == 2]
                    kk = sum(1 for d in dd if d['cls'] == 'window' and d['score'] >= t and any(_center_in(d, g) for g in doors)
                             and not any(_iou(d, w) >= 0.3 or _center_in(d, w) for w in allw))
                    n_on += kk
                    n_img += kk > 0
                    n_doors += 1
            wod.append(dict(thr=t, door_images=n_doors, window_boxes_on_doors_per_img=n_on / max(1, n_doors),
                            images_with_window_box_on_a_door=n_img / max(1, n_doors)))
        res['window_on_door_sweep'] = wod
    # door boxes landing on windows (not on a door): the confusion the window class should remove
    dsweep = []
    for t in (0.35, 0.5):
        n_on, n_img, n_doorbox = 0, 0, 0
        for dd, it in with_w:
            ws = wins(it)
            doors = [b[1:5] for b in it['boxes'] if int(b[0]) == 1]
            k = 0
            for d in dd:
                if d['cls'] != 'door' or d['score'] < t:
                    continue
                n_doorbox += 1
                if any(_iou(d, g) >= 0.5 or _center_in(d, g) for g in doors):
                    continue
                if any(_iou(d, w) >= 0.3 or _center_in(d, w) for w in ws):
                    k += 1
            n_on += k
            n_img += k > 0
        dsweep.append(dict(thr=t, images=len(with_w), door_boxes_on_windows_per_img=n_on / max(1, len(with_w)),
                           images_with_door_box_on_a_window=n_img / max(1, len(with_w)), door_boxes_total=n_doorbox))
    res['door_on_window_sweep'] = dsweep
    # image level, like the earlier quality check: photos of door look-alikes without a door box in the labels
    # (window photos, wardrobes, fridges, cupboards, closets): share of photos with any DOOR box
    look = collections.defaultdict(list)
    for k in ('oi_door', 'oi_neg', 'oi_window', 'oi_wneg'):
        for d, it in zip(D[k], sets[k]):
            if any(int(b[0]) == 1 for b in it['boxes']):
                continue
            if any(int(b[0]) == 2 and not b[5] for b in it['boxes']):
                look['window'].append(d)
            elif it.get('role') in ('wneg_wardrobe', 'wneg_refrigerator', 'wneg_cupboard', 'wneg_closet'):
                look[it['role'][5:]].append(d)
    res['door_on_lookalike_photos'] = {k: {'n': len(v), **{str(t): image_present_rate(v, 'door', t) for t in (0.35, 0.5)}}
                                       for k, v in sorted(look.items())}
    wo = [d for d, it in zip(D['oi_window'], sets['oi_window'])]
    res['fire_on_window_photos'] = {str(t): image_present_rate(wo, 'fire', t) for t in (0.4, 0.5)} | {'n': len(wo)}
    res['fire_on_wneg'] = {str(t): image_present_rate(D['oi_wneg'], 'fire', t) for t in (0.4, 0.5)} | {'n': len(D['oi_wneg'])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('model')
    ap.add_argument('--split', default='test')
    ap.add_argument('--out', default=None)
    ap.add_argument('--window-thr', type=float, default=0.5, help='window operating point for the per-role tables')
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
        'oi_window': keep(load_index(sp, ['window']), 'oi_window'),
        'oi_wneg': keep(load_index(sp, ['wneg']), 'oi_wneg'),
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
    window_eval(res, D, sets, args.window_thr)
    out = args.out or os.path.join(R, 'runs', f'eval_{sp}.json')
    json.dump(res, open(out, 'w'), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ('fire_sweep', 'door_sweep', 'oi_neg_fire_img_rate_by_role')}, indent=1))
    for row in fire_sweep:
        print('FIRE', ' '.join(f'{k}={v:.3f}' if isinstance(v, float) else f'{k}={v}' for k, v in row.items()))
    for row in door_sweep:
        print('DOOR', ' '.join(f'{k}={v:.3f}' if isinstance(v, float) else f'{k}={v}' for k, v in row.items()))
    for r, v in res['oi_neg_fire_img_rate_by_role'].items():
        print('ROLE', r, v)
    for row in res.get('window_sweep', []):
        print('WINDOW', ' '.join(f'{k}={v:.3f}' if isinstance(v, float) else f'{k}={v}' for k, v in row.items()))
    for row in res.get('window_on_door_sweep', []):
        print('WINDOW-ON-DOOR', ' '.join(f'{k}={v:.3f}' if isinstance(v, float) else f'{k}={v}' for k, v in row.items()))
    for row in res.get('door_on_window_sweep', []):
        print('DOOR-ON-WINDOW', ' '.join(f'{k}={v:.3f}' if isinstance(v, float) else f'{k}={v}' for k, v in row.items()))
    for r, v in res.get('door_on_lookalike_photos', {}).items():
        print('DOOR-ON-LOOKALIKE', r, v)
    for r, v in res.get('window_fp_by_role', {}).items():
        print('WINDOW-FP', r, v)


if __name__ == '__main__':
    main()
