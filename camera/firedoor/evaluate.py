"""Evaluation helpers: batched inference, VOC AP@0.5, operating-point precision/recall, FP/image."""
from __future__ import annotations

import numpy as np
import torch

from data import imread, preprocess
from decode import decode, CLASS_NAMES


@torch.no_grad()
def infer_arrays(model, rgbs, bs=16):
    """rgbs: list of HxWx3 uint8 RGB. Returns list of (heat, wh, off) numpy [2,GH,GW] with
    heat as sigmoid probability (works for train-mode and export-mode models)."""
    model.eval()
    out = []
    for i in range(0, len(rgbs), bs):
        x = torch.from_numpy(np.stack([preprocess(r) for r in rgbs[i:i + bs]]))
        heat, wh, off = model(x)
        if not getattr(model, 'export_mode', False):
            heat = torch.sigmoid(heat)
        for j in range(x.shape[0]):
            out.append((heat[j].numpy(), wh[j].numpy(), off[j].numpy()))
    return out


def infer_items(model, items, bs=16):
    res = []
    for i in range(0, len(items), 64):
        res += infer_arrays(model, [imread(it['path']) for it in items[i:i + 64]], bs)
    return res


def voc_ap(rec, prec):
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def _iou_xyxy(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def match(dets_per_img, items, cls_id, iou_thr=0.5):
    """Returns (scores, tp flags, n_gt). Group-of GT boxes (6th field = 1) are 'don't care':
    detections matching them are neither TP nor FP, and they are not counted in n_gt."""
    name = CLASS_NAMES[cls_id]
    scores, tps, n_gt = [], [], 0
    for dets, it in zip(dets_per_img, items):
        gts = [b for b in it['boxes'] if int(b[0]) == cls_id]
        real = [b[1:5] for b in gts if not (len(b) > 5 and b[5])]
        grp = [b[1:5] for b in gts if len(b) > 5 and b[5]]
        n_gt += len(real)
        used = [False] * len(real)
        for d in sorted([d for d in dets if d['cls'] == name], key=lambda d: -d['score']):
            db = (d['x'], d['y'], d['x'] + d['w'], d['y'] + d['h'])
            best, bj = 0.0, -1
            for j, g in enumerate(real):
                o = _iou_xyxy(db, g)
                if o > best:
                    best, bj = o, j
            if best >= iou_thr and not used[bj]:
                used[bj] = True
                scores.append(d['score']); tps.append(1)
            elif any(_iou_xyxy(db, g) >= iou_thr or
                     (db[0] >= g[0] - 0.02 and db[1] >= g[1] - 0.02 and db[2] <= g[2] + 0.02 and db[3] <= g[3] + 0.02)
                     for g in grp):
                continue
            else:
                scores.append(d['score']); tps.append(0)
    return np.array(scores), np.array(tps), n_gt


def ap_and_pr(dets_per_img, items, cls_id, thr, iou_thr=0.5):
    s, tp, n = match(dets_per_img, items, cls_id, iou_thr)
    if n == 0:
        return dict(ap=float('nan'), n_gt=0)
    o = np.argsort(-s, kind='stable')
    s, tp = s[o], tp[o]
    ctp = np.cumsum(tp)
    rec = ctp / n
    prec = ctp / np.maximum(1, np.arange(1, len(tp) + 1))
    sel = s >= thr
    r_at = float(tp[sel].sum() / n)
    p_at = float(tp[sel].sum() / max(1, sel.sum()))
    return dict(ap=voc_ap(rec, prec), n_gt=n, recall=r_at, precision=p_at, n_det=int(sel.sum()))


def decode_all(outs, min_score=0.05):
    return [decode(h, w, o, min_score=min_score) for h, w, o in outs]


def fp_per_image(dets_per_img, cls_name, thr):
    n = [sum(1 for d in dets if d['cls'] == cls_name and d['score'] >= thr) for dets in dets_per_img]
    return float(np.mean(n)), float(np.mean([k > 0 for k in n]))


def image_present_rate(dets_per_img, cls_name, thr):
    return float(np.mean([any(d['cls'] == cls_name and d['score'] >= thr for d in dets) for dets in dets_per_img]))


def max_scores(outs, cls_id):
    return np.array([float(h[cls_id].max()) for h, w, o in outs])
