"""Evaluate a checkpoint the way the device would use it.

    python3 -m ml.eval --weights ml/runs/synthetic_v0/best.pt --data VAL_DIR [--out metrics.json]
    python3 -m ml.eval --weights ... --data build/sim_dump      # C simulator dump (ps_sim --dump-dataset)
    python3 -m ml.eval --weights ... --synthetic 500 --seed 20000000

Reports, overall and per smoke slice (clear < 0.2, light 0.2-0.5, heavy > 0.5):
  * AP@0.5 per class (VOC all-point interpolation), using the device decoder
    mirror with a low threshold (0.05) and up to 100 detections per frame;
  * precision / recall at the device threshold 0.35 with the device cap of 16
    detections (exactly ps_centernet_decode());
  * person distance error: matched person detections at 0.35, distance from
    ps_estimate_distances() (f_px * 1.7 m / max(w, h)) vs labelled distance.
    `formula_on_gt_box` isolates the pinhole prior's own error from the model's.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from .dataset import DiskDataset, SynthDataset, collate, read_sample
from .decode import DEVICE_SCORE_THRESHOLD, centernet_decode, person_distance_from_box
from .model import PyroNet

CLASSES = ("fire", "person")
SLICES = {"all": (-1, 2), "clear": (-1, 0.2), "light": (0.2, 0.5), "heavy": (0.5, 2)}


def iou_xywh(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    iw, ih = x1 - x0, y1 - y0
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    return inter / (a[2] * a[3] + b[2] * b[3] - inter)


def match(preds, gts, iou_thr=0.5):
    """Greedy by score. preds: list of dict; gts: list of (box, dist). Returns list of
    (score, tp, gt_index or -1) and the number of GTs."""
    order = sorted(range(len(preds)), key=lambda i: -preds[i]["score"])
    used = [False] * len(gts)
    out = []
    for i in order:
        p = preds[i]
        pb = (p["x"], p["y"], p["w"], p["h"])
        best, bj = iou_thr, -1
        for j, (gb, _) in enumerate(gts):
            if used[j]:
                continue
            v = iou_xywh(pb, gb)
            if v >= best:
                best, bj = v, j
        if bj >= 0:
            used[bj] = True
        out.append((p["score"], bj >= 0, bj))
    return out


def average_precision(records, n_gt):
    if n_gt == 0:
        return None
    if not records:
        return 0.0
    records = sorted(records, key=lambda r: -r[0])
    tp = np.cumsum([r[1] for r in records])
    fp = np.cumsum([not r[1] for r in records])
    rec = tp / n_gt
    prec = tp / np.maximum(tp + fp, 1e-9)
    mrec = np.concatenate([[0], rec, [1]])
    mpre = np.concatenate([[0], prec, [0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.nonzero(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


@torch.no_grad()
def predict(model, ds, batch=64):
    """Yields (index, heat, wh, off) numpy per frame."""
    dl = torch.utils.data.DataLoader(ds, batch, shuffle=False, num_workers=2, collate_fn=collate)
    for b in dl:
        h, w, o = model(b["x"])
        for k, i in enumerate(b["idx"]):
            yield i, h[k].numpy(), w[k].numpy(), o[k].numpy()


def gt_of(ds, i):
    if isinstance(ds, DiskDataset):
        _, boxes, cls, dist = read_sample(ds.root, ds.ids[i])
        smoke = ds.smoke.get(ds.ids[i], 0.0)
    else:
        dc, boxes, cls, dist, smoke = ds.raw(i)
    return boxes, cls, dist, smoke


def evaluate(model, ds, outputs_fn=None):
    """outputs_fn(model, ds) -> iterable of (i, heat, wh, off); default float model."""
    model.eval()
    acc = {s: dict(ap_rec={c: [] for c in range(2)}, n_gt={c: 0 for c in range(2)},
                   tp={c: 0 for c in range(2)}, fp={c: 0 for c in range(2)},
                   dist_err=[], dist_rel=[], gt_formula_rel=[], frames=0) for s in SLICES}
    it = outputs_fn(model, ds) if outputs_fn else predict(model, ds)
    for i, h, w, o in it:
        boxes, cls, dist, smoke = gt_of(ds, i)
        lo_dets = centernet_decode(h, w, o, thr=0.05, max_det=100)
        dev_dets = centernet_decode(h, w, o, thr=DEVICE_SCORE_THRESHOLD)          # device behaviour
        for sname, (lo, hi) in SLICES.items():
            if not (lo <= smoke < hi) and sname != "all":
                continue
            a = acc[sname]
            a["frames"] += 1
            for c in range(2):
                gts = [(tuple(b), d) for b, k, d in zip(boxes, cls, dist) if k == c]
                a["n_gt"][c] += len(gts)
                a["ap_rec"][c] += match([d for d in lo_dets if d["cls"] == c], gts)
                m = match([d for d in dev_dets if d["cls"] == c], gts)
                a["tp"][c] += sum(r[1] for r in m)
                a["fp"][c] += sum(not r[1] for r in m)
                if c == 1:
                    preds = [d for d in dev_dets if d["cls"] == c]
                    order = sorted(range(len(preds)), key=lambda q: -preds[q]["score"])
                    for (score, tp, j), pi in zip(m, order):
                        if tp and gts[j][1] > 0:
                            p = preds[pi]
                            est = person_distance_from_box(p["w"], p["h"])
                            a["dist_err"].append(abs(est - gts[j][1]))
                            a["dist_rel"].append((est - gts[j][1]) / gts[j][1])
                    for gb, gd in gts:
                        if gd > 0:
                            a["gt_formula_rel"].append((person_distance_from_box(gb[2], gb[3]) - gd) / gd)
    out = {}
    for sname, a in acc.items():
        r = dict(frames=a["frames"], classes={})
        aps = []
        for c, cn in enumerate(CLASSES):
            ap = average_precision(a["ap_rec"][c], a["n_gt"][c])
            tp, fp, n = a["tp"][c], a["fp"][c], a["n_gt"][c]
            r["classes"][cn] = dict(n_gt=n, ap50=None if ap is None else round(ap, 4),
                                    precision_at_0_35=round(tp / (tp + fp), 4) if tp + fp else None,
                                    recall_at_0_35=round(tp / n, 4) if n else None, fp_at_0_35=fp)
            if ap is not None:
                aps.append(ap)
        r["map50"] = round(float(np.mean(aps)), 4) if aps else None
        de, dr, gf = np.array(a["dist_err"]), np.array(a["dist_rel"]), np.array(a["gt_formula_rel"])
        r["person_distance"] = dict(
            n=int(len(de)),
            mae_m=round(float(de.mean()), 3) if len(de) else None,
            median_abs_rel=round(float(np.median(np.abs(dr))), 3) if len(dr) else None,
            mean_rel_bias=round(float(dr.mean()), 3) if len(dr) else None,
            formula_on_gt_box_median_abs_rel=round(float(np.median(np.abs(gf))), 3) if len(gf) else None,
        )
        out[sname] = r
    return out


@torch.no_grad()
def save_samples(model, ds, path, n=8, seed=0):
    """Grid of n frames with GT (white) and device-threshold predictions (fire red, person cyan)."""
    from .viz import grid, render
    rng = np.random.default_rng(seed)
    with_obj = [i for i in rng.permutation(len(ds))[: n * 6] if len(gt_of(ds, int(i))[1])]
    without = [i for i in rng.permutation(len(ds))[: n * 3] if not len(gt_of(ds, int(i))[1])]
    pick = (with_obj[: n - 2] + without[:2])[:n]
    ims = []
    model.eval()
    for i in pick:
        i = int(i)
        boxes, cls, dist, smoke = gt_of(ds, i)
        dc = ds.raw(i)[0]
        h, w, o = (t[0].numpy() for t in model(ds[i]["x"][None]))
        dets = centernet_decode(h, w, o, thr=DEVICE_SCORE_THRESHOLD)
        cap = f"#{i} smoke {smoke:.2f}  GT {len(cls)}  pred {len(dets)}"
        ims.append(render(dc, [(int(c), *b) for c, b in zip(cls, boxes)], dets, caption=cap))
    grid(ims, 4).save(path)


def load_model(path):
    model = PyroNet()
    ck = torch.load(path, map_location="cpu")
    model.load_state_dict(ck["model"] if "model" in ck else ck)
    return model.eval()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", help="dir in on-disk format (synthetic, ps_sim --dump-dataset, or real)")
    ap.add_argument("--synthetic", type=int, default=0, help="evaluate on N fresh synthetic frames")
    ap.add_argument("--seed", type=int, default=20_000_000)
    ap.add_argument("--no-denoise", action="store_true", help="frames already went through device denoise")
    ap.add_argument("--out", help="write metrics JSON here")
    ap.add_argument("--samples", help="write a PNG grid of 8 frames with GT and predictions")
    args = ap.parse_args(argv)
    torch.set_num_threads(4)
    model = load_model(args.weights)
    if args.data:
        ds = DiskDataset(args.data, denoise=not args.no_denoise)
    else:
        ds = SynthDataset(args.synthetic or 500, seed0=args.seed, denoise=not args.no_denoise)
    if args.samples:
        save_samples(model, ds, args.samples)
    m = evaluate(model, ds)
    print(json.dumps(m, indent=1))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(m, f, indent=1)


if __name__ == "__main__":
    main()
