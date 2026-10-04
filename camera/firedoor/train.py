"""Train FireDoorNet on CPU (2 threads).

python3 train.py --out runs/main --epochs 14 --epoch-len 4000
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import time

os.environ.setdefault('OMP_NUM_THREADS', '2')
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import TrainSet, load_index
from evaluate import ap_and_pr, decode_all, fp_per_image, image_present_rate, infer_items
from model import FireDoorNet, load_keras_mnv2

R = os.path.dirname(os.path.abspath(__file__))
KERAS = os.path.join(R, '..', 'assets', 'keras', 'mobilenet_v2_0.5_224_no_top.h5')


def focal_loss(logits, target, mask):
    """CenterNet penalty-reduced focal loss, masked. Returns (sum, num_pos)."""
    p = torch.sigmoid(logits).clamp(1e-4, 1 - 1e-4)
    pos = (target >= 0.999).float() * mask
    neg = (1 - (target >= 0.999).float()) * mask
    pos_l = torch.log(p) * (1 - p) ** 2 * pos
    neg_l = torch.log(1 - p) * p ** 2 * (1 - target) ** 4 * neg
    return -(pos_l.sum() + neg_l.sum()), pos.sum()


def compute_loss(out, batch, w_wh=1.0, w_off=1.0, w_mil=0.3):
    heat_l, wh, off = out
    _, heat_t, mask, weak, wh_t, off_t, reg = batch
    fl, npos = focal_loss(heat_l, heat_t, mask)
    npos = npos.clamp(min=1.0)
    l_heat = fl / npos
    r = reg.unsqueeze(1)
    l_wh = ((wh - wh_t).abs() / wh_t.clamp(min=4.0) * r).sum() / npos
    l_off = ((off - off_t).abs() * r).sum() / npos
    # MIL: in tiles known to contain fire somewhere, the strongest fire logit should be positive
    has = weak.flatten(1).amax(1) > 0
    if has.any():
        m = heat_l[has, 0].masked_fill(weak[has] <= 0, -1e4).flatten(1).amax(1)
        l_mil = F.softplus(-m).mean()
    else:
        l_mil = heat_l.sum() * 0
    total = l_heat + w_wh * l_wh + w_off * l_off + w_mil * l_mil
    return total, dict(heat=l_heat.item(), wh=l_wh.item(), off=l_off.item(), mil=l_mil.item())


def freeze_early(model, n_blocks=3):
    mods = [model.backbone.stem] + [model.backbone.blocks[i] for i in range(n_blocks)]
    for m in mods:
        for p in m.parameters():
            p.requires_grad_(False)
    return mods


def set_train(model, frozen):
    model.train()
    for m in frozen:
        m.eval()


VAL_CACHE = {}


def validate(model, thr=0.4):
    if not VAL_CACHE:
        VAL_CACHE['fire'] = load_index('val', ['fire_box'])
        VAL_CACHE['door'] = load_index('val', ['door'])
        VAL_CACHE['neg'] = load_index('val', ['neg'])
        VAL_CACHE['weak'] = load_index('val', ['fire_weak'])
    res = {}
    dets = {k: decode_all(infer_items(model, v)) for k, v in VAL_CACHE.items()}
    f = ap_and_pr(dets['fire'], VAL_CACHE['fire'], 0, thr)
    d = ap_and_pr(dets['door'] + dets['neg'], VAL_CACHE['door'] + VAL_CACHE['neg'], 1, thr)
    res['fire_ap'], res['fire_R'], res['fire_P'] = f['ap'], f['recall'], f['precision']
    res['door_ap'], res['door_R'], res['door_P'] = d['ap'], d['recall'], d['precision']
    res['neg_fire_fp_img'], res['neg_fire_img_rate'] = fp_per_image(dets['neg'] + dets['door'], 'fire', thr)
    res['weak_fire_present'] = image_present_rate(dets['weak'], 'fire', thr)
    # selection score: APs, minus a penalty for fire false alarms on hard negatives
    res['score'] = 0.5 * f['ap'] + 0.3 * d['ap'] + 0.2 * res['weak_fire_present'] - 0.5 * res['neg_fire_img_rate']
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(R, 'runs', 'main'))
    ap.add_argument('--epochs', type=int, default=14)
    ap.add_argument('--epoch-len', type=int, default=4000)
    ap.add_argument('--bs', type=int, default=32)
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--freeze', type=int, default=3)
    ap.add_argument('--random-init', action='store_true')
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--max-minutes', type=float, default=1e9)
    ap.add_argument('--init', default=None, help='start from this checkpoint (stage-2 fine-tune)')
    ap.add_argument('--mix', default=None, help='JSON dict kind -> sampling weight')
    ap.add_argument('--w-mil', type=float, default=0.3)
    ap.add_argument('--mosaic', type=float, default=0.25)
    args = ap.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(0)
    os.makedirs(args.out, exist_ok=True)

    model = FireDoorNet()
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location='cpu'))
    elif not args.random_init:
        load_keras_mnv2(model.backbone, KERAS)
    frozen = freeze_early(model, args.freeze) if args.freeze else []
    model = model.to(memory_format=torch.channels_last)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)

    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if p.requires_grad:
            (no_decay if p.ndim <= 1 else decay).append(p)
    opt = torch.optim.AdamW([{'params': decay, 'weight_decay': 5e-4}, {'params': no_decay, 'weight_decay': 0}],
                            lr=args.lr, betas=(0.9, 0.99))
    ds = TrainSet(length=args.epoch_len, mix=json.loads(args.mix) if args.mix else None, mosaic_p=args.mosaic)
    dl = DataLoader(ds, batch_size=args.bs, num_workers=args.workers, drop_last=True,
                    persistent_workers=args.workers > 0, prefetch_factor=4 if args.workers else None)
    iters_per_epoch = len(dl)
    t0 = time.time()
    total = args.epochs * iters_per_epoch
    warm = min(300, total // 10) if not args.init else 50

    def progress(it):
        # whichever is further along: iterations or the wall-clock budget
        return max(it / total, (time.time() - t0) / 60 / args.max_minutes)

    def lr_at(it):
        if it < warm:
            return args.lr * (0.1 + 0.9 * it / warm)
        p = min(1.0, progress(it))
        return args.lr * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * p)))

    log = open(os.path.join(args.out, 'log.jsonl'), 'a')
    best, it = -1e9, 0
    t0 = time.time()
    for ep in range(args.epochs):
        set_train(model, frozen)
        agg, n, te = {}, 0, time.time()
        for batch in dl:
            for g in opt.param_groups:
                g['lr'] = lr_at(it)
            x = batch[0].contiguous(memory_format=torch.channels_last)
            out = model(x)
            loss, parts = compute_loss(out, batch, w_mil=args.w_mil)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            d = min(0.999, (1 + it) / (10 + it)) if not args.init else 0.998
            with torch.no_grad():
                for pe, pm in zip(ema.state_dict().values(), model.state_dict().values()):
                    if pe.dtype.is_floating_point:
                        pe.mul_(d).add_(pm.detach(), alpha=1 - d)
                    else:
                        pe.copy_(pm)
            for k, v in parts.items():
                agg[k] = agg.get(k, 0) + v
            n += 1
            it += 1
            if it % 25 == 0:
                print(f'ep {ep} it {it}/{total} loss {loss.item():.3f} ' +
                      ' '.join(f'{k} {v / n:.3f}' for k, v in agg.items()) +
                      f' {(time.time() - te) / n / args.bs * 1000:.0f} ms/img', flush=True)
        tv = time.time()
        res = validate(ema)
        res.update(epoch=ep, iters=it, train={k: v / n for k, v in agg.items()},
                   epoch_s=round(tv - te), val_s=round(time.time() - tv), elapsed_min=round((time.time() - t0) / 60, 1))
        print(json.dumps(res), flush=True)
        log.write(json.dumps(res) + '\n'); log.flush()
        torch.save(ema.state_dict(), os.path.join(args.out, 'last.pt'))
        if res['score'] > best:
            best = res['score']
            torch.save(ema.state_dict(), os.path.join(args.out, 'best.pt'))
            json.dump(res, open(os.path.join(args.out, 'best.json'), 'w'), indent=1)
        if progress(it) >= 0.995:
            print('time budget reached', flush=True)
            break


if __name__ == '__main__':
    main()
