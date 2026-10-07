"""Train the window branch of FireDoorWindowNet on CPU; the fire/door network stays frozen.

python3 train_window.py --trunk runs/stage2/best.pt --out runs/window_branch --max-minutes 35
python3 train_window.py --out runs/window_door_init --ch 64 --layers 1 --init-door --lr 1.5e-3 --max-minutes 35
python3 train_window.py --out runs/window_deep --ch 64 --layers 1 --init-door --deep --lr 1e-3 --max-minutes 50

Only the window head learns (its own small FPN neck on the frozen MobileNetV2 features), so the fire and
door heat maps of the shipped 2-class model are reproduced exactly. Same data pipeline and augmentation as
train.py (data.TrainSet); losses: CenterNet focal loss on the window channel (with its supervision mask),
L1 on box size and centre offset at window centres. With --init-door the branch (then laid out like the
trunk's own neck and head: --ch 64 --layers 1) starts as a copy of the door detector, window channel =
door channel, instead of from random weights. With --deep the branch also has its own trainable copy of
the backbone's last stage (blocks 13-16; started as a copy of the trunk's). Validation each epoch with train.validate (fire, door
and window numbers); every epoch is saved, the best window AP on validation is kept as best.pt.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import time

os.environ.setdefault('OMP_NUM_THREADS', os.environ.get('FD_THREADS', '3'))
import torch
from torch.utils.data import DataLoader

from data import TrainSet
from model import FireDoorWindowNet, init_window_from_door
from train import focal_loss, validate

R = os.path.dirname(os.path.abspath(__file__))


def window_loss(w_out, batch):
    hw, zw, ow = w_out
    _, heat_t, mask, weak, wh_t, off_t, reg = batch
    ht, m = heat_t[:, 2:3], mask[:, 2:3]
    fl, npos = focal_loss(hw, ht, m)
    npos = npos.clamp(min=1.0)
    r = ((ht[:, 0] >= 0.999).float() * m[:, 0]).unsqueeze(1)
    wh = torch.nn.functional.leaky_relu(zw, 0.01)
    l_wh = ((wh - wh_t).abs() / wh_t.clamp(min=4.0) * r).sum() / npos
    l_off = ((torch.sigmoid(ow) - off_t).abs() * r).sum() / npos
    l_heat = fl / npos
    return l_heat + l_wh + l_off, dict(heat=l_heat.item(), wh=l_wh.item(), off=l_off.item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trunk', default=os.path.join(R, 'runs', 'stage2', 'best.pt'))
    ap.add_argument('--out', default=os.path.join(R, 'runs', 'window_branch'))
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--epoch-len', type=int, default=9600)
    ap.add_argument('--bs', type=int, default=32)
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--warm', type=int, default=100)
    ap.add_argument('--ema', type=float, default=0.995)
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--threads', type=int, default=int(os.environ.get('FD_THREADS', '3')))
    ap.add_argument('--max-minutes', type=float, default=35)
    ap.add_argument('--mix', default='{"window": 0.45, "wneg": 0.2, "door": 0.2, "neg": 0.15}')
    ap.add_argument('--mosaic', type=float, default=0.25)
    ap.add_argument('--ch', type=int, default=48, help='window FPN channels')
    ap.add_argument('--head', type=int, default=64, help='window head channels')
    ap.add_argument('--layers', type=int, default=2, help='dws blocks in the window head')
    ap.add_argument('--init-door', action='store_true', help='start the window branch as a copy of the door detector')
    ap.add_argument('--deep', action='store_true', help="the branch gets its own trainable copy of the backbone's last stage")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    os.makedirs(args.out, exist_ok=True)

    model = FireDoorWindowNet(args.ch, args.head, args.layers, deep=args.deep)
    model.trunk.load_state_dict(torch.load(args.trunk, map_location='cpu'))
    if args.init_door:
        init_window_from_door(model)
    for p in model.trunk.parameters():
        p.requires_grad_(False)
    model = model.to(memory_format=torch.channels_last)
    ema = copy.deepcopy(model).eval()
    params = [p for p in model.window.parameters() if p.requires_grad]
    decay = [p for p in params if p.ndim > 1]
    no_decay = [p for p in params if p.ndim <= 1]
    opt = torch.optim.AdamW([{'params': decay, 'weight_decay': 5e-4}, {'params': no_decay, 'weight_decay': 0}],
                            lr=args.lr, betas=(0.9, 0.99))
    ds = TrainSet(length=args.epoch_len, mix=json.loads(args.mix), mosaic_p=args.mosaic)
    dl = DataLoader(ds, batch_size=args.bs, num_workers=args.workers, drop_last=True,
                    persistent_workers=args.workers > 0, prefetch_factor=4 if args.workers else None)
    total = args.epochs * len(dl)
    t0 = time.time()

    def progress(it):
        return max(it / total, (time.time() - t0) / 60 / args.max_minutes)

    def lr_at(it):
        if it < args.warm:
            return args.lr * (0.1 + 0.9 * it / args.warm)
        return args.lr * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * min(1.0, progress(it)))))

    log = open(os.path.join(args.out, 'log.jsonl'), 'a')
    log.write(json.dumps(dict(args=vars(args))) + '\n')
    best, it = -1.0, 0
    for ep in range(args.epochs):
        model.train()
        model.trunk.eval()            # frozen: BatchNorm statistics stay those of the shipped model
        agg, n, te = {}, 0, time.time()
        for batch in dl:
            for g in opt.param_groups:
                g['lr'] = lr_at(it)
            x = batch[0].contiguous(memory_format=torch.channels_last)
            with torch.no_grad():
                f8, f16, f32 = model.trunk.backbone(x)
            out = model.window(f8, f16, f32)
            loss, parts = window_loss(out, batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 10.0)
            opt.step()
            d = min(args.ema, (1 + it) / (10 + it))
            with torch.no_grad():
                for pe, pm in zip(ema.window.state_dict().values(), model.window.state_dict().values()):
                    if pe.dtype.is_floating_point:
                        pe.mul_(d).add_(pm.detach(), alpha=1 - d)
                    else:
                        pe.copy_(pm)
            for k, v in parts.items():
                agg[k] = agg.get(k, 0) + v
            n += 1
            it += 1
            if it % 25 == 0:
                print(f'ep {ep} it {it} loss {loss.item():.3f} ' + ' '.join(f'{k} {v / n:.3f}' for k, v in agg.items()) +
                      f' {(time.time() - te) / n / args.bs * 1000:.0f} ms/img lr {lr_at(it):.2e}', flush=True)
            if progress(it) >= 0.995:
                break
        tv = time.time()
        res = validate(ema)
        res.update(epoch=ep, iters=it, train={k: v / max(1, n) for k, v in agg.items()},
                   epoch_s=round(tv - te), val_s=round(time.time() - tv), elapsed_min=round((time.time() - t0) / 60, 1))
        print(json.dumps(res), flush=True)
        log.write(json.dumps(res) + '\n'); log.flush()
        sd = ema.state_dict()
        torch.save(sd, os.path.join(args.out, 'last.pt'))
        torch.save(sd, os.path.join(args.out, 'ep%02d.pt' % ep))
        if res['window_ap'] > best:
            best = res['window_ap']
            torch.save(sd, os.path.join(args.out, 'best.pt'))
            json.dump(res, open(os.path.join(args.out, 'best.json'), 'w'), indent=1)
        if progress(it) >= 0.995:
            print('time budget reached', flush=True)
            break


if __name__ == '__main__':
    main()
