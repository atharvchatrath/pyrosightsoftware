"""Train PyroNet (CPU-friendly). Checkpoints every epoch; --resume continues.

    python3 -m ml.train --out ml/runs/synthetic_v0 --data /path/to/cache --n-train 10000 --epochs 12
    python3 -m ml.train --out ml/runs/synthetic_v0 --resume          # after an interruption
    python3 -m ml.train --out ml/runs/finetune --data REAL_DIR --init ml/runs/synthetic_v0/best.pt

If --data does not exist (or is incomplete) it is filled with --n-train synthetic
frames first. Any directory in the on-disk format works (synthetic, simulator
dump, or real recordings).
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .dataset import DiskDataset, collate, generate_dir
from .model import PyroNet


def focal_loss(logits, target, alpha=2.0, beta=4.0):
    """CenterNet penalty-reduced focal loss on sigmoid(logits) vs gaussian targets."""
    p = torch.sigmoid(logits).clamp(1e-4, 1 - 1e-4)
    pos = target.eq(1).float()
    neg = 1 - pos
    pos_loss = torch.log(p) * (1 - p) ** alpha * pos
    neg_loss = torch.log(1 - p) * p ** alpha * (1 - target) ** beta * neg
    n = pos.sum().clamp(min=1)
    return -(pos_loss.sum() + neg_loss.sum()) / n


def detection_loss(out, batch, w_wh=0.1, w_off=1.0):
    heat, wh, off = out
    m = batch["mask"].unsqueeze(1)
    n = batch["mask"].sum().clamp(min=1)
    l_heat = focal_loss(heat, batch["heat"])
    l_wh = (F.l1_loss(wh * m, batch["wh"] * m, reduction="sum") / n)
    l_off = (F.l1_loss(off * m, batch["off"] * m, reduction="sum") / n)
    total = l_heat + w_wh * l_wh + w_off * l_off
    return total, dict(heat=l_heat.item(), wh=l_wh.item(), off=l_off.item())


def ensure_data(root, n, seed0, workers):
    if os.path.exists(os.path.join(root, "COMPLETE")):
        return
    print(f"generating {n} synthetic frames into {root} ...", flush=True)
    t = time.time()
    generate_dir(root, n, seed0=seed0, workers=workers)
    print(f"  done in {time.time() - t:.0f}s", flush=True)


def evaluate_loss(model, loader):
    model.eval()
    tot, k = 0.0, 0
    with torch.no_grad():
        for b in loader:
            loss, _ = detection_loss(model(b["x"].contiguous(memory_format=torch.channels_last)), b)
            tot += loss.item(); k += 1
    model.train()
    return tot / max(k, 1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="ml/runs/synthetic_v0")
    ap.add_argument("--data", default=None, help="training dir (on-disk format); synthetic cache if missing")
    ap.add_argument("--val", default=None, help="validation dir; synthetic if missing")
    ap.add_argument("--n-train", type=int, default=10000)
    ap.add_argument("--n-val", type=int, default=400)
    ap.add_argument("--train-seed", type=int, default=0)
    ap.add_argument("--val-seed", type=int, default=10_000_000)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--time-budget-min", type=float, default=0, help="stop after this many minutes (0 = off)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--init", default=None, help="initial weights (fine-tuning)")
    args = ap.parse_args(argv)

    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    os.makedirs(args.out, exist_ok=True)
    scratch = os.environ.get("PS_ML_CACHE", os.path.join(args.out, "cache"))
    data = args.data or os.path.join(scratch, f"train_{args.n_train}_{args.train_seed}")
    val = args.val or os.path.join(scratch, f"val_{args.n_val}_{args.val_seed}")
    if not args.data or not os.path.exists(os.path.join(data, "images")):
        ensure_data(data, args.n_train, args.train_seed, args.workers + 1)
    if not args.val or not os.path.exists(os.path.join(val, "images")):
        ensure_data(val, args.n_val, args.val_seed, args.workers + 1)

    train_ds = DiskDataset(data, augment=True, seed=1)
    val_ds = DiskDataset(val, augment=False)
    train_dl = DataLoader(train_ds, args.batch, shuffle=True, num_workers=args.workers, collate_fn=collate,
                          drop_last=True, persistent_workers=False)
    val_dl = DataLoader(val_ds, 64, shuffle=False, num_workers=args.workers, collate_fn=collate)

    model = PyroNet()
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location="cpu")["model"])
    # channels_last is ~10-30x faster than NCHW for this net on CPU (oneDNN).
    model = model.to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps_total = args.epochs * len(train_dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps_total, pct_start=0.15)
    start_epoch, best, history = 0, float("inf"), []
    ckpt_path = os.path.join(args.out, "last.pt")
    if args.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        start_epoch, best, history = ck["epoch"] + 1, ck["best"], ck["history"]
        print(f"resumed from epoch {ck['epoch']} (best val loss {best:.4f})", flush=True)

    t0 = time.time()
    for epoch in range(start_epoch, args.epochs):
        train_ds.set_epoch(epoch)
        model.train()
        te, agg, k = time.time(), dict(heat=0.0, wh=0.0, off=0.0, total=0.0), 0
        for i, b in enumerate(train_dl):
            loss, parts = detection_loss(model(b["x"].contiguous(memory_format=torch.channels_last)), b)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step(); sched.step()
            for kk, v in parts.items():
                agg[kk] += v
            agg["total"] += loss.item(); k += 1
            if i % 50 == 0:
                print(f"  ep {epoch} it {i}/{len(train_dl)} loss {loss.item():.3f} "
                      f"(heat {parts['heat']:.3f} wh {parts['wh']:.2f} off {parts['off']:.3f})", flush=True)
        vl = evaluate_loss(model, val_dl)
        rec = dict(epoch=epoch, train={kk: v / max(k, 1) for kk, v in agg.items()}, val_loss=vl,
                   lr=sched.get_last_lr()[0], epoch_s=round(time.time() - te, 1))
        history.append(rec)
        print(json.dumps(rec), flush=True)
        if vl < best:
            best = vl
            torch.save(dict(model=model.state_dict(), epoch=epoch, val_loss=vl), os.path.join(args.out, "best.pt"))
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                        epoch=epoch, best=best, history=history), ckpt_path + ".tmp")
        os.replace(ckpt_path + ".tmp", ckpt_path)
        with open(os.path.join(args.out, "train_history.json"), "w") as f:
            json.dump(history, f, indent=1)
        if args.time_budget_min and (time.time() - t0) / 60 > args.time_budget_min:
            print("time budget reached; stopping (use --resume to continue)", flush=True)
            break


if __name__ == "__main__":
    main()
