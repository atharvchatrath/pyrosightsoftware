"""Build an evaluation set from the C simulator (independent of ml/synth.py).

    python3 -m ml.sim_dataset --sim build/sim/ps_sim --out /tmp/sim_eval [--every 2]

Runs ps_sim --dump-dataset for each scenario x smoke level x seed and merges the
dumps into one directory in the on-disk format (renumbered ids, meta.csv smoke),
so it can be fed straight to `python3 -m ml.eval --data DIR`.
Frames are RAW (pre-denoise), like synthetic ones: keep the default denoise.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile

from .dataset import read_meta

SCENARIOS = ("corridor", "long", "crawl", "dropout")


def build(sim, out, smokes=(0.0, 0.35, 0.8), seeds=(1, 2), every=2):
    os.makedirs(os.path.join(out, "images"), exist_ok=True)
    os.makedirs(os.path.join(out, "labels"), exist_ok=True)
    n = 0
    with open(os.path.join(out, "meta.csv"), "w") as meta, tempfile.TemporaryDirectory() as tmp:
        meta.write("id,smoke\n")
        for sc in SCENARIOS:
            for sm in smokes:
                for seed in seeds:
                    d = os.path.join(tmp, f"{sc}_{sm}_{seed}")
                    subprocess.run([sim, "--scenario", sc, "--seed", str(seed), "--smoke", str(sm),
                                    "--dump-dataset", d, "--quiet", "--every", "1000000"],
                                   check=True, stdout=subprocess.DEVNULL)
                    m = read_meta(d)
                    ids = sorted(int(f[:-4]) for f in os.listdir(os.path.join(d, "images")))
                    for i in ids[::every]:
                        shutil.copy(os.path.join(d, "images", f"{i:06d}.bin"), os.path.join(out, "images", f"{n:06d}.bin"))
                        shutil.copy(os.path.join(d, "labels", f"{i:06d}.txt"), os.path.join(out, "labels", f"{n:06d}.txt"))
                        meta.write(f"{n},{m.get(i, sm):.2f}\n")
                        n += 1
    return n


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", default="build/sim/ps_sim")
    ap.add_argument("--out", required=True)
    ap.add_argument("--every", type=int, default=2, help="keep every Nth dumped frame")
    args = ap.parse_args(argv)
    meta = os.path.join(args.out, "meta.csv")
    if os.path.exists(meta):
        os.remove(meta)
    print(f"{build(args.sim, args.out, every=args.every)} frames -> {args.out}")


if __name__ == "__main__":
    main()
