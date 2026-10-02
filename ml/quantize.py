"""Post-training int8 quantisation for ESP-DL.

    python3 -m ml.quantize --weights ml/runs/synthetic_v0/best.pt --onnx ml/runs/synthetic_v0/model.onnx \
        --out-dir ml/runs/synthetic_v0 [--calib-dir DIR] [--n-calib 64] [--eval-dir DIR | --n-eval 300]

Steps
 1. Calibration set: N frames preprocessed exactly as on the device
    (median3x3 -> ps_thermal_model_code -> /255), saved as calib.npy [N,1,120,160] float32.
    Use real recordings (--calib-dir) as soon as you have them.
 2. esp-ppq (pip package `esp-ppq`, Espressif's PPQ fork) if importable:
    espdl_quantize_onnx(..., target="esp32p4", num_of_bits=8) -> model.espdl.
    If it is not installed, prints the exact commands to run instead.
 3. Always: a PyTorch fake-quant simulation of int8 PTQ with per-tensor
    power-of-two scales (ESP-DL style exponents) on weights and on every
    activation tensor (input, each conv/conv+ReLU output, adds, head outputs,
    sigmoid output), then mAP of float vs int8-sim on the eval set.
    Also reported: the same with per-output-channel po2 weight exponents, which is
    what esp-ppq's esp32p4 config actually uses. Biases stay float (int32 on device).
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
import torch
import torch.nn as nn

from .dataset import DiskDataset, SynthDataset
from .eval import evaluate
from .model import Add, PyroNet, fold_model


# ---------------------------------------------------------------- calibration data
def build_calib(n, calib_dir=None, seed=30_000_000):
    ds = DiskDataset(calib_dir) if calib_dir else SynthDataset(n, seed0=seed)
    idx = np.linspace(0, len(ds) - 1, min(n, len(ds))).astype(int)
    return np.stack([ds[int(i)]["x"].numpy() for i in idx]).astype(np.float32)


# ---------------------------------------------------------------- po2 fake quant
def po2_quant(t, exp, bits=8):
    s = 2.0 ** exp
    qmax = 2 ** (bits - 1) - 1
    return torch.clamp(torch.round(t / s), -qmax - 1, qmax) * s


def best_exponent(t, bits=8, search=3):
    """Exponent minimising MSE, starting from the max-abs exponent and trying smaller ones
    (clip the tail for finer steps)."""
    m = float(t.abs().max())
    if m == 0:
        return -bits
    e0 = math.ceil(math.log2(m / (2 ** (bits - 1) - 1)))
    best, be = None, e0
    for e in range(e0, e0 - search - 1, -1):
        err = float(((po2_quant(t, e, bits) - t) ** 2).mean())
        if best is None or err < best:
            best, be = err, e
    return be


class QuantSim:
    """Wraps a folded PyroNet; quantises weights in place and activations with hooks."""

    def __init__(self, folded: PyroNet, bits=8):
        self.m = folded.eval()
        self.bits = bits
        self.points = {}          # name -> module whose output is quantised
        for name, mod in self.m.named_modules():
            if isinstance(mod, nn.Sequential) and len(mod) and isinstance(mod[0], nn.Conv2d):
                self.points[name] = mod              # conv(+ReLU) block
            elif isinstance(mod, Add):
                self.points[name] = mod
            elif name in ("heat", "wh", "off"):
                self.points[name] = mod              # head conv outputs
        self.act_exp = {}
        self.w_exp = {}
        self._hooks = []
        self.quant_input = True

    def quantize_weights(self, per_channel=False):
        for name, mod in self.m.named_modules():
            if isinstance(mod, nn.Conv2d):
                w = mod.weight.data
                if per_channel:       # one po2 exponent per output channel (for comparison only)
                    es = [best_exponent(w[o], self.bits) for o in range(w.shape[0])]
                    self.w_exp[name] = es
                    mod.weight.data = torch.stack([po2_quant(w[o], e, self.bits) for o, e in enumerate(es)])
                else:
                    e = best_exponent(w, self.bits)
                    self.w_exp[name] = e
                    mod.weight.data = po2_quant(w, e, self.bits)

    @torch.no_grad()
    def calibrate(self, calib):
        store = {k: [] for k in self.points}
        hs = [mod.register_forward_hook(lambda m_, i, o, k=k: store[k].append(o.detach().clone()))
              for k, mod in self.points.items()]
        outs = []
        for x in torch.from_numpy(calib).split(16):
            outs.append(self.m(x)[2])
        for h in hs:
            h.remove()
        for k, v in store.items():
            self.act_exp[k] = best_exponent(torch.cat(v), self.bits)
        self.act_exp["input"] = best_exponent(torch.from_numpy(calib), self.bits)
        self.act_exp["off_sigmoid"] = -(self.bits - 1)      # [0,1) at 2^-7

    def enable(self):
        for k, mod in self.points.items():
            e = self.act_exp[k]
            self._hooks.append(mod.register_forward_hook(lambda m_, i, o, e=e: po2_quant(o, e, self.bits)))

    @torch.no_grad()
    def __call__(self, x):
        if self.quant_input:
            x = po2_quant(x, self.act_exp["input"], self.bits)
        heat, wh, off = self.m(x)
        return heat, wh, po2_quant(off, self.act_exp["off_sigmoid"], self.bits)


def outputs_with(fn):
    def gen(model, ds):
        dl = torch.utils.data.DataLoader(ds, 64, shuffle=False, num_workers=2,
                                         collate_fn=lambda b: (torch.stack([s["x"] for s in b]), [s["idx"] for s in b]))
        for x, idx in dl:
            h, w, o = fn(x)
            for k, i in enumerate(idx):
                yield i, h[k].numpy(), w[k].numpy(), o[k].numpy()
    return gen


# ---------------------------------------------------------------- esp-ppq
ESPPPQ_INSTRUCTIONS = """\
esp-ppq is not available here. To produce model.espdl on a machine with Python 3.10+:

    python3 -m pip install esp-ppq
    python3 - <<'EOF'
    import numpy as np, torch
    from torch.utils.data import DataLoader
    from esp_ppq.api import espdl_quantize_onnx
    calib = torch.from_numpy(np.load("{calib}"))            # [N,1,120,160], code/255
    dl = DataLoader(calib, batch_size=1, shuffle=False)
    espdl_quantize_onnx(onnx_import_file="{onnx}", espdl_export_file="{espdl}",
                        calib_dataloader=dl, calib_steps=len(calib), input_shape=[1, 1, 120, 160],
                        target="esp32p4", num_of_bits=8, collate_fn=lambda b: b.float(),
                        device="cpu", error_report=True, skip_export=False, export_test_values=True)
    EOF

Then add the .espdl to the firmware model partition (see firmware/ and ml/README.md).
"""


def run_espppq(onnx_path, calib_path, espdl_path):
    try:
        from esp_ppq.api import espdl_quantize_onnx
    except Exception as e:  # noqa: BLE001 - any import failure means "not available"
        print(ESPPPQ_INSTRUCTIONS.format(calib=calib_path, onnx=onnx_path, espdl=espdl_path))
        return dict(status="not_installed", error=repr(e)[:200])
    from torch.utils.data import DataLoader
    calib = torch.from_numpy(np.load(calib_path))
    dl = DataLoader(calib, batch_size=1, shuffle=False)
    try:
        espdl_quantize_onnx(onnx_import_file=onnx_path, espdl_export_file=espdl_path, calib_dataloader=dl,
                            calib_steps=len(calib), input_shape=[1, 1, 120, 160], target="esp32p4",
                            num_of_bits=8, collate_fn=lambda b: b.float(), device="cpu",
                            error_report=True, skip_export=False, export_test_values=True, verbose=0)
    except Exception as e:  # noqa: BLE001
        print(ESPPPQ_INSTRUCTIONS.format(calib=calib_path, onnx=onnx_path, espdl=espdl_path))
        return dict(status="failed", error=repr(e)[:500])
    return dict(status="ok", espdl=espdl_path, bytes=os.path.getsize(espdl_path))


# ---------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--onnx", default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--calib-dir", default=None)
    ap.add_argument("--n-calib", type=int, default=64)
    ap.add_argument("--eval-dir", default=None)
    ap.add_argument("--n-eval", type=int, default=300)
    ap.add_argument("--eval-seed", type=int, default=20_000_000)
    ap.add_argument("--skip-espppq", action="store_true")
    args = ap.parse_args(argv)
    torch.set_num_threads(4)
    os.makedirs(args.out_dir, exist_ok=True)

    calib_path = os.path.join(args.out_dir, "calib.npy")
    calib = build_calib(args.n_calib, args.calib_dir)
    np.save(calib_path, calib)
    print(f"calibration set {calib.shape} -> {calib_path}")

    model = PyroNet()
    ck = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(ck["model"] if "model" in ck else ck)
    model.eval()
    folded = fold_model(model)

    sim = QuantSim(fold_model(model))       # the reported scheme: per-tensor everything
    sim.quantize_weights()
    sim.calibrate(calib)
    sim.enable()

    sim_pc = QuantSim(fold_model(model))       # same, but per-output-channel weight exponents
    sim_pc.quantize_weights(per_channel=True)
    sim_pc.calibrate(calib)
    sim_pc.enable()

    ds = DiskDataset(args.eval_dir) if args.eval_dir else SynthDataset(args.n_eval, seed0=args.eval_seed)
    with torch.no_grad():
        m_float = evaluate(folded, ds, outputs_with(lambda x: folded(x)))
        m_q = evaluate(folded, ds, outputs_with(sim))
        m_pc = evaluate(folded, ds, outputs_with(sim_pc))
    res = dict(
        eval_set=args.eval_dir or f"synthetic n={args.n_eval} seed0={args.eval_seed}",
        scheme="int8 symmetric, per-tensor power-of-two scales (weights and activations), MSE-searched exponents",
        input_exponent=sim.act_exp["input"],
        note_input=("input code/255 in int8 with scale 2^%d keeps %s of the 256 code levels"
                    % (sim.act_exp["input"], "half" if sim.act_exp["input"] == -7 else "a quarter")),
        weight_exponents=sim.w_exp, activation_exponents=sim.act_exp,
        float=dict(map50=m_float["all"]["map50"],
                   ap50={c: v["ap50"] for c, v in m_float["all"]["classes"].items()}),
        int8_sim=dict(map50=m_q["all"]["map50"],
                      ap50={c: v["ap50"] for c, v in m_q["all"]["classes"].items()}),
        int8_sim_by_smoke={s: m_q[s]["map50"] for s in m_q},
        int8_sim_per_channel_weights=dict(map50=m_pc["all"]["map50"],
                                          ap50={c: v["ap50"] for c, v in m_pc["all"]["classes"].items()}),
    )
    if res["float"]["map50"] is not None and res["int8_sim"]["map50"] is not None:
        res["map50_drop"] = round(res["float"]["map50"] - res["int8_sim"]["map50"], 4)
        res["map50_drop_per_channel_weights"] = round(
            res["float"]["map50"] - res["int8_sim_per_channel_weights"]["map50"], 4)
    res["note"] = ("int8_sim = per-tensor po2 on weights and activations (strict). esp-ppq's esp32p4 "
                   "config (see model.json) uses per-channel po2 weights + per-tensor po2 activations, "
                   "which int8_sim_per_channel_weights mirrors; that is the closer estimate of the .espdl. "
                   "Per-tensor weight scales after BN folding cause most of the strict-sim loss.")
    if not args.skip_espppq and args.onnx:
        res["esp_ppq"] = run_espppq(args.onnx, calib_path, os.path.join(args.out_dir, "model.espdl"))
    with open(os.path.join(args.out_dir, "quant_report.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: v for k, v in res.items() if "exponents" not in k}, indent=1))


if __name__ == "__main__":
    main()
