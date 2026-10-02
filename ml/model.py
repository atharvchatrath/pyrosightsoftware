"""PyroNet: a small CenterNet-style detector for 160x120 thermal frames.

Device contract (core/include/pyrosight/ps_detect.h, ps_centernet_decode()):
  input  [1, 1, 120, 160] float = ps_thermal_model_code(deci-C) / 255
  heat   [1, 2, 30, 40]   raw class logits (0 fire, 1 person)
  wh     [1, 2, 30, 40]   box width, height in input pixels, >= 0 (ReLU in-model)
  off    [1, 2, 30, 40]   sub-cell centre offset in [0, 1) (Sigmoid in-model)
  stride 4.

Only ESP-DL friendly ops: Conv2d (incl. stride 2 and one (2,1) valid conv),
BatchNorm (folded before export), ReLU, nearest Upsample (Resize), Add, Sigmoid.

    120x160 --s2--> 60x80 (16) --s2--> 30x40 (24) ------------------------+
                                         |s2                              |
                                       15x20 (48) --------------+         |
                                         |s2                    |         |
                                       8x10 (96) -> up x2 16x20 |         |
                                         -> conv (2,1) valid -> 15x20 (+) |
                                                     conv, 1x1 -> up x2 (+)
                                                                  conv 3x3 (32)
                                                       heat / wh / off 1x1 heads
The (2,1) valid conv crops 16x20 back to 15x20 without a Slice/Pad op.
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

IN_H, IN_W = 120, 160
STRIDE = 4
NUM_CLASSES = 2
WH_SCALE = 16.0          # wh head predicts px/16 internally; folded into the conv at export


class CBR(nn.Sequential):
    def __init__(self, cin, cout, k=3, s=1, p=None, relu=True):
        if p is None:
            p = (k // 2) if isinstance(k, int) else (0, 0)
        layers = [nn.Conv2d(cin, cout, k, s, p, bias=False), nn.BatchNorm2d(cout)]
        if relu:
            layers.append(nn.ReLU(inplace=False))
        super().__init__(*layers)


class Add(nn.Module):
    """Explicit module so quantisation hooks can see element-wise adds."""

    def forward(self, a, b):
        return a + b


class PyroNet(nn.Module):
    def __init__(self, c=(16, 24, 48, 96), head_c=32):
        super().__init__()
        c1, c2, c3, c4 = c
        self.stem = CBR(1, c1, 3, 2)                       # 60x80
        self.s4 = nn.Sequential(CBR(c1, c2, 3, 2), CBR(c2, c2))       # 30x40
        self.s8 = nn.Sequential(CBR(c2, c3, 3, 2), CBR(c3, c3))       # 15x20
        self.s16 = nn.Sequential(CBR(c3, c4, 3, 2), CBR(c4, c4), CBR(c4, c4))  # 8x10
        self.up16 = nn.Upsample(scale_factor=2, mode="nearest")      # 16x20
        self.lat16 = CBR(c4, c3, (2, 1), 1, (0, 0))                 # 15x20 (valid conv crops a row)
        self.add8 = Add()
        self.fuse8 = CBR(c3, c3)
        self.red8 = CBR(c3, c2, 1)
        self.up8 = nn.Upsample(scale_factor=2, mode="nearest")       # 30x40
        self.add4 = Add()
        self.fuse4 = CBR(c2, head_c)
        self.heat = nn.Conv2d(head_c, NUM_CLASSES, 1)
        self.wh = nn.Conv2d(head_c, 2, 1)
        self.off = nn.Conv2d(head_c, 2, 1)
        self.register_buffer("wh_scale", torch.tensor(WH_SCALE))
        self.export_mode = False
        nn.init.constant_(self.heat.bias, -2.19)                    # prior p = 0.1
        nn.init.normal_(self.wh.weight, 0, 0.01)
        nn.init.constant_(self.wh.bias, 1.0)                        # 16 px
        nn.init.normal_(self.off.weight, 0, 0.01)
        nn.init.constant_(self.off.bias, 0.0)

    def forward(self, x):
        f2 = self.stem(x)
        f4 = self.s4(f2)
        f8 = self.s8(f4)
        f16 = self.s16(f8)
        p8 = self.fuse8(self.add8(f8, self.lat16(self.up16(f16))))
        p4 = self.fuse4(self.add4(f4, self.up8(self.red8(p8))))
        heat = self.heat(p4)
        z = self.wh(p4)
        if not self.export_mode:          # folded into the conv weights by fold_model()
            z = z * self.wh_scale
        # Training: leaky so cells pushed negative can recover. Device/export: plain ReLU.
        wh = F.relu(z) if (self.export_mode or not self.training) else F.leaky_relu(z, 0.01)
        off = torch.sigmoid(self.off(p4))
        return heat, wh, off


# ---------------------------------------------------------------- BN folding
def _fold_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    w = conv.weight.detach()
    b = conv.bias.detach() if conv.bias is not None else torch.zeros(w.shape[0])
    scale = bn.weight.detach() / torch.sqrt(bn.running_var + bn.eps)
    fused = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride,
                      conv.padding, conv.dilation, conv.groups, bias=True)
    fused.weight.data = w * scale.reshape(-1, 1, 1, 1)
    fused.bias.data = (b - bn.running_mean) * scale + bn.bias.detach()
    return fused


def fold_model(model: PyroNet) -> PyroNet:
    """Copy with every Conv+BN fused into one Conv (bias) and the wh scale folded
    into the wh conv. Output is numerically identical in eval mode."""
    m = copy.deepcopy(model).eval()

    def walk(mod):
        for name, child in mod.named_children():
            if isinstance(child, CBR):
                layers = [_fold_conv_bn(child[0], child[1])] + list(child)[2:]
                setattr(mod, name, nn.Sequential(*layers))
            else:
                walk(child)

    walk(m)
    s = float(m.wh_scale)
    m.wh.weight.data *= s
    m.wh.bias.data *= s
    m.wh_scale.fill_(1.0)
    m.export_mode = True
    return m


# ---------------------------------------------------------------- MAC count
def count_macs(model: nn.Module, shape=(1, 1, IN_H, IN_W)):
    """Multiply-accumulates of Conv layers (+ element-wise ops counted separately).
    An estimate: ignores BN (folded), activations, resize and memory traffic."""
    macs, elem = [0], [0]
    hooks = []

    def conv_hook(mod, inp, out):
        k = mod.kernel_size[0] * mod.kernel_size[1] * (mod.in_channels // mod.groups)
        macs[0] += out.numel() * k

    def add_hook(mod, inp, out):
        elem[0] += out.numel()

    for mod in model.modules():
        if isinstance(mod, nn.Conv2d):
            hooks.append(mod.register_forward_hook(conv_hook))
        elif isinstance(mod, Add):
            hooks.append(mod.register_forward_hook(add_hook))
    was = model.training
    model.eval()
    with torch.no_grad():
        model(torch.zeros(shape))
    model.train(was)
    for h in hooks:
        h.remove()
    params = sum(p.numel() for p in model.parameters())
    return dict(conv_macs=macs[0], elementwise_ops=elem[0], params=params)


if __name__ == "__main__":
    m = PyroNet()
    info = count_macs(m)
    print(f"params {info['params']:,}  conv MACs {info['conv_macs'] / 1e6:.1f} M")
    h, wh, off = m.eval()(torch.zeros(1, 1, IN_H, IN_W))
    print(h.shape, wh.shape, off.shape)
