"""FireDoorNet: MobileNetV2-0.5 (ImageNet init from Keras weights) + FPN-lite + CenterNet head.

Input  [1, 3, 256, 320] float32, NCHW, RGB, x = pixel / 127.5 - 1   (pixel in 0..255)
Outputs (stride 8, grid 32 x 40):
  heat [1, 3, 32, 40]  sigmoid probability per class (0 fire, 1 door, 2 window)
  wh   [1, 2, 32, 40]  box width, height in input pixels (ReLU, >= 0)
  off  [1, 2, 32, 40]  sub-cell centre offset in [0, 1) (Sigmoid), x then y

Only ops: Conv (incl. depthwise, groups=C), Clip(0,6) (=ReLU6), Relu, Add, Resize(nearest, x2), Sigmoid.
BatchNorm is folded at export. Padding is symmetric (PyTorch style); Keras' asymmetric 'same'
padding on stride-2 convs is replicated only when `keras_pad=True` (used for weight-mapping checks).
"""
from __future__ import annotations

import copy

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

IN_H, IN_W = 256, 320
STRIDE = 8
NUM_CLASSES = 3          # 0 fire, 1 door, 2 window (the window class was added on 2026-10-07)
CLASS_NAMES = ('fire', 'door', 'window')
WH_SCALE = 32.0
ALPHA = 0.5
DEEP_FROM = 13           # first backbone block of the last (stride 32) stage, see WindowBranch(deep=True)


def _md(v, d=8):
    n = max(d, int(v + d / 2) // d * d)
    if n < 0.9 * v:
        n += d
    return n


class ConvBN(nn.Sequential):
    def __init__(self, cin, cout, k=1, s=1, groups=1, act='relu6'):
        layers = [nn.Conv2d(cin, cout, k, s, k // 2, groups=groups, bias=False),
                  nn.BatchNorm2d(cout, eps=1e-3, momentum=0.01)]
        if act == 'relu6':
            layers.append(nn.ReLU6())
        elif act == 'relu':
            layers.append(nn.ReLU())
        super().__init__(*layers)


class InvRes(nn.Module):
    def __init__(self, cin, cout, s, t):
        super().__init__()
        mid = cin * t
        self.expand = ConvBN(cin, mid, 1) if t != 1 else None
        self.dw = ConvBN(mid, mid, 3, s, groups=mid)
        self.project = ConvBN(mid, cout, 1, act=None)
        self.res = (s == 1 and cin == cout)
        self.stride = s
        self.keras_pad = False

    def forward(self, x):
        y = self.expand(x) if self.expand is not None else x
        if self.stride == 2 and self.keras_pad:
            conv = self.dw[0]
            y = F.conv2d(F.pad(y, (0, 1, 0, 1)), conv.weight, None, 2, 0, 1, conv.groups)
            y = self.dw[2](self.dw[1](y))
        else:
            y = self.dw(y)
        y = self.project(y)
        return x + y if self.res else y


# (expansion t, channels c, repeats n, stride s) — MobileNetV2 table
CFG = [(1, 16, 1, 1), (6, 24, 2, 2), (6, 32, 3, 2), (6, 64, 4, 2), (6, 96, 3, 1), (6, 160, 3, 2), (6, 320, 1, 1)]


class MobileNetV2(nn.Module):
    """Features up to block 16 (no Conv_1 / classifier unless with_top)."""

    def __init__(self, alpha=ALPHA, with_top=False):
        super().__init__()
        c0 = _md(32 * alpha)
        self.stem = ConvBN(3, c0, 3, 2)
        blocks, cin = [], c0
        for t, c, n, s in CFG:
            cout = _md(c * alpha)
            for i in range(n):
                blocks.append(InvRes(cin, cout, s if i == 0 else 1, t))
                cin = cout
        self.blocks = nn.ModuleList(blocks)
        self.out_ch = cin
        self.with_top = with_top
        if with_top:
            self.conv_1 = ConvBN(cin, 1280, 1)
            self.logits = nn.Linear(1280, 1000)
        self.keras_pad = False

    def set_keras_pad(self, on):
        self.keras_pad = on
        for b in self.blocks:
            b.keras_pad = on

    def forward(self, x, taps=(5, 12, 16)):
        if self.keras_pad:
            conv = self.stem[0]
            x = self.stem[2](self.stem[1](F.conv2d(F.pad(x, (0, 1, 0, 1)), conv.weight, None, 2)))
        else:
            x = self.stem(x)
        feats = []
        for i, b in enumerate(self.blocks):
            x = b(x)
            if i in taps:
                feats.append(x)
        if self.with_top:
            y = self.conv_1(x).mean((2, 3))
            return self.logits(y)
        return feats


def load_keras_mnv2(model: MobileNetV2, h5path: str):
    """Map keras-applications MobileNetV2 (old 'mobl{i}_conv_{i}' naming) weights into model."""
    f = h5py.File(h5path, 'r')

    def g(layer, w):
        grp = f[layer]
        return np.array(grp[layer][w + ':0'])

    def conv(m, layer, depthwise=False):
        if depthwise:
            k = g(layer, 'depthwise_kernel')            # (kh, kw, C, 1)
            m.weight.data = torch.from_numpy(k.transpose(2, 3, 0, 1).copy())
        else:
            k = g(layer, 'kernel')                      # (kh, kw, cin, cout)
            m.weight.data = torch.from_numpy(k.transpose(3, 2, 0, 1).copy())

    def bn(m, layer):
        m.weight.data = torch.from_numpy(g(layer, 'gamma'))
        m.bias.data = torch.from_numpy(g(layer, 'beta'))
        m.running_mean.data = torch.from_numpy(g(layer, 'moving_mean'))
        m.running_var.data = torch.from_numpy(g(layer, 'moving_variance'))

    conv(model.stem[0], 'Conv1'); bn(model.stem[1], 'bn_Conv1')
    for i, b in enumerate(model.blocks):
        if i == 0:
            conv(b.dw[0], 'mobl0_conv_0_depthwise', True); bn(b.dw[1], 'bn0_conv_0_bn_depthwise')
            conv(b.project[0], 'mobl0_conv_0_project'); bn(b.project[1], 'bn0_conv_0_bn_project')
            continue
        conv(b.expand[0], f'mobl{i}_conv_{i}_expand'); bn(b.expand[1], f'bn{i}_conv_{i}_bn_expand')
        conv(b.dw[0], f'mobl{i}_conv_{i}_depthwise', True); bn(b.dw[1], f'bn{i}_conv_{i}_bn_depthwise')
        conv(b.project[0], f'mobl{i}_conv_{i}_project'); bn(b.project[1], f'bn{i}_conv_{i}_bn_project')
    if model.with_top:
        conv(model.conv_1[0], 'Conv_1'); bn(model.conv_1[1], 'Conv_1_bn')
        model.logits.weight.data = torch.from_numpy(g('Logits', 'kernel').T.copy())
        model.logits.bias.data = torch.from_numpy(g('Logits', 'bias'))
    return model


def dws(cin, cout):
    """Depthwise-separable 3x3 + 1x1, BN, ReLU."""
    return nn.Sequential(ConvBN(cin, cin, 3, 1, groups=cin, act='relu'), ConvBN(cin, cout, 1, act='relu'))


class FireDoorNet(nn.Module):
    def __init__(self, neck=64, alpha=ALPHA, num_classes=None):
        super().__init__()
        num_classes = NUM_CLASSES if num_classes is None else num_classes
        self.backbone = MobileNetV2(alpha)
        c8, c16, c32 = _md(32 * alpha), _md(96 * alpha), _md(320 * alpha)
        self.lat32 = ConvBN(c32, neck, 1, act='relu')
        self.lat16 = ConvBN(c16, neck, 1, act='relu')
        self.lat8 = ConvBN(c8, neck, 1, act='relu')
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.fuse16 = dws(neck, neck)
        self.fuse8 = dws(neck, neck)
        self.head = dws(neck, neck)
        self.heat = nn.Conv2d(neck, num_classes, 1)
        self.wh = nn.Conv2d(neck, 2, 1)
        self.off = nn.Conv2d(neck, 2, 1)
        self.register_buffer('wh_scale', torch.tensor(WH_SCALE))
        self.export_mode = False
        nn.init.constant_(self.heat.bias, -4.6)          # prior p = 0.01
        nn.init.normal_(self.heat.weight, 0, 0.01)
        nn.init.normal_(self.wh.weight, 0, 0.01)
        nn.init.constant_(self.wh.bias, 1.0)
        nn.init.normal_(self.off.weight, 0, 0.01)
        nn.init.constant_(self.off.bias, 0.0)

    def neck_head(self, f8, f16, f32):
        """heat logits, wh (scaled, before the final ReLU), off logits"""
        p16 = self.fuse16(self.lat16(f16) + self.up(self.lat32(f32)))
        p8 = self.fuse8(self.lat8(f8) + self.up(p16))
        h = self.head(p8)
        return self.heat(h), self.wh(h) * self.wh_scale, self.off(h)

    def forward(self, x):
        f8, f16, f32 = self.backbone(x)
        p16 = self.fuse16(self.lat16(f16) + self.up(self.lat32(f32)))
        p8 = self.fuse8(self.lat8(f8) + self.up(p16))
        h = self.head(p8)
        heat = self.heat(h)
        z = self.wh(h)
        if not self.export_mode:
            z = z * self.wh_scale
        if self.export_mode:
            return torch.sigmoid(heat), F.relu(z), torch.sigmoid(self.off(h))
        wh = F.leaky_relu(z, 0.01) if self.training else F.relu(z)
        return heat, wh, torch.sigmoid(self.off(h))


def load_expanding(model: 'FireDoorNet', path: str):
    """Load a checkpoint; a 2-class (fire, door) heat head is copied into the first two channels and
    the new class channels keep their fresh initialisation (weights N(0, 0.01), bias -4.6)."""
    sd = torch.load(path, map_location='cpu')
    own = model.state_dict()
    new = []
    for k, v in sd.items():
        if k in own and own[k].shape != v.shape and k.startswith('heat.'):
            t = own[k].clone()
            t[:v.shape[0]] = v
            sd[k] = t
            new.append(k)
    model.load_state_dict(sd)
    return new


class WindowBranch(nn.Module):
    """The window class: its own small FPN neck and CenterNet head on the (frozen) backbone features
    of FireDoorNet, so the fire and door outputs stay exactly those of the 2-class model.
    ch/head/layers: 48/64/2 (first version, trained from scratch) or 64/64/1 (the trunk's own neck and
    head layout, so it can start as a copy of the door detector, see init_window_from_door).
    deep: the branch also has its own trainable copy of the backbone's last stage (blocks 13-16, stride
    32) and reads its stride-32 features from that instead of the trunk's (frozen) ones: a window that
    fills much of the view needs features the fire/door network was never trained to make."""

    def __init__(self, ch=48, head=64, layers=2, alpha=ALPHA, deep=False):
        super().__init__()
        self.deep = nn.ModuleList(MobileNetV2(alpha).blocks[DEEP_FROM:]) if deep else None
        c8, c16, c32 = _md(32 * alpha), _md(96 * alpha), _md(320 * alpha)
        self.lat32 = ConvBN(c32, ch, 1, act='relu')
        self.lat16 = ConvBN(c16, ch, 1, act='relu')
        self.lat8 = ConvBN(c8, ch, 1, act='relu')
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.fuse16 = dws(ch, ch)
        self.fuse8 = dws(ch, ch)
        self.head = dws(ch, head) if layers == 1 else nn.Sequential(dws(ch, head), *[dws(head, head) for _ in range(layers - 1)])
        self.heat = nn.Conv2d(head, 1, 1)
        self.wh = nn.Conv2d(head, 2, 1)
        self.off = nn.Conv2d(head, 2, 1)
        self.register_buffer('wh_scale', torch.tensor(WH_SCALE))
        nn.init.constant_(self.heat.bias, -4.6)
        nn.init.normal_(self.heat.weight, 0, 0.01)
        nn.init.normal_(self.wh.weight, 0, 0.01)
        nn.init.constant_(self.wh.bias, 1.0)
        nn.init.normal_(self.off.weight, 0, 0.01)
        nn.init.constant_(self.off.bias, 0.0)

    def forward(self, f8, f16, f32):
        if self.deep is not None:
            f32 = f16
            for b in self.deep:
                f32 = b(f32)
        p16 = self.fuse16(self.lat16(f16) + self.up(self.lat32(f32)))
        p8 = self.fuse8(self.lat8(f8) + self.up(p16))
        h = self.head(p8)
        return self.heat(h), self.wh(h) * self.wh_scale, self.off(h)


def window_config(sd):
    """(ch, head, layers, deep) of the window branch in a FireDoorWindowNet state dict."""
    ch = sd['window.lat8.0.weight'].shape[0]
    head = sd['window.heat.weight'].shape[1]
    deep = any(k.startswith('window.deep.') for k in sd)
    if 'window.head.0.0.0.weight' not in sd:        # head = one dws block
        return ch, head, 1, deep
    return ch, head, sum(1 for i in range(16) if 'window.head.%d.0.0.weight' % i in sd), deep


def init_window_from_door(model: 'FireDoorWindowNet'):
    """Start the window branch (64/64/1 layout) as a copy of the trunk's neck and head, the window heat
    channel as a copy of the door channel: a door detector, which training then turns into a window one."""
    t, w = model.trunk, model.window
    for name in ('lat32', 'lat16', 'lat8', 'fuse16', 'fuse8', 'head', 'wh', 'off'):
        getattr(w, name).load_state_dict(getattr(t, name).state_dict())
    w.heat.weight.data.copy_(t.heat.weight.data[1:2])
    w.heat.bias.data.copy_(t.heat.bias.data[1:2])
    w.wh_scale.copy_(t.wh_scale)
    if w.deep is not None:
        for wb, tb in zip(w.deep, t.backbone.blocks[DEEP_FROM:]):
            wb.load_state_dict(tb.state_dict())


class FireDoorWindowNet(nn.Module):
    """FireDoorNet (fire, door; frozen) + WindowBranch. Outputs:
      heat  [B, 3, H, W]  0 fire, 1 door, 2 window (logits; sigmoid probabilities in export mode)
      wh    [B, 2, H, W]  box size for fire and door boxes  (exactly the 2-class model's)
      off   [B, 2, H, W]  centre offset for fire and door    (exactly the 2-class model's)
      wh_w  [B, 2, H, W]  box size for window boxes
      off_w [B, 2, H, W]  centre offset for window boxes
    so the fire and door boxes are those of the shipped 2-class model, bit for bit."""

    def __init__(self, ch=48, head=64, layers=2, deep=False):
        super().__init__()
        self.trunk = FireDoorNet(num_classes=2)
        self.window = WindowBranch(ch, head, layers, deep=deep)
        self.export_mode = False

    def forward(self, x):
        f8, f16, f32 = self.trunk.backbone(x)
        hf, zf, of = self.trunk.neck_head(f8, f16, f32)
        hw, zw, ow = self.window(f8, f16, f32)
        heat = torch.cat([hf, hw], 1)
        if self.export_mode:
            heat = torch.sigmoid(heat)
        return heat, F.relu(zf), torch.sigmoid(of), F.relu(zw), torch.sigmoid(ow)


def build_model(sd=None):
    """FireDoorNet (2-class) or FireDoorWindowNet from a state dict (keys 'trunk.' = the window model)."""
    if sd is not None and any(k.startswith('trunk.') for k in sd):
        m = FireDoorWindowNet(*window_config(sd))
    elif sd is not None:
        m = FireDoorNet(num_classes=sd['heat.weight'].shape[0])
    else:
        m = FireDoorWindowNet()
    if sd is not None:
        m.load_state_dict(sd)
    return m


# ---------------------------------------------------------------- BN folding
def _fold(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    w = conv.weight.detach()
    b = conv.bias.detach() if conv.bias is not None else torch.zeros(w.shape[0])
    scale = bn.weight.detach() / torch.sqrt(bn.running_var + bn.eps)
    fused = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride,
                      conv.padding, conv.dilation, conv.groups, bias=True)
    fused.weight.data = w * scale.reshape(-1, 1, 1, 1)
    fused.bias.data = (b - bn.running_mean) * scale + bn.bias.detach()
    return fused


def fold_model(model):
    m = copy.deepcopy(model).eval()

    def walk(mod):
        for name, child in mod.named_children():
            if isinstance(child, ConvBN):
                setattr(mod, name, nn.Sequential(_fold(child[0], child[1]), *list(child)[2:]))
            else:
                walk(child)
    walk(m)
    for sub in ([m.trunk, m.window] if isinstance(m, FireDoorWindowNet) else [m]):
        s = float(sub.wh_scale)
        sub.wh.weight.data *= s
        sub.wh.bias.data *= s
        sub.wh_scale.fill_(1.0)
    m.export_mode = True
    if isinstance(m, FireDoorWindowNet):
        m.trunk.export_mode = True
    return m


def count_macs(model, shape=(1, 3, IN_H, IN_W)):
    macs = [0]
    hooks = [m.register_forward_hook(lambda mod, i, o: macs.__setitem__(0, macs[0] + o.numel() * mod.kernel_size[0] * mod.kernel_size[1] * (mod.in_channels // mod.groups)))
             for m in model.modules() if isinstance(m, nn.Conv2d)]
    was = model.training
    model.eval()
    with torch.no_grad():
        model(torch.zeros(shape))
    model.train(was)
    for h in hooks:
        h.remove()
    return dict(conv_macs=macs[0], params=sum(p.numel() for p in model.parameters()))


if __name__ == '__main__':
    m = FireDoorNet()
    info = count_macs(m)
    print(f"params {info['params']:,}  conv MACs {info['conv_macs'] / 1e6:.1f} M")
    print([t.shape for t in m.eval()(torch.zeros(1, 3, IN_H, IN_W))])
