#!/usr/bin/env python3
"""Build ONNX test models for the op-list exporter (TorchScript exporter,
dynamo=False, like ml/export_onnx.py; onnxscript is not installed here).

  allops_opsetNN.onnx  every op the exporter claims to support, with
                       non-trivial BN statistics, asymmetric padding,
                       grouped/depthwise/dilated convs, ceil-mode pooling,
                       count_include_pad on/off, nearest and bilinear resize,
                       reshape/transpose/flatten, SE block, Gemm head.
  mnv2det.onnx         MobileNetV2 (width 0.5) backbone + upsampling
                       CenterNet-style head: the kind of model the fire/door
                       detector is likely to be (random weights).
  pyronet.onnx         the project's own thermal PyroNet (ml/model.py), BN folded.
  pads.onnx            hand-built graph with asymmetric / SAME_LOWER / SAME_UPPER pads.
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(2)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out', 'onnx')


def randomise_bn(m, g):
    for mod in m.modules():
        if isinstance(mod, nn.BatchNorm2d):
            n = mod.num_features
            mod.running_mean.copy_(torch.randn(n, generator=g) * 0.5)
            mod.running_var.copy_(torch.rand(n, generator=g) * 2 + 0.2)
            mod.weight.data.copy_(torch.randn(n, generator=g) * 0.5 + 1)
            mod.bias.data.copy_(torch.randn(n, generator=g) * 0.3)


class AllOps(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 16, 3, 2, 1, bias=False)
        self.b1 = nn.BatchNorm2d(16)
        self.dw = nn.Conv2d(16, 16, 3, 1, 1, groups=16, bias=False)
        self.b2 = nn.BatchNorm2d(16)
        self.gc = nn.Conv2d(16, 32, 3, 1, 1, groups=4)
        self.b3 = nn.BatchNorm2d(32)
        self.asym = nn.Conv2d(32, 32, 3, 2, 0)               # after ZeroPad2d((0,1,0,1)): TF-same style
        self.dil = nn.Conv2d(32, 24, 3, 1, (2, 3), dilation=(2, 3))
        self.hw = nn.Conv2d(24, 24, (3, 5), 1, (1, 2))
        self.dwm = nn.Conv2d(24, 48, 3, 1, 1, groups=24)     # depthwise with multiplier 2
        self.se1 = nn.Conv2d(48, 12, 1)
        self.se2 = nn.Conv2d(12, 48, 1)
        self.c5 = nn.Conv2d(48 + 32, 32, 1)
        self.b5 = nn.BatchNorm2d(32)
        self.up_c = nn.Conv2d(32, 16, 3, 1, 1)
        self.head = nn.Conv2d(16 + 16, 4, 1)
        self.fc = nn.Linear(32, 10)
        self.refl = nn.Conv2d(16, 8, 3, 1, 0)

    def forward(self, x):
        x1 = F.relu(self.b1(self.c1(x)))                       # /2, 16
        x2 = F.relu6(self.b2(self.dw(x1)))                     # Clip 0..6
        x3 = F.hardswish(self.b3(self.gc(x2)))                 # grouped conv
        x4 = self.asym(F.pad(x3, (0, 1, 0, 1)))                # /4
        x4 = F.leaky_relu(x4, 0.1)
        x5 = torch.sigmoid(self.dil(x4))
        x6 = torch.tanh(self.hw(x5)) + x5                      # Add
        x7 = self.dwm(x6)
        s = F.adaptive_avg_pool2d(x7, 1)                       # GlobalAveragePool
        s = F.hardsigmoid(self.se2(F.relu(self.se1(s))))
        x7 = x7 * s                                            # Mul (broadcast)
        p1 = F.max_pool2d(x4, 3, 1, 1)                         # same-size max pool
        x8 = torch.cat([x7, p1], 1)                            # Concat
        x8 = F.relu(self.b5(self.c5(x8)))
        x8 = torch.clamp(x8, -1.0, 4.0)                        # Clip general
        a1 = F.avg_pool2d(x8, 3, 1, 1, count_include_pad=True)
        a2 = F.avg_pool2d(x8, 3, 1, 1, count_include_pad=False)
        x9 = a1 * 0.5 + a2 * 0.25                              # Mul/Add by scalar
        m2 = F.max_pool2d(x9, 2, 2, ceil_mode=True)            # ceil mode
        u = F.interpolate(m2, scale_factor=2, mode='nearest')  # Resize nearest
        u = u[:, :, :x9.shape[2], :x9.shape[3]]                # Slice
        u = self.up_c(u + x9)
        u2 = F.interpolate(u, scale_factor=2, mode='bilinear', align_corners=False)
        u3 = F.interpolate(u, size=(x2.shape[2], x2.shape[3]), mode='nearest')
        r = self.refl(F.pad(x2, (1, 1, 1, 1), mode='reflect'))
        hm = torch.sigmoid(self.head(torch.cat([u2[:, :, :x2.shape[2], :x2.shape[3]], u3], 1)))
        hm = hm + F.avg_pool2d(r, 2, 2).mean(1, keepdim=True).repeat_interleave(4, 1)[:, :, :1, :1] * 0
        n, c, h, w = x8.shape
        seq = x8.reshape(n, c, h * w).transpose(1, 2)          # Reshape + Transpose -> [N, HW, C]
        cls = self.fc(torch.flatten(F.adaptive_avg_pool2d(x8, 1), 1))   # Flatten + Gemm
        cls = torch.softmax(cls, 1)
        return hm, seq, cls


def inv_res(cin, cout, s, t):
    hid = cin * t
    layers = []
    if t != 1:
        layers += [nn.Conv2d(cin, hid, 1, bias=False), nn.BatchNorm2d(hid), nn.ReLU6()]
    layers += [nn.Conv2d(hid, hid, 3, s, 1, groups=hid, bias=False), nn.BatchNorm2d(hid), nn.ReLU6(),
               nn.Conv2d(hid, cout, 1, bias=False), nn.BatchNorm2d(cout)]
    return nn.Sequential(*layers)


class IR(nn.Module):
    def __init__(self, cin, cout, s, t):
        super().__init__()
        self.b = inv_res(cin, cout, s, t)
        self.res = s == 1 and cin == cout

    def forward(self, x):
        y = self.b(x)
        return x + y if self.res else y


class MNV2Det(nn.Module):
    """MobileNetV2 x0.5 to stride 16 + light FPN + CenterNet heads (stride 4)."""

    def __init__(self, ncls=2):
        super().__init__()
        cfg = [(1, 8, 1, 1), (6, 16, 2, 2), (6, 16, 3, 2), (6, 32, 4, 2), (6, 48, 3, 1)]
        layers = [nn.Conv2d(3, 16, 3, 2, 1, bias=False), nn.BatchNorm2d(16), nn.ReLU6()]
        cin = 16
        self.taps = []
        for t, c, n, s in cfg:
            for i in range(n):
                layers.append(IR(cin, c, s if i == 0 else 1, t))
                cin = c
            self.taps.append(len(layers))
        self.body = nn.ModuleList(layers)
        self.lat8 = nn.Conv2d(16, 32, 1)
        self.lat4 = nn.Conv2d(16, 32, 1)
        self.top = nn.Conv2d(48, 32, 1)
        self.smooth = nn.Sequential(nn.Conv2d(32, 32, 3, 1, 1, groups=32), nn.Conv2d(32, 32, 1),
                                    nn.BatchNorm2d(32), nn.ReLU())
        self.heat = nn.Conv2d(32, ncls, 1)
        self.wh = nn.Conv2d(32, 2, 1)

    def forward(self, x):
        feats = {}
        for i, l in enumerate(self.body):
            x = l(x)
            feats[i + 1] = x
        c4 = feats[self.taps[1]]      # stride 4, 16 ch
        c8 = feats[self.taps[2]]      # stride 8, 16 ch
        c16 = feats[self.taps[4]]     # stride 16, 48 ch
        p = self.top(c16)
        p = F.interpolate(p, scale_factor=2, mode='nearest') + self.lat8(c8)
        p = F.interpolate(p, scale_factor=2, mode='nearest') + self.lat4(c4)
        p = self.smooth(p)
        return torch.sigmoid(self.heat(p)), F.relu(self.wh(p))


def export(model, x, path, names, opset):
    model.eval()
    torch.onnx.export(model, x, path, input_names=['input'], output_names=names, opset_version=opset,
                      do_constant_folding=True, dynamo=False)
    print('wrote', path)


def main():
    os.makedirs(OUT, exist_ok=True)
    g = torch.Generator().manual_seed(0)
    torch.manual_seed(0)
    m = AllOps()
    randomise_bn(m, g)
    x = torch.rand(1, 3, 96, 128)
    for opset in (11, 13, 17):
        export(m, x, os.path.join(OUT, 'allops_opset%d.onnx' % opset), ['heat', 'seq', 'cls'], opset)
    d = MNV2Det()
    randomise_bn(d, g)
    export(d, torch.rand(1, 3, 192, 256), os.path.join(OUT, 'mnv2det.onnx'), ['heat', 'wh'], 13)
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))
    from ml.model import PyroNet, fold_model
    p = PyroNet()
    randomise_bn(p, g)
    export(fold_model(p).eval(), torch.rand(1, 1, 120, 160), os.path.join(OUT, 'pyronet.onnx'),
           ['heat', 'wh', 'off'], 13)


def make_pads_model(path):
    """Hand-built ONNX graph: asymmetric Conv/depthwise/pool pads that PyTorch
    never emits (TF-style SAME on even sizes, SAME_LOWER, odd one-sided pads)."""
    import onnx
    from onnx import helper, TensorProto, numpy_helper
    rng = np.random.default_rng(1)
    inits, nodes = [], []

    def w(name, shape):
        inits.append(numpy_helper.from_array(rng.standard_normal(shape).astype(np.float32) * 0.3, name))
        return name
    nodes.append(helper.make_node('Conv', ['x', w('w1', [8, 3, 3, 3]), w('b1', [8])], ['c1'],
                                  strides=[2, 2], pads=[0, 0, 1, 1]))                      # TF SAME on even input
    nodes.append(helper.make_node('Conv', ['c1', w('w2', [8, 1, 3, 3])], ['c2'], group=8,
                                  pads=[1, 0, 0, 2]))                                      # depthwise, odd pads
    nodes.append(helper.make_node('Conv', ['c2', w('w3', [12, 8, 2, 3])], ['c3'], auto_pad='SAME_LOWER',
                                  strides=[1, 2]))
    nodes.append(helper.make_node('Conv', ['c3', w('w4', [12, 3, 2, 2])], ['c4a'], group=4, auto_pad='SAME_UPPER',
                                  strides=[2, 2]))
    nodes.append(helper.make_node('Conv', ['c4a', w('w5', [12, 12, 3, 3])], ['c4'], dilations=[2, 1],
                                  pads=[2, 0, 1, 1]))                                      # dilated, asymmetric
    nodes.append(helper.make_node('MaxPool', ['c4'], ['p1'], kernel_shape=[3, 2], strides=[2, 2], pads=[0, 1, 2, 0]))
    nodes.append(helper.make_node('AveragePool', ['c4'], ['p2'], kernel_shape=[3, 2], strides=[2, 2],
                                  pads=[0, 1, 2, 0], count_include_pad=0))
    nodes.append(helper.make_node('AveragePool', ['c4'], ['p3'], kernel_shape=[3, 2], strides=[2, 2],
                                  pads=[0, 1, 2, 0], count_include_pad=1))
    nodes.append(helper.make_node('Sub', ['p2', 'p3'], ['d']))
    nodes.append(helper.make_node('Add', ['d', 'p1'], ['y']))
    nodes.append(helper.make_node('Div', ['y', w('k', [12, 1, 1])], ['y2']))
    g = helper.make_graph(nodes, 'pads', [helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 3, 40, 52])],
                          [helper.make_tensor_value_info('y2', TensorProto.FLOAT, [1, 12, 'h', 'w'])], inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid('', 13)])
    m.ir_version = 8
    onnx.checker.check_model(m)
    onnx.save(m, path)
    print('wrote', path)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'pads':
        make_pads_model(os.path.join(OUT, 'pads.onnx'))
    else:
        main()
        make_pads_model(os.path.join(OUT, 'pads.onnx'))
