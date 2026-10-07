"""Export the trained FireDoorNet to ONNX (BN folded, fixed 1x3x256x320 input) and check parity.

python3 export_onnx.py runs/window_deep/best.pt export/firedoor.onnx     # fire, door, window (FireDoorWindowNet, shipped)
python3 export_onnx.py runs/stage2/best.pt export/firedoor_2class.onnx  # the earlier fire/door-only model
"""
from __future__ import annotations

import collections
import json
import os
import sys

import numpy as np
import onnx
import onnxruntime as ort
import torch

from data import imread, load_index, preprocess
from model import IN_H, IN_W, build_model, count_macs, fold_model

torch.set_num_threads(2)
R = os.path.dirname(os.path.abspath(__file__))


def main(ckpt, out):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    m = build_model(torch.load(ckpt, map_location='cpu'))
    m.eval()
    f = fold_model(m)
    x = torch.zeros(1, 3, IN_H, IN_W)
    names = ['heat', 'wh', 'off'] + (['wh_w', 'off_w'] if hasattr(m, 'window') else [])
    torch.onnx.export(f, x, out, opset_version=13, input_names=['input'], output_names=names,
                      do_constant_folding=True, dynamo=False)
    mo = onnx.load(out)
    # turn Constant nodes (Clip min/max, Resize roi/scales) into initializers: simpler for op-list runtimes
    keep = []
    for n in mo.graph.node:
        if n.op_type == 'Constant':
            t = onnx.helper.get_attribute_value(n.attribute[0])
            t.name = n.output[0]
            mo.graph.initializer.append(t)
        else:
            keep.append(n)
    del mo.graph.node[:]
    mo.graph.node.extend(keep)
    onnx.checker.check_model(mo)
    onnx.save(mo, out)
    ops = collections.Counter(n.op_type for n in mo.graph.node)
    sess = ort.InferenceSession(out, providers=['CPUExecutionProvider'])
    # parity on real images
    items = load_index('test', ['fire_box'])[:8] + load_index('test', ['door'])[:8] + load_index('test', ['neg'])[:8] + \
        load_index('test', ['window'])[:8]
    maxdiff = collections.defaultdict(float)
    for it in items:
        xi = torch.from_numpy(preprocess(imread(it['path']))[None])
        with torch.no_grad():
            ref = list(m(xi))
            ref[0] = torch.sigmoid(ref[0])
            fo = f(xi)
        got = sess.run(None, {'input': xi.numpy()})
        for name, a, b, c in zip(names, ref, fo, got):
            maxdiff[name + '_torch_vs_folded'] = max(maxdiff[name + '_torch_vs_folded'], float((a - b).abs().max()))
            maxdiff[name + '_torch_vs_onnx'] = max(maxdiff[name + '_torch_vs_onnx'], float(np.abs(a.numpy() - c).max()))
    n_params = sum(int(np.prod(t.dims)) for t in mo.graph.initializer)
    resize_attrs = [{a.name: onnx.helper.get_attribute_value(a) for a in n.attribute} for n in mo.graph.node if n.op_type == 'Resize']
    resize_inputs = [list(n.input) for n in mo.graph.node if n.op_type == 'Resize']
    clip_vals = sorted({float(onnx.numpy_helper.to_array(t)) for t in mo.graph.initializer
                        if t.name in {i for n in mo.graph.node if n.op_type == 'Clip' for i in n.input[1:]}})
    info = dict(onnx=os.path.relpath(out, R), checkpoint=os.path.relpath(ckpt, R), opset=13, ops=dict(ops), initializer_params=n_params,
                fp32_bytes=os.path.getsize(out), fp16_weight_bytes=2 * n_params,
                conv_macs=count_macs(m)['conv_macs'],
                resize_attrs=[{k: (v.decode() if isinstance(v, bytes) else v) for k, v in a.items()} for a in resize_attrs],
                resize_inputs=resize_inputs, clip_min_max_values=clip_vals, parity_max_abs_diff=dict(maxdiff),
                inputs=[(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim]) for i in mo.graph.input],
                outputs=[(o.name, [d.dim_value for d in o.type.tensor_type.shape.dim]) for o in mo.graph.output])
    print(json.dumps(info, indent=1))
    json.dump(info, open(out[:-5] + '.export.json', 'w'), indent=1)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
