"""Export the ONNX detector to the small binary format read by sim/nn_ref.c,
the float reference runtime the simulator and the browser demo use (the
device itself runs the int8 .espdl through ESP-DL).

    python3 -m ml.export_nnref --onnx ml/runs/synthetic_v0/model.onnx --out ml/runs/synthetic_v0/model.psnn

Format (little-endian):
    "PSNN" u32 version=1, u32 n_tensors, u32 n_ops, u32 out_heat, u32 out_wh, u32 out_off
    n_tensors x (u32 c, u32 h, u32 w)              tensor 0 is the input
    n_ops x op record:
        u32 type (1 conv, 2 relu, 3 resize-nearest, 4 add, 5 sigmoid)
        u32 in0, u32 in1 (0xFFFFFFFF if unused), u32 out
        u32 kh, kw, sh, sw, pad_top, pad_left, pad_bottom, pad_right
        u32 scale_h, scale_w                       resize only
        conv only: f32 weights[cout*cin*kh*kw], f32 bias[cout]
"""
import argparse
import struct

import numpy as np
import onnx
from onnx import numpy_helper, shape_inference

OPS = {"Conv": 1, "Relu": 2, "Resize": 3, "Add": 4, "Sigmoid": 5}
NONE = 0xFFFFFFFF


def export(onnx_path, out_path):
    m = shape_inference.infer_shapes(onnx.load(onnx_path))
    g = m.graph
    inits = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    shapes = {}
    for vi in list(g.input) + list(g.value_info) + list(g.output):
        dims = [d.dim_value for d in vi.type.tensor_type.shape.dim]
        if len(dims) == 4:
            shapes[vi.name] = dims[1:]
    ids = {g.input[0].name: 0}
    tensors = [shapes[g.input[0].name]]

    def tid(name):
        if name not in ids:
            ids[name] = len(tensors)
            tensors.append(shapes[name])
        return ids[name]

    ops = []
    for n in g.node:
        if n.op_type not in OPS:
            raise SystemExit(f"unsupported op {n.op_type}")
        a = {x.name: onnx.helper.get_attribute_value(x) for x in n.attribute}
        acts = [i for i in n.input if i and i not in inits]
        in0 = tid(acts[0])
        in1 = tid(acts[1]) if len(acts) > 1 else NONE
        out = tid(n.output[0])
        rec = [OPS[n.op_type], in0, in1, out] + [0] * 10
        blob = b""
        if n.op_type == "Conv":
            if a.get("group", 1) != 1 or a.get("dilations", [1, 1]) != [1, 1]:
                raise SystemExit("only group=1, dilation=1 convs are supported")
            w = inits[n.input[1]].astype(np.float32)
            b = inits[n.input[2]].astype(np.float32) if len(n.input) > 2 else np.zeros(w.shape[0], np.float32)
            kh, kw = w.shape[2], w.shape[3]
            sh, sw = a.get("strides", [1, 1])
            pt, pl, pb, pr = a.get("pads", [0, 0, 0, 0])
            rec[4:12] = [kh, kw, sh, sw, pt, pl, pb, pr]
            blob = w.tobytes() + b.tobytes()
        elif n.op_type == "Resize":
            if a.get("mode", b"nearest") != b"nearest":
                raise SystemExit("only nearest resize is supported")
            scales = inits[n.input[2]] if len(n.input) > 2 and n.input[2] in inits else None
            if scales is None:
                raise SystemExit("resize needs constant scales")
            rec[12:14] = [int(round(scales[2])), int(round(scales[3]))]
        ops.append(struct.pack("<14I", *rec) + blob)

    outs = [ids[o] for o in ("heat", "wh", "off")]
    with open(out_path, "wb") as f:
        f.write(b"PSNN" + struct.pack("<6I", 1, len(tensors), len(ops), *outs))
        for c, h, w in tensors:
            f.write(struct.pack("<3I", c, h, w))
        for o in ops:
            f.write(o)
    print(f"wrote {out_path}: {len(tensors)} tensors, {len(ops)} ops")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    export(a.onnx, a.out)
