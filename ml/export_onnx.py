"""Export a trained PyroNet to ONNX for ESP-DL / esp-ppq.

    python3 -m ml.export_onnx --weights ml/runs/synthetic_v0/best.pt --out ml/runs/synthetic_v0/model.onnx

* BatchNorm folded into the convolutions, wh scale folded into the wh conv;
* fixed shape: input "input" [1,1,120,160]; outputs "heat", "wh", "off" [1,2,30,40];
* checks the op set against the ESP-DL friendly allow-list;
* if onnxruntime is installed, checks ORT output against PyTorch.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from .model import IN_H, IN_W, PyroNet, count_macs, fold_model

ALLOWED_OPS = {"Conv", "Relu", "Add", "Resize", "Sigmoid", "MaxPool", "Concat", "Constant"}


def export(model: PyroNet, path: str, opset: int = 13):
    folded = fold_model(model).eval()
    x = torch.zeros(1, 1, IN_H, IN_W)
    torch.onnx.export(folded, x, path, input_names=["input"], output_names=["heat", "wh", "off"],
                      opset_version=opset, do_constant_folding=True, dynamic_axes=None, dynamo=False)
    _strip_identity(path)
    return folded


def _strip_identity(path):
    """The TorchScript exporter sometimes emits Identity nodes for de-duplicated
    initialisers; replace them by their initialiser so only real ops remain."""
    import onnx
    from onnx import numpy_helper
    m = onnx.load(path)
    inits = {i.name: i for i in m.graph.initializer}
    graph_outputs = {o.name for o in m.graph.output}
    for nd in list(m.graph.node):
        if nd.op_type != "Identity":
            continue
        src, dst = nd.input[0], nd.output[0]
        if src in inits and dst not in graph_outputs:
            t = numpy_helper.from_array(numpy_helper.to_array(inits[src]), dst)
            m.graph.initializer.append(t)
            m.graph.node.remove(nd)
        elif dst not in graph_outputs:
            for other in m.graph.node:
                for k, name in enumerate(other.input):
                    if name == dst:
                        other.input[k] = src
            m.graph.node.remove(nd)
    onnx.save(m, path)


def check(path, folded, ref_model=None, n=4, seed=0):
    import onnx
    m = onnx.load(path)
    onnx.checker.check_model(m)
    ops = sorted({nd.op_type for nd in m.graph.node})
    bad = [o for o in ops if o not in ALLOWED_OPS]
    report = dict(ops=ops, disallowed_ops=bad,
                  inputs=[(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim]) for i in m.graph.input],
                  outputs=[(o.name, [d.dim_value for d in o.type.tensor_type.shape.dim]) for o in m.graph.output])
    rng = np.random.default_rng(seed)
    xs = rng.uniform(0, 1, (n, 1, 1, IN_H, IN_W)).astype(np.float32)
    with torch.no_grad():
        if ref_model is not None:      # folding must not change the function
            ref_model.eval()
            d = max(float((a - b).abs().max()) for x in xs
                    for a, b in zip(ref_model(torch.from_numpy(x)), folded(torch.from_numpy(x))))
            report["fold_max_abs_diff"] = d
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        diffs = []
        for x in xs:
            o = sess.run(["heat", "wh", "off"], {"input": x})
            with torch.no_grad():
                t = folded(torch.from_numpy(x))
            diffs.append(max(float(np.abs(a - b.numpy()).max()) for a, b in zip(o, t)))
        report["onnxruntime_max_abs_diff"] = max(diffs)
    except ImportError:
        report["onnxruntime_max_abs_diff"] = "onnxruntime not installed"
    return report


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    model = PyroNet()
    ck = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(ck["model"] if "model" in ck else ck)
    model.eval()
    folded = export(model, args.out)
    rep = check(args.out, folded, model)
    rep.update(count_macs(folded))
    print(json.dumps(rep, indent=1))
    if rep["disallowed_ops"]:
        raise SystemExit(f"ops outside the ESP-DL allow-list: {rep['disallowed_ops']}")


if __name__ == "__main__":
    main()
