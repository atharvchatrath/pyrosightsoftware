"""Dump export/firedoor.onnx as a plain JSON op list (+ weights as base64 little-endian float16 or float32)
for simple JS runtimes:  python3 tools/onnx_to_oplist.py [fp16|fp32]  ->  export/firedoor_oplist_<dtype>.json
Node format: {op, inputs:[names], outputs:[names], attrs:{...}}; initializers: {name: {dims, dtype, b64}}.
"""
import base64, json, os, sys
import numpy as np, onnx
from onnx import numpy_helper
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
dt = sys.argv[1] if len(sys.argv) > 1 else 'fp16'
m = onnx.load(os.path.join(R, 'export/firedoor.onnx'))
inits = {}
for t in m.graph.initializer:
    a = numpy_helper.to_array(t)
    if a.dtype == np.float32 and a.size > 2 and dt == 'fp16':
        inits[t.name] = dict(dims=list(a.shape), dtype='float16', b64=base64.b64encode(a.astype('<f2').tobytes()).decode())
    else:
        inits[t.name] = dict(dims=list(a.shape), dtype=str(a.dtype), b64=base64.b64encode(a.astype(a.dtype.newbyteorder('<')).tobytes()).decode())
nodes = []
for n in m.graph.node:
    attrs = {}
    for a in n.attribute:
        v = onnx.helper.get_attribute_value(a)
        attrs[a.name] = v.decode() if isinstance(v, bytes) else (list(v) if isinstance(v, (list, tuple)) or hasattr(v, '__len__') and not isinstance(v, str) else v)
    nodes.append(dict(op=n.op_type, inputs=list(n.input), outputs=list(n.output), attrs=attrs))
out = dict(format='firedoor-oplist-v1', input=dict(name='input', shape=[1, 3, 256, 320], layout='NCHW'),
           outputs=[o.name for o in m.graph.output], nodes=nodes, initializers=inits)
p = os.path.join(R, f'export/firedoor_oplist_{dt}.json')
json.dump(out, open(p, 'w'))
print(p, os.path.getsize(p), 'bytes,', len(nodes), 'nodes')
