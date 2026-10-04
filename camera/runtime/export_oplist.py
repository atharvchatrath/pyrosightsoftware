#!/usr/bin/env python3
"""Convert an ONNX model or a TFLite flatbuffer into a PyroSight op list.

The op list is one JSON document (graph, shapes, weight table) plus one binary
weights blob. It is executed in the browser (and in node for tests) by
oplist.js on top of TF.js, using only ops the WebGL and CPU backends both
implement. Tensors are NHWC like TF.js; see FORMAT.md for the full contract.

    python3 export_oplist.py model.onnx  -o out/fire --weights float16
    python3 export_oplist.py face.tflite -o out/face --weights uint8

writes out/fire.oplist.json and out/fire.weights.bin (and, with --js,
out/fire.oplist.js which assigns both, base64, to a global for inlining).

Weight storage (--weights):
  float16  default; every tensor with >= --min-quant elements stored as IEEE
           half, the rest float32.
  uint8    per-output-channel affine: w = min[c] + q * scale[c].
  float32  lossless.
Biases, BN-derived vectors and int tensors are always float32/int32.

ONNX input is NCHW; by default the op list input is declared NHWC (what
tf.browser.fromPixels gives) and every 4-D activation is kept NHWC. Ops whose
meaning depends on memory order (Reshape, Flatten, Transpose, Softmax over
non-channel axes ...) get an explicit transpose back to ONNX order first, so
results match onnxruntime exactly; graph outputs are returned in ONNX layout
unless --nhwc-outputs is given.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import sys

import numpy as np

FUSABLE_ACTS = {'relu', 'relu6', 'sigmoid'}          # tf.fused.* activations we use
ELEMENTWISE_UNARY = {'relu', 'relu6', 'sigmoid', 'tanh', 'hard_sigmoid', 'hard_swish', 'clip',
                     'leaky_relu', 'exp', 'identity', 'neg', 'abs', 'sqrt'}


# --------------------------------------------------------------------------
# IR
# --------------------------------------------------------------------------
class OpList:
    def __init__(self):
        self.ops = []           # list of dict(op=..., inputs=[...], outputs=[...], **attrs)
        self.consts = {}        # name -> np.ndarray (already in TF.js layout)
        self.const_kind = {}    # name -> 'weight'|'bias'|'int'
        self.shapes = {}        # name -> list (stored layout)
        self.inputs = []        # [dict(name, shape, dtype)]
        self.outputs = []       # [dict(name, shape)]
        self.meta = {}
        self._n = 0

    def fresh(self, base):
        self._n += 1
        base = base.replace(':', '_')
        return '%s__%d' % (base, self._n)

    def add_const(self, name, arr, kind='weight'):
        arr = np.asarray(arr)
        if arr.dtype == np.float64 or arr.dtype == np.float16:
            arr = arr.astype(np.float32)
        if arr.dtype == np.int64:
            arr = arr.astype(np.int32)
        if arr.dtype == np.bool_:
            arr = arr.astype(np.int32)
        if name in self.consts:
            name = self.fresh(name)
        self.consts[name] = np.ascontiguousarray(arr)
        self.const_kind[name] = kind
        self.shapes[name] = list(arr.shape)
        return name

    def add_op(self, op, inputs, outputs, out_shapes, **attrs):
        d = dict(op=op, inputs=list(inputs), outputs=list(outputs))
        d.update(attrs)
        self.ops.append(d)
        for o, s in zip(outputs, out_shapes):
            self.shapes[o] = [int(v) for v in s]
        return d

    # ---------------------------------------------------------------- passes
    def consumers(self):
        c = {}
        for i, op in enumerate(self.ops):
            for t in op['inputs']:
                c.setdefault(t, []).append(i)
        return c

    def fuse_activations(self):
        """conv2d/depthwise/dense/add followed by its only consumer relu/relu6/sigmoid."""
        outs = {o['name'] for o in self.outputs}
        changed = True
        while changed:
            changed = False
            cons = self.consumers()
            producer = {}
            for i, op in enumerate(self.ops):
                for t in op['outputs']:
                    producer[t] = i
            for i, op in enumerate(self.ops):
                if op['op'] not in FUSABLE_ACTS:
                    continue
                src = op['inputs'][0]
                if src not in producer or src in outs:
                    continue
                p = self.ops[producer[src]]
                if p['op'] not in ('conv2d', 'depthwise_conv2d', 'dense') or p.get('act', 'linear') != 'linear':
                    continue
                if len(cons.get(src, [])) != 1:
                    continue
                p['act'] = op['op']
                p['outputs'] = [op['outputs'][0]]
                self.shapes[op['outputs'][0]] = self.shapes[src]
                del self.ops[i]
                changed = True
                break

    def remove_identities(self):
        outs = {o['name'] for o in self.outputs}
        ins = {i['name'] for i in self.inputs}
        # identity -> graph output: rename the producer's output instead
        cons = self.consumers()
        producer = {}
        for op in self.ops:
            for t in op['outputs']:
                producer[t] = op
        for op in list(self.ops):
            if op['op'] != 'identity' or op['outputs'][0] not in outs:
                continue
            src = op['inputs'][0]
            if src in ins or src in outs or src in self.consts or len(cons.get(src, [])) != 1 or src not in producer:
                continue
            p = producer[src]
            p['outputs'] = [op['outputs'][0] if t == src else t for t in p['outputs']]
            self.shapes[op['outputs'][0]] = self.shapes[src]
            self.ops.remove(op)
        alias = {}
        keep = []
        for op in self.ops:
            op['inputs'] = [alias.get(t, t) for t in op['inputs']]
            if op['op'] == 'identity' and op['outputs'][0] not in outs:
                alias[op['outputs'][0]] = op['inputs'][0]
                continue
            keep.append(op)
        self.ops = keep

    def prune(self):
        """Drop ops/consts not needed for the graph outputs."""
        need = {o['name'] for o in self.outputs}
        keep = []
        for op in reversed(self.ops):
            if any(o in need for o in op['outputs']):
                keep.append(op)
                need.update(op['inputs'])
        self.ops = list(reversed(keep))
        for c in list(self.consts):
            if c not in need:
                del self.consts[c]
                self.const_kind.pop(c, None)

    def optimise(self):
        self.remove_identities()
        self.fuse_activations()
        self.prune()

    # ---------------------------------------------------------------- write
    def serialise(self, storage='float16', min_quant=1024):
        blob = bytearray()

        def align(n=4):
            while len(blob) % n:
                blob.append(0)

        weights = []
        stats = dict(float32=0, float16=0, uint8=0, int32=0)
        for name, arr in self.consts.items():
            spec = dict(name=name, shape=[int(s) for s in arr.shape])
            kind = self.const_kind.get(name, 'weight')
            align(4)
            if arr.dtype.kind in 'iu':
                a = arr.astype('<i4')
                spec.update(dtype='int32', offset=len(blob), bytes=a.nbytes)
                blob += a.tobytes()
                stats['int32'] += a.nbytes
            elif kind != 'weight' or arr.size < min_quant or storage == 'float32':
                a = arr.astype('<f4')
                spec.update(dtype='float32', offset=len(blob), bytes=a.nbytes)
                blob += a.tobytes()
                stats['float32'] += a.nbytes
            elif storage == 'float16':
                a = arr.astype('<f2')
                if not np.all(np.isfinite(a[np.isfinite(arr)])):
                    raise ValueError('float16 overflow in %s' % name)
                spec.update(dtype='float16', offset=len(blob), bytes=a.nbytes)
                blob += a.tobytes()
                stats['float16'] += a.nbytes
            elif storage == 'uint8':
                axis = quant_axis(self.const_kind.get(name + '#layout', None), arr)
                ch = int(np.prod(arr.shape[axis:])) if arr.ndim else 1
                flat = arr.reshape(-1, ch).astype(np.float64)
                mn = flat.min(axis=0)
                mx = flat.max(axis=0)
                scale = (mx - mn) / 255.0
                scale[scale == 0] = 1.0
                q = np.clip(np.round((flat - mn) / scale), 0, 255).astype(np.uint8)
                spec.update(dtype='uint8', offset=len(blob), bytes=q.nbytes)
                blob += q.tobytes()
                align(4)
                mn32 = mn.astype('<f4')
                sc32 = scale.astype('<f4')
                spec['quant'] = dict(axis=axis, channels=ch, min_offset=len(blob), scale_offset=len(blob) + 4 * ch)
                blob += mn32.tobytes() + sc32.tobytes()
                stats['uint8'] += q.nbytes + 8 * ch
            else:
                raise ValueError(storage)
            weights.append(spec)
        align(4)
        doc = dict(
            format='pyrosight-oplist', version=1,
            inputs=self.inputs, outputs=self.outputs,
            weights=weights, weights_bytes=len(blob),
            ops=self.ops, shapes={k: v for k, v in self.shapes.items() if k not in self.consts},
            meta=self.meta,
        )
        doc['meta']['weight_storage'] = storage
        doc['meta']['weight_bytes_by_dtype'] = stats
        return doc, bytes(blob)


def quant_axis(layout_hint, arr):
    """Which trailing axes form the 'output channel' for per-channel uint8."""
    if arr.ndim <= 1:
        return 0
    if layout_hint == 'depthwise':      # [kh, kw, C, mult] -> channel = c*mult+m
        return arr.ndim - 2
    return arr.ndim - 1                 # HWIO conv, [K, M] dense: last axis


def same_pads(in_size, k, s, d=1):
    """TF 'SAME' padding (before, after) for one spatial dim."""
    ek = (k - 1) * d + 1
    out = -(-in_size // s)
    total = max((out - 1) * s + ek - in_size, 0)
    return total // 2, total - total // 2


def conv_out(in_size, k, s, d, pb, pa, ceil_mode=False):
    ek = (k - 1) * d + 1
    v = (in_size + pb + pa - ek) / s + 1
    return int(math.ceil(v) if ceil_mode else math.floor(v))


# --------------------------------------------------------------------------
# TFLite frontend
# --------------------------------------------------------------------------
TFL_ACT = {0: 'linear', 1: 'relu', 2: 'relu_n1_to_1', 3: 'relu6', 4: 'tanh', 5: 'sign_bit'}
TFL_DTYPE = {0: np.float32, 1: np.float16, 2: np.int32, 3: np.uint8, 4: np.int64, 9: np.int8, 7: np.int16}


def load_tflite(path):
    import tflite
    from tflite.BuiltinOperator import BuiltinOperator
    from tflite.BuiltinOptions import BuiltinOptions

    opnames = {v: k for k, v in BuiltinOperator.__dict__.items() if not k.startswith('_')}
    optnames = {v: k for k, v in BuiltinOptions.__dict__.items() if not k.startswith('_')}
    buf = open(path, 'rb').read()
    m = tflite.Model.GetRootAsModel(buf, 0)
    if m.SubgraphsLength() != 1:
        raise ValueError('only single-subgraph TFLite models are supported')
    sg = m.Subgraphs(0)
    g = OpList()
    g.meta['source'] = dict(kind='tflite', file=os.path.basename(path),
                            sha256=hashlib.sha256(buf).hexdigest())

    tnames = {}
    const_vals = {}
    for i in range(sg.TensorsLength()):
        t = sg.Tensors(i)
        name = t.Name().decode()
        tnames[i] = name
        shape = [int(s) for s in t.ShapeAsNumpy()] if t.ShapeLength() else []
        g.shapes[name] = shape
        b = m.Buffers(t.Buffer())
        if b is not None and b.DataLength() > 0:
            dt = TFL_DTYPE[t.Type()]
            arr = np.frombuffer(b.DataAsNumpy().tobytes(), dtype=dt).reshape(shape)
            q = t.Quantization()
            if q is not None and q.ScaleLength() > 0 and dt in (np.int8, np.uint8, np.int16):
                raise ValueError('integer-quantised TFLite weights are not supported (%s)' % name)
            const_vals[name] = arr

    def T(idx):
        return tnames[int(idx)]

    for i in sg.InputsAsNumpy():
        n = T(i)
        g.inputs.append(dict(name=n, shape=g.shapes[n], dtype='float32'))
    for i in sg.OutputsAsNumpy():
        n = T(i)
        g.outputs.append(dict(name=n, shape=g.shapes[n]))

    def opts(op, cls_name):
        import importlib
        mod = importlib.import_module('tflite.' + cls_name)
        o = getattr(mod, cls_name)()
        tab = op.BuiltinOptions()
        o.Init(tab.Bytes, tab.Pos)
        return o

    def const(name, kind='weight', arr=None, layout=None):
        a = const_vals[name] if arr is None else arr
        if name not in g.consts:
            g.add_const(name, a.astype(np.float32) if a.dtype in (np.float16, np.float64) else a, kind)
            if layout:
                g.const_kind[name + '#layout'] = layout
        return name

    def act_op(src, act, shape):
        if act == 'linear':
            return src
        if act not in ('relu', 'relu6', 'tanh'):
            raise ValueError('unsupported fused activation ' + act)
        out = g.fresh(src + '_' + act)
        g.add_op(act, [src], [out], [shape])
        return out

    rename = {}
    for k in range(sg.OperatorsLength()):
        op = sg.Operators(k)
        oc = m.OperatorCodes(op.OpcodeIndex())
        code = max(oc.BuiltinCode(), oc.DeprecatedBuiltinCode())
        name = opnames.get(code, str(code))
        ins = [T(i) if i >= 0 else None for i in op.InputsAsNumpy()]
        outs = [T(i) for i in op.OutputsAsNumpy()]
        ins = [rename.get(i, i) for i in ins]
        oshape = g.shapes[outs[0]]

        if name == 'DEQUANTIZE':
            src = ins[0]
            if src in const_vals:
                const_vals[outs[0]] = const_vals[src].astype(np.float32)
                continue
            g.add_op('cast_float', [src], outs, [oshape])
        elif name == 'CONV_2D':
            o = opts(op, 'Conv2DOptions')
            w = const_vals[ins[1]]                       # [O, kh, kw, I]
            hwio = np.transpose(w, (1, 2, 3, 0)).astype(np.float32)
            wn = const(ins[1] + '_hwio', arr=hwio, layout='hwio')
            ii = [ins[0], wn]
            if len(ins) > 2 and ins[2] is not None:
                ii.append(const(ins[2], 'bias'))
            pad = 'same' if o.Padding() == 0 else 'valid'
            g.add_op('conv2d', ii, outs, [oshape], strides=[o.StrideH(), o.StrideW()],
                     dilations=[o.DilationHFactor(), o.DilationWFactor()], pad=pad,
                     act=fusable(TFL_ACT[o.FusedActivationFunction()]), groups=1)
            _post_act(g, TFL_ACT[o.FusedActivationFunction()])
        elif name == 'DEPTHWISE_CONV_2D':
            o = opts(op, 'DepthwiseConv2DOptions')
            w = const_vals[ins[1]]                       # [1, kh, kw, C*mult]
            cin = g.shapes[ins[0]][3]
            mult = w.shape[3] // cin
            dw = w.reshape(w.shape[1], w.shape[2], cin, mult).astype(np.float32)
            wn = const(ins[1] + '_dw', arr=dw, layout='depthwise')
            ii = [ins[0], wn]
            if len(ins) > 2 and ins[2] is not None:
                ii.append(const(ins[2], 'bias'))
            pad = 'same' if o.Padding() == 0 else 'valid'
            g.add_op('depthwise_conv2d', ii, outs, [oshape], strides=[o.StrideH(), o.StrideW()],
                     dilations=[o.DilationHFactor(), o.DilationWFactor()], pad=pad,
                     act=fusable(TFL_ACT[o.FusedActivationFunction()]))
            _post_act(g, TFL_ACT[o.FusedActivationFunction()])
        elif name == 'FULLY_CONNECTED':
            o = opts(op, 'FullyConnectedOptions')
            w = const_vals[ins[1]]                       # [M, K]
            wn = const(ins[1] + '_km', arr=w.T.astype(np.float32), layout='dense')
            src = ins[0]
            k_ = w.shape[1]
            if len(g.shapes[src]) != 2:
                r = g.fresh(src + '_flat')
                g.add_op('reshape', [src], [r], [[-1, k_]], shape=[-1, k_])
                g.shapes[r] = [int(np.prod(g.shapes[src])) // k_, k_]
                src = r
            ii = [src, wn]
            if len(ins) > 2 and ins[2] is not None:
                ii.append(const(ins[2], 'bias'))
            g.add_op('dense', ii, outs, [oshape], act=fusable(TFL_ACT[o.FusedActivationFunction()]))
            _post_act(g, TFL_ACT[o.FusedActivationFunction()])
        elif name in ('ADD', 'MUL', 'SUB', 'DIV'):
            cls = {'ADD': 'AddOptions', 'MUL': 'MulOptions', 'SUB': 'SubOptions', 'DIV': 'DivOptions'}[name]
            act = 'linear'
            if op.BuiltinOptionsType() != 0:
                act = TFL_ACT[opts(op, cls).FusedActivationFunction()]
            ii = [const(x, 'bias') if x in const_vals else x for x in ins]
            tmp = outs[0] if act == 'linear' else g.fresh(outs[0] + '_pre')
            g.add_op(name.lower(), ii, [tmp], [oshape])
            if act != 'linear':
                g.add_op(act, [tmp], outs, [oshape])
        elif name in ('RELU', 'RELU6', 'LOGISTIC', 'TANH', 'HARD_SWISH', 'RELU_N1_TO_1'):
            m_ = {'RELU': 'relu', 'RELU6': 'relu6', 'LOGISTIC': 'sigmoid', 'TANH': 'tanh',
                  'HARD_SWISH': 'hard_swish'}
            if name == 'RELU_N1_TO_1':
                g.add_op('clip', ins[:1], outs, [oshape], min=-1.0, max=1.0)
            else:
                g.add_op(m_[name], ins[:1], outs, [oshape])
        elif name == 'LEAKY_RELU':
            g.add_op('leaky_relu', ins[:1], outs, [oshape], alpha=float(opts(op, 'LeakyReluOptions').Alpha()))
        elif name in ('PAD', 'PADV2'):
            p = const_vals[ins[1]].astype(int).tolist()
            val = 0.0
            if name == 'PADV2' and len(ins) > 2:
                val = float(const_vals[ins[2]].reshape(-1)[0])
            g.add_op('pad', ins[:1], outs, [oshape], pads=p, value=val)
        elif name in ('MAX_POOL_2D', 'AVERAGE_POOL_2D'):
            o = opts(op, 'Pool2DOptions')
            g.add_op('max_pool' if name == 'MAX_POOL_2D' else 'avg_pool', ins[:1], outs, [oshape],
                     k=[o.FilterHeight(), o.FilterWidth()], strides=[o.StrideH(), o.StrideW()],
                     pad='same' if o.Padding() == 0 else 'valid', count_include_pad=False)
            _post_act(g, TFL_ACT[o.FusedActivationFunction()])
        elif name == 'RESHAPE':
            shp = list(oshape)
            g.add_op('reshape', ins[:1], outs, [oshape], shape=shp)
        elif name == 'CONCATENATION':
            o = opts(op, 'ConcatenationOptions')
            ax = o.Axis()
            rank = len(oshape)
            ii = [const(x, 'bias') if x in const_vals else x for x in ins]
            g.add_op('concat', ii, outs, [oshape], axis=ax % rank)
            _post_act(g, TFL_ACT[o.FusedActivationFunction()])
        elif name in ('RESIZE_NEAREST_NEIGHBOR', 'RESIZE_BILINEAR'):
            cls = 'ResizeNearestNeighborOptions' if name == 'RESIZE_NEAREST_NEIGHBOR' else 'ResizeBilinearOptions'
            o = opts(op, cls)
            g.add_op('resize', ins[:1], outs, [oshape],
                     mode='nearest' if name == 'RESIZE_NEAREST_NEIGHBOR' else 'bilinear',
                     size=[oshape[1], oshape[2]], align_corners=bool(o.AlignCorners()),
                     half_pixel=bool(o.HalfPixelCenters()))
        elif name == 'MEAN':
            axes = [int(a) for a in np.atleast_1d(const_vals[ins[1]])]
            keep = bool(opts(op, 'ReducerOptions').KeepDims())
            g.add_op('mean', ins[:1], outs, [oshape], axes=axes, keepdims=keep)
        elif name == 'SOFTMAX':
            g.add_op('softmax', ins[:1], outs, [oshape], axis=len(oshape) - 1,
                     beta=float(opts(op, 'SoftmaxOptions').Beta()))
        elif name == 'TRANSPOSE':
            g.add_op('transpose', ins[:1], outs, [oshape], perm=const_vals[ins[1]].astype(int).tolist())
        else:
            raise ValueError('unsupported TFLite op %s (%s)' % (name, optnames.get(op.BuiltinOptionsType())))
    g.optimise()
    return g


def fusable(act):
    return act if act in ('linear', 'relu', 'relu6') else 'linear'


def _post_act(g, act):
    """For fused activations TF.js cannot fuse, append a separate op."""
    if act in ('linear', 'relu', 'relu6'):
        return
    op = g.ops[-1]
    out = op['outputs'][0]
    tmp = g.fresh(out + '_pre')
    op['outputs'][0] = tmp
    g.shapes[tmp] = g.shapes[out]
    if act == 'tanh':
        g.add_op('tanh', [tmp], [out], [g.shapes[out]])
    elif act == 'relu_n1_to_1':
        g.add_op('clip', [tmp], [out], [g.shapes[out]], min=-1.0, max=1.0)
    else:
        raise ValueError('unsupported fused activation ' + act)


# --------------------------------------------------------------------------
# ONNX frontend
# --------------------------------------------------------------------------
NHWC_FROM_NCHW = [0, 2, 3, 1]
NCHW_FROM_NHWC = [0, 3, 1, 2]


def load_onnx(path, input_layout='nhwc', nhwc_outputs=False):
    import onnx
    from onnx import numpy_helper, helper
    import onnxruntime as ort

    model = onnx.load(path)
    raw = open(path, 'rb').read()
    graph = model.graph
    opset = max([o.version for o in model.opset_import if o.domain in ('', 'ai.onnx')] or [13])
    g = OpList()
    g.meta['source'] = dict(kind='onnx', file=os.path.basename(path), opset=opset,
                            sha256=hashlib.sha256(raw).hexdigest())

    init = {i.name: numpy_helper.to_array(i) for i in graph.initializer}
    graph_inputs = [i for i in graph.input if i.name not in init]

    # --- probe every intermediate shape (and constant-foldable value) with onnxruntime
    probe = onnx.ModelProto()
    probe.CopyFrom(model)
    del probe.graph.output[:]
    seen = set()
    for nd in probe.graph.node:
        for o in nd.output:
            if o and o not in seen:
                seen.add(o)
                probe.graph.output.append(helper.make_empty_tensor_value_info(o))
    feeds = {}
    for i in graph_inputs:
        dims = [d.dim_value if d.dim_value > 0 else 1 for d in i.type.tensor_type.shape.dim]
        et = i.type.tensor_type.elem_type
        dt = {1: np.float32, 7: np.int64, 6: np.int32, 10: np.float16}.get(et, np.float32)
        feeds[i.name] = np.zeros(dims, dt)
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    so.log_severity_level = 3
    sess = ort.InferenceSession(probe.SerializeToString(), so, providers=['CPUExecutionProvider'])
    names = [o.name for o in sess.get_outputs()]
    vals = sess.run(names, feeds)
    probe_vals = dict(zip(names, vals))
    oshape = {k: list(v.shape) for k, v in probe_vals.items()}
    for k, v in feeds.items():
        oshape[k] = list(v.shape)

    # value tracking: name -> ('const', ndarray) or ('act', stored_name, layout)
    V = {}
    for k, v in init.items():
        V[k] = ('const', v)
    for i in graph_inputs:
        s = oshape[i.name]
        if len(s) == 4 and input_layout == 'nhwc':
            g.inputs.append(dict(name=i.name, shape=[s[0], s[2], s[3], s[1]], dtype='float32', layout='nhwc',
                                 onnx_shape=s))
            g.shapes[i.name] = [s[0], s[2], s[3], s[1]]
            V[i.name] = ('act', i.name, 'nhwc')
        else:
            g.inputs.append(dict(name=i.name, shape=s, dtype='float32', layout='native', onnx_shape=s))
            g.shapes[i.name] = s
            V[i.name] = ('act', i.name, 'native')

    def is_const(n):
        return n in V and V[n][0] == 'const'

    def cval(n):
        return V[n][1]

    def nhwc(n):
        """stored name of activation n in NHWC layout (4-D)."""
        v = V[n]
        if v[0] == 'const':
            a = v[1]
            if a.ndim == 4:
                return g.add_const(g.fresh(n), np.transpose(a, NHWC_FROM_NCHW).astype(np.float32), 'bias')
            return g.add_const(g.fresh(n), a.astype(np.float32), 'bias')
        if v[2] == 'nhwc':
            return v[1]
        s = g.shapes[v[1]]
        if len(s) != 4:
            raise ValueError('expected 4-D tensor for %s, got %s' % (n, s))
        key = ('nhwc', v[1])
        if key in cache:
            return cache[key]
        out = g.fresh(n + '_nhwc')
        if s[2] * s[3] == 1 or s[1] == 1:   # same memory order: a reshape is enough
            g.add_op('reshape', [v[1]], [out], [[s[0], s[2], s[3], s[1]]], shape=[s[0], s[2], s[3], s[1]])
        else:
            g.add_op('transpose', [v[1]], [out], [[s[0], s[2], s[3], s[1]]], perm=NHWC_FROM_NCHW)
        cache[key] = out
        return out

    def native(n):
        v = V[n]
        if v[0] == 'const':
            a = v[1]
            return g.add_const(g.fresh(n), a.astype(np.float32) if a.dtype.kind == 'f' else a, 'bias')
        if v[2] == 'native':
            return v[1]
        s = g.shapes[v[1]]
        key = ('native', v[1])
        if key in cache:
            return cache[key]
        out = g.fresh(n + '_nchw')
        if s[1] * s[2] == 1 or s[3] == 1:   # same memory order: a reshape is enough
            g.add_op('reshape', [v[1]], [out], [[s[0], s[3], s[1], s[2]]], shape=[s[0], s[3], s[1], s[2]])
        else:
            g.add_op('transpose', [v[1]], [out], [[s[0], s[3], s[1], s[2]]], perm=NCHW_FROM_NHWC)
        cache[key] = out
        return out

    cache = {}

    def layout_of(n):
        v = V[n]
        return 'const' if v[0] == 'const' else v[2]

    def set_act(name, stored, layout):
        V[name] = ('act', stored, layout)

    def attr(nd, key, default=None):
        for a in nd.attribute:
            if a.name == key:
                return helper.get_attribute_value(a)
        return default

    def sdecode(v):
        return v.decode() if isinstance(v, bytes) else v

    for nd in graph.node:
        t = nd.op_type
        ins = list(nd.input)
        outs = list(nd.output)
        o0 = outs[0] if outs else None

        # ---- constant folding: every input constant (or a Shape of a static tensor)
        if t == 'Constant':
            V[o0] = ('const', probe_vals[o0])
            continue
        if t == 'Shape':
            V[o0] = ('const', probe_vals[o0])
            continue
        if all((not x) or is_const(x) for x in ins) and t not in ('Identity',):
            for o in outs:
                if o:
                    V[o] = ('const', probe_vals[o])
            continue

        if t == 'Identity':
            V[o0] = V[ins[0]]
            continue

        if t == 'Conv':
            x = nhwc(ins[0])
            w = cval(ins[1]).astype(np.float32)
            b = cval(ins[2]).astype(np.float32) if len(ins) > 2 and ins[2] else None
            group = attr(nd, 'group', 1)
            ks = list(w.shape[2:])
            strides = list(attr(nd, 'strides', [1, 1]))
            dil = list(attr(nd, 'dilations', [1, 1]))
            pads = conv_pads(nd, attr, sdecode, g.shapes[x][1:3], ks, strides, dil)
            cin = g.shapes[x][3]
            O = w.shape[0]
            ii = [x]
            if group == 1:
                ii.append(g.add_const(g.fresh(ins[1]), np.transpose(w, (2, 3, 1, 0)), 'weight'))
                g.const_kind[ii[-1] + '#layout'] = 'hwio'
                op = 'conv2d'
            elif group == cin and w.shape[1] == 1:
                mult = O // cin
                dw = np.transpose(w.reshape(cin, mult, ks[0], ks[1]), (2, 3, 0, 1))
                ii.append(g.add_const(g.fresh(ins[1]), dw, 'weight'))
                g.const_kind[ii[-1] + '#layout'] = 'depthwise'
                op = 'depthwise_conv2d'
            else:
                ii.append(g.add_const(g.fresh(ins[1]), np.transpose(w, (2, 3, 1, 0)), 'weight'))
                g.const_kind[ii[-1] + '#layout'] = 'hwio'
                op = 'conv2d'
            if b is not None:
                ii.append(g.add_const(g.fresh(ins[2]), b, 'bias'))
            s = oshape[o0]
            out = g.fresh(o0)
            g.add_op(op, ii, [out], [[s[0], s[2], s[3], s[1]]], strides=strides, dilations=dil,
                     pad=pads, act='linear', groups=int(group) if op == 'conv2d' else 1)
            set_act(o0, out, 'nhwc')
            continue

        if t == 'BatchNormalization':
            scale, bias, mean, var = [cval(n).astype(np.float64) for n in ins[1:5]]
            eps = attr(nd, 'epsilon', 1e-5)
            k = scale / np.sqrt(var + eps)
            c = bias - mean * k
            src = V[ins[0]]
            # fold into the producing conv when we are its only consumer
            prod = g.ops[-1] if g.ops else None
            if (src[0] == 'act' and src[2] == 'nhwc' and prod is not None and prod['outputs'][0] == src[1]
                    and prod['op'] in ('conv2d', 'depthwise_conv2d') and prod.get('act') == 'linear'
                    and _single_use(graph, ins[0])):
                wname = prod['inputs'][1]
                w = g.consts[wname].astype(np.float64)
                if prod['op'] == 'conv2d':
                    w = w * k.reshape(1, 1, 1, -1)
                else:
                    kh, kw, C, mult = w.shape
                    w = w * k.reshape(1, 1, C, mult)
                g.consts[wname] = w.astype(np.float32)
                if len(prod['inputs']) > 2:
                    bn = prod['inputs'][2]
                    g.consts[bn] = (g.consts[bn].astype(np.float64) * k + c).astype(np.float32)
                else:
                    prod['inputs'].append(g.add_const(g.fresh(ins[2]), c.astype(np.float32), 'bias'))
                V[o0] = src
                continue
            if layout_of(ins[0]) == 'nhwc' or len(oshape[ins[0]]) == 4:
                x = nhwc(ins[0])
                kk = g.add_const(g.fresh(o0 + '_k'), k.astype(np.float32), 'bias')
                cc = g.add_const(g.fresh(o0 + '_c'), c.astype(np.float32), 'bias')
                lay = 'nhwc'
            else:
                x = native(ins[0])
                rank = len(oshape[ins[0]])
                shp = [1, -1] + [1] * (rank - 2)
                kk = g.add_const(g.fresh(o0 + '_k'), k.astype(np.float32).reshape(shp), 'bias')
                cc = g.add_const(g.fresh(o0 + '_c'), c.astype(np.float32).reshape(shp), 'bias')
                lay = 'native'
            m1 = g.fresh(o0 + '_mul')
            out = g.fresh(o0)
            g.add_op('mul', [x, kk], [m1], [g.shapes[x]])
            g.add_op('add', [m1, cc], [out], [g.shapes[x]])
            set_act(o0, out, lay)
            continue

        if t in ('Relu', 'Sigmoid', 'Tanh', 'HardSigmoid', 'HardSwish', 'LeakyRelu', 'Clip', 'Exp', 'Neg',
                 'Abs', 'Sqrt', 'Elu'):
            src = V[ins[0]]
            x = src[1]
            out = g.fresh(o0)
            if t == 'Clip':
                lo = attr(nd, 'min', None)
                hi = attr(nd, 'max', None)
                if len(ins) > 1 and ins[1]:
                    lo = float(cval(ins[1]))
                if len(ins) > 2 and ins[2]:
                    hi = float(cval(ins[2]))
                lo = -3.4e38 if lo is None else float(lo)
                hi = 3.4e38 if hi is None else float(hi)
                if lo == 0 and hi == 6:
                    g.add_op('relu6', [x], [out], [g.shapes[x]])
                elif lo == 0 and hi >= 3.4e38:
                    g.add_op('relu', [x], [out], [g.shapes[x]])
                else:
                    g.add_op('clip', [x], [out], [g.shapes[x]], min=lo, max=hi)
            elif t == 'HardSigmoid':
                g.add_op('hard_sigmoid', [x], [out], [g.shapes[x]], alpha=float(attr(nd, 'alpha', 0.2)),
                         beta=float(attr(nd, 'beta', 0.5)))
            elif t == 'LeakyRelu':
                g.add_op('leaky_relu', [x], [out], [g.shapes[x]], alpha=float(attr(nd, 'alpha', 0.01)))
            elif t == 'Elu':
                if abs(float(attr(nd, 'alpha', 1.0)) - 1.0) > 1e-6:
                    raise ValueError('Elu alpha != 1 unsupported')
                g.add_op('elu', [x], [out], [g.shapes[x]])
            else:
                g.add_op({'Relu': 'relu', 'Sigmoid': 'sigmoid', 'Tanh': 'tanh', 'HardSwish': 'hard_swish',
                          'Exp': 'exp', 'Neg': 'neg', 'Abs': 'abs', 'Sqrt': 'sqrt'}[t], [x], [out], [g.shapes[x]])
            set_act(o0, out, src[2])
            continue

        if t in ('Add', 'Mul', 'Sub', 'Div', 'Max', 'Min', 'Pow'):
            a, b = ins[0], ins[1]
            la, lb = layout_of(a), layout_of(b)
            rank_o = len(oshape[o0])
            want = 'nhwc' if (rank_o == 4 and 'nhwc' in (la, lb)) else 'native'
            ii = []
            for n in (a, b):
                if is_const(n):
                    c = cval(n)
                    c = c.astype(np.float32) if c.dtype.kind == 'f' else c.astype(np.float32)
                    if want == 'nhwc' and c.ndim > 0 and c.size > 1:
                        c = c.reshape([1] * (4 - c.ndim) + list(c.shape))
                        c = np.transpose(c, NHWC_FROM_NCHW)
                    if c.size == 1:
                        c = c.reshape([])
                    ii.append(g.add_const(g.fresh(n), c, 'bias'))
                else:
                    ii.append(nhwc(n) if want == 'nhwc' else native(n))
            s = oshape[o0]
            out = g.fresh(o0)
            g.add_op(t.lower() if t not in ('Max', 'Min') else ('maximum' if t == 'Max' else 'minimum'),
                     ii, [out], [[s[0], s[2], s[3], s[1]] if want == 'nhwc' else s])
            set_act(o0, out, want)
            continue

        if t == 'Concat':
            axis = attr(nd, 'axis')
            rank = len(oshape[o0])
            axis = axis % rank
            lays = {layout_of(n) for n in ins}
            s = oshape[o0]
            out = g.fresh(o0)
            if rank == 4 and 'nhwc' in lays:
                ii = [nhwc(n) for n in ins]
                g.add_op('concat', ii, [out], [[s[0], s[2], s[3], s[1]]], axis=[0, 3, 1, 2][axis])
                set_act(o0, out, 'nhwc')
            else:
                ii = [native(n) for n in ins]
                g.add_op('concat', ii, [out], [s], axis=axis)
                set_act(o0, out, 'native')
            continue

        if t in ('Resize', 'Upsample'):
            x = nhwc(ins[0])
            s = oshape[o0]
            mode = sdecode(attr(nd, 'mode', 'nearest'))
            ctm = sdecode(attr(nd, 'coordinate_transformation_mode', 'half_pixel' if t == 'Resize' else 'asymmetric'))
            nmode = sdecode(attr(nd, 'nearest_mode', 'round_prefer_floor'))
            if t == 'Upsample' or opset < 11:
                ctm, nmode = 'asymmetric', 'floor'
            out = g.fresh(o0)
            ih, iw = g.shapes[x][1:3]
            oh, ow = s[2], s[3]
            if mode == 'nearest':
                integer_up = oh % ih == 0 and ow % iw == 0
                if ctm == 'asymmetric' and nmode == 'floor':
                    ac, hp = False, False
                elif ctm in ('half_pixel', 'pytorch_half_pixel') and nmode == 'round_prefer_ceil':
                    ac, hp = False, True
                elif integer_up and ctm in ('half_pixel', 'pytorch_half_pixel', 'asymmetric', 'tf_half_pixel_for_nn'):
                    ac, hp = False, False   # every mode picks floor(o / s) for integer up-scaling
                elif ctm == 'align_corners':
                    ac, hp = True, False
                else:
                    raise ValueError('unsupported nearest Resize %s/%s' % (ctm, nmode))
                g.add_op('resize', [x], [out], [[s[0], oh, ow, s[1]]], mode='nearest', size=[oh, ow],
                         align_corners=ac, half_pixel=hp)
            elif mode in ('linear', 'bilinear'):
                ac = ctm == 'align_corners'
                hp = ctm in ('half_pixel', 'pytorch_half_pixel')
                g.add_op('resize', [x], [out], [[s[0], oh, ow, s[1]]], mode='bilinear', size=[oh, ow],
                         align_corners=ac, half_pixel=hp)
            else:
                raise ValueError('unsupported Resize mode ' + mode)
            set_act(o0, out, 'nhwc')
            continue

        if t in ('MaxPool', 'AveragePool'):
            x = nhwc(ins[0])
            ks = list(attr(nd, 'kernel_shape'))
            strides = list(attr(nd, 'strides', [1, 1]))
            dil = list(attr(nd, 'dilations', [1, 1]))
            if dil != [1, 1]:
                raise ValueError('dilated pooling unsupported')
            pads = conv_pads(nd, attr, sdecode, g.shapes[x][1:3], ks, strides, dil)
            ceil_mode = bool(attr(nd, 'ceil_mode', 0))
            s = oshape[o0]
            if ceil_mode:   # extend bottom/right so floor() gives the ceil() size
                if pads in ('same', 'valid'):
                    pads = [0, 0, 0, 0] if pads == 'valid' else [*same_pads(g.shapes[x][1], ks[0], strides[0]),
                                                                  *same_pads(g.shapes[x][2], ks[1], strides[1])]
                ih, iw = g.shapes[x][1:3]
                need_h = (s[2] - 1) * strides[0] + ks[0] - (ih + pads[0] + pads[1])
                need_w = (s[3] - 1) * strides[1] + ks[1] - (iw + pads[2] + pads[3])
                pads = [pads[0], pads[1] + max(0, need_h), pads[2], pads[3] + max(0, need_w)]
            out = g.fresh(o0)
            cip = bool(attr(nd, 'count_include_pad', 0))
            g.add_op('max_pool' if t == 'MaxPool' else 'avg_pool', [x], [out], [[s[0], s[2], s[3], s[1]]],
                     k=ks, strides=strides, pad=pads, count_include_pad=cip)
            set_act(o0, out, 'nhwc')
            continue

        if t in ('GlobalAveragePool', 'GlobalMaxPool'):
            x = nhwc(ins[0])
            s = oshape[o0]
            out = g.fresh(o0)
            g.add_op('mean' if t == 'GlobalAveragePool' else 'max', [x], [out], [[s[0], 1, 1, s[1]]],
                     axes=[1, 2], keepdims=True)
            set_act(o0, out, 'nhwc')
            continue

        if t in ('ReduceMean', 'ReduceMax', 'ReduceSum'):
            axes = attr(nd, 'axes', None)
            if axes is None and len(ins) > 1 and ins[1]:
                axes = cval(ins[1]).tolist()
            keep = bool(attr(nd, 'keepdims', 1))
            rank = len(oshape[ins[0]])
            axes = sorted(a % rank for a in axes)
            opn = {'ReduceMean': 'mean', 'ReduceMax': 'max', 'ReduceSum': 'sum'}[t]
            out = g.fresh(o0)
            if layout_of(ins[0]) == 'nhwc' and keep:
                x = nhwc(ins[0])
                s = oshape[o0]
                g.add_op(opn, [x], [out], [[s[0], s[2], s[3], s[1]]], axes=[[0, 3, 1, 2][a] for a in axes],
                         keepdims=True)
                set_act(o0, out, 'nhwc')
            else:
                x = native(ins[0])
                g.add_op(opn, [x], [out], [oshape[o0]], axes=axes, keepdims=keep)
                set_act(o0, out, 'native')
            continue

        if t == 'Pad':
            mode = sdecode(attr(nd, 'mode', 'constant'))
            if len(ins) > 1 and ins[1]:
                p = cval(ins[1]).astype(int).tolist()
            else:
                p = list(attr(nd, 'pads'))
            val = 0.0
            if len(ins) > 2 and ins[2]:
                val = float(cval(ins[2]).reshape(-1)[0])
            elif attr(nd, 'value', None) is not None:
                val = float(attr(nd, 'value'))
            if len(ins) > 3 and ins[3]:
                raise ValueError('Pad with axes input unsupported')
            r = len(p) // 2
            pp = [[p[i], p[i + r]] for i in range(r)]
            out = g.fresh(o0)
            s = oshape[o0]
            if layout_of(ins[0]) == 'nhwc' and r == 4:
                x = nhwc(ins[0])
                pp = [pp[0], pp[2], pp[3], pp[1]]
                shp = [s[0], s[2], s[3], s[1]]
                lay = 'nhwc'
            else:
                x = native(ins[0])
                shp = s
                lay = 'native'
            if mode not in ('constant', 'reflect'):
                raise ValueError('Pad mode %s unsupported' % mode)
            g.add_op('pad', [x], [out], [shp], pads=pp, value=val, mode=mode)
            set_act(o0, out, lay)
            continue

        if t in ('Reshape', 'Flatten', 'Squeeze', 'Unsqueeze'):
            s = oshape[o0]
            x = native(ins[0])
            out = g.fresh(o0)
            g.add_op('reshape', [x], [out], [s], shape=list(s))
            set_act(o0, out, 'native')
            continue

        if t == 'Transpose':
            perm = list(attr(nd, 'perm'))
            src = V[ins[0]]
            s = oshape[o0]
            if src[2] == 'nhwc' and perm == NHWC_FROM_NCHW:
                set_act(o0, src[1], 'native')          # already stored that way
                continue
            if src[2] == 'native' and perm == NCHW_FROM_NHWC and len(g.shapes[src[1]]) == 4:
                set_act(o0, src[1], 'nhwc')
                continue
            x = native(ins[0])
            out = g.fresh(o0)
            g.add_op('transpose', [x], [out], [s], perm=perm)
            set_act(o0, out, 'native')
            continue

        if t in ('Gemm', 'MatMul'):
            a = native(ins[0])
            if t == 'Gemm':
                alpha = float(attr(nd, 'alpha', 1.0))
                beta = float(attr(nd, 'beta', 1.0))
                if attr(nd, 'transA', 0):
                    raise ValueError('Gemm transA unsupported')
            else:
                alpha, beta = 1.0, 1.0
            if is_const(ins[1]) and len(g.shapes[a]) == 2:
                w = cval(ins[1]).astype(np.float32)
                if t == 'Gemm' and attr(nd, 'transB', 0):
                    w = w.T
                wn = g.add_const(g.fresh(ins[1]), np.ascontiguousarray(w * alpha), 'weight')
                g.const_kind[wn + '#layout'] = 'dense'
                ii = [a, wn]
                if len(ins) > 2 and ins[2]:
                    c = cval(ins[2]).astype(np.float32) * beta
                    ii.append(g.add_const(g.fresh(ins[2]), np.broadcast_to(c, (w.shape[1],)).copy(), 'bias'))
                out = g.fresh(o0)
                g.add_op('dense', ii, [out], [oshape[o0]], act='linear')
            else:
                b = native(ins[1])
                out = g.fresh(o0)
                g.add_op('matmul', [a, b], [out], [oshape[o0]],
                         transpose_b=bool(t == 'Gemm' and attr(nd, 'transB', 0)))
                if t == 'Gemm' and (alpha != 1.0 or (len(ins) > 2 and ins[2])):
                    raise ValueError('general Gemm with alpha/bias on activations unsupported')
            set_act(o0, out, 'native')
            continue

        if t == 'Softmax':
            axis = attr(nd, 'axis', -1 if opset >= 13 else 1)
            rank = len(oshape[o0])
            axis = axis % rank
            out = g.fresh(o0)
            if layout_of(ins[0]) == 'nhwc' and rank == 4 and axis == 1:
                x = nhwc(ins[0])
                g.add_op('softmax', [x], [out], [g.shapes[x]], axis=3)
                set_act(o0, out, 'nhwc')
            else:
                if opset < 13 and axis != rank - 1:
                    raise ValueError('Softmax opset<13 with axis != last unsupported')
                x = native(ins[0])
                g.add_op('softmax', [x], [out], [oshape[o0]], axis=axis)
                set_act(o0, out, 'native')
            continue

        if t in ('Split', 'Slice'):
            x = native(ins[0])
            rank = len(oshape[ins[0]])
            inshape = oshape[ins[0]]
            if t == 'Split':
                axis = attr(nd, 'axis', 0) % rank
                if len(ins) > 1 and ins[1]:
                    sizes = cval(ins[1]).astype(int).tolist()
                elif attr(nd, 'split', None) is not None:
                    sizes = list(attr(nd, 'split'))
                else:
                    sizes = [oshape[o][axis] for o in outs]
                start = 0
                for o, sz in zip(outs, sizes):
                    begin = [0] * rank
                    begin[axis] = start
                    size = list(inshape)
                    size[axis] = sz
                    out = g.fresh(o)
                    g.add_op('slice', [x], [out], [size], begin=begin, size=size)
                    set_act(o, out, 'native')
                    start += sz
            else:
                starts = cval(ins[1]).astype(np.int64).tolist()
                ends = cval(ins[2]).astype(np.int64).tolist()
                axes = cval(ins[3]).astype(int).tolist() if len(ins) > 3 and ins[3] else list(range(len(starts)))
                steps = cval(ins[4]).astype(int).tolist() if len(ins) > 4 and ins[4] else [1] * len(starts)
                if any(st != 1 for st in steps):
                    raise ValueError('Slice with steps != 1 unsupported')
                begin = [0] * rank
                size = list(inshape)
                for st, en, ax in zip(starts, ends, axes):
                    ax %= rank
                    d = inshape[ax]
                    st = max(0, min(d, st + d if st < 0 else st))
                    en = max(0, min(d, en + d if en < 0 else en))
                    begin[ax] = st
                    size[ax] = max(0, en - st)
                out = g.fresh(o0)
                g.add_op('slice', [x], [out], [size], begin=begin, size=size)
                set_act(o0, out, 'native')
            continue

        if t == 'Tile':
            reps = cval(ins[1]).astype(int).tolist()
            s = oshape[o0]
            out = g.fresh(o0)
            if layout_of(ins[0]) == 'nhwc' and len(reps) == 4:
                x = nhwc(ins[0])
                g.add_op('tile', [x], [out], [[s[0], s[2], s[3], s[1]]], reps=[reps[0], reps[2], reps[3], reps[1]])
                set_act(o0, out, 'nhwc')
            else:
                x = native(ins[0])
                g.add_op('tile', [x], [out], [s], reps=reps)
                set_act(o0, out, 'native')
            continue

        raise ValueError('unsupported ONNX op %s (node %s)' % (t, nd.name))

    for o in graph.output:
        v = V[o.name]
        if v[0] == 'const':
            raise ValueError('constant graph output %s' % o.name)
        if nhwc_outputs and v[2] == 'nhwc':
            stored = v[1]
            lay = 'nhwc'
        else:
            stored = native(o.name)
            lay = 'native'
        if stored != o.name:
            g.add_op('identity', [stored], [o.name], [g.shapes[stored]])
        g.outputs.append(dict(name=o.name, shape=g.shapes[o.name], layout=lay, onnx_shape=oshape[o.name]))
    g.optimise()
    return g


def _single_use(graph, name):
    n = 0
    for nd in graph.node:
        n += sum(1 for i in nd.input if i == name)
    n += sum(1 for o in graph.output if o.name == name)
    return n == 1


def conv_pads(nd, attr, sdecode, in_hw, ks, strides, dil):
    auto = sdecode(attr(nd, 'auto_pad', 'NOTSET'))
    if auto in ('SAME_UPPER', 'SAME_LOWER'):
        ph = same_pads(in_hw[0], ks[0], strides[0], dil[0])
        pw = same_pads(in_hw[1], ks[1], strides[1], dil[1])
        if auto == 'SAME_LOWER':
            ph, pw = ph[::-1], pw[::-1]
        return [ph[0], ph[1], pw[0], pw[1]]
    if auto == 'VALID':
        return 'valid'
    p = list(attr(nd, 'pads', [0, 0, 0, 0]))
    t_, l_, b_, r_ = p
    return [int(t_), int(b_), int(l_), int(r_)]


# --------------------------------------------------------------------------
def write(g, out_prefix, storage, min_quant, js=False, js_global='PS_OPLIST_ASSETS', js_key=None):
    doc, blob = g.serialise(storage, min_quant)
    os.makedirs(os.path.dirname(os.path.abspath(out_prefix)), exist_ok=True)
    with open(out_prefix + '.oplist.json', 'w') as f:
        json.dump(doc, f, separators=(',', ':'))
    with open(out_prefix + '.weights.bin', 'wb') as f:
        f.write(blob)
    if js:
        key = js_key or os.path.basename(out_prefix)
        with open(out_prefix + '.oplist.js', 'w') as f:
            f.write('(globalThis.%s = globalThis.%s || {})[%s] = {json: %s, weightsB64: "%s"};\n' % (
                js_global, js_global, json.dumps(key), json.dumps(doc, separators=(',', ':')),
                base64.b64encode(blob).decode()))
    return doc, blob


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('model', help='.onnx or .tflite')
    ap.add_argument('-o', '--out', required=True, help='output prefix (writes PREFIX.oplist.json + PREFIX.weights.bin)')
    ap.add_argument('--weights', default='float16', choices=['float16', 'uint8', 'float32'])
    ap.add_argument('--min-quant', type=int, default=1024,
                    help='tensors with fewer elements stay float32 (default 1024)')
    ap.add_argument('--input-layout', default='nhwc', choices=['nhwc', 'nchw'],
                    help='ONNX only: declare 4-D inputs NHWC (default) or keep NCHW')
    ap.add_argument('--nhwc-outputs', action='store_true', help='ONNX only: leave 4-D outputs NHWC')
    ap.add_argument('--js', action='store_true', help='also write PREFIX.oplist.js (base64, for inlining)')
    ap.add_argument('--meta', action='append', default=[], help='KEY=JSON stored in meta (e.g. mean=[0.5])')
    a = ap.parse_args(argv)
    if a.model.endswith('.tflite'):
        g = load_tflite(a.model)
    elif a.model.endswith('.onnx'):
        g = load_onnx(a.model, 'nhwc' if a.input_layout == 'nhwc' else 'native', a.nhwc_outputs)
    else:
        raise SystemExit('model must be .onnx or .tflite')
    for kv in a.meta:
        k, v = kv.split('=', 1)
        g.meta[k] = json.loads(v)
    doc, blob = write(g, a.out, a.weights, a.min_quant, a.js)
    ops = {}
    for op in doc['ops']:
        ops[op['op']] = ops.get(op['op'], 0) + 1
    print('wrote %s.oplist.json (%d ops: %s) and %s.weights.bin (%d bytes, %s)' % (
        a.out, len(doc['ops']), ', '.join('%s x%d' % kv for kv in sorted(ops.items())), a.out, len(blob),
        doc['meta']['weight_bytes_by_dtype']))


if __name__ == '__main__':
    main()
