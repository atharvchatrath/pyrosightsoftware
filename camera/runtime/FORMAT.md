# PyroSight op-list format (version 1)

An op list is a small, framework-free description of a feed-forward CNN that
`oplist.js` executes with TF.js (WebGL or CPU backend). It is produced by
`export_oplist.py` from an ONNX model or a TFLite flatbuffer and consists of
two parts:

* `NAME.oplist.json` - the graph (this document),
* `NAME.weights.bin` - one little-endian binary blob with every constant.

For inlining into a page, `export_oplist.py --js` also writes
`NAME.oplist.js`, which assigns `{json, weightsB64}` to
`globalThis.PS_OPLIST_ASSETS[NAME]`; `PSOpList.loadEmbedded(asset, {tf})`
loads that shape directly.

## Tensor conventions

* Every 4-D activation is **NHWC** (TF.js layout). ONNX inputs declared NCHW
  are re-declared NHWC (`inputs[i].layout = "nhwc"`, `onnx_shape` keeps the
  original). Pass `--input-layout nchw` to keep NCHW inputs.
* Graph outputs keep the **source layout** (ONNX outputs come back NCHW,
  bit-for-bit comparable with onnxruntime) unless the model was exported with
  `--nhwc-outputs`; `outputs[i].layout` says which.
* Shapes are static: they are the shapes seen when the source model was
  probed (ONNX: one onnxruntime run with zeros; TFLite: tensor shapes).
  Ops that depend on memory order (`reshape`, `transpose`, `softmax` on a
  non-last axis ...) are emitted in ONNX order with explicit transposes, so
  results equal the source model's.
* Data type of every activation is float32.

## JSON document

```jsonc
{
  "format": "pyrosight-oplist",
  "version": 1,
  "inputs":  [{"name": "input", "shape": [1, 128, 128, 3], "dtype": "float32",
               "layout": "nhwc", "onnx_shape": [1, 3, 128, 128]}],
  "outputs": [{"name": "heat", "shape": [1, 2, 30, 40], "layout": "native"}],
  "weights": [ /* weight table, below */ ],
  "weights_bytes": 230552,
  "ops":     [ /* executed in order, below */ ],
  "shapes":  {"tensor name": [1, 64, 64, 24], ...},   // informational
  "meta":    {"source": {"kind": "tflite", "file": "...", "sha256": "..."},
              "weight_storage": "float16", "weight_bytes_by_dtype": {...}, ...}
}
```

## Weight table

Each entry names one constant tensor, already in TF.js layout:

| field    | meaning |
|----------|---------|
| `name`   | tensor name used by `ops[].inputs` |
| `shape`  | TF.js shape |
| `dtype`  | storage type: `float32`, `float16`, `uint8`, `int32` |
| `offset` | byte offset in the blob (always a multiple of 4) |
| `bytes`  | stored byte length |
| `quant`  | only for `uint8`: `{axis, channels, min_offset, scale_offset}` |

Decoding to float32:

* `float32`: little-endian IEEE single.
* `float16`: little-endian IEEE half (`oplist.js` decodes with a 64k-entry table).
* `uint8`, per-output-channel affine: the tensor is viewed as
  `[outer, channels]` with `channels = prod(shape[axis:])` (the fastest
  varying index is the channel); `w[i] = min[c] + q[i] * scale[c]`,
  `c = i % channels`. `min` and `scale` are float32 arrays of length
  `channels` at `min_offset` / `scale_offset`. The channel axis is the output
  channel: last axis of a conv filter `[kh, kw, in, out]` and of a dense
  matrix `[K, M]`; for a depthwise filter `[kh, kw, C, mult]` it is
  `C*mult` (`axis = 2`).
* `int32`: little-endian int32 (shape vectors etc.).

`export_oplist.py --weights {float16|uint8|float32}` picks the storage of
"weight" tensors with at least `--min-quant` (default 1024) elements; biases,
BatchNorm-derived vectors and small tensors always stay float32.

## Ops

Every op is `{"op": NAME, "inputs": [...], "outputs": [one name], ...attrs}`.
Inputs are names of graph inputs, weights or earlier outputs. TF.js calls in
brackets.

| op | inputs | attributes | TF.js |
|----|--------|------------|-------|
| `conv2d` | x, filter `[kh,kw,in/groups,out]`, bias? | `strides [sh,sw]`, `dilations [dh,dw]`, `pad`, `act`, `groups` | `tf.fused.conv2d`; `groups > 1` splits x and filter and concatenates |
| `depthwise_conv2d` | x, filter `[kh,kw,C,mult]`, bias? | `strides`, `dilations`, `pad`, `act` | `tf.fused.depthwiseConv2d` |
| `dense` | x `[N,K]`, w `[K,M]`, bias? | `act` | `tf.fused.matMul` |
| `matmul` | a, b | `transpose_b` | `tf.matMul` |
| `add` `sub` `mul` `div` `pow` `maximum` `minimum` | a, b | - | broadcasting (`tf.add` ...) |
| `relu` `relu6` `sigmoid` `tanh` `elu` `exp` `neg` `abs` `sqrt` | x | - | same name |
| `leaky_relu` | x | `alpha` | `tf.leakyRelu` |
| `clip` | x | `min`, `max` | `tf.clipByValue` |
| `hard_sigmoid` | x | `alpha`, `beta` | `clip(alpha*x+beta, 0, 1)` |
| `hard_swish` | x | - | `x * relu6(x+3) / 6` |
| `concat` | x... | `axis` | `tf.concat` |
| `reshape` | x | `shape` | `tf.reshape` |
| `transpose` | x | `perm` | `tf.transpose` |
| `slice` | x | `begin`, `size` | `tf.slice` |
| `tile` | x | `reps` | `tf.tile` |
| `pad` | x | `pads [[before,after] per dim]`, `value`, `mode` (`constant`/`reflect`) | `tf.pad` / `tf.mirrorPad` |
| `max_pool` `avg_pool` | x | `k [kh,kw]`, `strides`, `pad`, `count_include_pad` | `tf.maxPool` / `tf.avgPool` |
| `mean` `max` `sum` | x | `axes`, `keepdims` | reductions (GlobalAveragePool = `mean` over `[1,2]`) |
| `softmax` | x | `axis`, `beta`? | `tf.softmax` (transposed for a non-last axis) |
| `resize` | x | `mode` (`nearest`/`bilinear`), `size [h,w]`, `align_corners`, `half_pixel` | `tf.image.resizeNearestNeighbor` / `resizeBilinear` |
| `identity` | x | - | `clone` |
| `cast_float` | x | - | `tf.cast(x, 'float32')` |

`act` is `linear`, `relu`, `relu6` or `sigmoid` (fused by TF.js).

`pad` of `conv2d` / `depthwise_conv2d` / pooling is `"same"` (TF semantics:
extra row/column at the bottom/right), `"valid"`, or explicit
`[top, bottom, left, right]`. At run time an explicit pad that equals what
`same` would give for the actual input size runs as `same`; any other
explicit pad is applied exactly: convolutions get a zero `tf.pad` and run
`valid`; max/avg pooling passes it to TF.js as explicit padding (padded cells
are ignored by max and excluded from the average, i.e. ONNX
`count_include_pad = 0`); with `count_include_pad` the zeros are padded
first. ONNX `ceil_mode` is converted to extra bottom/right padding at export.

## What the exporter folds or rejects

* ONNX: BatchNormalization is folded into the preceding conv when it is the
  conv's only consumer (else `mul` + `add`); `Constant`, `Shape` and any node
  whose inputs are all constant are evaluated at export (onnxruntime probe);
  `Identity` is aliased; Gemm/MatMul with a constant matrix become `dense`.
  A following ReLU/ReLU6(Clip 0..6)/Sigmoid is fused into conv/dense.
* TFLite: `DEQUANTIZE` of float16 constants is folded into float32 weights
  (re-stored per `--weights`); fused activations NONE/RELU/RELU6 are fused,
  TANH/RELU_N1_TO_1 become separate ops. Integer-quantised (int8/uint8)
  TFLite models are rejected.
* Unsupported ops raise an error naming the node.

ONNX ops accepted: Conv (incl. depthwise, grouped, dilated, asymmetric pads,
auto_pad), BatchNormalization, Relu, Clip, Sigmoid, HardSigmoid, HardSwish,
LeakyRelu, Tanh, Elu, Exp, Neg, Abs, Sqrt, Add, Sub, Mul, Div, Pow, Max, Min,
Concat, Resize/Upsample (nearest and linear), MaxPool, AveragePool,
GlobalAveragePool, GlobalMaxPool, ReduceMean/Max/Sum, Pad (constant,
reflect), Reshape, Flatten, Squeeze, Unsqueeze, Transpose, Gemm, MatMul,
Softmax, Split, Slice (step 1), Tile, Identity, plus constant-only subgraphs.

TFLite ops accepted: CONV_2D, DEPTHWISE_CONV_2D, FULLY_CONNECTED, ADD, SUB,
MUL, DIV, RELU, RELU6, RELU_N1_TO_1, LOGISTIC, TANH, HARD_SWISH, LEAKY_RELU,
PAD, PADV2, MAX_POOL_2D, AVERAGE_POOL_2D, RESHAPE, CONCATENATION,
RESIZE_NEAREST_NEIGHBOR, RESIZE_BILINEAR, MEAN, SOFTMAX, TRANSPOSE,
DEQUANTIZE (float16 constants).

## COCO-SSD re-pack (`repack_cocossd.py`)

The person model is not an op list but a TF.js graph model whose weights use
the same weight-table encoding:

```jsonc
{
  "format": "pyrosight-tfjs-graph", "version": 1,
  "modelTopology": { /* TF.js GraphDef JSON, preprocessor removed, compacted */ },
  "weightSpecs": [{"name", "shape", "dtype": "float32"|"int32"}],  // order of weightData
  "packed": [ /* weight table as above, same names and order */ ],
  "weights_bytes": 3442288,
  "input": {"name": "image", "shape": [1, 300, 300, 3], "preprocess": "..."},
  "outputs": {"scores": "Postprocessor/Slice", "boxes": "Postprocessor/ExpandDims_1"},
  "classes": [{"index": 0, "coco_id": 1, "name": "person"}],
  "meta": {...}
}
```

`people.js` decodes `packed` with `PSOpList.decodeWeights`, concatenates the
float32 arrays in `weightSpecs` order and calls
`tf.loadGraphModel(tf.io.fromMemory({modelTopology, weightSpecs, weightData}))`.
