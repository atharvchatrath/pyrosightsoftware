// Reference TF.js interpreter for export/firedoor_oplist_*.json, checked against onnxruntime outputs.
//   node tools/tfjs_oplist_check.js export/firedoor_oplist_fp16.json
// It shows the one TF.js gotcha: ONNX pads are SYMMETRIC [1,1,1,1] on 3x3 convs (including stride 2), so use
// explicit padding [[0,0],[1,1],[1,1],[0,0]]. Never use 'same': TF 'same' pads 0 top/left, 1 bottom/right for stride 2.
const fs = require('fs');
const path = require('path');
const tf = require(path.join(__dirname, '..', '..', 'node_modules', '@tensorflow', 'tfjs'));
const D = require(path.join(__dirname, '..', 'decode.js'));

function f16ToF32(u16) {
  const out = new Float32Array(u16.length);
  for (let i = 0; i < u16.length; i++) {
    const h = u16[i], s = (h & 0x8000) ? -1 : 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    out[i] = e === 0 ? s * Math.pow(2, -14) * (m / 1024) : e === 31 ? (m ? NaN : s * Infinity) : s * Math.pow(2, e - 15) * (1 + m / 1024);
  }
  return out;
}

function loadInit(t) {
  const buf = Buffer.from(t.b64, 'base64');
  const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.length);
  if (t.dtype === 'float16') return f16ToF32(new Uint16Array(ab));
  if (t.dtype === 'float32') return new Float32Array(ab);
  throw new Error('unsupported dtype ' + t.dtype);
}

function buildRunner(g) {
  // pre-convert weights: Conv W [O, I/g, kh, kw] -> conv2d [kh, kw, I, O] or depthwise [kh, kw, C, 1]
  const W = {};
  for (const n of g.nodes) {
    if (n.op !== 'Conv') continue;
    const t = g.initializers[n.inputs[1]], [O, Ig, kh, kw] = t.dims;
    const w = tf.tensor4d(loadInit(t), [O, Ig, kh, kw]);
    const dw = n.attrs.group > 1;
    if (dw && !(n.attrs.group === O && Ig === 1)) throw new Error('only depthwise grouping supported');
    W[n.inputs[1]] = dw ? w.transpose([2, 3, 0, 1]) : w.transpose([2, 3, 1, 0]);
    W[n.inputs[2]] = tf.tensor1d(loadInit(g.initializers[n.inputs[2]]));
    w.dispose();
  }
  const scalar = (name) => loadInit(g.initializers[name])[0];
  return function run(inputNCHW) {
    return tf.tidy(() => {
      const env = { [g.input.name]: tf.tensor4d(inputNCHW, g.input.shape).transpose([0, 2, 3, 1]) };   // NHWC inside
      for (const n of g.nodes) {
        const x = env[n.inputs[0]];
        let y;
        if (n.op === 'Conv') {
          const [pt, pl, pb, pr] = n.attrs.pads, s = n.attrs.strides;
          const pad = (pt | pl | pb | pr) ? [[0, 0], [pt, pb], [pl, pr], [0, 0]] : 'valid';
          const dw = n.attrs.group > 1;
          y = dw ? tf.depthwiseConv2d(pad === 'valid' ? x : tf.pad(x, pad), W[n.inputs[1]], s, 'valid')
                 : tf.conv2d(x, W[n.inputs[1]], s, pad);
          y = tf.add(y, W[n.inputs[2]]);
        } else if (n.op === 'Clip') y = tf.clipByValue(x, scalar(n.inputs[1]), scalar(n.inputs[2]));
        else if (n.op === 'Relu') y = tf.relu(x);
        else if (n.op === 'Add') y = tf.add(x, env[n.inputs[1]]);
        else if (n.op === 'Sigmoid') y = tf.sigmoid(x);
        else if (n.op === 'Resize') {     // nearest, asymmetric, floor, scales [1,1,2,2]
          const sc = loadInit(g.initializers[n.inputs[2]]);
          y = tf.image.resizeNearestNeighbor(x, [x.shape[1] * sc[2], x.shape[2] * sc[3]], false, false);
        } else if (n.op === 'Identity') y = x || tf.tensor(loadInit(g.initializers[n.inputs[0]]), g.initializers[n.inputs[0]].dims);
        else if (n.op === 'Mul') y = tf.mul(x, env[n.inputs[1]] || tf.scalar(scalar(n.inputs[1])));
        else if (n.op === 'Concat') {     // the window model's heat: fire, door (trunk) + window (branch) channels
          if (n.attrs.axis !== 1) throw new Error('Concat only on the channel axis');
          y = tf.concat(n.inputs.map((i) => env[i]), 3);
        } else throw new Error('op ' + n.op);
        env[n.outputs[0]] = y;
      }
      return g.outputs.map((o) => env[o].transpose([0, 3, 1, 2]).dataSync());   // back to NCHW, Float32Array
    });
  };
}

(async () => {
  await tf.setBackend('cpu');
  const g = JSON.parse(fs.readFileSync(process.argv[2]));
  const run = buildRunner(g);
  const S = path.join(__dirname, '..', 'export', 'sample_io');
  const rd = (f) => { const b = fs.readFileSync(path.join(S, f)); return new Float32Array(b.buffer, b.byteOffset, b.length / 4); };
  for (const name of ['fire', 'door', 'window']) {
    if (!fs.existsSync(path.join(S, name + '_input_1x3x256x320.f32'))) continue;
    const t0 = Date.now();
    const [heat, wh, off, whW, offW] = run(rd(name + '_input_1x3x256x320.f32'));
    const ms = Date.now() - t0;
    const C = heat.length / 1280;     // classes: 2 (fire, door) or 3 (+ window)
    const ref = ['heat', 'wh', 'off'].map((o) => rd(`${name}_${o}_1x${o === 'heat' ? C : 2}x32x40.f32`));
    const md = [heat, wh, off].map((a, k) => a.reduce((m, v, i) => Math.max(m, Math.abs(v - ref[k][i])), 0));
    const dets = D.decode(heat, wh, off, whW ? { windowWh: whW, windowOff: offW } : {});
    const exp = JSON.parse(fs.readFileSync(path.join(S, name + '_expected.json'))).detections_default_thresholds;
    console.log(JSON.stringify({ name, weights: path.basename(process.argv[2]), maxAbsDiff: { heat: md[0], wh: md[1], off: md[2] },
      tfjs_dets: dets.map((d) => [d.cls, +d.score.toFixed(3), +d.x.toFixed(3), +d.y.toFixed(3), +d.w.toFixed(3), +d.h.toFixed(3)]),
      ort_dets: exp.map((d) => [d.cls, +d.score.toFixed(3), +d.x.toFixed(3), +d.y.toFixed(3), +d.w.toFixed(3), +d.h.toFixed(3)]), first_run_ms: ms }));
  }
  const x = rd('fire_input_1x3x256x320.f32');
  const t1 = Date.now();
  for (let i = 0; i < 3; i++) run(x);
  console.log(JSON.stringify({ backend: tf.getBackend(), steady_state_ms_per_frame: (Date.now() - t1) / 3 }));
})();
