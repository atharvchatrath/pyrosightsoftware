#!/usr/bin/env node
// Fire/door parity: op-list runtime (TF.js CPU backend in node) vs onnxruntime.
//   node tests/firedoor_parity.js PREFIX[,PREFIX2...] [--ref tests/out/firedoor_ref] [--json out.json]
// PREFIX.oplist.json + PREFIX.weights.bin from runtime/export_oplist.py (--nhwc-outputs or not).
// Compares raw heat/wh/off (max abs error) and the decoded boxes (firedoor/decode.js,
// default thresholds fire 0.50 / door 0.35 / window 0.50, and the hysteresis floor 0.35 for every class).
const path = require('path');
const fs = require('fs');
const CAMERA = path.join(__dirname, '..');
const tf = require(path.join(CAMERA, 'node_modules/@tensorflow/tfjs'));
require(path.join(CAMERA, 'runtime/oplist.js'));
const tfile = require(path.join(CAMERA, 'runtime/tests/tensorfile.js'));
const dec = require(path.join(CAMERA, 'firedoor/decode.js'));

function toNCHW(data, shape) {   // [1,H,W,C] -> [1,C,H,W]
  const [, H, W, C] = shape, out = new Float32Array(data.length);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) for (let c = 0; c < C; c++) out[c * H * W + y * W + x] = data[(y * W + x) * C + c];
  return out;
}
function iou(a, b) { return dec.iou(a, b); }

function compareBoxes(ref, got) {
  // greedy one-to-one match by class and IoU
  const used = new Set();
  let matched = 0, minIoU = 1, maxDScore = 0;
  for (const r of ref) {
    let best = -1, bi = 0;
    got.forEach((g, j) => { if (!used.has(j) && g.cls === r.cls) { const v = iou(r, g); if (v > bi) { bi = v; best = j; } } });
    if (best >= 0 && bi > 0.5) { used.add(best); matched++; minIoU = Math.min(minIoU, bi); maxDScore = Math.max(maxDScore, Math.abs(r.score - got[best].score)); }
  }
  return { ref: ref.length, got: got.length, matched, minIoU: matched ? minIoU : null, maxDScore };
}

async function main() {
  const args = process.argv.slice(2);
  let refPrefix = path.join(__dirname, 'out/firedoor_ref'), jsonOut = null;
  const prefixes = [];
  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--ref') refPrefix = args[++i];
    else if (args[i] === '--json') jsonOut = args[++i];
    else prefixes.push(...args[i].split(','));
  }
  await tf.setBackend('cpu');
  const ref = tfile.load(refPrefix);
  const report = { ref: path.relative(CAMERA, refPrefix), cases: ref.meta.cases.length, models: [] };
  for (const prefix of prefixes) {
    const json = JSON.parse(fs.readFileSync(prefix + '.oplist.json', 'utf8'));
    const model = PSOpList.load(json, fs.readFileSync(prefix + '.weights.bin'), { tf });
    const r = { model: path.relative(CAMERA, prefix), storage: json.meta.weight_storage, maxAbs: {}, boxes: {}, perCase: [], ms: [] };
    const box = { default: { ref: 0, got: 0, matched: 0, minIoU: 1, maxDScore: 0 }, floor035: { ref: 0, got: 0, matched: 0, minIoU: 1, maxDScore: 0 } };
    for (const c of ref.meta.cases) {
      const inp = ref.arrays['in/' + c];
      const x = tf.tensor(inp.data, inp.shape);
      const t0 = Date.now();
      const out = model.run(x);
      const vals = {};
      for (const o of json.outputs) {
        const d = out[o.name].dataSync();
        vals[o.name] = o.layout === 'nhwc' ? toNCHW(d, o.shape) : Float32Array.from(d);
      }
      r.ms.push(Date.now() - t0);
      tf.dispose(Object.values(out)); x.dispose();
      const pc = { case: c, label: Buffer.from(Array.from(ref.arrays['label/' + c].data)).toString().trim() };
      for (const k of Object.keys(vals)) {
        if (!ref.arrays['out/' + c + '/' + k]) continue;
        const a = vals[k], b = ref.arrays['out/' + c + '/' + k].data;
        let m = 0;
        for (let i = 0; i < a.length; i++) { const d = Math.abs(a[i] - b[i]); if (!(d <= m)) m = d; }
        pc[k] = m; r.maxAbs[k] = Math.max(r.maxAbs[k] || 0, m);
      }
      for (const [key, opts] of [['default', {}], ['floor035', { thresholds: { fire: 0.35, door: 0.35, window: 0.35 } }]]) {
        const R = (k) => ref.arrays['out/' + c + '/' + k] && ref.arrays['out/' + c + '/' + k].data;
        const rb = dec.decode(R('heat'), R('wh'), R('off'), Object.assign({ windowWh: R('wh_w'), windowOff: R('off_w') }, opts));
        const gb = dec.decode(vals.heat, vals.wh, vals.off, Object.assign({ windowWh: vals.wh_w || null, windowOff: vals.off_w || null }, opts));
        const cmp = compareBoxes(rb, gb);
        const B = box[key];
        B.ref += cmp.ref; B.got += cmp.got; B.matched += cmp.matched;
        if (cmp.minIoU != null) B.minIoU = Math.min(B.minIoU, cmp.minIoU);
        B.maxDScore = Math.max(B.maxDScore, cmp.maxDScore);
        if (key === 'default') pc.boxes = cmp.ref + '/' + cmp.got + ' matched ' + cmp.matched;
        for (const d of rb) { const k2 = key + '_' + d.cls; B.byClass = B.byClass || {}; B.byClass[d.cls] = (B.byClass[d.cls] || 0) + 1; }
      }
      r.perCase.push(pc);
    }
    r.boxes = box;
    const s = r.ms.slice(1).sort((a, b) => a - b);
    r.medianMs = s[Math.floor(s.length / 2)];
    r.tensorsBeforeDispose = tf.memory().numTensors;
    model.dispose();
    r.tensorsAfterDispose = tf.memory().numTensors;
    delete r.ms;
    report.models.push(r);
    console.log(r.model, r.storage, 'maxAbs', JSON.stringify(r.maxAbs), 'boxes', JSON.stringify(box), 'median', r.medianMs, 'ms');
  }
  if (jsonOut) fs.writeFileSync(jsonOut, JSON.stringify(report, null, 1));
}
main().catch((e) => { console.error(e); process.exit(1); });
