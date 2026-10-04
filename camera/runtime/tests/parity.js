#!/usr/bin/env node
// Parity test: op-list runtime (TF.js CPU backend in node) vs a reference
// produced by ref_outputs.py.
//   node parity.js MODEL_PREFIX REF_PREFIX [--outputs-map a=b,...] [--json result.json]
// MODEL_PREFIX.oplist.json + MODEL_PREFIX.weights.bin; prints max abs error
// per output (and relative to the output's range) plus one-run CPU timing.
const path = require('path');
const fs = require('fs');
const tf = require(path.join(__dirname, '../../node_modules/@tensorflow/tfjs'));
require('../oplist.js');
const tfile = require('./tensorfile.js');

async function main() {
  const args = process.argv.slice(2);
  const modelPrefix = args[0];
  const refPrefix = args[1];
  let jsonOut = null;
  const omap = {};
  for (let i = 2; i < args.length; i++) {
    if (args[i] === '--json') jsonOut = args[++i];
    else if (args[i] === '--outputs-map') {
      for (const kv of args[++i].split(',')) { const [a, b] = kv.split('='); omap[a] = b; }
    }
  }
  await tf.setBackend('cpu');
  const json = JSON.parse(fs.readFileSync(modelPrefix + '.oplist.json', 'utf8'));
  const wbuf = fs.readFileSync(modelPrefix + '.weights.bin');
  const t0 = Date.now();
  const model = PSOpList.load(json, wbuf, { tf });
  const tLoad = Date.now() - t0;
  const ref = tfile.load(refPrefix);
  const res = { model: path.basename(modelPrefix), storage: json.meta.weight_storage, cases: {}, maxAbs: {}, maxRel: {} };
  const times = [];
  for (const c of ref.meta.cases) {
    const inp = ref.arrays['in/' + c];
    const x = tf.tensor(inp.data, inp.shape);
    const t1 = Date.now();
    const out = model.run(x);
    const vals = {};
    for (const k in out) vals[k] = out[k].dataSync();
    times.push(Date.now() - t1);
    res.cases[c] = {};
    for (const o of json.outputs) {
      const refName = omap[o.name] || o.name;
      const r = ref.arrays['out/' + c + '/' + refName];
      if (!r) throw new Error('no reference for ' + refName);
      const v = vals[o.name];
      if (v.length !== r.data.length) throw new Error('size mismatch ' + o.name + ' ' + v.length + ' vs ' + r.data.length);
      let m = 0, lo = Infinity, hi = -Infinity;
      for (let i = 0; i < v.length; i++) {
        const d = Math.abs(v[i] - r.data[i]);
        if (!(d <= m)) m = d;   // NaN-safe
        if (r.data[i] < lo) lo = r.data[i];
        if (r.data[i] > hi) hi = r.data[i];
      }
      res.cases[c][o.name] = m;
      res.maxAbs[o.name] = Math.max(res.maxAbs[o.name] || 0, m);
      res.maxRel[o.name] = Math.max(res.maxRel[o.name] || 0, m / Math.max(hi - lo, 1e-12));
    }
    tf.dispose(Object.values(out));
    x.dispose();
  }
  res.loadMs = tLoad;
  res.firstRunMs = times[0];
  res.medianRunMs = times.slice(1).sort((a, b) => a - b)[Math.floor((times.length - 1) / 2)] || times[0];
  res.numTensorsAfter = tf.memory().numTensors;
  res.weightTensors = Object.keys(model.weights).length;
  model.dispose();
  res.numTensorsAfterDispose = tf.memory().numTensors;
  console.log(JSON.stringify(res, null, 1));
  if (jsonOut) fs.writeFileSync(jsonOut, JSON.stringify(res, null, 1));
}
main().catch((e) => { console.error(e); process.exit(1); });
