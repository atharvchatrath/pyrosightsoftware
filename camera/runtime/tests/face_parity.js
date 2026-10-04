#!/usr/bin/env node
// BlazeFace checks (TF.js CPU backend in node):
//  A. op list + people.js decoding on the *identical* 128x128 inputs as the
//     Python TFLite reference (face_ref_tflite.py): decoding/NMS exactness.
//  B. full people.js pipeline (letterbox from the raw RGB image) against the
//     official MediaPipe Tasks FaceDetector (face_ref_mediapipe.py).
//   node face_parity.js FACE_PREFIX RGB_PREFIX [--json out.json]
const path = require('path');
const fs = require('fs');
const tf = require(path.join(__dirname, '../../node_modules/@tensorflow/tfjs'));
require('../oplist.js');
require('../people.js');
const tfile = require('./tensorfile.js');

function iou(a, b) {
  const ix = Math.max(0, Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x));
  const iy = Math.max(0, Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y));
  const i = ix * iy;
  return i / (a.w * a.h + b.w * b.h - i);
}

function clampBox(r) {
  const c = (v) => Math.min(1, Math.max(0, v));
  const x0 = c(r.x), y0 = c(r.y), x1 = c(r.x + r.w), y1 = c(r.y + r.h);
  return Object.assign({}, r, { x: x0, y: y0, w: x1 - x0, h: y1 - y0 });
}

function compare(refs, ours, thr, clamp) {
  if (clamp) refs = refs.map(clampBox);
  // greedy match by IoU
  const st = { ref: 0, ours: 0, matched: 0, iouSum: 0, iouMin: 1, maxCoord: 0, maxScore: 0, maxKp: 0, unmatchedRef: [], unmatchedOurs: [] };
  refs.forEach((r) => { if (r.score >= thr) st.ref++; });
  ours.forEach((o) => { if (o.score >= thr) st.ours++; });
  const used = new Set();
  for (const r of refs) {
    if (r.score < thr) continue;
    let best = -1, bi = 0;
    ours.forEach((o, j) => { if (!used.has(j)) { const v = iou(r, o); if (v > bi) { bi = v; best = j; } } });
    if (best < 0 || bi < 0.5) { st.unmatchedRef.push(r.score); continue; }
    used.add(best);
    const o = ours[best];
    st.matched++;
    st.iouSum += bi;
    st.iouMin = Math.min(st.iouMin, bi);
    st.maxCoord = Math.max(st.maxCoord, Math.abs(o.x - r.x), Math.abs(o.y - r.y), Math.abs(o.w - r.w), Math.abs(o.h - r.h));
    st.maxScore = Math.max(st.maxScore, Math.abs(o.score - r.score));
    if (r.kp && o.kp) r.kp.forEach((p, k) => { st.maxKp = Math.max(st.maxKp, Math.abs(p[0] - o.kp[k][0]), Math.abs(p[1] - o.kp[k][1])); });
  }
  ours.forEach((o, j) => { if (!used.has(j) && o.score >= thr) st.unmatchedOurs.push(o.score); });
  return st;
}

function merge(acc, st) {
  for (const k of ['ref', 'ours', 'matched', 'iouSum']) acc[k] = (acc[k] || 0) + st[k];
  acc.iouMin = Math.min(acc.iouMin === undefined ? 1 : acc.iouMin, st.iouMin);
  for (const k of ['maxCoord', 'maxScore', 'maxKp']) acc[k] = Math.max(acc[k] || 0, st[k]);
  acc.unmatchedRef = (acc.unmatchedRef || []).concat(st.unmatchedRef);
  acc.unmatchedOurs = (acc.unmatchedOurs || []).concat(st.unmatchedOurs);
  return acc;
}

async function main() {
  const [facePrefix, rgbPrefix] = process.argv.slice(2);
  const ji = process.argv.indexOf('--json');
  await tf.setBackend('cpu');
  const face = { json: JSON.parse(fs.readFileSync(facePrefix + '.oplist.json', 'utf8')), weights: fs.readFileSync(facePrefix + '.weights.bin') };
  const O = { faceMinScore: 0.5, faceIoU: 0.3, maxFaces: 100 };
  await PSPeople.init(tf, { face }, O);
  const I = PSPeople._internals;
  const out = { face: path.basename(facePrefix) };

  // ---- A: identical inputs
  const refT = JSON.parse(fs.readFileSync(path.join(__dirname, 'out/face_ref_tflite.json'), 'utf8'));
  const inputs = tfile.load(path.join(__dirname, 'out/face_ref_tflite_inputs'));
  let A = {};
  refT.images.forEach((im, k) => {
    const a = inputs.arrays['in/img' + k];
    const x = tf.tensor(a.data, a.shape);
    const o = PSPeople.faceModel.run(x);
    const dets = I.decodeFaces(o.regressors.dataSync(), o.classificators.dataSync(), Object.assign({}, PSPeople.DEFAULTS, O));
    tf.dispose(Object.values(o)); x.dispose();
    const ours = dets.map((d) => ({ score: d.score, x: d.x0, y: d.y0, w: d.x1 - d.x0, h: d.y1 - d.y0, kp: d.kp }));
    const refs = im.faces.map((f) => ({ score: f.score, x: f.lb[0], y: f.lb[1], w: f.lb[2] - f.lb[0], h: f.lb[3] - f.lb[1], kp: f.lb_kp }));
    A = merge(A, compare(refs, ours, 0.5));
  });
  A.meanIoU = A.iouSum / Math.max(1, A.matched);
  out.identicalInputs_vs_pythonTFLite = A;

  // ---- B: full pipeline vs official MediaPipe
  const refM = JSON.parse(fs.readFileSync(path.join(__dirname, 'out/face_ref_mediapipe.json'), 'utf8'));
  const meta = JSON.parse(fs.readFileSync(rgbPrefix + '.json', 'utf8'));
  const fd = fs.openSync(rgbPrefix + '.bin', 'r');
  const byFile = {};
  for (const im of meta.images) byFile[im.file] = im;
  let B = {}, C = {};
  const times = [];
  for (const im of refM.images) {
    const rel = im.file.split('/testdata/people/')[1];
    const e = byFile[rel];
    if (!e) throw new Error('no rgb for ' + rel);
    const n = e.width * e.height * 3;
    const buf = Buffer.alloc(n);
    fs.readSync(fd, buf, 0, n, e.offset);
    const x = tf.tensor3d(new Uint8Array(buf.buffer, buf.byteOffset, n), [e.height, e.width, 3], 'int32');
    const t0 = Date.now();
    const r = await PSPeople.detect(x, O);
    times.push(Date.now() - t0);
    x.dispose();
    const ours = r.detections.filter((d) => d.cls === 'face').map((d) => ({ score: d.score, x: d.x, y: d.y, w: d.w, h: d.h, kp: d.keypoints }));
    B = merge(B, compare(im.faces, ours, 0.5, true));
    // and vs the Python TFLite pipeline (same letterbox recipe as people.js)
    const pt = refT.images.find((q) => q.file === im.file);
    C = merge(C, compare(pt.faces.map((f) => ({ score: f.score, x: f.x, y: f.y, w: f.w, h: f.h, kp: f.kp })), ours, 0.5, true));
  }
  B.meanIoU = B.iouSum / Math.max(1, B.matched);
  C.meanIoU = C.iouSum / Math.max(1, C.matched);
  out.fullPipeline_vs_mediapipeTasks = B;
  out.fullPipeline_vs_pythonTFLitePipeline = C;
  times.sort((a, b) => a - b);
  out.medianDetectMs = times[Math.floor(times.length / 2)];
  console.log(JSON.stringify(out, null, 1));
  if (ji > 0) fs.writeFileSync(process.argv[ji + 1], JSON.stringify(out, null, 1));
}
main().catch((e) => { console.error(e); process.exit(1); });
