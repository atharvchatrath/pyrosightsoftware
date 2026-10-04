#!/usr/bin/env node
// Run person/face detectors over the decoded test corpus (TF.js CPU backend)
// and store raw detections at low thresholds for eval_people.js.
//
//   node run_people_corpus.js RGB_PREFIX VARIANT OUT.json
// VARIANT:
//   orig            the unmodified COCO-SSD graph (18 MB float32, with its own
//                   preprocessor) + the coco-ssd npm package's detect()
//   repack:PREFIX   a repack_cocossd.py output (PREFIX.json/.weights.bin) via people.js
//   face:PREFIX     a BlazeFace op list (PREFIX.oplist.json/.weights.bin) via people.js
//   full:PERSON_PREFIX:FACE_PREFIX   both, through PSPeople.detect (display list too)
const path = require('path');
const fs = require('fs');
const tf = require(path.join(__dirname, '../../node_modules/@tensorflow/tfjs'));
require('../oplist.js');
require('../people.js');

const LOW = { personMinScore: 0.2, faceMinScore: 0.5, faceAttachMinScore: 0.5 };

function loadRGB(prefix) {
  const meta = JSON.parse(fs.readFileSync(prefix + '.json', 'utf8'));
  const fd = fs.openSync(prefix + '.bin', 'r');
  return { meta, fd };
}
function readImage(c, im) {
  const n = im.width * im.height * 3;
  const buf = Buffer.alloc(n);
  fs.readSync(c.fd, buf, 0, n, im.offset);
  return tf.tensor3d(new Uint8Array(buf.buffer, buf.byteOffset, n), [im.height, im.width, 3], 'int32');
}

async function loadOrig() {
  const src = path.join(__dirname, '../../assets/cocossd');
  const mj = JSON.parse(fs.readFileSync(path.join(src, 'model.json'), 'utf8'));
  const wm = mj.weightsManifest[0];
  const data = Buffer.concat(wm.paths.map((p) => fs.readFileSync(path.join(src, p))));
  const model = await tf.loadGraphModel(tf.io.fromMemory({
    modelTopology: mj.modelTopology, weightSpecs: wm.weights,
    weightData: data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength),
  }));
  const cocoSsd = require(path.join(__dirname, '../../node_modules/@tensorflow-models/coco-ssd'));
  const od = new cocoSsd.ObjectDetection('lite_mobilenet_v2', 'memory://unused');
  od.model = model;
  return od;
}

async function main() {
  const [rgbPrefix, variant, outPath] = process.argv.slice(2);
  await tf.setBackend('cpu');
  const c = loadRGB(rgbPrefix);
  const res = { variant, backend: tf.getBackend(), images: [] };
  let od = null;
  const parts = variant.split(':');
  const kind = parts[0];
  if (kind === 'orig') {
    od = await loadOrig();
  } else {
    const assets = {};
    if (kind === 'repack' || kind === 'full') {
      const p = parts[1];
      assets.person = { json: JSON.parse(fs.readFileSync(p + '.json', 'utf8')), weights: fs.readFileSync(p + '.weights.bin') };
    }
    if (kind === 'face' || kind === 'full') {
      const p = kind === 'face' ? parts[1] : parts[2];
      assets.face = { json: JSON.parse(fs.readFileSync(p + '.oplist.json', 'utf8')), weights: fs.readFileSync(p + '.weights.bin') };
    }
    const info = await PSPeople.init(tf, assets, LOW);
    res.init = info;
  }
  const times = [];
  let nt0 = null;
  const LIMIT = +process.env.LIMIT || Infinity;
  for (const im of c.meta.images.slice(0, LIMIT)) {
    const x = readImage(c, im);
    const t0 = Date.now();
    let entry;
    if (od) {
      const dets = await od.detect(x, 100, 0.2);
      entry = {
        file: im.file, width: im.width, height: im.height,
        persons: dets.filter((d) => d.class === 'person').map((d) => ({
          score: d.score, x: d.bbox[0] / im.width, y: d.bbox[1] / im.height,
          w: d.bbox[2] / im.width, h: d.bbox[3] / im.height })),
        other: dets.filter((d) => d.class !== 'person').map((d) => ({ cls: d.class, score: d.score })),
      };
    } else {
      const r = await PSPeople.detect(x, Object.assign({ maxPersons: 100 }, LOW));
      entry = {
        file: im.file, width: im.width, height: im.height,
        persons: r.detections.filter((d) => d.cls === 'person').map(strip),
        faces: r.detections.filter((d) => d.cls === 'face').map(strip),
        ms: r.ms,
      };
    }
    times.push(Date.now() - t0);
    x.dispose();
    res.images.push(entry);
    if (nt0 === null) nt0 = tf.memory().numTensors;
    if (res.images.length % 20 === 0) process.stderr.write(variant + ' ' + res.images.length + '\n');
  }
  times.sort((a, b) => a - b);
  res.medianMs = times[Math.floor(times.length / 2)];
  res.numTensorsAfterFirst = nt0;
  res.numTensorsEnd = tf.memory().numTensors;
  fs.writeFileSync(outPath, JSON.stringify(res));
  console.log(variant, 'images', res.images.length, 'median ms', res.medianMs, 'tensors', nt0, '->', res.numTensorsEnd);
}

function strip(d) {
  const o = { score: +d.score.toFixed(5), x: +d.x.toFixed(5), y: +d.y.toFixed(5), w: +d.w.toFixed(5), h: +d.h.toFixed(5) };
  if (d.keypoints) o.kp = d.keypoints.map((p) => [+p[0].toFixed(4), +p[1].toFixed(4)]);
  return o;
}

main().catch((e) => { console.error(e); process.exit(1); });
