#!/usr/bin/env node
// Distance-label sanity check, step 2: run the page's face detector (PSPeople, TF.js CPU) on the
// close-up photos and compare the BlazeFace box width with (a) the Open Images face box and
// (b) the distance between the eye keypoints. With an average adult eye (pupil) distance of
// about 0.063 m, box_width / eye_distance * 0.063 is the real-world width the box covers,
// which people.js assumes to be faceWidthM = 0.16 m when it turns box width into distance.
//   node tests/face_scale_check.js OUTDIR
const path = require('path'), fs = require('fs');
const CAMERA = path.join(__dirname, '..');
const tf = require(path.join(CAMERA, 'node_modules/@tensorflow/tfjs'));
require(path.join(CAMERA, 'runtime/oplist.js')); require(path.join(CAMERA, 'runtime/people.js'));
require(path.join(CAMERA, 'runtime/dist/people_assets.js'));
(async () => {
  const dir = process.argv[2];
  const meta = JSON.parse(fs.readFileSync(path.join(dir, 'closeups.json')));
  const bin = fs.readFileSync(path.join(dir, 'closeups.bin'));
  await tf.setBackend('cpu');
  await PSPeople.init(tf, globalThis.PS_PEOPLE_ASSETS);
  const ratioGt = [], implied = [];
  for (const m of meta) {
    const n = m.w * m.h * 3;
    const t = tf.tensor3d(new Uint8Array(bin.buffer, bin.byteOffset + m.offset, n), [m.h, m.w, 3], 'int32');
    const r = await PSPeople.detect(t, { person: false });
    t.dispose();
    for (const f of r.detections.filter((d) => d.cls === 'face')) {
      const wpx = f.w * m.w;
      const [re, le] = f.keypoints;
      const eye = Math.hypot((re[0] - le[0]) * m.w, (re[1] - le[1]) * m.h);
      if (eye > 4) implied.push(wpx / eye * 0.063);
      for (const g of m.faces) {
        const ix = Math.max(0, Math.min(g[0] + g[2], f.x + f.w) - Math.max(g[0], f.x)), iy = Math.max(0, Math.min(g[1] + g[3], f.y + f.h) - Math.max(g[1], f.y));
        const iou = ix * iy / (g[2] * g[3] + f.w * f.h - ix * iy);
        if (iou > 0.3) ratioGt.push(f.w / g[2]);
      }
    }
  }
  const med = (a) => { const b = a.slice().sort((x, y) => x - y); return b[b.length >> 1]; };
  const q = (a, p) => { const b = a.slice().sort((x, y) => x - y); return b[Math.floor(p * (b.length - 1))]; };
  const res = { images: meta.length, faces: implied.length, matchedToOpenImagesFace: ratioGt.length,
    boxWidthOverOpenImagesFaceWidth: { median: med(ratioGt), p25: q(ratioGt, 0.25), p75: q(ratioGt, 0.75) },
    impliedBoxWidthM_from_eyes_0063: { median: med(implied), p25: q(implied, 0.25), p75: q(implied, 0.75) },
    assumedFaceWidthM: PSPeople.DEFAULTS.faceWidthM };
  console.log(JSON.stringify(res, null, 1));
  fs.writeFileSync(path.join(__dirname, 'out', 'face_scale_check.json'), JSON.stringify(res, null, 1));
})();
