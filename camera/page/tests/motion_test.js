#!/usr/bin/env node
// Node test of page/motion.js (no browser):
//  1. phase correlation recovers known shifts (synthetic shifts of real frames);
//  2. PSR of unrelated frame pairs vs matching pairs (sets psrOk);
//  3. MotionTracker on the rotating-pan clip (tests/out/pan_grey128.bin from
//     make_clips.py): yaw error per frame, and where a way-out mark placed at
//     the centre of frame 0 is drawn later (px error at 640 px width);
//  4. orientationToYawPitch on known poses; direction phrases.
//   node tests/motion_test.js [--json tests/out/motion_test.json]
const fs = require('fs');
const path = require('path');
const M = require('../motion.js');

const OUT = path.join(__dirname, 'out');
const clips = JSON.parse(fs.readFileSync(path.join(OUT, 'clips.json')));
const n = 128;
const raw = new Float32Array(fs.readFileSync(path.join(OUT, 'pan_grey128.bin')).buffer.slice(0));
const frames = [];
for (let i = 0; i < clips.pan.frames; i++) frames.push(raw.subarray(i * n * n, (i + 1) * n * n));
const res = {};
let fails = 0;
function check(name, ok, detail) {
  console.log((ok ? 'PASS ' : 'FAIL ') + name + (detail ? '  ' + detail : ''));
  if (!ok) fails++;
}

// ---------------------------------------------------------------- 1. shifts
function shifted(img, dx, dy) {   // bilinear shift: content moves by (+dx, +dy)
  const o = new Float32Array(n * n);
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
    const sx = Math.min(n - 1.001, Math.max(0, x - dx)), sy = Math.min(n - 1.001, Math.max(0, y - dy));
    const x0 = Math.floor(sx), y0 = Math.floor(sy), fx = sx - x0, fy = sy - y0;
    const a = img[y0 * n + x0], b = img[y0 * n + x0 + 1], c = img[(y0 + 1) * n + x0], d = img[(y0 + 1) * n + x0 + 1];
    o[y * n + x] = a * (1 - fx) * (1 - fy) + b * fx * (1 - fy) + c * (1 - fx) * fy + d * fx * fy;
  }
  return o;
}
const pc = new M.PhaseCorrelator(n);
let rng = 12345;
const rnd = () => {   // mulberry32
  rng = (rng + 0x6D2B79F5) | 0;
  let t = Math.imul(rng ^ (rng >>> 15), 1 | rng);
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
};
const shiftErr = [], goodPsr = [];
for (let k = 0; k < 40; k++) {
  const img = frames[(k * 7) % frames.length];
  const dx = (rnd() - 0.5) * 30, dy = (rnd() - 0.5) * 20;
  const noisy = shifted(img, dx, dy).map((v) => v + (rnd() - 0.5) * 10);
  const r = pc.correlate(pc.prepare(img), pc.prepare(noisy));
  shiftErr.push(Math.hypot(r.dx - dx, r.dy - dy));
  goodPsr.push(r.psr);
}
shiftErr.sort((a, b) => a - b);
res.shift = { n: shiftErr.length, medianErrPx: shiftErr[20], maxErrPx: shiftErr[39], minPsr: Math.min(...goodPsr) };
check('phase correlation: synthetic shifts up to +-15 px', shiftErr[39] < 0.5,
  `median err ${shiftErr[20].toFixed(3)} px, max ${shiftErr[39].toFixed(3)} px (128-px frame), min PSR ${res.shift.minPsr.toFixed(1)}`);

// ---------------------------------------------------------------- 2. PSR
// unrelated pairs: frames from the still start vs the hold at 120 deg (no overlap), plus noise images
const badPsr = [];
const A = frames.slice(0, 20), B = frames.slice(45, 55);
for (const a of A.slice(0, 10)) for (const b of B.slice(0, 5)) badPsr.push(pc.correlate(pc.prepare(a), pc.prepare(b)).psr);
for (let k = 0; k < 20; k++) {
  const z = new Float32Array(n * n).map(() => rnd() * 255);
  badPsr.push(pc.correlate(pc.prepare(frames[k]), pc.prepare(z)).psr);
}
// consecutive frames during the pan (3-4 deg per frame)
const panPsr = [];
for (let i = 21; i < frames.length - 1; i++) panPsr.push(pc.correlate(pc.prepare(frames[i - 1]), pc.prepare(frames[i])).psr);
res.psr = { unrelatedMax: Math.max(...badPsr), unrelatedMedian: badPsr.sort((a, b) => a - b)[badPsr.length >> 1],
  consecutiveMin: Math.min(...panPsr), consecutiveMedian: panPsr.sort((a, b) => a - b)[panPsr.length >> 1],
  psrOk: M.TRACK_DEFAULTS.psrOk };
check('PSR separates unrelated from matching frames', res.psr.unrelatedMax < M.TRACK_DEFAULTS.psrOk && res.psr.consecutiveMin > M.TRACK_DEFAULTS.psrOk,
  `unrelated max ${res.psr.unrelatedMax.toFixed(1)} (median ${res.psr.unrelatedMedian.toFixed(1)}), consecutive min ${res.psr.consecutiveMin.toFixed(1)} (median ${res.psr.consecutiveMedian.toFixed(1)}), threshold ${M.TRACK_DEFAULTS.psrOk}`);

// ---------------------------------------------------------------- 3. tracker on the pan clip
function runTracker(step, label) {
  const W = 640, H = 480, fov = clips.pan.fovDeg;
  const tr = new M.MotionTracker({ fovDeg: fov });
  tr.hasMark = true;
  let mark = null;
  const rows = [];
  let t = 0;
  for (let i = 0; i < frames.length; i += step) {
    t += 1000 / 15 * step;
    const r = tr.update(frames[i], W, H, t);
    if (i === 0) mark = Object.assign(M.pointToDirection(0.5, 0.5, tr.pose, W, H, fov), { w: 14, h: 28 });
    const truth = clips.pan.yaw[i];
    const p = M.project(mark, tr.pose, W, H, fov);
    // where the mark really is: yaw 0 seen from camera yaw `truth`
    const f = M.focalPx(W, H, fov);
    const trueInView = Math.abs(truth) < 80 && Math.abs(W / 2 + f * Math.tan(-truth * Math.PI / 180) - W / 2) <= W / 2;
    const trueX = W / 2 + f * Math.tan(-truth * Math.PI / 180);
    rows.push({ i, truth, yaw: tr.pose.yaw, err: M.wrap180(tr.pose.yaw - truth), state: r.state,
      psr: r.match ? r.match.psr : null, inView: p.inView, trueInView,
      markErrPx: p.inView && trueInView ? p.centre.x * W - trueX : null });
  }
  const errs = rows.map((r) => Math.abs(r.err));
  const markErrs = rows.filter((r) => r.markErrPx !== null).map((r) => Math.abs(r.markErrPx));
  const viewAgree = rows.filter((r) => r.inView === r.trueInView).length;
  const out = { step, frames: rows.length, maxYawErrDeg: Math.max(...errs), finalYawErrDeg: rows[rows.length - 1].err,
    yawAt120: rows.filter((r) => r.truth === 120).map((r) => +r.yaw.toFixed(2)).slice(0, 3),
    maxMarkErrPx: markErrs.length ? Math.max(...markErrs) : null, inViewAgreement: viewAgree + '/' + rows.length,
    states: rows.reduce((a, r) => (a[r.state] = (a[r.state] || 0) + 1, a), {}), rekeys: tr.rekeys };
  res['pan_' + label] = out;
  check(`tracker on rotating pan (${label}): yaw error`, out.maxYawErrDeg < 3,
    `max ${out.maxYawErrDeg.toFixed(2)} deg, back at start ${out.finalYawErrDeg.toFixed(2)} deg, at 120 deg reads ${out.yawAt120.join(', ')}, states ${JSON.stringify(out.states)}, rekeys ${out.rekeys}`);
  check(`tracker on rotating pan (${label}): mark position`, out.maxMarkErrPx !== null && out.maxMarkErrPx < 12,
    `max ${out.maxMarkErrPx && out.maxMarkErrPx.toFixed(1)} px at 640 px; in-view agrees ${out.inViewAgreement}`);
  return rows;
}
runTracker(1, 'every frame');
runTracker(2, 'every 2nd frame (8-12 deg steps)');

// tracker given an unrelated scene for 2 s while a mark exists: must go 'lost', then recover when the view returns
{
  const tr = new M.MotionTracker({ fovDeg: 65 });
  tr.hasMark = true;
  let t = 0;
  for (let i = 0; i < 10; i++) tr.update(frames[i], 640, 480, t += 66);
  const yawBefore = tr.pose.yaw;
  const states = [];
  for (let k = 0; k < 30; k++) states.push(tr.update(new Float32Array(n * n).map(() => rnd() * 255), 640, 480, t += 66).state);
  const back = tr.update(frames[12], 640, 480, t += 66);
  res.lost = { statesDuringNoise: states.join(','), recovered: back.state, yawAfter: tr.pose.yaw, yawBefore };
  check('tracker: unrelated view -> weak then lost, recovers on return', states.includes('lost') && back.state === 'ok' &&
    Math.abs(tr.pose.yaw - yawBefore) < 1, `states ${states[0]}..${states[states.length - 1]}, back: ${back.state}, yaw ${yawBefore.toFixed(2)} -> ${tr.pose.yaw.toFixed(2)}`);
}

// a mark made while the picture does not match the key frame (just after a cut / whip-pan) re-anchors
// the tracker on the current view: it must not go 'lost' while the camera keeps looking there
{
  const tr = new M.MotionTracker({ fovDeg: 65 });
  let t = 0;
  for (let i = 0; i < 10; i++) tr.update(frames[i], 640, 480, t += 66);
  const scene = new Float32Array(n * n).map(() => rnd() * 255);   // a different, textured view
  const before = [];
  for (let k = 0; k < 6; k++) before.push(tr.update(scene, 640, 480, t += 66).state);
  const re = tr.reanchor();
  tr.hasMark = true;
  const after = [];
  for (let k = 0; k < 40; k++) after.push(tr.update(scene, 640, 480, t += 66).state);
  res.reanchor = { before: before.join(','), reanchored: re, after: [...new Set(after)].join(',') };
  check('tracker: marking while the picture does not match re-anchors (no false "lost")', before.includes('weak') && re &&
    after.every((x) => x === 'ok'), `before ${before[before.length - 1]}, after mark ${res.reanchor.after} over ${after.length} frames`);
}

// a large object (a person 46 % of the width, 92 % of the height) walks across a still view, twice:
// its motion must not be taken for a camera turn (before: the pose followed it, up to 254 deg in 45 s)
{
  const tr = new M.MotionTracker({ fovDeg: 65 });
  const bg = frames[0], fg = frames[60];
  const pw = 59, ph = 118;
  let t = 0, maxErr = 0;
  const states = {};
  const feed = (img) => {
    const r = tr.update(img, 640, 480, t += 1000 / 15);
    states[r.state] = (states[r.state] || 0) + 1;
    if (r.state !== 'lost') maxErr = Math.max(maxErr, Math.abs(tr.pose.yaw));
    return r;
  };
  for (let i = 0; i < 10; i++) feed(bg.map((v) => v + (rnd() - 0.5) * 4));
  tr.hasMark = true;
  for (let pass = 0; pass < 2; pass++) {
    for (let x0 = -pw; x0 <= n; x0 += 4) {
      const img = bg.map((v) => v + (rnd() - 0.5) * 4);
      for (let y = n - ph - 2; y < n - 2; y++) for (let x = Math.max(0, x0); x < Math.min(n, x0 + pw); x++) img[y * n + x] = fg[y * n + (x - x0 + 30)];
      feed(img);
    }
    for (let i = 0; i < 30; i++) feed(bg.map((v) => v + (rnd() - 0.5) * 4));
  }
  const end = tr.update(bg, 640, 480, t += 66);
  res.walker = { maxYawErrDeg: maxErr, finalYaw: tr.pose.yaw, finalState: end.state, states, rejected: tr.rejected };
  check('tracker: a person walking close past a still camera does not turn the pose', maxErr < 1 && Math.abs(tr.pose.yaw) < 0.5 && end.state === 'ok',
    `max yaw ${maxErr.toFixed(2)} deg (not lost), final ${tr.pose.yaw.toFixed(2)} deg ${end.state}, states ${JSON.stringify(states)}, matches refused ${tr.rejected}`);
}

// a busy page (CPU detectors) gives frames 1.1-2.7 s apart while the camera turns 18 deg between
// them: the tracker must not report a confident ('ok') wrong pose; back at the start it recovers
{
  const tr = new M.MotionTracker({ fovDeg: 65 });
  let t = 0;
  for (let i = 0; i < 20; i++) tr.update(frames[i], 640, 480, t += 66);
  tr.hasMark = true;
  const rows = [];
  for (const [i, dt] of [[26, 2700], [32, 1100], [38, 2700], [44, 1100], [50, 2700], [44, 1100], [38, 2700], [32, 1100], [26, 2700], [20, 1100], [0, 2700], [1, 66], [2, 66]]) {
    const r = tr.update(frames[i], 640, 480, t += dt);
    rows.push({ i, truth: clips.pan.yaw[i], yaw: +tr.pose.yaw.toFixed(2), state: r.state });
  }
  const okWrong = rows.filter((r) => r.state === 'ok' && Math.abs(M.wrap180(r.yaw - r.truth)) > 3);
  const last = rows[rows.length - 1];
  res.gaps = { rows, okWrong: okWrong.length };
  check('tracker: frames far apart during a fast turn never give a confident wrong pose', okWrong.length === 0 &&
    last.state === 'ok' && Math.abs(last.yaw - last.truth) < 1,
    `${okWrong.length} confident wrong; states ${rows.map((r) => r.state).join(',')}; back at start ${last.yaw} deg (${last.state})`);
}

// sensor fusion: camera sees noise (no vision), sensor turns 30 deg right -> pose follows the sensor
{
  const tr = new M.MotionTracker({ fovDeg: 65 });
  let t = 0;
  // sensor starts moving at step 0 (yaw 10 -> pose 0), then turns 30 deg right and 5 deg up while the camera sees noise
  for (let i = 0; i < 5; i++) { tr.sensor(10, 0, t); tr.update(frames[i], 640, 480, t += 66); }
  for (let k = 0; k <= 30; k++) { tr.sensor(10 + k, k / 6, t); tr.update(new Float32Array(n * n).map(() => rnd() * 255), 640, 480, t += 66); }
  res.sensor = { yaw: tr.pose.yaw, pitch: tr.pose.pitch, source: tr.source(t) };
  check('fusion: vision lost, sensor carries the pose', Math.abs(tr.pose.yaw - 29) < 0.5 && Math.abs(tr.pose.pitch - 29 / 6) < 0.3,
    `yaw ${tr.pose.yaw.toFixed(2)} (expect 29: sensor 40 minus its first moving reading 11), pitch ${tr.pose.pitch.toFixed(2)} (expect 4.83), source ${res.sensor.source}`);
}

// ---------------------------------------------------------------- 4. orientation + phrases
{
  const o = M.orientationToYawPitch;
  const cases = [
    // [alpha, beta, gamma, front, expected yaw, expected pitch, what]
    [0, 90, 0, false, 0, 0, 'portrait upright, back camera, facing alpha origin'],
    [30, 90, 0, false, -30, 0, 'turned 30 deg left (alpha +30)'],
    [-45, 90, 0, false, 45, 0, 'turned 45 deg right'],
    [0, 60, 0, false, 0, -30, 'tilted down 30 deg (beta 60)'],
    [0, 90, 0, true, 180, 0, 'front camera looks the other way'],
    [0, 0, -90, false, 90, 0, 'flat device rolled onto its left edge (gamma -90): back camera points along +x (east)'],
  ];
  const errs = cases.map(([a, b, g, fr, ey, ep, what]) => {
    const r = o(a, b, g, fr);
    const e = Math.max(Math.abs(M.wrap180(r.yaw - ey)), Math.abs(r.pitch - ep));
    if (e > 0.01) console.log('   ', what, JSON.stringify(r));
    return e;
  });
  check('orientationToYawPitch on known poses', Math.max(...errs) < 0.01, `max err ${Math.max(...errs).toExponential(1)} deg over ${cases.length} poses`);
  const ph = [[0, 'ahead.'], [40, 'ahead, to your left.'], [-90, 'to your right.'], [180, 'behind you.'], [-150, 'behind you, to the right.']];
  check('direction phrases (device sectors)', ph.every(([b, s]) => M.directionPhrase(b) === s));
  check('side phrases (person/fire)', M.sidePhrase(0.5, 65) === 'ahead.' && M.sidePhrase(0.1, 65) === 'ahead, to your left.' && M.sidePhrase(0.9, 65) === 'ahead, to your right.');
}

// ---------------------------------------------------------------- speed
{
  const t0 = process.hrtime.bigint();
  const tr = new M.MotionTracker();
  for (let i = 0; i < frames.length; i++) tr.update(frames[i], 640, 480, i * 66);
  const ms = Number(process.hrtime.bigint() - t0) / 1e6 / frames.length;
  res.msPerUpdate = ms;
  console.log(`INFO tracker update: ${ms.toFixed(2)} ms per frame in node (128x128, one or two correlations)`);
}

const jsonOut = process.argv.includes('--json') ? process.argv[process.argv.indexOf('--json') + 1] : null;
if (jsonOut) fs.writeFileSync(jsonOut, JSON.stringify(res, null, 1));
console.log(fails ? `${fails} FAILED` : 'all passed');
process.exit(fails ? 1 : 0);
