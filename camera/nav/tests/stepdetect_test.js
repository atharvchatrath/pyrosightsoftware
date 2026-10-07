/*
 * Step detector on synthetic phone accelerometer traces (accelerationIncludingGravity,
 * ~60 Hz with timing jitter), and the orientation -> heading conversion.
 *
 *   node camera/nav/tests/stepdetect_test.js
 *
 * Walking (hand-held and in a pocket) at 1.4-2.2 steps/s must count within
 * 5% (+-2 steps); standing still, pocket jostle while standing, shaking (4-8
 * Hz) and knocks every 2 s must count at most 2 steps; knocks every 0.6 s (a
 * walking rhythm) at most 4 out of 66.
 */
'use strict';
const path = require('path');
const PSNav = require(path.join(__dirname, '..', 'dist', 'ps_nav.js'));

const G = 9.81;
function makeRand(seed) {
  const r = PSNav.DemoWalker.mulberry32(seed);
  const n = () => { let u = 0; while (u === 0) u = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * r()); };
  return { r, n };
}

/** Run a trace generator through the detector. gen(t, R) -> {x, y, z} (m/s^2). */
function run(gen, seconds, seed, hz) {
  const R = makeRand(seed), det = new PSNav.StepDetector();
  let t = 0, steps = 0;
  const dtMs = 1000 / (hz || 60);
  while (t < seconds * 1000) {
    t += dtMs * (0.8 + 0.4 * R.r());          // event timing jitter
    const a = gen(t / 1000, R);
    steps += det.push(a.x, a.y, a.z, t).length;
  }
  return { steps, det };
}

// phone upright in the hand (gravity along +y), slightly tilted; bounce mostly vertical
function walking(f, amp, opts) {
  opts = opts || {};
  const ph = opts.phase || 0.3, harm = opts.harm === undefined ? 0.35 : opts.harm, noise = opts.noise || 0.25;
  let jit = 0;
  return (t, R) => {
    jit += 0.002 * R.n();                       // slowly varying cadence
    const w = 2 * Math.PI * f * t + jit;
    const vert = amp * Math.sin(w) + harm * amp * Math.sin(2 * w + ph) + noise * R.n();
    const lat = 0.4 * amp * Math.sin(w / 2) + noise * R.n();
    const fwd = 0.3 * amp * Math.sin(w + 1.2) + noise * R.n();
    const tilt = opts.tilt || 0.15;
    return { x: lat, y: (G + vert) * Math.cos(tilt) + fwd * Math.sin(tilt), z: fwd * Math.cos(tilt) - (G + vert) * Math.sin(tilt) };
  };
}
const still = (noise) => (t, R) => ({ x: noise * R.n(), y: G + noise * R.n(), z: noise * R.n() });
function pocketJostle() {
  // standing with the phone in a pocket: slow random sway (AR(1), < 2 Hz) + shifting weight
  let a = 0, b = 0;
  return (t, R) => {
    a = 0.97 * a + 0.12 * R.n(); b = 0.95 * b + 0.1 * R.n();
    return { x: 0.6 * G * Math.sin(0.2 * a) + 0.1 * R.n(), y: G * Math.cos(0.2 * a) + b, z: 0.4 * G + 0.3 * b };
  };
}
const shake = (f, amp) => (t, R) => {
  const s = amp * Math.sin(2 * Math.PI * f * t) + 0.5 * R.n();
  return { x: s, y: G + 0.6 * s + 0.3 * R.n(), z: 0.3 * s };
};
const knocks = (every) => (t, R) => {
  const k = (t % every) < 0.03 ? 18 : 0;   // a 30 ms knock every `every` s
  return { x: 0.1 * R.n(), y: G + k + 0.1 * R.n(), z: 0.1 * R.n() };
};

const cases = [];
const T = 40;
for (const f of [1.4, 1.6, 1.8, 2.0, 2.2]) {
  cases.push({ name: `walk hand ${f.toFixed(1)} Hz`, gen: walking(f, 2.2), expect: f * T, walking: true });
  cases.push({ name: `walk hand slow-soft ${f.toFixed(1)} Hz`, gen: walking(f, 1.3, { noise: 0.2 }), expect: f * T, walking: true });
  cases.push({ name: `walk pocket ${f.toFixed(1)} Hz`, gen: walking(f, 4.5, { harm: 0.6, noise: 0.6, tilt: 1.2 }), expect: f * T, walking: true });
}
cases.push({ name: 'walk 1.8 Hz at 30 Hz sampling', gen: walking(1.8, 2.2), expect: 1.8 * T, walking: true, hz: 30 });
cases.push({ name: 'standing still (quiet)', gen: still(0.05), expect: 0 });
cases.push({ name: 'standing still (hand tremor)', gen: still(0.2), expect: 0 });
cases.push({ name: 'pocket jostle, standing', gen: pocketJostle(), expect: 0 });
for (const f of [4, 5, 6, 8]) cases.push({ name: `shaking ${f} Hz`, gen: shake(f, 8), expect: 0 });
cases.push({ name: 'shaking 5 Hz violent', gen: shake(5, 20), expect: 0 });
// a knock rhythm at walking cadence is the hardest case: allow <= 4 false steps out of 66 knocks
cases.push({ name: 'knocks every 0.6 s', gen: knocks(0.6), expect: 0, allow: 4 });
cases.push({ name: 'knocks every 2 s', gen: knocks(2), expect: 0 });

let ok = true;
for (const c of cases) {
  const seeds = [1, 2, 3];
  const counts = seeds.map((s) => run(c.gen, T, s * 7 + c.name.length, c.hz).steps);
  let pass;
  if (c.walking) pass = counts.every((n) => Math.abs(n - c.expect) <= Math.max(2, 0.05 * c.expect));
  else pass = counts.every((n) => n <= (c.allow || 2));
  ok = ok && pass;
  console.log(`${c.name.padEnd(30)} expected ${String(Math.round(c.expect)).padStart(3)}  counted ${counts.map((n) => String(n).padStart(3)).join(' ')}  ${pass ? 'ok' : 'FAIL'}`);
}

// walking -> standing -> shaking -> walking, one stream
{
  const segs = [[walking(1.8, 2.2), 20], [still(0.1), 10], [shake(6, 10), 8], [walking(1.6, 2.0), 20]];
  const R = makeRand(5), det = new PSNav.StepDetector();
  let t = 0, steps = 0, t0 = 0;
  for (const [gen, dur] of segs) { const end = t0 + dur * 1000; while (t < end) { t += 16.7; const a = gen(t / 1000, R); steps += det.push(a.x, a.y, a.z, t).length; } t0 = end; }
  const expect = 1.8 * 20 + 1.6 * 20, pass = Math.abs(steps - expect) <= 0.06 * expect;
  ok = ok && pass;
  console.log(`${'mixed walk/stand/shake/walk'.padEnd(30)} expected ${String(expect).padStart(3)}  counted ${String(steps).padStart(3)}  ${pass ? 'ok' : 'FAIL'}`);
}

// ---- heading from DeviceOrientation (alpha, beta, gamma) -> ps_quat_to_yaw semantics
const D2R = Math.PI / 180;
const core = new PSNav.Core();
let hok = true;
const check = (label, cond) => { hok = hok && cond; console.log(`${label.padEnd(60)} ${cond ? 'ok' : 'FAIL'}`); };
for (const beta of [0, 45, 80, 90]) {
  const y0 = PSNav.orientationYaw(10, beta, 0, 0).yaw, y1 = PSNav.orientationYaw(40, beta, 0, 0).yaw;
  check(`beta ${beta}: alpha +30 (turn left) -> yaw +30`, Math.abs(PSNav.wrapPi(y1 - y0) / D2R - 30) < 0.5);
}
{
  // landscape, phone upright, rotated 90 deg counter-clockwise in the screen plane
  // (screen.orientation.angle 90): R = Rz(heading) * Rx(90) * Rz(90)
  const qa = (ax, deg) => { const h = deg * D2R / 2, q = [Math.cos(h), 0, 0, 0]; q[ax] = Math.sin(h); return q; };
  const hold = (hdg) => PSNav.quatMul(PSNav.quatMul(qa(3, hdg), qa(1, 90)), qa(3, 90));
  const a = PSNav.yawFromQuat(hold(20), 90), b = PSNav.yawFromQuat(hold(-10), 90);
  check('landscape upright (screen angle 90): turn right 30 -> yaw -30', Math.abs(PSNav.wrapPi(b.yaw - a.yaw) / D2R + 30) < 0.5 && a.level > 0.9);
  const c = PSNav.yawFromQuat(hold(20), 0);
  check('same hold without the screen correction is degenerate (level ~0)', c.level < 0.1);
}
{
  const q = PSNav.eulerToQuat(70, 30, -20), jsYaw = PSNav.quatToYaw(q), cYaw = core.quatToYaw(q[0], q[1], q[2], q[3]);
  check('quatToYaw (JS) equals ps_quat_to_yaw (C)', Math.abs(jsYaw - cYaw) < 1e-6);
}
ok = ok && hok;
console.log(ok ? 'PASS step detector + heading' : 'FAIL step detector + heading');
process.exit(ok ? 0 : 1);
