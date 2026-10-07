/*
 * Direction words and the camera-picture EXIT box, with known bearings.
 *
 *   node camera/nav/tests/geom_test.js
 *
 * - PSNav.directionWord and the guidance word agree with the device's
 *   ps_direction_phrase() for every bearing from -360 to 360 deg in 0.25 deg steps.
 * - exitBox(): side of the picture for unmirrored (back camera) and mirrored
 *   (front camera, looks back at the user) previews, edge arrows outside the
 *   field of view, nothing when unreliable / at the door, and the box size
 *   equals ps_display.c draw_exit() on the device's own geometry.
 */
'use strict';
const path = require('path');
const PSNav = require(path.join(__dirname, '..', 'dist', 'ps_nav.js'));

let ok = true, n = 0;
const check = (label, cond, extra) => { n++; if (!cond) { ok = false; console.log('FAIL ' + label + (extra ? ' ' + extra : '')); } };
const core = new PSNav.Core();

// ---- direction words vs the device
let mism = 0;
for (let b = -360; b <= 360; b += 0.25) {
  const c = core.directionPhrase(b) - PSNav.PHRASE.DIR_AHEAD;
  if (PSNav.DIR_WORDS[c] !== PSNav.directionWord(b)) mism++;
}
check('directionWord == ps_direction_phrase for 2881 bearings', mism === 0, mism + ' mismatches');
const known = [[0, 'ahead'], [22, 'ahead'], [23, 'ahead-left'], [45, 'ahead-left'], [67, 'ahead-left'], [68, 'left'], [90, 'left'],
  [135, 'behind-left'], [170, 'behind'], [180, 'behind'], [-180, 'behind'], [-150, 'behind-right'], [-90, 'right'], [-30, 'ahead-right'], [-22, 'ahead']];
for (const [b, w] of known) check(`bearing ${b} -> ${w}`, PSNav.directionWord(b) === w, 'got ' + PSNav.directionWord(b));
console.log('direction words: ' + known.map(([b, w]) => b + '=' + w).join(', '));

// phrase text through the core: walk 10 m ahead, then turn 90 deg; the door is behind
// where they started walking, so after a LEFT turn it is on their LEFT, after a RIGHT turn on their RIGHT.
for (const [turn, want] of [[1, 'left'], [-1, 'right']]) {
  const c = new PSNav.Core();
  c.onYaw(0, 0); c.markEntry(0);
  for (let t = 10; t < 20000; t += 10) { c.onYaw(0, t); c.onAccel(2, t); if (t % 550 === 0) c.onStep(t); c.tick(t); }
  for (let t = 20000; t < 21000; t += 10) { c.onAccel(0.2, t); c.onYaw(turn * Math.PI / 2 * (t - 20000) / 1000, t); c.tick(t); }
  for (let t = 21000; t < 21500; t += 10) { c.onAccel(0.2, t); c.onYaw(turn * Math.PI / 2, t); c.tick(t); }
  const g = c.guidance();
  const text = 'Way out is to your ' + want + '.';
  check(`walk ahead, turn ${want}: "${text}"`, g.word === want && g.phrase === text, JSON.stringify([g.word, g.phrase, g.routeBearingDeg]));
  console.log(`walk 10 m ahead, turn 90 ${want}: ${g.phrase} (route bearing ${g.routeBearingDeg.toFixed(1)} deg, + = left)`);
}

// ---- exit box
const G = (b, d, extra) => Object.assign({ valid: true, state: 'tracking', levelId: 0, exitIsNext: true, homeBearingDeg: b, routeBearingDeg: b, homeDistM: d, routeDistM: d }, extra || {});
const cx = (r) => r.x + r.w / 2;
const rows = [];
const T = (label, g, o, expect) => {
  const r = PSNav.exitBox(g, o);
  let pass = r.kind === expect.kind;
  if (pass && expect.side) pass = r.side === expect.side;
  if (pass && expect.half) pass = expect.half === 'left' ? cx(r) < 0.5 : cx(r) > 0.5;
  if (pass && expect.behind !== undefined) pass = r.behind === expect.behind;
  check(label, pass, JSON.stringify(r));
  rows.push(`${label.padEnd(58)} -> ${r.kind}${r.kind === 'box' ? ' centre x ' + cx(r).toFixed(3) : r.kind === 'edge' ? ' ' + r.side + ' edge' + (r.behind ? ' (behind)' : '') : ' (' + r.reason + ')'}`);
};
// back camera (unmirrored): sees what the walker faces
T('back cam, door 20 deg LEFT, 5 m', G(20, 5), { mirrored: false }, { kind: 'box', half: 'left' });
T('back cam, door 20 deg RIGHT, 5 m', G(-20, 5), { mirrored: false }, { kind: 'box', half: 'right' });
T('back cam, door 90 deg left (outside 65 deg view)', G(90, 5), { mirrored: false }, { kind: 'edge', side: 'left', behind: false });
T('back cam, door 120 deg right', G(-120, 5), { mirrored: false }, { kind: 'edge', side: 'right', behind: true });
T('back cam, door behind-left 170 deg', G(170, 5), { mirrored: false }, { kind: 'edge', side: 'left', behind: true });
// front camera (mirrored preview, looks back at the walker)
T('front cam, door BEHIND-LEFT 160 deg: in its view, left half', G(160, 5), { mirrored: true }, { kind: 'box', half: 'left' });
T('front cam, door BEHIND-RIGHT -160 deg: right half', G(-160, 5), { mirrored: true }, { kind: 'box', half: 'right' });
T('front cam, door ahead-left 20 deg: not in view, left edge', G(20, 5), { mirrored: true }, { kind: 'edge', side: 'left' });
T('front cam, door ahead-right -20 deg: right edge', G(-20, 5), { mirrored: true }, { kind: 'edge', side: 'right' });
T('front cam, door left 90 deg: left edge', G(90, 5), { mirrored: true }, { kind: 'edge', side: 'left' });
// mirrored preview of a camera that looks forward (e.g. a mirrored eyepiece view)
T('mirrored, looks forward, door 20 deg left -> right half', G(20, 5), { mirrored: true, cameraLooksBack: false }, { kind: 'box', half: 'right' });
// nothing to draw
T('unreliable -> none', G(0, 5, { levelId: 2 }), {}, { kind: 'none' });
T('at the door (0.6 m) -> none', G(0, 0.6), {}, { kind: 'none' });
T('idle -> none', { valid: false, state: 'idle' }, {}, { kind: 'none' });
T('door not next, route leg 70 deg left -> left edge', G(70, 8, { exitIsNext: false }), {}, { kind: 'edge', side: 'left' });
{
  const r = PSNav.exitBox(G(10, 8, { exitIsNext: false, routeBearingDeg: 10 }), {});
  check('door not next, route leg 10 deg left in view -> route marker left of centre', r.kind === 'route' && r.x < 0.5, JSON.stringify(r));
  const m = PSNav.exitBox(G(170, 8, { exitIsNext: false }), { mirrored: true });
  check('front cam, route leg behind-left in view -> route marker left of centre', m.kind === 'route' && m.x < 0.5, JSON.stringify(m));
  rows.push(`route leg 10 deg left, back cam: route marker x ${r.x.toFixed(3)}; 170 deg, front cam: x ${m.x.toFixed(3)}`);
}
// size: device draw_exit() with its own geometry (160 x 120, hfov 57, camera 1.5 m)
{
  const W = 160, H = 120, hf = 57, b = 12, d = 6;
  const a = b * Math.PI / 180, f = (W / 2) / Math.tan(hf / 2 * Math.PI / 180);
  let depth = Math.max(0.5, d * Math.cos(a));
  const dcx = W / 2 - f * Math.tan(a), dw = Math.max(4, f * 0.9 / depth);
  const top = H / 2 - f * (2.0 - 1.5) / depth, bot = H / 2 + f * 1.5 / depth;
  const r = PSNav.exitBox(G(b, d), { hfovDeg: hf, width: W, height: H });
  const err = Math.max(Math.abs(r.px.x - (dcx - dw / 2)), Math.abs(r.px.w - dw), Math.abs(r.px.y - top), Math.abs(r.px.h - (bot - top)));
  check('box geometry equals ps_display.c draw_exit (px)', err < 1e-6, 'err ' + err);
  rows.push(`device geometry 160x120, hfov 57, 12 deg left, 6 m: box x ${r.px.x.toFixed(2)} y ${r.px.y.toFixed(2)} w ${r.px.w.toFixed(2)} h ${r.px.h.toFixed(2)} px (device formula, max diff ${err.toExponential(1)})`);
}
rows.forEach((r) => console.log(r));
console.log(`${n} checks`);
console.log(ok ? 'PASS direction words + EXIT box' : 'FAIL direction words + EXIT box');
process.exit(ok ? 0 : 1);
