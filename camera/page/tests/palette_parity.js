#!/usr/bin/env node
/*
 * The eyepiece view's palettes must be the device's palettes, value for value.
 *
 *     node page/tests/palette_parity.js
 *
 * The page colours camera brightness with the same three lookup tables the
 * firmware builds in core/src/ps_display.c (build_luts): white-hot, ironbow
 * and amber night. This test rebuilds them from the C source's own constants
 * and compares against the table the page actually ships, so the view cannot
 * drift into showing a colour ramp the eyepiece never produces.
 *
 * Two details decide every near-miss here, and both are in the C:
 *   - the amber ramp uses integer division (truncating), not rounding;
 *   - the ironbow interpolation runs in float32, so Math.fround is required.
 * Without them 190 of 768 values, and then 6 of 768, came out one level off.
 *
 * Not covered: the panel then packs these to RGB565, so the hardware shows a
 * quantised version of what the page draws.
 */
'use strict';
const fs = require('fs');
const path = require('path');

const ROOT = path.join(__dirname, '..', '..', '..');
const appSrc = fs.readFileSync(path.join(__dirname, '..', 'app.js'), 'utf8');
const dispSrc = fs.readFileSync(path.join(ROOT, 'core', 'src', 'ps_display.c'), 'utf8');

/* ---- the page's table, taken from the shipped source ---- */
const m = appSrc.match(/const EYE_PALETTES = \(function \(\) \{[\s\S]*?\n  \}\)\(\);/);
if (!m) { console.error('EYE_PALETTES not found in page/app.js'); process.exit(1); }
const EYE_PALETTES = new Function(m[0].replace('const EYE_PALETTES =', 'return') + '\n')();

/* ---- the device's table, from the C source's own numbers ---- */
const ironM = dispSrc.match(/static const uint8_t iron\[5\]\[3\] = \{([\s\S]*?)\};/);
if (!ironM) { console.error('iron[] not found in core/src/ps_display.c'); process.exit(1); }
const nums = ironM[1].match(/\d+/g).map(Number);
const iron = [];
for (let i = 0; i < 15; i += 3) iron.push(nums.slice(i, i + 3));

const fr = Math.fround;
const expect = { white: [], iron: [], night: [] };
for (let v = 0; v < 256; v++) {
  expect.white.push([v, v, v]);
  const t = fr(fr(v / 255) * 4), k = Math.min(3, Math.max(0, Math.floor(t))), f = fr(t - k);
  expect.iron.push([0, 1, 2].map((c) =>
    new Uint8Array([fr(iron[k][c] + fr((iron[k + 1][c] - iron[k][c]) * f))])[0]));
  const a = (v * 205 / 255) | 0;
  expect.night.push([a, (a * 140 / 255) | 0, 0]);
}

let bad = 0;
const order = ['white', 'iron', 'night'];
for (let p = 0; p < 3; p++) {
  const lut = EYE_PALETTES[p].lut, want = expect[order[p]];
  for (let v = 0; v < 256; v++) {
    const got = [lut[v * 3], lut[v * 3 + 1], lut[v * 3 + 2]];
    if (got.join() !== want[v].join()) {
      if (bad < 8) console.error(`  ${EYE_PALETTES[p].name} v=${v}: page ${got} device ${want[v]}`);
      bad++;
    }
  }
}
console.log(`palette_parity: 768 values compared, ${bad} differences`);
if (EYE_PALETTES.length !== 3) { console.error('expected 3 palettes'); bad++; }
process.exit(bad ? 1 : 0);
