// node tools/parity_decode.js runs/parity  -> compares decode.js against decode.py results
const fs = require('fs');
const path = require('path');
const D = require(path.join(__dirname, '..', 'decode.js'));
const dir = process.argv[2];
const cases = JSON.parse(fs.readFileSync(path.join(dir, 'py.json')));
const N = 2 * 32 * 40;
let nBoxes = 0, maxDiff = 0, mismatches = 0, nhwcMismatch = 0;
for (const c of cases) {
  const buf = fs.readFileSync(path.join(dir, c.k + '.bin'));
  const all = new Float32Array(buf.buffer, buf.byteOffset, buf.length / 4);
  const heat = all.subarray(0, N), wh = all.subarray(N, 2 * N), off = all.subarray(2 * N, 3 * N);
  const js = D.decode(heat, wh, off, { minScore: c.minScore });
  if (js.length !== c.py.length) { mismatches++; console.log('count mismatch', c.k, js.length, c.py.length); continue; }
  for (let i = 0; i < js.length; i++) {
    const a = js[i], b = c.py[i];
    if (a.cls !== b.cls) mismatches++;
    for (const k of ['score', 'x', 'y', 'w', 'h']) maxDiff = Math.max(maxDiff, Math.abs(a[k] - b[k]));
    nBoxes++;
  }
  // NHWC: transpose and decode again, must be identical
  const toNHWC = (a) => { const o = new Float32Array(N); for (let ch = 0; ch < 2; ch++) for (let p = 0; p < 1280; p++) o[p * 2 + ch] = a[ch * 1280 + p]; return o; };
  const js2 = D.decode(toNHWC(heat), toNHWC(wh), toNHWC(off), { minScore: c.minScore, layout: 'NHWC' });
  if (JSON.stringify(js2) !== JSON.stringify(js)) nhwcMismatch++;
}
console.log(JSON.stringify({ cases: cases.length, boxes: nBoxes, maxAbsDiff: maxDiff, mismatches, nhwcMismatch }));
