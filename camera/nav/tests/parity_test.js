/*
 * Parity: the device's navigation C code compiled natively (gcc) vs the same
 * code as JavaScript in dist/ps_nav.js (zig -> wasm -> wasm2js), fed the same
 * sample sequence. Compares every state value, every breadcrumb and every
 * spoken-alert phrase at each checkpoint.
 *
 *   node camera/nav/tests/parity_test.js [--keep DIR]
 *
 * Sequences: three auto-demo walks (demo walker IMU stream, seeds 1-3), and a
 * synthetic stress run: ~600 m staircase walk (the trail reaches the 128-crumb
 * limit and is thinned), a 2 s IMU outage (state LOST, "follow hose"), crawling
 * (accel without steps), "where is out?", a re-marked entry and a reset.
 * Integer state (state, level, steps, crumbs, return target, phrases) must
 * match exactly; floats within 1e-3 (absolute, or relative for large values).
 */
'use strict';
const path = require('path');
const fs = require('fs');
const os = require('os');
const { execFileSync } = require('child_process');
const PSNav = require(path.join(__dirname, '..', 'dist', 'ps_nav.js'));

const NAV = path.join(__dirname, '..');
const ROOT = path.join(NAV, '..', '..');
const keepIdx = process.argv.indexOf('--keep');
const work = keepIdx > 0 ? process.argv[keepIdx + 1] : fs.mkdtempSync(path.join(os.tmpdir(), 'psnav-parity-'));
fs.mkdirSync(work, { recursive: true });

// ---- record command streams
function recordingCore(cmds) {
  const c = new PSNav.Core();
  const wrap = (name, fmt) => { const f = c[name].bind(c); c[name] = function () { cmds.push(fmt.apply(null, arguments)); return f.apply(null, arguments); }; };
  wrap('onYaw', (y, t) => 'Y ' + y + ' ' + t);
  wrap('onAccel', (m, t) => 'A ' + m + ' ' + t);
  wrap('onStep', (t) => 'S ' + t);
  wrap('tick', (t) => 'K ' + t);
  wrap('markEntry', (t) => 'M ' + t);
  wrap('whereOut', (t) => 'W ' + t);
  wrap('reset', () => 'R');
  return c;
}

function demoSequence(seed) {
  const cmds = [];
  let now = 0;
  const nav = new PSNav.Navigator({ now: () => now, demoSpeed: 1 });
  nav.core = recordingCore(cmds);
  nav.useDemo({ seed });
  nav.autoDemo(seed);
  let lastP = 0;
  while (nav.walker.phase !== 'done' && now < 400000) {
    now += 50;
    nav.update();
    if (nav.coreT - lastP >= 500) { cmds.push('P'); lastP = nav.coreT; }
  }
  cmds.push('P');
  return cmds;
}

function stressSequence() {
  const cmds = [];
  const rnd = PSNav.DemoWalker.mulberry32(99);
  let t = 0, yaw = 0.7, raw = 0.7;
  const tick = () => { cmds.push('K ' + t); };
  cmds.push('Y ' + raw + ' ' + t); cmds.push('M ' + t);
  // staircase walk (alternating ~90 deg left/right turns, never looping back) so the
  // trail outgrows PS_NAV_MAX_CRUMBS (128) and gets thinned
  let nextTurn = 3000, target = yaw, stepPh = 0, leftNext = true;
  for (let i = 0; i < 60000; i++) {           // 600 s at 100 Hz
    t += 10;
    if (t > nextTurn) { target += (leftNext ? 1 : -1) * (1.3 + 0.5 * rnd()); leftNext = !leftNext; nextTurn = t + 2000 + rnd() * 4000; }
    yaw += Math.max(-0.015, Math.min(0.015, target - yaw));
    raw = PSNav.wrapPi(yaw + 0.001 * (rnd() - 0.5));
    const outage = t > 200000 && t < 202000;
    const crawl = t > 300000 && t < 320000;
    if (!outage) {
      cmds.push('Y ' + raw + ' ' + t);
      cmds.push('A ' + (crawl ? 1.9 : 2.1 + 0.3 * rnd()) + ' ' + t);
      if (!crawl) { stepPh += 0.017; if (stepPh >= 1) { stepPh -= 1; cmds.push('S ' + t); } }
    }
    tick();
    if (t === 250000 || t === 450000) cmds.push('W ' + t);
    if (t === 500000) cmds.push('M ' + t);
    if (t % 1000 === 0) cmds.push('P');
  }
  cmds.push('R'); cmds.push('P');
  t += 10; cmds.push('Y 0.1 ' + t); cmds.push('M ' + t); t += 10; tick(); cmds.push('P');
  return cmds;
}

// ---- replay through a fresh JS core
function replayJS(cmds) {
  const c = new PSNav.Core(), out = [];
  for (const line of cmds) {
    const p = line.split(' ');
    switch (p[0]) {
      case 'Y': c.onYaw(+p[1], +p[2]); break;
      case 'A': c.onAccel(+p[1], +p[2]); break;
      case 'S': c.onStep(+p[1]); break;
      case 'K': c.tick(+p[1]); break;
      case 'M': c.markEntry(+p[1]); break;
      case 'W': c.whereOut(+p[1]); break;
      case 'R': c.reset(); break;
      case 'P': {
        const st = c.raw(), cr = c.crumbs(), al = c.popAlerts();
        out.push({ st, cr: cr.flatMap((q) => [q.x, q.y, q.headingSigma]), al: al.map((a) => a.parts.join(',')) });
        break;
      }
    }
  }
  return out;
}

function parseNative(text) {
  return text.trim().split('\n').map((l) => {
    const [a, b, c] = l.slice(2).split('|');
    return {
      st: a.trim().split(/\s+/).map(Number),
      cr: b.trim() ? b.trim().split(/\s+/).map(Number) : [],
      al: c.trim() ? c.trim().split(';').map((s) => s.trim()).filter(Boolean).map((s) => s.split(/\s+/).join(',')) : [],
    };
  });
}

// ---- native build
const exe = path.join(work, 'parity_native');
execFileSync('gcc', ['-O2', '-std=c99', '-ffp-contract=off', '-I' + path.join(ROOT, 'core', 'include'),
  ...['ps_nav.c', 'ps_config.c', 'ps_alerts.c'].map((f) => path.join(ROOT, 'core', 'src', f)),
  path.join(NAV, 'src', 'nav_api.c'), path.join(NAV, 'tests', 'parity_native.c'), '-lm', '-o', exe]);

const INT_IDX = new Set([0, 1, 8, 9, 14, 15, 16, 18, 19, 20, 24]);
let allOk = true;
const report = [];
const seqs = [['demo seed 1', demoSequence(1)], ['demo seed 2', demoSequence(2)], ['demo seed 3', demoSequence(3)], ['stress', stressSequence()]];
for (const [name, cmds] of seqs) {
  const file = path.join(work, name.replace(/\s+/g, '_') + '.txt');
  fs.writeFileSync(file, cmds.join('\n') + '\n');
  const nat = parseNative(execFileSync(exe, [file], { maxBuffer: 1 << 28 }).toString());
  const js = replayJS(cmds);
  let maxDiff = 0, maxPos = 0, intMismatch = 0, crumbMismatch = 0, alertMismatch = 0, alerts = 0;
  const states = new Set(), levels = new Set();
  let maxCrumbs = 0;
  if (nat.length !== js.length) { allOk = false; report.push(name + ': checkpoint count differs'); continue; }
  for (let k = 0; k < js.length; k++) {
    const a = js[k], b = nat[k];
    states.add(b.st[0]); levels.add(b.st[9]); maxCrumbs = Math.max(maxCrumbs, b.st[15]);
    for (let i = 0; i < 26; i++) {
      const d = Math.abs(a.st[i] - b.st[i]);
      if (INT_IDX.has(i)) { if (d > 0) intMismatch++; continue; }
      const rel = d / Math.max(1, Math.abs(b.st[i]));
      maxDiff = Math.max(maxDiff, rel);
      if (i === 10 || i === 11) maxPos = Math.max(maxPos, d);
    }
    if (a.cr.length !== b.cr.length) crumbMismatch++;
    else for (let i = 0; i < a.cr.length; i++) maxDiff = Math.max(maxDiff, Math.abs(a.cr[i] - b.cr[i]) / Math.max(1, Math.abs(b.cr[i])));
    alerts += b.al.length;
    if (a.al.join(';') !== b.al.join(';')) alertMismatch++;
  }
  const ok = maxDiff <= 1e-3 && intMismatch === 0 && crumbMismatch === 0 && alertMismatch === 0;
  allOk = allOk && ok;
  report.push(`${name.padEnd(12)} ${cmds.length} commands, ${js.length} checkpoints, ${alerts} alerts, states seen ${[...states].join('/')}, levels ${[...levels].join('/')}, max crumbs ${maxCrumbs}: ` +
    `max float diff ${maxDiff.toExponential(2)} (position ${maxPos.toExponential(2)} m), integer mismatches ${intMismatch}, crumb-count mismatches ${crumbMismatch}, alert mismatches ${alertMismatch} -> ${ok ? 'OK' : 'MISMATCH'}`);
}
report.forEach((l) => console.log(l));
console.log(allOk ? 'PASS parity C vs JS' : 'FAIL parity C vs JS');
if (keepIdx < 0) fs.rmSync(work, { recursive: true, force: true });
process.exit(allOk ? 0 : 1);
