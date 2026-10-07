/*
 * Closed-loop demo walk: the auto demo (scripted walk in, look around, then
 * the walker follows ONLY the device's arrow back out; the hose line if the
 * device says its estimate is unreliable) over many seeds.
 *
 *   node camera/nav/tests/walker_test.js [nSeeds=30] [--json out.json]
 *
 * Pass: >= 80% of runs end within 1.5 m of the door (the point where the
 * entry was marked), every run within 3.0 m, every run ends (no timeout), and
 * the estimate visibly differs from the truth (> 1 m) in most runs, so the
 * IMU error model is not a no-op. The numbers are reported, not tuned to pass.
 *
 * The 3.0 m bound was 2.5 m until core commit a6f9d7b (nav: make the arrow
 * usable on long, crawled and dropout routes), which deliberately trades some
 * "follow the hose" endings for arrow endings 1.5-3 m from the door.
 */
'use strict';
const path = require('path');
const fs = require('fs');
const PSNav = require(path.join(__dirname, '..', 'dist', 'ps_nav.js'));

const args = process.argv.slice(2);
const N = parseInt(args.find((a) => /^\d+$/.test(a)) || '30', 10);
const jsonOut = args.includes('--json') ? args[args.indexOf('--json') + 1] : null;

function runSeed(seed) {
  let now = 0;
  const nav = new PSNav.Navigator({ now: () => now, demoSpeed: 1 });
  nav.useDemo({ seed });
  nav.autoDemo(seed);
  const levels = new Set();
  let followHose = 0, alerts = [];
  nav.on('alert', (a) => { alerts.push(a.text); });
  let maxEst = 0, inboundM = 0, minConf = 1, tIn = 0, sawOut = false;
  while (nav.walker.phase !== 'done' && now < 600000) {
    now += 100;
    const g = nav.update();
    levels.add(g.level);
    if (nav.walker.phase === 'outbound' && !sawOut) { sawOut = true; inboundM = g.distWalkedM; tIn = nav.coreT / 1000; }
  }
  const r = nav.walker.result || { how: 'none', doorErrorM: nav.walker.doorDistance() };
  return {
    seed, how: r.how, doorErrorM: r.doorErrorM, hose: !!r.hose,
    maxEstErrM: nav.stats.maxEstErrM, minConfidence: nav.stats.minConfidence,
    unreliableEntries: nav.stats.unreliableEntries, followHoseAlerts: nav.stats.followHoseAlerts,
    inboundM, tInS: tIn, totalS: nav.coreT / 1000, strideBias: nav.walker.err.strideBias, driftDps: nav.walker.err.driftDps,
    degradedSeen: levels.has('DEGRADED'), alerts,
  };
}

const res = [];
const t0 = Date.now();
for (let s = 1; s <= N; s++) res.push(runSeed(s));
const ms = Date.now() - t0;
const ok = res.filter((r) => r.doorErrorM <= 1.5);
const mean = (a) => a.reduce((x, y) => x + y, 0) / Math.max(1, a.length);
const pct = (a, p) => { const b = a.slice().sort((x, y) => x - y); return b[Math.min(b.length - 1, Math.floor(p * (b.length - 1) + 0.5))]; };
const errs = res.map((r) => r.doorErrorM), est = res.map((r) => r.maxEstErrM);
for (const r of res) {
  console.log(`seed ${String(r.seed).padStart(2)}  end ${r.how.padEnd(7)} door err ${r.doorErrorM.toFixed(2)} m  max est err ${r.maxEstErrM.toFixed(2)} m  ` +
    `min conf ${r.minConfidence.toFixed(2)}  unreliable ${r.unreliableEntries}  hose alerts ${r.followHoseAlerts}  ` +
    `in ${r.inboundM.toFixed(1)} m/${r.tInS.toFixed(0)} s  total ${r.totalS.toFixed(0)} s  bias ${(r.strideBias * 100).toFixed(1)}% drift ${r.driftDps.toFixed(3)} deg/s`);
}
const summary = {
  runs: N, within1_5m: ok.length, successRate: ok.length / N,
  doorErrorM: { mean: mean(errs), median: pct(errs, 0.5), p90: pct(errs, 0.9), max: Math.max(...errs) },
  maxEstErrM: { mean: mean(est), median: pct(est, 0.5), max: Math.max(...est) },
  runsWithEstErrOver1m: est.filter((e) => e > 1).length,
  runsUnreliable: res.filter((r) => r.unreliableEntries > 0).length,
  runsFollowHoseAlert: res.filter((r) => r.followHoseAlerts > 0).length,
  runsDegraded: res.filter((r) => r.degradedSeen).length,
  runsEndedByHose: res.filter((r) => r.how === 'hose').length,
  minConfidence: { mean: mean(res.map((r) => r.minConfidence)), min: Math.min(...res.map((r) => r.minConfidence)) },
  inboundM: mean(res.map((r) => r.inboundM)),
  wallMs: ms,
};
console.log(JSON.stringify(summary, null, 1));
if (jsonOut) fs.writeFileSync(jsonOut, JSON.stringify({ summary, runs: res }, null, 1));
const pass = summary.successRate >= 0.8 && summary.doorErrorM.max <= 3.0 &&
  res.every((r) => r.how !== 'none' && r.how !== 'timeout') && summary.runsWithEstErrOver1m >= N / 2;
console.log(pass ? 'PASS walker closed loop' : 'FAIL walker closed loop');
process.exit(pass ? 0 : 1);
