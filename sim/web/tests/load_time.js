#!/usr/bin/env node
// Load time of the simulator page with and without the embedded camera page, measured the same way
// on both, alternating runs (A B A B ...), a fresh browser context each run.
//
//   node sim/web/tests/load_time.js --old OLD.html [--new sim/web/dist/pyrosight_sim.html] [--runs 7]
//        [--configs file,file-4x,net-fast,net-slow] [--out sim/web/tests/out/load_time.json]
//
// file      file:// (both files read from disk at once), no throttling
// file-4x   the same with Chromium's 4x CPU throttling (a slower phone; main thread only)
// net-fast  served over http from 127.0.0.1, throttled to 20 Mbit/s, 40 ms latency
// net-slow  the same at 5 Mbit/s, 100 ms latency
// Per run: time to the first simulator frame (the "Loading" overlay hidden, right before the first
// frame is drawn: the simulator is ready), first contentful paint, DOMContentLoaded, load, and the
// longest main-thread task / total time in tasks over 50 ms until load + 2 s.
const fs = require('fs');
const os = require('os');
const path = require('path');
const http = require('http');
const { execSync } = require('child_process');
const pw = require(path.join(execSync('npm root -g').toString().trim(), 'playwright'));

const argv = process.argv.slice(2);
const opt = (k, d) => { const i = argv.indexOf(k); return i >= 0 ? argv[i + 1] : d; };
const PAGES = { old: path.resolve(opt('--old', '')), new: path.resolve(opt('--new', path.join(__dirname, '..', 'dist', 'pyrosight_sim.html'))) };
if (opt('--old', null) === null) { console.error('--old OLD.html is required (the page before the change)'); process.exit(2); }
const RUNS = +opt('--runs', 7);
const CONFIGS = opt('--configs', 'file,file-4x,net-fast,net-slow').split(',');
const OUT = path.resolve(opt('--out', path.join(__dirname, 'out', 'load_time.json')));
const NET = { 'net-fast': { down: 20e6 / 8, up: 5e6 / 8, latency: 40 }, 'net-slow': { down: 5e6 / 8, up: 1e6 / 8, latency: 100 } };

const INIT = `
  window.__lt = { overlayHidden: null, longtasks: [] };
  try { new PerformanceObserver((l) => { for (const e of l.getEntries()) window.__lt.longtasks.push([Math.round(e.startTime), Math.round(e.duration)]); }).observe({ type: 'longtask', buffered: true }); } catch (e) {}
  new MutationObserver((ms) => {
    for (const m of ms) if (m.target.id === 'overlay' && m.target.hidden && window.__lt.overlayHidden == null) window.__lt.overlayHidden = performance.now();
  }).observe(document, { subtree: true, attributes: true, attributeFilter: ['hidden'] });
`;

let server = null, base = null;
async function startServer() {
  server = http.createServer((req, res) => {
    const k = req.url.replace(/^\//, '').replace(/\.html$/, '');
    if (!PAGES[k]) { res.writeHead(404); res.end(); return; }
    const buf = fs.readFileSync(PAGES[k]);
    res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8', 'Content-Length': buf.length, 'Cache-Control': 'no-store' });
    res.end(buf);
  });
  await new Promise((r) => server.listen(0, '127.0.0.1', r));
  base = 'http://127.0.0.1:' + server.address().port + '/';
}

async function once(browser, cfg, which) {
  const ctx = await browser.newContext({ viewport: { width: 1280, height: 800 } });
  await ctx.addInitScript(INIT + 'try { speechSynthesis.speak = () => {}; } catch (e) {}');
  const page = await ctx.newPage();
  const cdp = await ctx.newCDPSession(page);
  if (cfg === 'file-4x') await cdp.send('Emulation.setCPUThrottlingRate', { rate: 4 });
  if (NET[cfg]) {
    await cdp.send('Network.enable');
    await cdp.send('Network.emulateNetworkConditions', { offline: false, latency: NET[cfg].latency, downloadThroughput: NET[cfg].down, uploadThroughput: NET[cfg].up });
  }
  const url = NET[cfg] ? base + which + '.html' : 'file://' + PAGES[which];
  await page.goto(url, { waitUntil: 'load', timeout: 300000 });
  await page.waitForFunction(() => window.__lt.overlayHidden != null, null, { timeout: 300000 });
  await page.waitForTimeout(2000);
  const r = await page.evaluate(() => {
    const n = performance.getEntriesByType('navigation')[0];
    const fcp = performance.getEntriesByName('first-contentful-paint')[0];
    const lt = window.__lt.longtasks.filter((t) => t[0] < n.loadEventEnd + 2000);
    return { firstFrame: Math.round(window.__lt.overlayHidden), fcp: fcp ? Math.round(fcp.startTime) : null, dcl: Math.round(n.domContentLoadedEventEnd),
      load: Math.round(n.loadEventEnd), responseEnd: Math.round(n.responseEnd), longest: lt.reduce((m, t) => Math.max(m, t[1]), 0),
      longSum: lt.reduce((s, t) => s + t[1], 0), longN: lt.length, engine: document.getElementById('engine').textContent };
  });
  await ctx.close();
  return r;
}

const med = (a) => { const b = a.slice().sort((x, y) => x - y); return b.length ? b[b.length >> 1] : null; };

(async () => {
  if (CONFIGS.some((c) => NET[c])) await startServer();
  const browser = await pw.chromium.launch({ args: ['--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] });
  const out = { pages: { old: { file: PAGES.old, bytes: fs.statSync(PAGES.old).size }, new: { file: PAGES.new, bytes: fs.statSync(PAGES.new).size } },
    runs: RUNS, loadavg: os.loadavg(), cpus: os.cpus().length, when: new Date().toISOString(), configs: {} };
  for (const cfg of CONFIGS) {
    const res = { old: [], new: [] };
    await once(browser, cfg, 'old');   // warm-up (disk cache, browser start)
    for (let i = 0; i < RUNS; i++) {
      for (const which of i % 2 ? ['new', 'old'] : ['old', 'new']) res[which].push(await once(browser, cfg, which));
    }
    const sum = {};
    for (const which of ['old', 'new']) {
      sum[which] = {};
      for (const k of ['firstFrame', 'fcp', 'dcl', 'load', 'responseEnd', 'longest', 'longSum']) sum[which][k] = med(res[which].map((r) => r[k]));
      sum[which].firstFrameAll = res[which].map((r) => r.firstFrame);
      sum[which].engine = res[which][0].engine;
    }
    const pct = (k) => sum.old[k] ? Math.round((sum.new[k] / sum.old[k] - 1) * 1000) / 10 : null;
    sum.change_pct = { firstFrame: pct('firstFrame'), fcp: pct('fcp'), dcl: pct('dcl'), load: pct('load') };
    out.configs[cfg] = { summary: sum, raw: res };
    console.log(cfg + ': median first frame old ' + sum.old.firstFrame + ' ms / new ' + sum.new.firstFrame + ' ms (' + sum.change_pct.firstFrame + ' %); FCP ' +
      sum.old.fcp + ' / ' + sum.new.fcp + '; DCL ' + sum.old.dcl + ' / ' + sum.new.dcl + '; load ' + sum.old.load + ' / ' + sum.new.load +
      '; longest task ' + sum.old.longest + ' / ' + sum.new.longest + ' ms; tasks>50ms ' + sum.old.longSum + ' / ' + sum.new.longSum + ' ms');
  }
  out.loadavgEnd = os.loadavg();
  fs.mkdirSync(path.dirname(OUT), { recursive: true });
  fs.writeFileSync(OUT, JSON.stringify(out, null, 1));
  await browser.close();
  if (server) server.close();
})().catch((e) => { console.error(e); process.exit(1); });
