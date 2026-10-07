#!/usr/bin/env node
// Responsiveness of the PyroSight Camera page: are taps handled at once, while the detectors load
// and while they run? (Extends the lead's camcheck/freeze.cjs.)
//
//   taskset -c 2,3 node tests/responsiveness_test.js [--page dist/pyrosight_camera.html]
//       [--configs webgl,cpu,webgl-4x,...] [--clip DIR/e2e_still_face.y4m] [--out tests/out/responsiveness.json]
//       [--det-clicks 8] [--det-secs 14] [--label NAME] [--tmp DIR for the CSP host pages] [--nav]
//
// For every configuration: open the page (file://, or the fragment inside a sandboxed iframe with a
// strict Content-Security-Policy), then click at fixed times after navigation:
//   0.5 s  "Eyepiece view"   (a pure UI toggle: its text changes at once when the click is handled)
//   1.0 s  "Start camera"    (the real flow: the button turns into "Starting camera…")
//   2.0 s, 5.0 s  "Eyepiece view"
//   then, once boxes are coming in, --det-clicks more "Eyepiece view" clicks spread over detection.
// Clicks are raw mouse events at precomputed coordinates (page.mouse.click), so nothing waits for the
// page before the click is sent. In the page an init script records, per click:
//   handlerDelay  event.timeStamp -> the click reaching the page's listeners (main thread queueing)
//   visibleMs     event.timeStamp -> the first animation frame after the handlers ran (+ one task),
//                 i.e. when the changed button text can be on screen ("click to visible response")
//   eventTiming   the browser's own Event Timing duration for that click (>= 16 ms entries only)
// and over the whole run: long tasks (> 50 ms) on the main thread, split into load (first paint ->
// first boxes) and detection; long animation frames; a 50 ms timer probe; detection update times.
//
// Note: Chromium's CPU throttling (Emulation.setCPUThrottlingRate) slows only the page's main thread,
// not a dedicated worker (measured in this repo: busy loop main 62 -> 254 ms at 4x, worker 70 -> 60 ms),
// so throttled runs show main-thread cost; the "-1core" configs pin the whole browser to one CPU core
// so that a worker competes with the main thread for it.
const fs = require('fs');
const path = require('path');
const { execSync, spawnSync } = require('child_process');
const pw = require(path.join(execSync('npm root -g').toString().trim(), 'playwright'));

const CAMERA = path.join(__dirname, '..');
const OUT = path.join(__dirname, 'out');
fs.mkdirSync(OUT, { recursive: true });
const argv = process.argv.slice(2);
const opt = (k, d) => { const i = argv.indexOf(k); return i >= 0 ? argv[i + 1] : d; };
const PAGE = path.resolve(opt('--page', path.join(CAMERA, 'dist', 'pyrosight_camera.html')));
const FRAGMENT = path.resolve(opt('--fragment', PAGE.replace(/\.html$/, '.fragment.html')));
const SCRATCH = '/tmp/claude-0/-home-claude/069cc723-8434-5b03-8db8-df5ff0accd4e/scratchpad';
const CLIP = opt('--clip', path.join(SCRATCH, 'clips', 'e2e_still_face.y4m'));
const VIDEO = opt('--video', path.join(CAMERA, 'page', 'tests', 'out', 'mixed_640x480.webm'));
const RESULTS = path.resolve(opt('--out', path.join(OUT, 'responsiveness.json')));
const LABEL = opt('--label', path.basename(path.dirname(PAGE)) + '/' + path.basename(PAGE));
const DET_CLICKS = +opt('--det-clicks', 8);
const DET_SECS = +opt('--det-secs', 14);
// --nav: once boxes come in, start the navigation auto demo (8x, restarted each time it ends) and force
// its map and arrow to draw even when scrolled off screen (worst case), for the whole detection phase
const NAV = argv.includes('--nav');
// the CSP host pages (each holds a copy of the 8 MB fragment) and the taskset wrapper go here, not into tests/out
const TMP = path.resolve(opt('--tmp', fs.existsSync(SCRATCH) ? path.join(SCRATCH, 'resp_hosts') : path.join(require('os').tmpdir(), 'ps-resp')));
fs.mkdirSync(TMP, { recursive: true });

const STRICT_CSP = "default-src 'none'; script-src 'unsafe-inline' blob:; worker-src blob:; img-src data: blob:; media-src data: blob:; style-src 'unsafe-inline'";
const NOWORKER_CSP = STRICT_CSP.replace("worker-src blob:", "worker-src 'none'");

// gpu: WebGL on SwiftShader allowed (true) or all WebGL removed (false); rate: main-thread CPU throttling;
// engine: 'main' forces the main-thread fallback (?engine=main); host: 'file' | 'csp' | 'csp-noworker';
// cores: run the browser under taskset -c <cores> (else inherits this process's affinity)
const CONFIGS = {
  'webgl': { gpu: true, rate: 1 },
  'webgl-4x': { gpu: true, rate: 4 },
  'webgl-6x': { gpu: true, rate: 6 },
  'cpu': { gpu: false, rate: 1 },
  'cpu-4x': { gpu: false, rate: 4 },
  'cpu-6x': { gpu: false, rate: 6 },
  'webgl-main': { gpu: true, rate: 1, engine: 'main' },
  'webgl-4x-main': { gpu: true, rate: 4, engine: 'main' },
  'webgl-6x-main': { gpu: true, rate: 6, engine: 'main' },
  'cpu-main': { gpu: false, rate: 1, engine: 'main' },
  'cpu-4x-main': { gpu: false, rate: 4, engine: 'main' },
  'csp-iframe': { gpu: true, rate: 1, host: 'csp' },
  'csp-iframe-4x': { gpu: true, rate: 4, host: 'csp' },
  'csp-iframe-noworker': { gpu: true, rate: 1, host: 'csp-noworker' },
  'csp-iframe-noworker-4x': { gpu: true, rate: 4, host: 'csp-noworker' },
  'webgl-4x-1core': { gpu: true, rate: 4, cores: '2' },
  'cpu-1core': { gpu: false, rate: 1, cores: '2' },
};
const NAMES = (opt('--configs', 'webgl,cpu,webgl-4x,webgl-6x,webgl-main,cpu-main,webgl-4x-main,csp-iframe,csp-iframe-noworker')).split(',');

const INIT = `
(() => {
  if (window.__resp) return;
  const R = window.__resp = { lt: [], loaf: [], ev: [], clicks: [], fcp: null, probe: [], firstDet: null, origin: performance.timeOrigin };
  const obs = (type, fn, extra) => { try { new PerformanceObserver((l) => l.getEntries().forEach(fn)).observe(Object.assign({ type, buffered: true }, extra || {})); } catch (e) { /* unsupported */ } };
  obs('longtask', (e) => R.lt.push([Math.round(e.startTime), Math.round(e.duration)]));
  obs('long-animation-frame', (e) => R.loaf.push([Math.round(e.startTime), Math.round(e.duration), Math.round(e.blockingDuration || 0), (e.scripts || []).filter((x) => x.duration > 30).map((x) => (x.invoker || '') + ' ' + (x.sourceFunctionName || '') + ' ' + Math.round(x.duration)).join('; ')]));
  obs('event', (e) => { if (e.name === 'click' || e.name === 'pointerdown' || e.name === 'pointerup' || e.name === 'mousedown' || e.name === 'mouseup')
    R.ev.push({ name: e.name, start: +e.startTime.toFixed(1), procStart: +e.processingStart.toFixed(1), procEnd: +e.processingEnd.toFixed(1), dur: e.duration, id: e.target && e.target.id || null }); }, { durationThreshold: 16 });
  obs('paint', (e) => { if (e.name === 'first-contentful-paint') R.fcp = e.startTime; });
  let last = performance.now();
  setInterval(() => { const n = performance.now(); const late = n - last - 50; if (late > 30) R.probe.push([Math.round(n), Math.round(late)]); last = n; }, 50);
  // pointer-up of every click (fired even on a disabled button), to pair the test's clicks with the page's records
  R.ups = [];
  addEventListener('pointerup', (e) => { R.ups.push({ ts: +e.timeStamp.toFixed(1), at: +performance.now().toFixed(1), id: e.target && (e.target.id || e.target.tagName) || null }); }, true);
  // a fixed probe button (top right), added to the PyroSight document only: the same click -> text
  // change -> paint path as the page's own buttons, at a place no layout change can move
  const addProbe = () => {
    if (document.getElementById('__resp_probe') || !document.getElementById('ps-root') || !document.body) return false;
    const b = document.createElement('button');
    b.id = '__resp_probe'; b.textContent = 'P0';
    b.style.cssText = 'position:fixed;top:0;right:0;width:56px;height:44px;z-index:2147483647;opacity:.7;font:12px monospace;min-height:0;padding:0';
    b.addEventListener('click', () => { b.textContent = b.textContent === 'P0' ? 'P1' : 'P0'; });
    document.body.appendChild(b);
    return true;
  };
  if (!addProbe()) { const mo = new MutationObserver(() => { if (addProbe()) mo.disconnect(); }); mo.observe(document, { childList: true, subtree: true }); }
  addEventListener('click', (e) => {
    const b = e.target && e.target.closest ? e.target.closest('button') : null;
    const eng = document.getElementById('engine');
    const rec = { id: b ? b.id : (e.target && e.target.id) || null, ts: +e.timeStamp.toFixed(1), at: +performance.now().toFixed(1), text0: b ? b.textContent : null,
      engine: eng ? eng.textContent : null, appReady: !!window.PSCamera };
    R.clicks.push(rec);
    requestAnimationFrame(() => {
      rec.raf = +performance.now().toFixed(1);
      rec.text1 = b ? b.textContent : null;
      const ch = new MessageChannel();
      ch.port1.onmessage = () => { rec.after = +performance.now().toFixed(1); };
      ch.port2.postMessage(0);
    });
  }, true);
})();
`;

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const median = (a) => { const b = a.filter((v) => v != null).sort((x, y) => x - y); return b.length ? b[b.length >> 1] : null; };
const pct = (a, q) => { const b = a.filter((v) => v != null).sort((x, y) => x - y); return b.length ? b[Math.min(b.length - 1, Math.floor(q * b.length))] : null; };

function hostFor(cfg) {
  if (!cfg.host || cfg.host === 'file') return 'file://' + PAGE + (cfg.engine ? '?engine=' + cfg.engine : '');
  const frag = fs.readFileSync(FRAGMENT, 'utf8');
  const csp = cfg.host === 'csp-noworker' ? NOWORKER_CSP : STRICT_CSP;
  const tag = cfg.host + (cfg.engine ? '_' + cfg.engine : '');
  const inner = path.join(TMP, 'resp_inner_' + tag + '.html');
  const host = path.join(TMP, 'resp_host_' + tag + '.html');
  const engineInit = cfg.engine ? '<script>window.PS_ENGINE = ' + JSON.stringify(cfg.engine) + ';</script>' : '';
  fs.writeFileSync(inner, '<!doctype html><html><head><meta charset=utf8><meta name=viewport content="width=device-width,initial-scale=1">' +
    '<meta http-equiv="Content-Security-Policy" content="' + csp + '"><style>html,body{margin:0}</style>' + engineInit + '</head><body>' + frag + '</body></html>');
  fs.writeFileSync(host, '<!doctype html><meta name="viewport" content="width=device-width, initial-scale=1"><style>html,body{margin:0;overflow:hidden}</style>' +
    '<iframe sandbox="allow-scripts" allow="camera" style="display:block;width:100vw;height:100vh;border:0" src="' + inner + '"></iframe>');
  return 'file://' + host;
}

async function launch(cfg) {
  const args = ['--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream', '--use-file-for-fake-video-capture=' + CLIP];
  if (cfg.gpu) args.push('--enable-unsafe-swiftshader', '--ignore-gpu-blocklist');
  else args.push('--disable-gpu', '--disable-software-rasterizer', '--disable-webgl', '--disable-webgl2');
  const o = { args };
  if (cfg.cores) {
    // wrap the browser binary with taskset so every Chromium process shares the given core(s)
    const exe = pw.chromium.executablePath();
    const wrap = path.join(TMP, 'chromium_taskset_' + cfg.cores.replace(/\W/g, '_') + '.sh');
    fs.writeFileSync(wrap, '#!/bin/sh\nexec taskset -c ' + cfg.cores + ' "' + exe + '" "$@"\n');
    fs.chmodSync(wrap, 0o755);
    o.executablePath = wrap;
  }
  return pw.chromium.launch(o);
}

const ctxOpts = { viewport: { width: 390, height: 844 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true,
  userAgent: 'Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0 Mobile Safari/537.36' };
const SPEECH_STUB = `try { if (window.speechSynthesis) { speechSynthesis.speak = () => {}; speechSynthesis.cancel = () => {}; } } catch (e) {}`;

// button centres in page coordinates, from a copy of the page with scripts disabled (same layout
// as the live page before its scripts run), so the measured run never waits for the page to answer
async function layout(browser, url) {
  const ctx = await browser.newContext(Object.assign({}, ctxOpts, { javaScriptEnabled: false }));
  const page = await ctx.newPage();
  await page.goto(url, { waitUntil: 'load' });
  const fr = page.frames().length > 1 ? page.frames()[1] : page.mainFrame();
  await fr.waitForSelector('#eyepiece', { state: 'attached' });
  const out = {};
  for (const id of ['start', 'eyepiece', 'voice', 'file-btn']) {
    const bb = await fr.locator('#' + id).boundingBox();
    out[id] = bb && { x: bb.x + bb.width / 2, y: bb.y + bb.height / 2, h: bb.height };
  }
  await ctx.close();
  return out;
}

async function runConfig(name) {
  const cfg = CONFIGS[name];
  if (!cfg) throw new Error('unknown config ' + name);
  const url = hostFor(cfg);
  const browser = await launch(cfg);
  const res = { config: name, cfg, url: path.basename(url), page: LABEL };
  try {
    const pos = await layout(browser, url);
    res.buttons = pos;
    const ctx = await browser.newContext(ctxOpts);
    await ctx.addInitScript(SPEECH_STUB + INIT + (cfg.engineGlobal ? 'window.PS_ENGINE=' + JSON.stringify(cfg.engineGlobal) + ';' : ''));
    const page = await ctx.newPage();
    const errors = [], external = [];
    page.on('pageerror', (e) => errors.push(e.message));
    page.on('request', (r) => { const u = r.url(); if (!/^(file|data|blob):/.test(u)) external.push(u); });
    const cdp = await ctx.newCDPSession(page);
    if (cfg.rate > 1) await cdp.send('Emulation.setCPUThrottlingRate', { rate: cfg.rate });
    const t0 = Date.now();
    await page.goto(url, { waitUntil: 'commit' });
    const frameOf = () => (page.frames().length > 1 ? page.frames()[1] : page.mainFrame());
    // an out-of-process iframe needs its own throttling session
    let frameThrottled = false;
    const throttleFrame = async () => {
      if (cfg.rate <= 1 || frameThrottled || page.frames().length < 2) return;
      try { const s = await ctx.newCDPSession(page.frames()[1]); await s.send('Emulation.setCPUThrottlingRate', { rate: cfg.rate }); frameThrottled = 'own session'; } catch (e) { frameThrottled = 'same process as the host'; }
    };
    const clicks = [], acks = [];
    const PROBE = { x: ctxOpts.viewport.width - 28, y: 22 };
    let fr = null;
    // raw input events, not awaited: each click goes out at its time even while the page is busy
    const send = (type, x, y) => cdp.send('Input.dispatchMouseEvent', Object.assign({ type, x, y }, type === 'mouseMoved' ? {} : { button: 'left', clickCount: 1 }));
    const where = async (id) => {      // live position when the page answers within 60 ms, else the last known one
      if (id === 'probe') return PROBE;
      if (fr && !fr.isDetached()) {
        const bb = await Promise.race([fr.locator('#' + id).boundingBox().catch(() => null), sleep(60).then(() => null)]);
        if (bb) pos[id] = { x: bb.x + bb.width / 2, y: bb.y + bb.height / 2, h: bb.height };
      }
      return pos[id];
    };
    const clickAt = async (ms, id, label) => {
      const wait = ms - (Date.now() - t0);
      if (wait > 0) await sleep(wait);
      await throttleFrame();
      const p = await where(id);
      const s = Date.now();
      const c = { label, id, sentAtMs: s - t0 };
      clicks.push(c);
      acks.push(Promise.all([send('mouseMoved', p.x, p.y), send('mousePressed', p.x, p.y), send('mouseReleased', p.x, p.y)]).then(() => { c.ackMs = Date.now() - s; }, (e) => { c.ackError = String(e); }));
    };
    await clickAt(500, 'eyepiece', '0.5 s');
    await clickAt(600, 'probe', '0.6 s probe');
    await clickAt(1000, 'start', '1.0 s start');
    await clickAt(1100, 'probe', '1.1 s probe');
    await clickAt(2000, 'probe', '2 s probe');
    await clickAt(3000, 'probe', '3 s probe');
    await clickAt(5000, 'probe', '5 s probe');
    fr = frameOf();
    await clickAt(5200, 'eyepiece', '5.2 s');
    await clickAt(8000, 'probe', '8 s probe');
    // camera running? (a sandboxed iframe may refuse it: then a video file is analysed instead)
    await Promise.all(acks);
    await fr.waitForFunction(() => window.PSCamera, null, { timeout: 120000 });
    let srcKind = null;
    for (let i = 0; i < 40 && !srcKind; i++) {
      srcKind = await fr.evaluate(() => { const s = window.PSCamera.state; return s.src ? s.src.kind : (s.cameraError ? 'error:' + s.cameraError : null); });
      if (!srcKind) await sleep(250);
    }
    res.startClickHandled = !!srcKind && srcKind !== 'error:';
    if (!srcKind) {      // the 1.0 s click was lost (no handler yet): click once more, normally
      res.startLost = true;
      await fr.click('#start');
      await fr.waitForFunction(() => window.PSCamera.state.src || window.PSCamera.state.cameraError, null, { timeout: 60000 });
      srcKind = await fr.evaluate(() => window.PSCamera.state.src ? window.PSCamera.state.src.kind : 'error:' + window.PSCamera.state.cameraError);
    }
    res.source = srcKind;
    if (/^error/.test(srcKind)) {
      await fr.setInputFiles('#file', VIDEO);
      res.source = srcKind + ' -> video file';
    }
    const tDetWait = Date.now();
    await fr.waitForFunction(() => (window.__psDetections && window.__psDetections.length >= 2) || (window.PSCamera.state.loadError), null, { timeout: 300000 });
    res.firstBoxesAfterNavMs = Date.now() - t0;
    res.waitedForBoxesMs = Date.now() - tDetWait;
    if (NAV) {
      res.nav = await fr.evaluate(() => {
        if (!window.PSCamera.navDemo) return { error: 'no navigation in this page' };
        const P = window.PSCamera;
        P.navDemo(true);
        P.nav.setSpeed(8);
        P.navUi.visible = true;
        // the 8x demo finishes in well under the detection phase: start it again each time it ends
        // so navigation keeps running for the whole measurement; count overlay kinds drawn meanwhile
        window.__navLoad = { restarts: 0, kinds: {} };
        P.nav.on('demo', (e) => {
          if (e.phase !== 'done') return;
          setTimeout(() => { if (!P.navUi.on) return; P.navDemo(true); P.nav.setSpeed(8); P.navUi.visible = true; window.__navLoad.restarts++; }, 300);
        });
        (function tick() { const o = P.navUi.overlay; const k = o ? o.kind : 'null'; window.__navLoad.kinds[k] = (window.__navLoad.kinds[k] || 0) + 1; setTimeout(tick, 250); })();
        return { started: true };
      });
    }
    // main-thread busy time during detection (CDP Performance metrics; the page's own renderer)
    const metric = async () => { try { const m = await cdp.send('Performance.getMetrics'); const g = (n) => (m.metrics.find((x) => x.name === n) || {}).value; return { task: g('TaskDuration'), script: g('ScriptDuration'), t: Date.now() }; } catch (e) { return null; } };
    try { await cdp.send('Performance.enable'); } catch (e) { /* ignore */ }
    const m0 = await metric();
    // clicks during detection, spread over the update cycle
    const tDet = Date.now() - t0;
    const gap = Math.max(700, Math.round(DET_SECS * 1000 / Math.max(1, DET_CLICKS)));
    for (let i = 0; i < DET_CLICKS; i++) {
      await clickAt(tDet + 400 + i * gap + Math.round((i * 397) % 300), 'eyepiece', 'detection #' + (i + 1));
      await clickAt(tDet + 400 + i * gap + Math.round((i * 397) % 300) + Math.round(gap / 2), 'probe', 'detection probe #' + (i + 1));
    }
    await Promise.all(acks);
    await sleep(1500);
    const m1 = await metric();
    if (m0 && m1 && m0.task != null) res.mainThreadDuringDetection = { busyPct: +(100 * (m1.task - m0.task) / ((m1.t - m0.t) / 1000)).toFixed(1), scriptPct: +(100 * (m1.script - m0.script) / ((m1.t - m0.t) / 1000)).toFixed(1), note: cfg.host && cfg.host !== 'file' ? 'host page renderer (an isolated iframe may not be included)' : undefined };
    if (NAV) {
      res.nav = Object.assign(res.nav || {}, await fr.evaluate(() => {
        const n = window.PSCamera.nav;
        return n ? { input: n.input.mode, phase: n.walker && n.walker.phase, steps: n.g.steps, stalls: n.stalls, overlay: window.PSCamera.navUi.overlay && window.PSCamera.navUi.overlay.kind, restarts: window.__navLoad && window.__navLoad.restarts, overlayKinds: window.__navLoad && window.__navLoad.kinds } : null;
      }));
    }
    const data = await fr.evaluate(() => {
      const R = window.__resp, s = window.PSCamera.state, d = window.__psDetections || [];
      return { R, fcp: R.fcp, timeOriginDelta: 0, engine: document.getElementById('engine').textContent, sEngine: document.getElementById('s-engine').textContent,
        backend: s.backend, engineMode: s.engineMode || 'main (old page)', engineInfo: s.engineInfo || null, workerTimes: s.workerTimes || null, workerError: s.workerError || null, loadMs: s.loadMs, loadError: s.loadError ? String(s.loadError) : null,
        dets: d.map((r) => ({ i: r.i, t: r.t, ms: r.ms, backend: r.backend, people: r.people.length, fire: r.fire.length, door: r.door.length, tensors: r.tensors })), now: performance.now() };
    });
    res.frameThrottle = frameThrottled || (cfg.rate > 1 && page.frames().length > 1 ? 'not checked' : undefined);
    res.engine = { mode: data.engineMode, backend: data.backend, info: data.engineInfo, workerTimes: data.workerTimes, workerError: data.workerError, loadMs: Math.round(data.loadMs || 0), loadError: data.loadError, status: data.sEngine };
    res.errors = errors; res.external = external;
    // per click: the pointer-up whose wall-clock time (timeOrigin + timeStamp) is closest after the send;
    // its click record has the same timestamp
    const pr = data.R.clicks, ups = data.R.ups, origin = data.R.origin, used = new Set();
    res.clicks = clicks.map((c) => {
      const sent = t0 + c.sentAtMs;
      let u = null;
      for (const x of ups) { const d = origin + x.ts - sent; if (!used.has(x) && d > -50 && d < 2000 && (!u || Math.abs(d) < Math.abs(origin + u.ts - sent))) u = x; }
      if (!u) return Object.assign(c, { lost: true });
      used.add(u);
      const r = pr.find((x) => Math.abs(x.ts - u.ts) < 2);
      const want = c.id === 'probe' ? '__resp_probe' : c.id;
      if (!r) return Object.assign(c, { hit: u.id, upDelayMs: Math.round(u.at - u.ts), noClick: true, wrongTarget: u.id !== want });
      const ev = data.R.ev.filter((e) => e.name === 'click' && Math.abs(e.start - r.ts) < 2);
      return Object.assign(c, {
        hit: r.id, wrongTarget: r.id !== want, handlerDelayMs: Math.round(r.at - r.ts), visibleMs: r.after != null ? Math.round(r.after - r.ts) : null,
        eventTimingMs: ev.length ? ev[0].dur : '<16', changed: r.text0 !== r.text1, text: r.text0 + ' -> ' + r.text1, engineText: r.engine, appReady: r.appReady,
      });
    });
    const fcp = data.fcp || 0;
    const firstDet = data.dets.length ? data.dets[0].t + (data.dets[0].ms ? data.dets[0].ms.total : 0) : data.now;
    const lt = data.R.lt.filter((x) => x[0] + x[1] > fcp);
    const sum = (a) => a.reduce((p, x) => p + x[1], 0);
    const desc = (a) => ({ n: a.length, maxMs: a.length ? Math.max(...a.map((x) => x[1])) : 0, totalMs: sum(a), over200: a.filter((x) => x[1] > 200).length });
    const loadLt = lt.filter((x) => x[0] < firstDet), detLt = lt.filter((x) => x[0] >= firstDet);
    const detSpan = (data.now - firstDet) / 1000;
    res.fcpMs = Math.round(fcp);
    res.timeline = { firstDetPageMs: Math.round(firstDet), detT: data.dets.slice(0, 4).map((d) => [d.t, d.ms && d.ms.total]), clicksPage: pr.map((c) => [c.id, c.ts, c.at]), ups: data.R.ups };
    res.longTasks = { beforeFcp: desc(data.R.lt.filter((x) => x[0] + x[1] <= fcp)), afterFcp: desc(lt), load: desc(loadLt),
      detection: Object.assign(desc(detLt), { seconds: +detSpan.toFixed(1), blockedPct: +(100 * sum(detLt) / Math.max(1, detSpan * 1000)).toFixed(1) }),
      top: lt.slice().sort((a, b) => b[1] - a[1]).slice(0, 8) };
    res.loaf = { afterFcp: data.R.loaf.filter((x) => x[0] > fcp).length, maxBlockingMs: Math.max(0, ...data.R.loaf.filter((x) => x[0] > fcp).map((x) => x[2])),
      top: data.R.loaf.filter((x) => x[0] > fcp).sort((a, b) => b[2] - a[2]).slice(0, 5) };
    res.timerProbe = { lateOver100: data.R.probe.filter((p) => p[1] > 100).length, maxLateMs: Math.max(0, ...data.R.probe.map((p) => p[1])) };
    const dets = data.dets.slice(1);
    const ts = data.dets.map((d) => d.t);
    res.detection = { updates: data.dets.length, backend: [...new Set(data.dets.map((d) => d.backend))].join(','),
      medianTotalMs: median(dets.map((d) => d.ms && d.ms.total)), p90TotalMs: pct(dets.map((d) => d.ms && d.ms.total), 0.9),
      medianPersonMs: median(dets.map((d) => d.ms && d.ms.person)), medianFaceMs: median(dets.map((d) => d.ms && d.ms.face)), medianFiredoorMs: median(dets.map((d) => d.ms && d.ms.firedoor)),
      updatesPerSec: ts.length > 2 ? +((ts.length - 2) * 1000 / (ts[ts.length - 1] - ts[1])).toFixed(2) : null,
      tensors: [...new Set(dets.map((d) => d.tensors))].join('/'), peopleBoxes: dets.filter((d) => d.people > 0).length + '/' + dets.length };
    const lat = res.clicks.filter((c) => c.visibleMs != null && !c.wrongTarget);
    res.summary = {
      clickVisibleMs: Object.fromEntries(res.clicks.map((c) => [c.label, c.lost ? 'LOST' : c.wrongTarget ? 'hit ' + c.hit + (c.visibleMs != null ? ' ' + c.visibleMs : '') : c.visibleMs])),
      worstVisibleMs: lat.length ? Math.max(...lat.map((c) => c.visibleMs)) : null,
      detectionClickMedianMs: median(lat.filter((c) => /^detection/.test(c.label)).map((c) => c.visibleMs)),
      lost: res.clicks.filter((c) => c.lost || c.wrongTarget || (c.id !== 'probe' && c.changed === false)).map((c) => c.label),
      startHandledAfterMs: (res.clicks.find((c) => c.id === 'start') || {}).visibleMs,
    };
    await ctx.close();
  } catch (e) {
    res.error = String(e && e.stack || e);
  }
  await browser.close();
  return res;
}

function line(r) {
  if (r.error) return r.config + ': ERROR ' + r.error.split('\n')[0];
  const s = r.summary, L = r.longTasks, d = r.detection;
  return [r.config.padEnd(24) + ' engine ' + r.engine.mode + '/' + r.engine.backend + ', FCP ' + r.fcpMs + ' ms, boxes after ' + (r.firstBoxesAfterNavMs / 1000).toFixed(1) + ' s',
    '   click->visible ms: ' + Object.entries(s.clickVisibleMs).map(([k, v]) => k + ' ' + v).join(', ') + (s.lost.length ? '  NOT HANDLED/CHANGED: ' + s.lost.join(', ') : ''),
    '   long tasks after first paint: ' + L.afterFcp.n + ' (max ' + L.afterFcp.maxMs + ' ms, ' + L.afterFcp.over200 + ' over 200 ms); load: max ' + L.load.maxMs + ' ms total ' + L.load.totalMs +
      ' ms; detection: max ' + L.detection.maxMs + ' ms, blocked ' + L.detection.blockedPct + ' % of ' + L.detection.seconds + ' s' +
      (r.mainThreadDuringDetection ? '; main thread busy ' + r.mainThreadDuringDetection.busyPct + ' % (script ' + r.mainThreadDuringDetection.scriptPct + ' %)' : ''),
    '   detection: ' + d.updates + ' updates on ' + d.backend + ', median ' + d.medianTotalMs + ' ms (people ' + d.medianPersonMs + ', faces ' + d.medianFaceMs + ', fire/door ' + d.medianFiredoorMs +
      '), ' + d.updatesPerSec + ' updates/s, tensors ' + d.tensors + ', people boxed ' + d.peopleBoxes +
      (r.errors.length ? ', page errors ' + r.errors.length : '') + (r.external.length ? ', EXTERNAL ' + r.external.length : '')].join('\n');
}

(async () => {
  let prev = {};
  try { prev = JSON.parse(fs.readFileSync(RESULTS)); } catch (e) { prev = {}; }
  const out = Object.assign({ runs: {} }, prev);
  for (const n of NAMES) {
    const r = await runConfig(n);
    out.runs[LABEL + ' | ' + n] = Object.assign(r, { when: new Date().toISOString() });
    console.log(line(r));
    fs.writeFileSync(RESULTS, JSON.stringify(out, null, 1));
  }
  console.log('wrote ' + RESULTS);
})();
