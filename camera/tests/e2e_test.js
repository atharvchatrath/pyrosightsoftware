#!/usr/bin/env node
// End-to-end test of the integrated PyroSight Camera page (camera/dist/pyrosight_camera.html)
// with Chromium's fake camera fed from the barcoded clips of tests/make_e2e_clips.py.
//
//   python3 tests/make_e2e_clips.py --clips-dir DIR
//   taskset -c 2,3 node tests/e2e_test.js [--secs 60] [--only faces,fire,windows,...] [--cpu-only] [--no-cpu]
//
// For every clip: open the page from file://, press Start camera, wait for the detectors, then
// record every detection update for --secs seconds through the page's test hook
// (window.__psDetections: boxes, per-model ms, backend, tf.memory().numTensors). A capture hook
// (window.__psCaptureHook, set by this test) reads the frame-number barcode from the analysed
// frame, so each update is compared with the ground truth boxes of the image on screen.
// The same is repeated with WebGL disabled (CPU fallback). Screenshots go to camera/shots.
// Results: tests/out/e2e_results.json (+ a printed summary).
const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');
const pw = require(path.join(execSync('npm root -g').toString().trim(), 'playwright'));

const CAMERA = path.join(__dirname, '..');
const DIST_DEFAULT = path.join(CAMERA, 'dist', 'pyrosight_camera.html');
const OUT = path.join(__dirname, 'out');
const SHOTS = path.join(CAMERA, 'shots');
const MAN = JSON.parse(fs.readFileSync(path.join(OUT, 'e2e_clips.json')));
fs.mkdirSync(SHOTS, { recursive: true });

const argv = process.argv.slice(2);
const opt = (k, d) => { const i = argv.indexOf(k); return i >= 0 ? argv[i + 1] : d; };
const SECS = +opt('--secs', 60);
const DIST = path.resolve(opt('--page', DIST_DEFAULT));   // --page: another build (e.g. an older one, for comparison)
const ONLY = opt('--only', null);
const CPU_ONLY = argv.includes('--cpu-only');
const NO_CPU = argv.includes('--no-cpu');
const RESULTS = path.resolve(OUT, opt('--out', 'e2e_results.json'));   // a bare name goes into tests/out

const SPEECH_STUB = `
  window.__spoken = [];
  try { if (window.speechSynthesis) { speechSynthesis.speak = (u) => { window.__spoken.push(u.text); }; speechSynthesis.cancel = () => {}; } } catch (e) {}
`;
// reads the frame-number barcode (see make_e2e_clips.py) from the frame handed to the detectors
const CAPTURE_HOOK = `
  window.__psCaptureHook = function (cap, ctx) {
    const sx = cap.width / 640, sy = cap.height / 480;
    const d = ctx.getImageData(0, Math.round(476 * sy), cap.width, 1).data;
    const bits = [];
    for (let i = 0; i < 12; i++) { const x = Math.round((128 + i * 32 + 16) * sx); bits.push(d[4 * x + 1] > 128 ? 1 : 0); }
    let idx = 0, par = 0;
    for (let i = 0; i < 11; i++) { idx |= bits[i] << i; par ^= bits[i]; }
    return { frame: par === bits[11] ? idx : null, w: cap.width, h: cap.height };
  };
`;

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const median = (a) => { const b = a.filter((v) => v != null).sort((x, y) => x - y); return b.length ? b[b.length >> 1] : null; };
const pct = (a, q) => { const b = a.filter((v) => v != null).sort((x, y) => x - y); return b.length ? b[Math.min(b.length - 1, Math.floor(q * b.length))] : null; };
function iou(a, b) {
  const ix = Math.max(0, Math.min(a[0] + a[2], b.x + b.w) - Math.max(a[0], b.x));
  const iy = Math.max(0, Math.min(a[1] + a[3], b.y + b.h) - Math.max(a[1], b.y));
  const i = ix * iy, u = a[2] * a[3] + b.w * b.h - i;
  return u > 0 ? i / u : 0;
}
const centreIn = (g, b) => { const cx = b.x + b.w / 2, cy = b.y + b.h / 2; return cx >= g[0] && cx <= g[0] + g[2] && cy >= g[1] && cy <= g[1] + g[3]; };

async function launch(clipFile, cpu) {
  const args = ['--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream', '--use-file-for-fake-video-capture=' + clipFile];
  // (--disable-webgl alone still leaves WebGL on OffscreenCanvas through SwiftShader; this combination removes it)
  if (cpu) args.push('--disable-gpu', '--disable-software-rasterizer', '--disable-webgl', '--disable-webgl2');
  else args.push('--enable-unsafe-swiftshader', '--ignore-gpu-blocklist');
  return pw.chromium.launch({ args });
}

function segOf(clip, frame) {
  if (frame == null) return null;
  const f = frame % clip.n_frames;
  return clip.segments.find((s) => f >= s.frames[0] && f <= s.frames[1]) || null;
}

function score(name, clip, recs) {
  const s = { updates: recs.length, barcodeOk: 0, person: { gtUpdates: 0, hit: 0, gtPersons: 0, personsFound: 0, boxes: 0, unmatchedBoxes: 0, labels: {} },
    fire: { gtUpdates: 0, any: 0, hit: 0, boxes: 0, falseBoxes: 0, falseUpdates: 0, noGtUpdates: 0 },
    door: { gtUpdates: 0, any: 0, hit: 0, boxes: 0, otherBoxes: 0, onWindow: 0 },
    window: { gtUpdates: 0, any: 0, hit: 0, boxes: 0, otherBoxes: 0, offWindow: 0, gtWindows: 0, windowsFound: 0, imagesEverHit: 0, images: 0 },
    perSegment: {} };
  const segHit = new Map();
  for (const r of recs) {
    const seg = segOf(clip, r.capture && r.capture.frame);
    if (!seg) continue;
    s.barcodeOk++;
    const gt = seg.gt || {};
    const ps = s.perSegment[seg.image] || (s.perSegment[seg.image] = { updates: 0, person: 0, fire: 0, door: 0, window: 0, maxScore: {} });
    const win = r.window || [];
    ps.updates++;
    if (r.people.length) ps.person++;
    if (r.fire.length) ps.fire++;
    if (r.door.length) ps.door++;
    if (win.length) ps.window++;
    for (const k of ['fire', 'door', 'window']) for (const b of (r[k] || [])) ps.maxScore[k] = Math.max(ps.maxScore[k] || 0, b.score);
    // people (white boxes): judge only on clips whose images have people ground truth
    if (gt.person) {
      const persons = gt.person, faces = gt.face || [];
      s.person.boxes += r.people.length;
      for (const b of r.people) {
        s.person.labels[b.label] = (s.person.labels[b.label] || 0) + 1;
        if (!persons.some((g) => iou(g, b) >= 0.1) && !faces.some((g) => iou(g, b) >= 0.1)) s.person.unmatchedBoxes++;
      }
      if (persons.length) {
        s.person.gtUpdates++;
        s.person.gtPersons += persons.length;
        const used = new Set();
        for (const g of persons) {
          const j = r.people.findIndex((b, k) => !used.has(k) && iou(g, b) >= 0.3);
          if (j >= 0) { used.add(j); s.person.personsFound++; }
        }
        if (r.people.some((b) => persons.some((g) => iou(g, b) >= 0.3) || faces.some((g) => iou(g, b) >= 0.3))) s.person.hit++;
      }
    }
    // fire (purple)
    s.fire.boxes += r.fire.length;
    if (gt.fire && gt.fire.length) {
      s.fire.gtUpdates++;
      if (r.fire.length) s.fire.any++;
      if (r.fire.some((b) => gt.fire.some((g) => iou(g, b) >= 0.3 || centreIn(g, b)))) s.fire.hit++;
    } else if (gt.fire_present) {
      s.fire.gtUpdates++;
      if (r.fire.length) s.fire.any++;
    } else {
      s.fire.noGtUpdates++;
      s.fire.falseBoxes += r.fire.length;
      if (r.fire.length) s.fire.falseUpdates++;
    }
    // doors (green)
    s.door.boxes += r.door.length;
    if (gt.door && gt.door.length) {
      s.door.gtUpdates++;
      if (r.door.length) s.door.any++;
      if (r.door.some((b) => gt.door.some((g) => iou(g, b) >= 0.3 || centreIn(g, b)))) s.door.hit++;
    } else s.door.otherBoxes += r.door.length;
    // windows (green, WINDOW)
    s.window.boxes += win.length;
    if (gt.window && gt.window.length) {
      s.window.gtUpdates++;
      if (win.length) s.window.any++;
      const on = win.filter((b) => gt.window.some((g) => iou(g, b) >= 0.3 || centreIn(g, b)));
      if (on.length) { s.window.hit++; segHit.set(seg.image, true); } else if (!segHit.has(seg.image)) segHit.set(seg.image, false);
      s.window.offWindow += win.length - on.length;
      s.window.gtWindows += gt.window.length;
      s.window.windowsFound += gt.window.filter((g) => win.some((b) => iou(g, b) >= 0.3 || centreIn(g, b))).length;
      // a DOOR box on a window (not on a door): the confusion the window class should remove
      s.door.onWindow += r.door.filter((b) => gt.window.some((g) => iou(g, b) >= 0.3 || centreIn(g, b)) &&
        !(gt.door || []).some((g) => iou(g, b) >= 0.3 || centreIn(g, b))).length;
    } else s.window.otherBoxes += win.length;
  }
  s.window.images = segHit.size;
  s.window.imagesEverHit = [...segHit.values()].filter(Boolean).length;
  return s;
}

function timing(recs) {
  const ms = (k) => recs.map((r) => r.ms[k]);
  const t = recs.map((r) => r.t);
  return {
    totalMedian: median(ms('total')), totalP90: pct(ms('total'), 0.9),
    personMedian: median(ms('person')), faceMedian: median(ms('face')), firedoorMedian: median(ms('firedoor')),
    updatesPerSec: recs.length > 1 ? +((recs.length - 1) * 1000 / (t[t.length - 1] - t[0])).toFixed(3) : null,
  };
}

async function canvasPoint(page, nx, ny) {
  return page.evaluate(([nx, ny]) => {
    const st = window.PSCamera.state, c = document.getElementById('view'), r = c.getBoundingClientRect(), fit = st.fit;
    const x = st.src.mirror ? 1 - nx : nx;
    return { x: r.left + (fit.x + x * fit.w) * r.width / c.width, y: r.top + (fit.y + ny * fit.h) * r.height / c.height };
  }, [nx, ny]);
}

async function runClip(name, cpu) {
  const clip = MAN.clips[name];
  const clipFile = path.join(clip.dir || MAN.clips_dir, clip.file);
  const tag = (cpu ? 'cpu_' : '') + name;
  const browser = await launch(clipFile, cpu);
  const ctx = await browser.newContext({ viewport: { width: 1280, height: 900 }, deviceScaleFactor: 1 });
  await ctx.addInitScript(SPEECH_STUB + CAPTURE_HOOK);
  const page = await ctx.newPage();
  const rec = { external: [], errors: [], console: [] };
  page.on('request', (r) => { const u = r.url(); if (!/^(file|data|blob):/.test(u)) rec.external.push(u); });
  page.on('pageerror', (e) => rec.errors.push(e.message));
  page.on('console', (m) => { if (m.type() === 'error' || m.type() === 'warning') rec.console.push(m.type() + ': ' + m.text().slice(0, 300)); });
  const res = { clip: name, cpu, file: clip.file, shots: [], actions: {} };
  const t0 = Date.now();
  try {
    await page.goto('file://' + DIST);
    await page.click('#start');
    await page.waitForFunction(() => window.PSCamera && (window.__psLastDetections || window.PSCamera.state.loadError), null, { timeout: 300000 });
    const st0 = await page.evaluate(() => { const s = window.PSCamera.state; return {
      backend: s.backend, backendInfo: s.backendInfo, loadMs: Math.round(s.loadMs), loadError: s.loadError && String(s.loadError),
      fd: s.fd && { name: s.fd.name, stub: !!s.fd.stub, hysteresis: s.fd.hysteresis || null }, fdError: s.fdError && String(s.fdError),
      src: s.src && { kind: s.src.kind, w: s.src.w, h: s.src.h, mirror: s.src.mirror, name: s.src.name } }; });
    res.start = st0;
    res.firstUpdateAfterStartMs = Date.now() - t0;
    const i0 = await page.evaluate(() => window.__psLastDetections.i);
    const want = { faces: 'people', group: 'people', fire: 'fire', doors: 'door', lights: 'fire', nopeople: 'people', firevideo: null, windows: 'window' }[name];
    const tEnd = Date.now() + SECS * 1000;
    let shotTaken = false, marked = false;
    while (Date.now() < tEnd) {
      await sleep(700);
      const last = await page.evaluate(() => window.__psLastDetections);
      if (!last) continue;
      let has = want && last[want] && last[want].length > 0;
      if (want === 'people' && last.fire.length) has = false;   // the key person shot shows people only
      const tapList = name === 'doors' ? last.door : name === 'windows' ? (last.window || []) : [];
      const needSync = (has && !shotTaken) || (!cpu && !marked && tapList.length);
      let inSync = clip.segments.length <= 1;
      if (needSync && clip.segments.length > 1) {
        // only when the picture on screen is the image the boxes were found on (slideshow cuts)
        const shown = await page.evaluate(() => {
          const v = document.getElementById('cam-video');
          if (!v.videoWidth) return null;
          const c = document.createElement('canvas'); c.width = v.videoWidth; c.height = v.videoHeight;
          const g = c.getContext('2d'); g.drawImage(v, 0, 0);
          return window.__psCaptureHook(c, g).frame;
        });
        const a = segOf(clip, shown), b = segOf(clip, last.capture && last.capture.frame);
        inSync = !!(a && b && a === b);
      }
      has = has && inSync;
      // key screenshots (never of the CCTV news footage)
      if (has && !shotTaken && name !== 'firevideo') {
        await sleep(150);
        const file = (name === 'windows' ? 'window_e2e_' + (cpu ? 'cpu_' : '') : 'e2e_' + tag + '_') + (want === 'people' ? 'white' : want === 'fire' ? 'purple' : 'green') + '.png';
        await page.locator('#screen').screenshot({ path: path.join(SHOTS, file) });
        res.shots.push({ file, last: { people: last.people, fire: last.fire, door: last.door, window: last.window, capture: last.capture } });
        shotTaken = true;
      }
      // doors / windows: tap the detected door or window to mark the way out there (green EXIT box)
      if (!cpu && !marked && tapList.length && inSync) {
        const d = tapList[0];
        const p = await canvasPoint(page, d.x + d.w / 2, d.y + d.h / 2);
        await page.mouse.click(p.x, p.y);
        await sleep(400);
        const m = await page.evaluate(() => ({ mark: window.PSCamera.state.mark, exit: window.PSCamera.exitInfo(), log: window.PSCamera.state.log.slice(0, 3).map((l) => l.text) }));
        const act = { box: d, mark: m.mark && { door: m.mark.door, window: m.mark.window, at: m.mark.at, how: m.mark.how }, inView: m.exit && m.exit.inView, trusted: m.exit && m.exit.trusted, log: m.log };
        const file = name === 'windows' ? 'window_e2e_exit_marked.png' : 'e2e_exit_marked_on_door.png';
        if (name === 'windows') res.actions.markWindow = act; else res.actions.markDoor = act;
        await page.locator('#screen').screenshot({ path: path.join(SHOTS, file) });
        res.shots.push({ file });
        marked = true;
      }
    }
    // faces: Mark way out (button) + eyepiece view at the end
    if (name === 'faces' && !cpu) {
      await page.click('#mark');
      await sleep(600);
      const ex = await page.evaluate(() => ({ exit: window.PSCamera.exitInfo(), mark: window.PSCamera.state.mark }));
      res.actions.markButton = { inView: ex.exit && ex.exit.inView, trusted: ex.exit && ex.exit.trusted, door: ex.mark && ex.mark.door };
      await page.locator('#screen').screenshot({ path: path.join(SHOTS, 'e2e_exit_marked_centre.png') });
      await page.click('#eyepiece');
      await sleep(1500);
      await page.locator('#screen').screenshot({ path: path.join(SHOTS, 'e2e_eyepiece.png') });
      res.actions.eyepiece = await page.evaluate(() => {
        const c = document.getElementById('view'), g = c.getContext('2d'), fit = window.PSCamera.state.fit;
        const d = g.getImageData(Math.round(fit.x + fit.w * 0.3), Math.round(fit.y + fit.h * 0.3), 40, 40).data;
        let maxChroma = 0;
        for (let i = 0; i < d.length; i += 4) maxChroma = Math.max(maxChroma, Math.abs(d[i] - d[i + 1]), Math.abs(d[i + 1] - d[i + 2]));
        return { maxChroma, palette: window.PSCamera.state.palette, button: document.getElementById('palette').textContent };
      });
      await page.screenshot({ path: path.join(SHOTS, 'e2e_page_desktop.png'), fullPage: true });
    }
    const all = await page.evaluate(() => (window.__psDetections || []).slice());
    const recs = all.filter((r) => r.i > i0);
    const fin = await page.evaluate(() => { const s = window.PSCamera.state; return {
      // TF.js runs in the page's detector worker now (no tf global here): the count it reports with each update
      numTensors: s.numTensors != null ? s.numTensors : (window.tf ? tf.memory().numTensors : null), engine: s.engineMode, fps: s.fps, errors: s.stats.errors.slice(0, 5), log: s.log.slice(0, 12).map((l) => l.text),
      status: { engine: document.getElementById('s-engine').textContent, fd: document.getElementById('s-fd').textContent,
        speed: document.getElementById('s-speed').textContent, seen: document.getElementById('s-seen').textContent },
      overflow: document.documentElement.scrollWidth - window.innerWidth }; });
    res.final = fin;
    res.timing = timing(recs);
    const tens = recs.map((r) => r.tensors);
    res.tensors = { first: tens[0], last: tens[tens.length - 1], min: Math.min(...tens), max: Math.max(...tens), n: tens.length };
    res.backendsSeen = [...new Set(recs.map((r) => r.backend + (r.software ? '(software)' : '')))];
    res.score = score(name, clip, recs);
    res.records = recs.map((r) => ({ i: r.i, frame: r.capture && r.capture.frame, ms: r.ms, tensors: r.tensors,
      people: r.people.map((b) => b.label), fire: r.fire.map((b) => b.score), door: r.door.map((b) => b.score), window: (r.window || []).map((b) => b.score) }));
  } catch (e) {
    res.error = String(e && e.stack || e);
  }
  res.external = rec.external;
  res.pageErrors = rec.errors;
  res.consoleWarnings = rec.console.slice(0, 10);
  res.wallMs = Date.now() - t0;
  await browser.close();
  return res;
}

function summary(r) {
  if (r.error) return (r.cpu ? 'CPU  ' : 'WebGL ') + r.clip + ': ERROR ' + r.error.split('\n')[0];
  const s = r.score, t = r.timing;
  const bits = [(r.cpu ? 'CPU  ' : 'WebGL') + ' ' + r.clip.padEnd(9) + ' backend ' + r.start.backend + (r.start.backendInfo && r.start.backendInfo.software ? '(sw)' : '') +
    ' load ' + r.start.loadMs + ' ms, ' + s.updates + ' updates (' + s.barcodeOk + ' matched to a frame), median ' + t.totalMedian + ' ms (p90 ' + t.totalP90 +
    '; people ' + t.personMedian + ', faces ' + t.faceMedian + ', fire/door ' + t.firedoorMedian + '), tensors ' + r.tensors.min + '..' + r.tensors.max +
    ', external ' + r.external.length + ', errors ' + r.pageErrors.length];
  if (s.person.gtUpdates) bits.push('   white: updates with a person boxed ' + s.person.hit + '/' + s.person.gtUpdates + ', GT persons found ' + s.person.personsFound + '/' + s.person.gtPersons + ', boxes not on a person ' + s.person.unmatchedBoxes + '/' + s.person.boxes);
  else if (r.clip === 'nopeople') bits.push('   white boxes on people-free photos: ' + r.records.filter((x) => x.people.length).length + '/' + s.updates + ' updates');
  if (s.fire.gtUpdates) bits.push('   purple: updates with FIRE ' + s.fire.any + '/' + s.fire.gtUpdates + (r.clip === 'firevideo' ? '' : ', on the fire ' + s.fire.hit + '/' + s.fire.gtUpdates));
  if (s.fire.noGtUpdates) bits.push('   purple on no-fire images: ' + s.fire.falseUpdates + '/' + s.fire.noGtUpdates + ' updates (' + s.fire.falseBoxes + ' boxes)');
  if (s.door.gtUpdates) bits.push('   green: updates with DOOR ' + s.door.any + '/' + s.door.gtUpdates + ', on a door ' + s.door.hit + '/' + s.door.gtUpdates);
  else if (s.door.otherBoxes) bits.push('   DOOR boxes on images without door ground truth: ' + s.door.otherBoxes);
  if (s.window.gtUpdates) bits.push('   green: updates with WINDOW ' + s.window.any + '/' + s.window.gtUpdates + ', on a window ' + s.window.hit + '/' + s.window.gtUpdates +
    ', GT windows found ' + s.window.windowsFound + '/' + s.window.gtWindows + ', images boxed at least once ' + s.window.imagesEverHit + '/' + s.window.images +
    ', WINDOW boxes not on a window ' + s.window.offWindow + '/' + s.window.boxes + ', DOOR boxes on a window ' + s.door.onWindow);
  else if (s.window.otherBoxes) bits.push('   WINDOW boxes on images without window ground truth: ' + s.window.otherBoxes);
  return bits.join('\n');
}

(async () => {
  const names = ONLY ? ONLY.split(',') : ['faces', 'group', 'nopeople', 'fire', 'doors', 'lights', 'firevideo', 'windows'];
  const cpuNames = ONLY ? names.filter((n) => ['faces', 'fire', 'doors'].includes(n)) : ['faces', 'fire', 'doors'];
  let prev = {};
  try { prev = JSON.parse(fs.readFileSync(RESULTS)); } catch (e) { prev = {}; }
  const out = Object.assign({ page: path.relative(CAMERA, DIST), pageBytes: fs.statSync(DIST).size, secsPerClip: SECS, runs: {} }, prev, { secsPerClip: SECS, pageBytes: fs.statSync(DIST).size });
  out.runs = out.runs || {};
  const jobs = [];
  if (!CPU_ONLY) for (const n of names) jobs.push([n, false]);
  if (!NO_CPU) for (const n of cpuNames) jobs.push([n, true]);
  for (const [n, cpu] of jobs) {
    const r = await runClip(n, cpu);
    out.runs[(cpu ? 'cpu_' : 'webgl_') + n] = r;
    console.log(summary(r));
    fs.writeFileSync(RESULTS, JSON.stringify(out, null, 1));
  }
  console.log('wrote ' + RESULTS);
})();
