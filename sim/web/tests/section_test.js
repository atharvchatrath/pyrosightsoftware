#!/usr/bin/env node
// Browser tests of the simulator page's "Try it on your own camera" section (sim/web/page.html,
// built by sim/web/build_web.py). Headless Chromium via Playwright, Chromium's fake camera fed
// from the barcoded still clips of camera/tests/make_e2e_clips.py.
//
//   node sim/web/tests/section_test.js [--clips DIR] [--shots DIR] [--tmp DIR] [--baseline OLD_pyrosight_sim.html]
//   (make the clips first: python3 camera/tests/make_e2e_clips.py --only still_face,still_fire,still_door,still_window)
//   ONLY=sim,camera,close,denied,layout,csp node sim/web/tests/section_test.js
//
// sim     the simulator still runs (frames advance, no errors), same visible text as --baseline,
//         nothing of the camera page parsed before the tap
// camera  Start camera with e2e_still_face / _fire / _door / _window: WHITE / PURPLE / GREEN DOOR / GREEN
//         WINDOW boxes inside the frame (its window.__psLastDetections and the box colours on its canvas)
// close   Close camera stops the tracks and removes the frame; removing the frame alone ends them;
//         the simulator is paused while the camera is open and runs again after
// denied  camera refused: guidance, Save button only with window.claude downloads, the saved file
//         is byte-identical to dist/pyrosight_sim.html, a declined save does nothing, the camera
//         page's "Open photo or video" still works inside the frame
// layout  390x844 phone and 1280x800 desktop, light and dark: no horizontal scroll (page and frame),
//         buttons tappable (elementFromPoint) and >= 44 px, screenshots (--shots)
// csp     sandboxed iframe host with a strict CSP (frame-src 'self' blob: data: about:), the same with
//         frame-src 'none', an opaque-origin sandbox (no camera possible), and a host of another
//         origin that does not allow the camera (like claude.ai) with a downloads stub
// failure the camera frame never answers (guidance + Save, frame removed, simulator running again);
//         the stored camera page missing (a clear message, no frame)
// Every run also records every request: anything but file:, data:, blob:, about: fails.
// Results: sim/web/tests/out/section_results.json
const fs = require('fs');
const os = require('os');
const path = require('path');
const crypto = require('crypto');
const { execSync } = require('child_process');
const pw = require(path.join(execSync('npm root -g').toString().trim(), 'playwright'));

const WEB = path.join(__dirname, '..');
const ROOT = path.join(WEB, '..', '..');
const argv = process.argv.slice(2);
const opt = (k, d) => { const i = argv.indexOf(k); return i >= 0 ? argv[i + 1] : d; };
const PAGE = path.resolve(opt('--page', path.join(WEB, 'dist', 'pyrosight_sim.html')));
const FRAG = PAGE.replace(/\.html$/, '.fragment.html');
const CLIPS = opt('--clips', process.env.PS_CLIPS || null);   // default: where camera/tests/make_e2e_clips.py put each clip
const SHOTS = path.resolve(opt('--shots', path.join(os.tmpdir(), 'pyrosight_section_shots')));
const TMP = fs.mkdtempSync(path.join(path.resolve(opt('--tmp', os.tmpdir())), 'ps_section_'));   // host pages for the csp tests (removed at the end)
const BASELINE = opt('--baseline', null);
const PHOTO = path.join(ROOT, 'camera', 'testdata', 'fire', 'images', 'firenet_img_1.jpg');
const OUT = path.join(__dirname, 'out');
const ONLY = (process.env.ONLY || 'sim,camera,close,denied,layout,csp,failure').split(',');
fs.mkdirSync(SHOTS, { recursive: true });
fs.mkdirSync(OUT, { recursive: true });

const STRICT_CSP = "default-src 'none'; script-src 'unsafe-inline' blob:; worker-src blob:; img-src data: blob:; media-src data: blob:; style-src 'unsafe-inline'; frame-src 'self' blob: data: about:";
const NOFRAME_CSP = STRICT_CSP.replace("frame-src 'self' blob: data: about:", "frame-src 'none'");
const COLORS = { people: [255, 255, 255], fire: [200, 80, 255], door: [40, 255, 80], window: [40, 255, 80] };
// Chromium's notes about features a host does not allow (the camera page asks for the camera and the
// motion sensors; microphone is 'none' on purpose)
const ALLOWED_CONSOLE = [/(Potential )?[Pp]ermissions policy violation: \w+ is not allowed in this document/];
const SPEECH_STUB = `try { if (window.speechSynthesis) { speechSynthesis.speak = () => {}; speechSynthesis.cancel = () => {}; } } catch (e) {}`;
// claude.ai downloads capability stub: window.__saved gets {filename, text}
const CLAUDE_STUB = (mode) => `
  window.__saves = [];
  window.claude = { use: (name) => Promise.resolve(name !== 'downloads' || ${JSON.stringify(mode)} === 'null' ? null : Object.freeze({
    save: (req) => {
      const rec = { filename: req.filename, isBlob: req.data instanceof Blob };
      window.__saves.push(rec);
      if (${JSON.stringify(mode)} === 'declined') return Promise.reject({ code: 'declined', message: 'no' });
      return (req.data instanceof Blob ? req.data.text() : Promise.resolve(String(req.data))).then((t) => { rec.text = t; return { status: 'saved' }; });
    } })) };`;

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const results = { page: PAGE, bytes: fs.statSync(PAGE).size, started: new Date().toISOString(), checks: [], runs: {} };
let failed = 0;
function check(name, ok, detail) {
  results.checks.push({ name, ok: !!ok, detail: detail === undefined ? null : detail });
  if (!ok) failed++;
  console.log((ok ? 'PASS ' : 'FAIL ') + name + (detail === undefined ? '' : '  ' + (typeof detail === 'string' ? detail : JSON.stringify(detail))));
}
const sha = (buf) => crypto.createHash('sha256').update(buf).digest('hex');
const PAGE_SHA = sha(fs.readFileSync(PAGE));

// the clips of camera/tests/make_e2e_clips.py: in --clips DIR if given, else where its manifest
// (camera/tests/out/e2e_clips.json) says each one is, else in its default folder camera/tests/out/clips
function clipDir(name) {
  const e2e = path.join(ROOT, 'camera', 'tests', 'out');
  try {
    const m = JSON.parse(fs.readFileSync(path.join(e2e, 'e2e_clips.json')));
    const c = m.clips[name];
    if (c && (c.dir || m.clips_dir)) return c.dir || m.clips_dir;
  } catch (e) { /* no manifest yet */ }
  return path.join(e2e, 'clips');
}
function clip(name) { return path.join(CLIPS || clipDir(name), 'e2e_' + name + '.y4m'); }
async function launch(o) {
  o = o || {};
  const args = ['--enable-unsafe-swiftshader', '--ignore-gpu-blocklist', '--use-fake-device-for-media-stream'];
  if (o.clip) args.push('--use-file-for-fake-video-capture=' + clip(o.clip));
  if (!o.deny) args.push('--use-fake-ui-for-media-stream');
  return pw.chromium.launch({ args });
}
const PHONE = { viewport: { width: 390, height: 844 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true,
  userAgent: 'Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0 Mobile Safari/537.36' };
const DESKTOP = { viewport: { width: 1280, height: 800 } };

// a page with request / error recording
async function newPage(browser, ctxOpts, init) {
  const ctx = await browser.newContext(ctxOpts || DESKTOP);
  await ctx.addInitScript(SPEECH_STUB);
  if (init) await ctx.addInitScript(init);
  const page = await ctx.newPage();
  page.__touch = !!(ctxOpts && ctxOpts.hasTouch);
  const rec = { external: [], errors: [], console: [] };
  page.on('request', (r) => { const u = r.url(); if (!/^(file|data|blob|about):/.test(u) && !(rec.allow || []).includes(u)) rec.external.push(u); });
  page.on('pageerror', (e) => rec.errors.push(String(e && e.stack || e).slice(0, 400)));
  page.on('console', (m) => {
    if (m.type() !== 'error') return;
    const t = m.text();
    if (ALLOWED_CONSOLE.some((re) => re.test(t))) rec.console.push('(allowed) ' + t); else rec.errors.push('console: ' + t);
  });
  return { ctx, page, rec };
}
function recOk(name, rec) {
  check(name + ': no request outside the page', rec.external.length === 0, rec.external.slice(0, 5));
  check(name + ': no page errors', rec.errors.length === 0, rec.errors.slice(0, 5));
}

// the frame of the simulator page that holds the simulator (the page itself, or the iframe of a host)
function simFrame(page) { return page.frames().find((f) => { try { return /pyrosight_sim|ps_section_inner/.test(f.url()); } catch (e) { return false; } }) || page.mainFrame(); }
// the camera frame inside it
function camFrame(page, sim) { return page.frames().find((f) => f.parentFrame() === sim && f.url() === 'about:srcdoc') || null; }

async function waitSim(fr, timeout) {
  await fr.waitForFunction(() => /per frame/.test(document.getElementById('engine').textContent), null, { timeout: timeout || 90000 });
}
async function simClock(fr) { return fr.evaluate(() => document.getElementById('clock').textContent); }

async function tapButton(page, fr, sel) {
  // scroll it into view, check it is the topmost element at its centre, then tap / click it there
  const el = fr.locator(sel);
  await el.scrollIntoViewIfNeeded();
  await sleep(400);
  const bb = await el.boundingBox();
  const hit = await fr.evaluate((s) => {
    const b = document.querySelector(s), r = b.getBoundingClientRect();
    const e = document.elementFromPoint(r.left + r.width / 2, r.top + r.height / 2);
    return { top: e === b || b.contains(e), h: r.height, w: r.width };
  }, sel);
  if (page.__touch) await page.touchscreen.tap(bb.x + bb.width / 2, bb.y + bb.height / 2);
  else await page.mouse.click(bb.x + bb.width / 2, bb.y + bb.height / 2);
  return hit;
}

async function waitDet(cf, kind, timeout) {
  return cf.waitForFunction((k) => { const d = window.__psLastDetections; return d && d[k] && d[k].length > 0 ? d : null; }, kind, { timeout: timeout || 180000, polling: 300 })
    .then((h) => h.jsonValue()).catch(() => null);
}

// exact box colour on the camera page's canvas, in a band around the detected box's outline
async function boxColour(cf, kind, det) {
  return cf.evaluate(([kind, col, det]) => {
    const st = window.PSCamera.state, c = document.getElementById('view'), fit = st.fit;
    if (!fit) return { ok: false, why: 'no fit' };
    const g = c.getContext('2d'), im = g.getImageData(0, 0, c.width, c.height).data;
    const b = det[kind][0], x = st.src && st.src.mirror ? 1 - b.x - b.w : b.x;
    const r = { x0: fit.x + x * fit.w, y0: fit.y + b.y * fit.h, x1: fit.x + (x + b.w) * fit.w, y1: fit.y + (b.y + b.h) * fit.h };
    const band = 14;
    let near = 0, all = 0;
    for (let y = 0; y < c.height; y++) for (let xx = 0; xx < c.width; xx++) {
      const i = 4 * (y * c.width + xx);
      if (im[i] !== col[0] || im[i + 1] !== col[1] || im[i + 2] !== col[2] || im[i + 3] !== 255) continue;
      all++;
      const inOuter = xx >= r.x0 - band && xx <= r.x1 + band && y >= r.y0 - band - 30 && y <= r.y1 + band;
      const inInner = xx > r.x0 + band && xx < r.x1 - band && y > r.y0 + band && y < r.y1 - band;
      if (inOuter && !inInner) near++;
    }
    return { ok: near >= 20, near, all, canvas: [c.width, c.height] };
  }, [kind, COLORS[kind], det]);
}

async function frameLayout(cf) {
  return cf.evaluate(() => ({ sw: document.documentElement.scrollWidth, cw: document.documentElement.clientWidth, w: innerWidth, h: innerHeight,
    theme: document.documentElement.getAttribute('data-theme') }));
}
async function pageLayout(fr) {
  return fr.evaluate(() => ({ sw: document.documentElement.scrollWidth, cw: document.documentElement.clientWidth, w: innerWidth }));
}

// ------------------------------------------------------------------------------------------ sim
async function testSim() {
  const browser = await launch({ clip: 'still_face' });
  const { ctx, page, rec } = await newPage(browser, DESKTOP);
  const t0 = Date.now();
  await page.goto('file://' + PAGE);
  await waitSim(page);
  const startMs = Date.now() - t0;
  const c1 = await simClock(page);
  await sleep(3000);
  const c2 = await simClock(page);
  const st = await page.evaluate(() => ({ engine: document.getElementById('engine').textContent, palette: document.getElementById('palette').textContent,
    pscamera: typeof window.PSCamera, tf: typeof window.tf, iframes: document.querySelectorAll('iframe').length,
    store: (() => { const b = document.getElementById('ps-cam-store'); if (!b) return null; const k = {};
      b.childNodes.forEach((n) => { k[n.nodeType] = (k[n.nodeType] || 0) + 1; }); return { children: k, elements: b.children.length, hidden: b.hidden }; })(),
    scripts: document.scripts.length,
    shell: !!document.getElementById('ps-sim-shell'), section: !!document.getElementById('camera') }));
  check('sim: simulator runs (frames advance)', c1 !== c2, { clock: [c1, c2], engine: st.engine, startMs });
  check('sim: opens in ironbow', /Ironbow/.test(st.palette), st.palette);
  // stored as comments only (node type 8, plus the line breaks between them), no elements
  check('sim: camera page stored as inert text, not parsed before the tap', st.pscamera === 'undefined' && st.tf === 'undefined' && st.iframes === 0 &&
    st.store && st.store.elements === 0 && st.store.children[8] > 10 && Object.keys(st.store.children).every((t) => t === '8' || t === '3') && st.store.hidden, st);
  // the camera text read back from the blocks is exactly camera/dist/pyrosight_camera.html (as built)
  const cam = await page.evaluate(() => { const s = window.PSCamSection.cameraPageText();
    return { len: s.length, want: +document.getElementById('ps-cam-store').getAttribute('data-len'), head: s.slice(0, 15), text: s }; });
  const camFile = path.join(ROOT, 'camera', 'dist', 'pyrosight_camera.html');
  const camSame = fs.existsSync(camFile) && sha(Buffer.from(cam.text)) === sha(fs.readFileSync(camFile));
  check('sim: embedded camera page reads back intact', cam.len === cam.want && cam.head === '<!doctype html>', { len: cam.len, sameAsCurrentCameraDist: camSame });
  // Save rebuild = the file itself
  const parts = await page.evaluate(() => window.PSCamSection.pageParts().join(''));
  check('sim: page rebuilt for "Save this page" is byte-identical to the file', sha(Buffer.from(parts)) === PAGE_SHA, { bytes: parts.length });
  recOk('sim', rec);
  if (BASELINE) {
    // same visible text as the baseline page (markup only: scripts off), apart from the new section
    const texts = {};
    for (const [k, f] of [['old', BASELINE], ['new', PAGE]]) {
      const c2x = await browser.newContext(Object.assign({ javaScriptEnabled: false }, DESKTOP));
      const p = await c2x.newPage();
      await p.goto('file://' + path.resolve(f));
      texts[k] = await p.evaluate(() => { const w = document.querySelector('.wrap').cloneNode(true); w.querySelectorAll('#camera,#cam-jump').forEach((e) => e.remove()); return w.textContent.replace(/\s+/g, ' ').trim(); });
      await c2x.close();
    }
    check('sim: same visible text as the baseline page (outside the new section)', texts.old === texts.new, texts.old === texts.new ? undefined : { old: texts.old.length, new: texts.new.length });
    // the main script: the same code once \uXXXX escapes are read back
    const mainOf = (s) => { const m = s.match(/<script>\n\(function \(\) \{\n  "use strict";\n  \/\/ Layout of api_state[\s\S]*?<\/script>/); return m ? m[0] : null; };
    const oldMain = mainOf(fs.readFileSync(BASELINE, 'utf8')), newMain = mainOf(fs.readFileSync(PAGE, 'utf8'));
    const unesc = (s) => s.replace(/\\u([0-9a-f]{4})/g, (m, h) => String.fromCharCode(parseInt(h, 16)));
    check('sim: simulator script unchanged (apart from \\u escapes)', oldMain && newMain && unesc(newMain) === oldMain);
  }
  await ctx.close();
  await browser.close();
}

// ------------------------------------------------------------------------------------ camera
async function openCamera(page, fr, label) {
  const hit = await tapButton(page, fr, '#cam-start');
  check(label + ': Start camera is tappable', hit.top && hit.h >= 44, hit);
  const t0 = Date.now();
  await fr.waitForFunction(() => window.PSCamSection.ready, null, { timeout: 60000 });
  const readyMs = Date.now() - t0;
  const cf = camFrame(page, fr);
  return { cf, readyMs, t0 };
}

async function testCamera() {
  const want = { still_face: 'people', still_fire: 'fire', still_door: 'door', still_window: 'window' };
  for (const [clipName, kind] of Object.entries(want)) {
    const browser = await launch({ clip: clipName });
    const { ctx, page, rec } = await newPage(browser, DESKTOP);
    await page.goto('file://' + PAGE);
    await waitSim(page);
    const { cf, readyMs, t0 } = await openCamera(page, page.mainFrame(), 'camera ' + clipName);
    const running = await page.waitForFunction(() => { const i = window.PSCamSection.info; return i && i.running; }, null, { timeout: 30000 }).then(() => true, () => false);
    check('camera ' + clipName + ': camera running in the frame', running, { readyMs });
    const det = await waitDet(cf, kind);
    const ms = Date.now() - t0;
    const col = det ? await boxColour(cf, kind, det) : null;
    const brief = det && { people: det.people.map((b) => b.label), fire: det.fire.map((b) => b.score), door: det.door.map((b) => b.score),
      window: (det.window || []).map((b) => b.score), backend: det.backend, engine: det.engine };
    check('camera ' + clipName + ': ' + kind + ' detected inside the frame', !!det, { firstBoxMs: ms, boxes: brief });
    check('camera ' + clipName + ': ' + { people: 'WHITE', fire: 'PURPLE', door: 'GREEN', window: 'GREEN' }[kind] + ' box drawn', col && col.ok, col);
    await sleep(600);
    await page.locator('#camera').screenshot({ path: path.join(SHOTS, 'camera_' + clipName + '_desktop.png') });
    results.runs['camera_' + clipName] = { readyMs, firstBoxMs: ms, boxes: brief, colour: col };
    recOk('camera ' + clipName, rec);
    await ctx.close();
    await browser.close();
  }
}

// ------------------------------------------------------------------------------------- close
async function testClose() {
  const browser = await launch({ clip: 'still_face' });
  const { ctx, page, rec } = await newPage(browser, DESKTOP);
  await page.goto('file://' + PAGE);
  await waitSim(page);
  const { cf } = await openCamera(page, page.mainFrame(), 'close');
  await page.waitForFunction(() => { const i = window.PSCamSection.info; return i && i.running; }, null, { timeout: 30000 });
  const during = await page.evaluate(() => document.getElementById('play').textContent);
  const c1 = await simClock(page);
  await sleep(1500);
  const c2 = await simClock(page);
  check('close: simulator paused while the camera is open', during === 'Play' && c1 === c2, { play: during, clock: [c1, c2], status: await page.textContent('#cam-status') });
  await page.evaluate(() => { window.__tr = window.PSCamSection.frame.contentWindow.PSCamera.state.stream.getTracks(); });
  const hit = await tapButton(page, page.mainFrame(), '#cam-close');
  await sleep(900);
  const after = await page.evaluate(() => ({ tracks: window.__tr.map((t) => t.readyState), iframes: document.querySelectorAll('iframe').length,
    start: !document.getElementById('cam-start').hidden, close: document.getElementById('cam-close').hidden, play: document.getElementById('play').textContent }));
  check('close: Close camera tappable', hit.top && hit.h >= 44, hit);
  check('close: tracks ended, frame removed', after.tracks.length > 0 && after.tracks.every((s) => s === 'ended') && after.iframes === 0 && after.start && after.close, after);
  const c3 = await simClock(page);
  await sleep(1500);
  const c4 = await simClock(page);
  check('close: simulator runs again after closing', after.play === 'Pause' && c3 !== c4, { clock: [c3, c4] });
  // open again; then remove the frame by itself (no stop call): the tracks still end
  await openCamera(page, page.mainFrame(), 'close (again)');
  await page.waitForFunction(() => { const i = window.PSCamSection.info; return i && i.running; }, null, { timeout: 30000 });
  const tr2 = await page.evaluate(async () => {
    const f = window.PSCamSection.frame, tr = f.contentWindow.PSCamera.state.stream.getTracks();
    const before = tr.map((t) => t.readyState);
    f.remove();
    await new Promise((r) => setTimeout(r, 500));
    return { before, after: tr.map((t) => t.readyState) };
  });
  check('close: removing the frame alone ends the camera tracks', tr2.before.every((s) => s === 'live') && tr2.after.every((s) => s === 'ended'), tr2);
  await page.evaluate(() => window.PSCamSection.close());
  recOk('close', rec);
  await ctx.close();
  await browser.close();
}

// ------------------------------------------------------------------------------------ denied
async function testDenied() {
  const browser = await launch({ clip: 'still_face', deny: true });
  for (const mode of ['none', 'saved', 'declined', 'null']) {
    const { ctx, page, rec } = await newPage(browser, PHONE, mode === 'none' ? null : CLAUDE_STUB(mode));
    await page.goto('file://' + PAGE);
    await waitSim(page);
    const { cf } = await openCamera(page, page.mainFrame(), 'denied/' + mode);
    const shown = await page.waitForFunction(() => !document.getElementById('cam-help').hidden && window.PSCamSection.helpKind, null, { timeout: 30000 }).then((h) => h.jsonValue(), () => null);
    const st = await page.evaluate(() => ({ title: document.getElementById('cam-help-title').textContent, text: document.getElementById('cam-help-text').textContent,
      save: !document.getElementById('cam-save-row').hidden, status: document.getElementById('cam-status').textContent, err: window.PSCamSection.info && window.PSCamSection.info.error }));
    check('denied/' + mode + ': guidance shown', shown === 'blocked' && /blocked/.test(st.title), st);
    const wantSave = mode === 'saved' || mode === 'declined';
    check('denied/' + mode + ': Save button ' + (wantSave ? 'shown' : 'hidden'), st.save === wantSave, { save: st.save });
    if (mode === 'none') check('denied/none: tells how to allow the camera / open it in a browser', /site settings/.test(st.text) && /Open photo or video/.test(st.text));
    if (wantSave) check('denied/' + mode + ': tells to save and open in a browser', /Save this page/.test(st.text) && /Chrome/.test(st.text));
    await sleep(300);
    await page.screenshot({ path: path.join(SHOTS, 'denied_' + mode + '_phone.png') });
    if (wantSave) {
      const hit = await tapButton(page, page.mainFrame(), '#cam-save');
      check('denied/' + mode + ': Save tappable', hit.top && hit.h >= 44, hit);
      await page.waitForFunction(() => window.__saves.length > 0 && (window.__saves[0].text !== undefined || true), null, { timeout: 15000 });
      await sleep(1500);
      const sv = await page.evaluate(() => ({ n: window.__saves.length, filename: window.__saves[0].filename, isBlob: window.__saves[0].isBlob, text: window.__saves[0].text || null,
        msg: document.getElementById('cam-save-msg').textContent }));
      if (mode === 'saved') {
        check('denied/saved: saved file is byte-identical to dist/pyrosight_sim.html', sv.text && sha(Buffer.from(sv.text)) === PAGE_SHA && sv.filename === 'pyrosight.html' && sv.isBlob,
          { filename: sv.filename, bytes: sv.text && sv.text.length, blob: sv.isBlob });
        check('denied/saved: says where the file went', /Saved as pyrosight\.html/.test(sv.msg), sv.msg);
      } else {
        check('denied/declined: a declined save does nothing', sv.n === 1 && sv.msg === '', { msg: sv.msg });
      }
    }
    if (mode === 'none') {
      // the camera page's own photo path inside the frame
      await cf.setInputFiles('#file', PHOTO);
      const d = await cf.waitForFunction(() => { const d = window.__psLastDetections; return d && d.src === 'image' ? d : null; }, null, { timeout: 180000, polling: 300 }).then((h) => h.jsonValue(), () => null);
      check('denied/none: "Open photo or video" works inside the frame', !!d, d && { fire: d.fire.length, people: d.people.length, door: d.door.length });
      await page.locator('#camera').screenshot({ path: path.join(SHOTS, 'denied_photo_phone.png') });
    }
    recOk('denied/' + mode, rec);
    await ctx.close();
  }
  await browser.close();
}

// ------------------------------------------------------------------------------------ layout
async function testLayout() {
  const browser = await launch({ clip: 'still_face' });
  for (const [dev, o] of [['phone', PHONE], ['desktop', DESKTOP]]) {
    for (const scheme of ['light', 'dark']) {
      const tag = dev + '_' + scheme;
      const { ctx, page, rec } = await newPage(browser, Object.assign({ colorScheme: scheme }, o));
      await page.goto('file://' + PAGE);
      await waitSim(page);
      const l0 = await pageLayout(page);
      check('layout ' + tag + ': no horizontal scroll', l0.sw <= l0.cw, l0);
      // the header link brings the section into view
      const jump = await tapButton(page, page.mainFrame(), '#cam-jump');
      await sleep(900);
      const vis = await page.evaluate(() => { const r = document.getElementById('cam-start').getBoundingClientRect(); return { top: r.top, bottom: r.bottom, vh: innerHeight }; });
      check('layout ' + tag + ': "Try it on your own camera" link works and is tappable', jump.top && jump.h >= 44 && vis.top >= 0 && vis.bottom <= vis.vh, { jump, vis });
      await page.screenshot({ path: path.join(SHOTS, 'section_' + tag + '.png') });
      const { cf } = await openCamera(page, page.mainFrame(), 'layout ' + tag);
      const det = await waitDet(cf, 'people');
      await sleep(1000);
      const l1 = await pageLayout(page), fl = await frameLayout(cf);
      const fh = await page.evaluate(() => { const f = window.PSCamSection.frame, r = f.getBoundingClientRect(); return { w: r.width, h: r.height, left: r.left }; });
      check('layout ' + tag + ': camera on, white box', !!det && det.people.length > 0);
      check('layout ' + tag + ': no horizontal scroll with the camera open (page and frame)', l1.sw <= l1.cw && fl.sw <= fl.cw, { page: l1, frame: fl, frameBox: fh });
      const inner = await cf.evaluate(() => ({ sh: document.documentElement.scrollHeight, ih: innerHeight, sy: (scrollTo(0, 99999), scrollY) }));
      check('layout ' + tag + ': frame as tall as the camera page (no inner scroll)', inner.sh <= inner.ih && inner.sy === 0, { frameBox: fh.h, inner });
      const closeHit = await page.evaluate(() => { const b = document.getElementById('cam-close'); b.scrollIntoView({ block: 'center' }); const r = b.getBoundingClientRect();
        const e = document.elementFromPoint(r.left + r.width / 2, r.top + r.height / 2); return { top: e === b, h: r.height }; });
      check('layout ' + tag + ': Close camera tappable', closeHit.top && closeHit.h >= 44, closeHit);
      await page.evaluate(() => window.PSCamSection.frame.scrollIntoView({ block: 'start' }));
      await sleep(500);
      await page.screenshot({ path: path.join(SHOTS, 'camera_' + tag + '.png') });
      await page.evaluate(() => document.getElementById('camera').scrollIntoView({ block: 'start' }));
      await sleep(300);
      await page.screenshot({ path: path.join(SHOTS, 'camera_top_' + tag + '.png') });
      if (dev === 'desktop' && scheme === 'light') {
        // a host theme switch reaches the frame
        await page.evaluate(() => document.documentElement.setAttribute('data-theme', 'dark'));
        await sleep(500);
        const th = await frameLayout(cf);
        check('layout: data-theme passed into the frame', th.theme === 'dark', th.theme);
        await page.evaluate(() => document.documentElement.removeAttribute('data-theme'));
      }
      results.runs['layout_' + tag] = { page: l1, frame: fl, frameBox: fh };
      recOk('layout ' + tag, rec);
      await ctx.close();
    }
  }
  await browser.close();
}

// --------------------------------------------------------------------------------------- csp
// Host pages: the fragment inside a page with a Content-Security-Policy (meta tag), framed by an
// outer page through a sandboxed iframe. http hosts are served from 127.0.0.1 (a secure context, so a
// same-origin sandbox can grant the camera like an https artifact host would); file hosts give the
// sandboxed frame an opaque origin, where Chromium never allows the camera.
let server = null, serverUrl = null;
async function startServer() {
  if (server) return;
  const http = require('http');
  server = http.createServer((req, res) => {
    const f = path.join(TMP, path.basename(decodeURIComponent(req.url.split('?')[0])));
    if (!f.startsWith(TMP) || !fs.existsSync(f)) { res.writeHead(404); res.end(); return; }
    res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
    fs.createReadStream(f).pipe(res);
  });
  await new Promise((r) => server.listen(0, '127.0.0.1', r));
  serverUrl = 'http://127.0.0.1:' + server.address().port + '/';
}
async function cspHost(cs) {
  if (cs.http) await startServer();
  const frag = fs.readFileSync(FRAG, 'utf8');
  const inner = 'ps_section_inner_' + cs.tag + '.html', host = 'ps_section_host_' + cs.tag + '.html';
  fs.writeFileSync(path.join(TMP, inner), '<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">' +
    '<meta http-equiv="Content-Security-Policy" content="' + cs.csp + '">' + (cs.init ? '<script>' + cs.init + '</script>' : '') + '</head><body>' + frag + '</body></html>');
  fs.writeFileSync(path.join(TMP, host), '<!doctype html><meta name="viewport" content="width=device-width, initial-scale=1"><style>html,body{margin:0;overflow:hidden}</style>' +
    '<iframe sandbox="' + cs.sandbox + '"' + (cs.allow ? ' allow="' + cs.allow + '"' : '') +
    ' style="display:block;width:100vw;height:100vh;border:0" src="' + (cs.crossOrigin ? serverUrl.replace('127.0.0.1', 'localhost') : '') + inner + '"></iframe>');
  if (cs.http) return serverUrl + host;
  return 'file://' + path.join(TMP, host);
}

async function testCsp() {
  const ALLOW = 'camera *; accelerometer *; gyroscope *; magnetometer *; fullscreen *';
  const SAME = 'allow-scripts allow-same-origin', OPAQUE = 'allow-scripts';
  const cases = [
    { tag: 'strict', csp: STRICT_CSP, http: true, sandbox: SAME, allow: ALLOW, expect: 'camera' },
    { tag: 'frame-src-none', csp: NOFRAME_CSP, http: true, sandbox: SAME, allow: ALLOW, expect: 'either' },
    { tag: 'opaque-sandbox', csp: STRICT_CSP, http: false, sandbox: OPAQUE, allow: ALLOW, expect: 'blocked' },
    // like claude.ai: the page in a frame of another origin that is not given the camera, with the downloads capability
    { tag: 'no-camera-allowed', csp: STRICT_CSP, http: true, crossOrigin: true, sandbox: SAME, allow: null, expect: 'blocked', init: CLAUDE_STUB('saved'), save: true },
  ];
  for (const cs of cases) {
    const browser = await launch({ clip: 'still_face' });
    const { ctx, page, rec } = await newPage(browser, PHONE);
    const url = await cspHost(cs);
    if (cs.http) { const inner = url.replace(/ps_section_host_/, 'ps_section_inner_'); rec.allow = [url, inner, inner.replace('127.0.0.1', 'localhost')]; }
    await page.goto(url);
    await page.waitForTimeout(500);
    const fr = simFrame(page);
    await waitSim(fr, 120000);
    const eng = await fr.evaluate(() => document.getElementById('engine').textContent);
    const c1 = await simClock(fr);
    await sleep(2000);
    check('csp ' + cs.tag + ': simulator runs', c1 !== await simClock(fr), eng);
    const hit = await tapButton(page, fr, '#cam-start');
    check('csp ' + cs.tag + ': Start camera tappable', hit.top, hit);
    // the frame opens and the camera runs, or the section explains what to do
    const outcome = await fr.waitForFunction(() => {
      const s = window.PSCamSection, i = s.info;
      if (i && i.running) return 'camera';
      if (!document.getElementById('cam-help').hidden && s.helpKind) return s.helpKind;
      return null;
    }, null, { timeout: 90000, polling: 250 }).then((h) => h.jsonValue(), () => 'timeout');
    const help = await fr.evaluate(() => ({ hidden: document.getElementById('cam-help').hidden, title: document.getElementById('cam-help-title').textContent,
      text: document.getElementById('cam-help-text').textContent, save: !document.getElementById('cam-save-row').hidden, ready: window.PSCamSection.ready,
      error: window.PSCamSection.info && window.PSCamSection.info.error }));
    const cf = camFrame(page, fr);
    const origins = cf && await cf.evaluate(() => ({ frame: String(self.origin), parent: (() => { try { return String(parent.location.origin); } catch (e) { return 'cross-origin'; } })() })).catch(() => null);
    let det = null;
    if (outcome === 'camera') det = cf && await waitDet(cf, 'people');
    const ok = cs.expect === 'either' ? (outcome === 'camera' ? !!det : outcome === 'frame' || outcome === 'blocked') :
      cs.expect === 'camera' ? outcome === 'camera' && !!det : outcome === cs.expect && help.save === !!cs.save;
    check('csp ' + cs.tag + ': ' + (cs.expect === 'camera' ? 'camera and boxes work' : cs.expect === 'blocked' ? 'guidance shown' + (cs.save ? ' with Save' : '') : 'camera works or guidance shown'), ok,
      { outcome, people: det && det.people.length, origins, help: help.hidden ? null : { title: help.title, save: help.save, error: help.error } });
    if (cs.save && help.save) {
      await tapButton(page, fr, '#cam-save');
      await sleep(2000);
      const sv = await fr.evaluate(() => window.__saves.map((s) => ({ filename: s.filename, text: s.text })));
      check('csp ' + cs.tag + ': Save gives the standalone page', sv.length === 1 && sv[0].text && sha(Buffer.from(sv[0].text)) === PAGE_SHA, { n: sv.length, bytes: sv[0] && sv[0].text && sv[0].text.length });
    }
    await sleep(500);
    await page.screenshot({ path: path.join(SHOTS, 'csp_' + cs.tag + '_phone.png') });
    // Close camera works here too (frame removed; camera tracks ended when there was one)
    if (outcome !== 'frame') {
      const tr = outcome === 'camera' ? await fr.evaluate(() => { try { window.__tr = window.PSCamSection.frame.contentWindow.PSCamera.state.stream.getTracks(); return window.__tr.length; } catch (e) { return -1; } }) : 0;
      await tapButton(page, fr, '#cam-close');
      await sleep(900);
      const after = await fr.evaluate(() => ({ iframes: document.querySelectorAll('iframe').length, tracks: window.__tr ? window.__tr.map((t) => t.readyState) : null }));
      check('csp ' + cs.tag + ': Close camera removes the frame' + (tr > 0 ? ' and ends the tracks' : ''), after.iframes === 0 && (tr > 0 ? after.tracks.every((x) => x === 'ended') : true), after);
    }
    results.runs['csp_' + cs.tag] = { outcome, engine: eng, people: det && det.people.length, origins };
    recOk('csp ' + cs.tag, rec);
    await ctx.close();
    await browser.close();
  }
  if (server) server.close();
}

// ---------------------------------------------------------------------------------- failure
async function testFailure() {
  const browser = await launch({ clip: 'still_face' });
  {  // the frame never says hello (as if the viewer blocked it): every srcdoc is replaced by a plain page
    const { ctx, page, rec } = await newPage(browser, PHONE, CLAUDE_STUB('saved') + `
      Object.defineProperty(HTMLIFrameElement.prototype, 'srcdoc', { configurable: true, set(v) { this.setAttribute('srcdoc', '<p>blocked</p>'); }, get() { return this.getAttribute('srcdoc'); } });`);
    await page.goto('file://' + PAGE);
    await waitSim(page);
    await openCameraNoWait(page);
    const t0 = Date.now();
    const kind = await page.waitForFunction(() => !document.getElementById('cam-help').hidden && window.PSCamSection.helpKind, null, { timeout: 30000 }).then((h) => h.jsonValue(), () => null);
    const afterMs = Date.now() - t0;
    await sleep(800);   // a closed frame is hidden at once and removed 0.5 s later (time for the camera to stop)
    const st = await page.evaluate(() => ({ title: document.getElementById('cam-help-title').textContent, save: !document.getElementById('cam-save-row').hidden,
      iframes: document.querySelectorAll('iframe').length, start: !document.getElementById('cam-start').hidden, play: document.getElementById('play').textContent }));
    check('failure/no answer: guidance with Save, frame removed, simulator running again', kind === 'frame' && st.save && st.iframes === 0 && st.start && st.play === 'Pause',
      Object.assign({ kind, afterMs }, st));
    await page.locator('#camera').screenshot({ path: path.join(SHOTS, 'failure_frame_phone.png') });
    recOk('failure/no answer', rec);
    await ctx.close();
  }
  {  // the stored camera page is missing (for example a host that strips comments)
    const { ctx, page, rec } = await newPage(browser, PHONE);
    await page.goto('file://' + PAGE);
    await waitSim(page);
    await page.evaluate(() => { const b = document.getElementById('ps-cam-store'); while (b.firstChild) b.removeChild(b.firstChild); });
    await openCameraNoWait(page);
    const kind = await page.waitForFunction(() => !document.getElementById('cam-help').hidden && window.PSCamSection.helpKind, null, { timeout: 10000 }).then((h) => h.jsonValue(), () => null);
    const st = await page.evaluate(() => ({ text: document.getElementById('cam-help-text').textContent, iframes: document.querySelectorAll('iframe').length,
      start: !document.getElementById('cam-start').hidden && !document.getElementById('cam-start').disabled, play: document.getElementById('play').textContent }));
    check('failure/missing camera page: says so, no frame, simulator untouched', /^other:/.test(kind || '') && /missing/.test(st.text) && st.iframes === 0 && st.start && st.play === 'Pause', st);
    recOk('failure/missing', rec);
    await ctx.close();
  }
  await browser.close();
}
async function openCameraNoWait(page) {
  const hit = await tapButton(page, page.mainFrame(), '#cam-start');
  check('failure: Start camera tappable', hit.top, hit);
}

(async () => {
  const all = { sim: testSim, camera: testCamera, close: testClose, denied: testDenied, layout: testLayout, csp: testCsp, failure: testFailure };
  for (const k of Object.keys(all)) {
    if (!ONLY.includes(k)) continue;
    const t0 = Date.now();
    try { await all[k](); } catch (e) { check(k + ': ran to the end', false, String(e && e.stack || e).slice(0, 600)); }
    console.log('-- ' + k + ' took ' + Math.round((Date.now() - t0) / 1000) + ' s');
  }
  results.finished = new Date().toISOString();
  results.passed = results.checks.filter((c) => c.ok).length;
  results.failed = failed;
  fs.writeFileSync(path.join(OUT, 'section_results' + (process.env.ONLY ? '_' + ONLY.join('_') : '') + '.json'), JSON.stringify(results, null, 1));
  console.log(`\n${results.passed} of ${results.checks.length} checks passed; screenshots in ${SHOTS}`);
  fs.rmSync(TMP, { recursive: true, force: true });
  process.exit(failed ? 1 : 0);
})();
