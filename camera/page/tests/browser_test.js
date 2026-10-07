#!/usr/bin/env node
// Headless Chromium tests of the PyroSight Camera page with a fake webcam.
//   python3 tests/make_clips.py && python3 build_page.py && (fire/door test build, see README)
//   node tests/browser_test.js                 # all scenarios
//   ONLY=desktop,pan node tests/browser_test.js
// Scenarios: desktop, mobile, cpu, pan, firedoor, upload, denied, csp, switch, sensor.
// Writes tests/out/browser_test.json and screenshots into shots/.
// No GPU here: WebGL runs on SwiftShader (software), so timings are pessimistic.
const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');
const pw = require(path.join(execSync('npm root -g').toString().trim(), 'playwright'));

const PAGE_DIR = path.join(__dirname, '..');
const OUT = path.join(__dirname, 'out');
const SHOTS = path.join(PAGE_DIR, 'shots');
const DIST = path.join(PAGE_DIR, 'dist', 'pyrosight_camera.html');
const FD_DIST = path.join(PAGE_DIR, 'build', 'dist_firedoor_test', 'pyrosight_camera.html');
const CLIPS = JSON.parse(fs.readFileSync(path.join(OUT, 'clips.json')));
const clip = (n) => path.join(OUT, n + '_640x480.y4m');
fs.mkdirSync(SHOTS, { recursive: true });

const checks = [];
function check(scn, name, ok, detail) {
  checks.push({ scn, name, ok: !!ok, detail });
  console.log((ok ? 'PASS ' : 'FAIL ') + scn + ': ' + name + (detail !== undefined ? '  ' + (typeof detail === 'string' ? detail : JSON.stringify(detail)) : ''));
}

const SPEECH_STUB = `
  window.__spoken = [];
  try { if (window.speechSynthesis) { speechSynthesis.speak = (u) => { window.__spoken.push(u.text); }; speechSynthesis.cancel = () => {}; } } catch (e) {}
  window.__dialogs = 0;
  window.alert = window.confirm = window.prompt = () => { window.__dialogs++; };
`;

async function launch(clipName, opts) {
  opts = opts || {};
  const args = ['--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'];
  // (a video file replaces the fake devices with one camera, so the multi-camera test uses Chromium's built-in test pattern)
  if (opts.devices) args.push('--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream=device-count=' + opts.devices);
  else if (clipName) args.push('--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream', '--use-file-for-fake-video-capture=' + clip(clipName));
  return pw.chromium.launch({ args });
}

async function newPage(browser, o) {
  o = o || {};
  const ctx = await browser.newContext({
    viewport: o.viewport || { width: 1280, height: 900 }, deviceScaleFactor: o.dpr || 1,
    isMobile: !!o.mobile, hasTouch: !!o.mobile, colorScheme: o.dark ? 'dark' : 'light',
    userAgent: o.mobile ? 'Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0 Mobile Safari/537.36' : undefined,
  });
  await ctx.addInitScript(SPEECH_STUB + (o.init || ''));
  const page = await ctx.newPage();
  const rec = { external: [], errors: [], console: [] };
  page.on('request', (r) => { const u = r.url(); if (!/^(file|data|blob):/.test(u)) rec.external.push(u); });
  page.on('pageerror', (e) => rec.errors.push(e.message));
  page.on('console', (m) => { if (m.type() === 'error' || m.type() === 'warning') rec.console.push(m.type() + ': ' + m.text().slice(0, 200)); });
  page.on('dialog', (d) => { rec.errors.push('dialog: ' + d.message()); d.dismiss(); });
  return { ctx, page, rec };
}

const S = (page) => page.evaluate(() => {
  const s = window.PSCamera.state;
  return { backend: s.backend, software: s.backendInfo && s.backendInfo.software, modelsReady: s.modelsReady, loadMs: s.loadMs,
    loadError: s.loadError && String(s.loadError), inferences: s.stats.inferences, tensors: s.stats.tensors.slice(), ms: s.stats.ms.slice(),
    labels: s.stats.labels.slice(), errors: s.stats.errors.slice(), trackStates: s.stats.trackStates, log: s.log.map((l) => l.text),
    spoken: window.__spoken.slice(), dialogs: window.__dialogs, src: s.src && { kind: s.src.kind, w: s.src.w, h: s.src.h, mirror: s.src.mirror },
    mark: s.mark, fd: s.fd && { name: s.fd.name, stub: !!s.fd.stub }, fps: s.fps,
    overflow: document.documentElement.scrollWidth - window.innerWidth,
    results: s.results && { people: s.results.people.map((d) => ({ x: d.x, y: d.y, w: d.w, h: d.h, label: d.label, src: d.src, score: d.score })),
      fire: s.results.fire, door: s.results.door } };
});
const waitInf = (page, n, timeout) => page.waitForFunction((k) => window.PSCamera && (window.PSCamera.state.stats.inferences >= k || window.PSCamera.state.loadError),
  n, { timeout: timeout || 300000 });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const shot = (page, name, full) => page.screenshot({ path: path.join(SHOTS, name), fullPage: full !== false });
const median = (a) => { const b = a.slice().sort((x, y) => x - y); return b.length ? b[b.length >> 1] : null; };
function flat(t) { return t.length >= 3 && t.slice(1).every((v) => v === t[1]); }

// ------------------------------------------------------------------ scenarios
const SCN = {};

SCN.desktop = async () => {
  const browser = await launch('still_fire');
  const { page, rec } = await newPage(browser);
  const r = {};
  try {
    await page.goto('file://' + DIST);
    await sleep(500);
    await shot(page, 'desktop_light_before_start.png');
    await page.click('#start');
    await waitInf(page, 8);
    let s = await S(page);
    r.start = s;
    check('desktop', 'detectors load (WebGL or CPU backend)', !s.loadError && (s.backend === 'webgl' || s.backend === 'cpu'), s.backend + (s.software ? ' (software)' : '') + ', load ' + Math.round(s.loadMs) + ' ms');
    check('desktop', 'white person box with distance label on the person', s.labels.slice(-4).every((l) => /\d\.\dM/.test(l)), s.labels.slice(-4));
    check('desktop', 'fire/door reported as stub in this build', s.fd && s.fd.stub, s.fd);
    check('desktop', 'tensor count flat across inferences', flat(s.tensors), s.tensors);
    await shot(page, 'desktop_light.png');
    // voice + alerts
    await page.click('#voice');
    await sleep(300);
    await page.click('#mark');
    await sleep(800);
    await shot(page, 'desktop_marked.png');
    await page.click('#whereout');
    await sleep(300);
    s = await S(page);
    r.afterMark = { log: s.log, spoken: s.spoken, mark: s.mark };
    check('desktop', 'voice speaks; mark and where-out phrases', s.spoken.includes('Voice on.') && s.spoken.includes('Way out marked.') && s.spoken.includes('Way out is ahead.'), s.spoken);
    check('desktop', 'person call-out said once while the person stays in view (rate limit)', s.log.filter((t) => /^Person ahead/.test(t)).length === 1, s.log);
    // eyepiece
    await page.click('#eyepiece');
    await sleep(1500);
    await shot(page, 'desktop_eyepiece.png');
    const grey = await page.evaluate(() => {
      const c = document.getElementById('view'), g = c.getContext('2d'), d = g.getImageData(c.width * 0.25, c.height * 0.3, 40, 40).data;
      let maxChroma = 0;
      for (let i = 0; i < d.length; i += 4) maxChroma = Math.max(maxChroma, Math.abs(d[i] - d[i + 1]), Math.abs(d[i + 1] - d[i + 2]));
      return maxChroma;
    });
    check('desktop', 'eyepiece view renders grey', grey <= 2, 'max channel difference in a patch ' + grey);
    await page.click('#eyepiece');
    // letterboxing: a short window caps the picture height (72vh), so the 4:3 frame is pillarboxed
    await page.setViewportSize({ width: 1280, height: 560 });
    await sleep(1200);
    const lb = await page.evaluate(() => {
      const st = window.PSCamera.state, c = document.getElementById('view'), r = c.getBoundingClientRect();
      const d = st.results && st.results.people[0];
      return { fit: st.fit, canvas: [c.width, c.height], css: [r.width, r.height], src: [st.src.w, st.src.h], person: d && { x: d.x, y: d.y, w: d.w, h: d.h } };
    });
    await shot(page, 'desktop_letterboxed.png', false);
    r.letterbox = lb;
    check('desktop', 'resize: picture letterboxed inside the canvas with its own aspect ratio', lb.fit.x > 10 && Math.abs(lb.fit.w / lb.fit.h - lb.src[0] / lb.src[1]) < 0.01 &&
      Math.abs(lb.canvas[0] - lb.css[0]) <= 1 && Math.abs(lb.canvas[1] - lb.css[1]) <= 1, lb);
    await page.setViewportSize({ width: 1280, height: 900 });
    await sleep(800);
    // dark
    await page.emulateMedia({ colorScheme: 'dark' });
    await sleep(800);
    await shot(page, 'desktop_dark.png');
    const bg = await page.evaluate(() => getComputedStyle(document.body).backgroundColor);
    check('desktop', 'dark mode switches the page background', bg === 'rgb(14, 17, 21)', bg);
    // hidden tab pauses detection
    await sleep(500);
    const n0 = (await S(page)).inferences;
    await page.evaluate(() => { Object.defineProperty(document, 'hidden', { configurable: true, get: () => true }); document.dispatchEvent(new Event('visibilitychange')); });
    await sleep(5000);
    const n1 = (await S(page)).inferences;
    await sleep(5000);
    const n2 = (await S(page)).inferences;
    await page.evaluate(() => { delete document.hidden; document.dispatchEvent(new Event('visibilitychange')); });
    await waitInf(page, n2 + 2, 60000);
    const n3 = (await S(page)).inferences;
    r.hidden = { n0, n1, n2, n3 };
    check('desktop', 'detection pauses while the tab is hidden and resumes', n2 === n1 && n1 - n0 <= 1 && n3 >= n2 + 2, r.hidden);
    s = await S(page);
    r.end = { tensors: s.tensors, inferences: s.inferences, ms: s.ms.slice(-5), trackStates: s.trackStates, overflow: s.overflow };
    check('desktop', 'tensor count flat over the whole session', flat(s.tensors), s.inferences + ' inferences, tensors ' + [...new Set(s.tensors)].join('/'));
    check('desktop', 'no horizontal scroll at 1280 px', s.overflow <= 0, s.overflow);
    check('desktop', 'no network requests, page errors or dialogs', !rec.external.length && !rec.errors.length && !s.dialogs, { external: rec.external, errors: rec.errors });
    r.timing = { medianTotalMs: median(s.ms.map((m) => m.total)), medianPersonMs: median(s.ms.filter((m) => m.person != null).map((m) => m.person)),
      medianFaceMs: median(s.ms.map((m) => m.face)) };
    r.console = rec.console.slice(0, 10);
  } finally { await browser.close(); }
  return r;
};

SCN.mobile = async () => {
  const r = {};
  for (const dark of [false, true]) {
    const browser = await launch('still_fire');
    const { page, rec } = await newPage(browser, { viewport: { width: 390, height: 844 }, dpr: 2, mobile: true, dark });
    const tag = dark ? 'dark' : 'light';
    try {
      await page.goto('file://' + DIST);
      await sleep(500);
      await shot(page, `mobile_${tag}_before_start.png`);
      const o0 = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
      await page.click('#start');
      await waitInf(page, 4);
      await sleep(300);
      const s = await S(page);
      await shot(page, `mobile_${tag}.png`);
      await shot(page, `mobile_${tag}_viewport.png`, false);
      r[tag] = { overflowBefore: o0, overflow: s.overflow, mirror: s.src && s.src.mirror, labels: s.labels.slice(-3), tensors: s.tensors };
      check('mobile', `390 px ${tag}: no horizontal scroll`, o0 <= 0 && s.overflow <= 0, { before: o0, running: s.overflow });
      check('mobile', `390 px ${tag}: person box shown, back camera not mirrored`, s.labels.slice(-2).every((l) => /M/.test(l)) && s.src.mirror === false, s.labels.slice(-2));
      check('mobile', `390 px ${tag}: no errors, no network`, !rec.external.length && !rec.errors.length, rec);
    } finally { await browser.close(); }
  }
  return r;
};

SCN.cpu = async () => {
  const browser = await launch('still_fire');
  const { page, rec } = await newPage(browser);
  const r = {};
  try {
    await page.goto('file://' + DIST + '?backend=cpu');
    await page.click('#start');
    await waitInf(page, 7, 400000);
    const s = await S(page);
    r.backend = s.backend; r.tensors = s.tensors; r.ms = s.ms; r.labels = s.labels;
    check('cpu', 'CPU fallback backend runs', s.backend === 'cpu', s.backend);
    check('cpu', 'person model every 2nd update on CPU, faces every update', s.ms.filter((m) => m.person == null).length >= 2 && s.ms.every((m) => m.face != null), s.ms.map((m) => m.person == null ? 'face only' : 'person+face'));
    check('cpu', 'tensor count flat', flat(s.tensors), s.tensors);
    check('cpu', 'person box kept on face-only updates', s.labels.every((l) => /M/.test(l)), s.labels);
    r.timing = { medianTotalMs: median(s.ms.map((m) => m.total)), personMs: median(s.ms.filter((m) => m.person != null).map((m) => m.person)), faceMs: median(s.ms.map((m) => m.face)) };
    check('cpu', 'no errors, no network', !rec.external.length && !rec.errors.length, rec);
  } finally { await browser.close(); }
  return r;
};

const TRACK_HOOK = `
  window.__trk = [];
  (function () {
    const c = document.createElement('canvas'); c.width = 120; c.height = 6;
    const g = c.getContext('2d', { willReadFrequently: true });
    window.__psTrackHook = function (el, pose, r) {
      g.drawImage(el, 0, el.videoHeight - 6, 120, 6, 0, 0, 120, 6);
      const d = g.getImageData(0, 0, 120, 6).data;
      let idx = 0;
      for (let b = 0; b < 10; b++) if (d[(3 * 120 + b * 12 + 6) * 4] > 128) idx |= 1 << b;
      const e = window.PSCamera.exitInfo();
      window.__trk.push({ t: performance.now(), idx, yaw: pose.yaw, pitch: pose.pitch, state: r.state, psr: r.match ? r.match.psr : null,
        exit: e ? { inView: e.inView, cx: e.centre ? e.centre.x : null, cy: e.centre ? e.centre.y : null, ry: e.rel.yaw, trusted: e.trusted } : null });
    };
  })();
`;

SCN.pan = async () => {
  const browser = await launch('pan');
  const { page, rec } = await newPage(browser, { init: TRACK_HOOK });
  const r = {};
  const yaw = CLIPS.pan.yaw, W = 640, H = 480, fov = CLIPS.pan.fovDeg;
  const f = (Math.max(W, H) / 2) / Math.tan(fov * Math.PI / 360);
  try {
    await page.goto('file://' + DIST + '?mirror=0');
    await page.click('#start');
    await waitInf(page, 1);
    // mark while the clip is in its still opening (frames 0-19, yaw 0)
    await page.waitForFunction(() => { const t = window.__trk; return t.length && t[t.length - 1].idx >= 2 && t[t.length - 1].idx <= 12; }, null, { timeout: 60000 });
    const markInfo = await page.evaluate(() => { window.PSCamera.markWayOut(0.5, 0.5, 'centre'); const t = window.__trk; return { idx: t[t.length - 1].idx, n: t.length, poseYaw: window.PSCamera.tracker.pose.yaw }; });
    await sleep(700);
    await shot(page, 'pan_marked.png', false);
    // out of view: frames 40-60 (yaw 60-120)
    await page.waitForFunction(() => { const t = window.__trk; return t.length && t[t.length - 1].idx >= 45 && t[t.length - 1].idx <= 55; }, null, { timeout: 60000 });
    await shot(page, 'pan_arrow.png', false);
    const midLog = await page.evaluate(() => window.PSCamera.state.log.map((l) => l.text));
    // back in view near the end of the clip (frames 95-110, yaw 0-20)
    await page.waitForFunction(() => { const t = window.__trk; return t.length && t[t.length - 1].idx >= 100 && t[t.length - 1].idx <= 112; }, null, { timeout: 60000 });
    await shot(page, 'pan_back.png', false);
    await sleep(9000);   // one more loop of the clip
    const trk = await page.evaluate(() => window.__trk);
    const s = await S(page);
    const rows = trk.slice(markInfo.n);
    const markYaw = yaw[markInfo.idx], markPoseYaw = markInfo.poseYaw;
    const errs = [], yawErr = [], agree = [];
    let gaps = 0, prev = null;
    for (const t of rows) {
      if (t.idx >= yaw.length) continue;
      const rel = markYaw - yaw[t.idx];
      const trueX = W / 2 + f * Math.tan(rel * Math.PI / 180);
      const trueIn = Math.abs(rel) < 80 && trueX >= 0 && trueX <= W;
      if (t.exit) {
        agree.push(t.exit.inView === trueIn);
        if (t.exit.inView && trueIn) errs.push(Math.abs(t.exit.cx * W - trueX));
        // turn since the mark: measured vs true
        yawErr.push(Math.abs(((t.yaw - markPoseYaw) - (yaw[t.idx] - markYaw) + 540) % 360 - 180));
      }
      if (prev !== null) { const d = (t.idx - prev + yaw.length) % yaw.length; if (d > 1) gaps++; }
      prev = t.idx;
    }
    r.mark = markInfo;
    r.updates = rows.length;
    r.frameGaps = gaps;
    r.states = rows.reduce((a, t) => (a[t.state] = (a[t.state] || 0) + 1, a), {});
    r.markErrPx = { median: median(errs), max: errs.length ? Math.max(...errs) : null, n: errs.length };
    r.yawErrDeg = { median: median(yawErr), max: yawErr.length ? Math.max(...yawErr) : null };
    r.inViewAgreement = agree.filter(Boolean).length + '/' + agree.length;
    r.midLog = midLog;
    r.log = s.log;
    r.psr = { median: median(rows.filter((t) => t.psr != null).map((t) => t.psr)), min: Math.min(...rows.filter((t) => t.psr != null).map((t) => t.psr)) };
    check('pan', 'way-out mark follows a 120-degree camera turn and back', r.markErrPx.max !== null && r.markErrPx.max < 25 && r.yawErrDeg.max < 5,
      `mark centre error median ${r.markErrPx.median && r.markErrPx.median.toFixed(1)} px, max ${r.markErrPx.max && r.markErrPx.max.toFixed(1)} px (640 px frame, ${r.markErrPx.n} frames); yaw error max ${r.yawErrDeg.max && r.yawErrDeg.max.toFixed(2)} deg; states ${JSON.stringify(r.states)}; skipped-frame gaps ${gaps}/${rows.length}`);
    check('pan', 'in view / out of view agrees with truth', agree.filter(Boolean).length >= agree.length - 3, r.inViewAgreement);
    check('pan', 'spoken direction once the way out is out of view (it is to the left)', s.log.some((t) => /^Way out is (ahead, to your left|to your left|behind you, to the left)\./.test(t)), s.log);
    check('pan', 'no errors, no network', !rec.external.length && !rec.errors.length, rec);
  } finally { await browser.close(); }
  return r;
};

SCN.firedoor = async () => {
  const r = {};
  if (!fs.existsSync(FD_DIST)) { check('firedoor', 'fire/door test build exists', false, FD_DIST); return r; }
  {
    const browser = await launch('still_fire');
    const { page, rec } = await newPage(browser);
    try {
      await page.goto('file://' + FD_DIST);
      await page.click('#start');
      await waitInf(page, 5);
      await sleep(300);
      const s = await S(page);
      await shot(page, 'firedoor_fire.png');
      r.fire = { labels: s.labels, fd: s.fd, tensors: s.tensors, ms: s.ms.slice(-3), fire: s.results.fire, log: s.log };
      check('firedoor', 'purple FIRE box on the fire photo', s.labels.slice(-3).filter((l) => /FIRE/.test(l)).length >= 2, s.labels.slice(-3));
      check('firedoor', 'fire box on the fire half of the frame (raw x > 0.5)', s.results.fire.length && s.results.fire.every((d) => d.x + d.w / 2 > 0.5), s.results.fire);
      check('firedoor', 'tensor count flat with the fire/door model', flat(s.tensors), s.tensors);
      check('firedoor', 'fire call-out spoken with its side', s.log.some((t) => /^Fire ahead/.test(t)), s.log);
      await page.click('#eyepiece');
      await sleep(1200);
      await shot(page, 'firedoor_fire_eyepiece.png');
      check('firedoor', 'no errors, no network (fire)', !rec.external.length && !rec.errors.length, rec);
    } finally { await browser.close(); }
  }
  {
    const browser = await launch('still_door');
    const { page, rec } = await newPage(browser);
    try {
      await page.goto('file://' + FD_DIST);
      await page.click('#start');
      await waitInf(page, 5);
      let s = await S(page);
      r.door = { labels: s.labels, door: s.results.door };
      check('firedoor', 'green DOOR box on the door photo', s.labels.slice(-3).some((l) => /DOOR/.test(l)), s.labels.slice(-3));
      const tap = await page.evaluate(() => {
        const st = window.PSCamera.state, d = st.results && st.results.door[0], fit = st.fit, c = document.getElementById('view');
        if (!d || !fit) return null;
        let nx = d.x + d.w / 2; if (st.src.mirror) nx = 1 - nx;
        const r = c.getBoundingClientRect(), k = r.width / c.width;
        return { x: r.left + (fit.x + nx * fit.w) * k, y: r.top + (fit.y + (d.y + d.h / 2) * fit.h) * k };
      });
      if (tap) {
        await page.mouse.click(tap.x, tap.y);
        await sleep(800);
        s = await S(page);
        r.door.mark = s.mark;
        check('firedoor', 'tapping a detected door marks the way out at the door', s.mark && s.mark.door === true, s.mark);
        await shot(page, 'firedoor_door_marked.png');
      } else check('firedoor', 'tapping a detected door marks the way out at the door', false, 'no door detected to tap');
      check('firedoor', 'no errors, no network (door)', !rec.external.length && !rec.errors.length, rec);
    } finally { await browser.close(); }
  }
  return r;
};

SCN.upload = async () => {
  const r = {};
  const browser = await launch(null);
  try {
    for (const [build, tag, vp, dark] of [[DIST, 'stub', { width: 1280, height: 900 }, false], [FD_DIST, 'firedoor_mobile_dark', { width: 390, height: 844 }, true]]) {
      if (!fs.existsSync(build)) continue;
      const { ctx, page, rec } = await newPage(browser, { viewport: vp, dark, dpr: vp.width < 500 ? 2 : 1, mobile: vp.width < 500 });
      await page.goto('file://' + build);
      await page.click('#start');      // no camera on this browser: NotFoundError path
      await page.waitForSelector('#notice:not([hidden])', { timeout: 20000 });
      const notice = await page.textContent('#notice-text');
      await page.setInputFiles('#file', path.join(OUT, 'person_fire.jpg'));
      await waitInf(page, 1);
      await sleep(400);
      let s = await S(page);
      await shot(page, `upload_photo_${tag}.png`);
      r[tag] = { notice, photo: { src: s.src, labels: s.labels, fire: s.results.fire.length } };
      check('upload', `${tag}: no camera -> notice offers the file fallback`, /No camera/.test(notice), notice);
      check('upload', `${tag}: photo analysed (person box)`, s.src.kind === 'image' && /M/.test(s.labels[s.labels.length - 1]), s.labels);
      if (tag !== 'stub') check('upload', `${tag}: photo analysed (fire box)`, s.results.fire.length > 0, s.results.fire);
      const n0 = s.inferences;
      await page.setInputFiles('#file', path.join(OUT, 'mixed_640x480.webm'));
      await waitInf(page, n0 + 3);
      s = await S(page);
      await shot(page, `upload_video_${tag}.png`);
      r[tag].video = { src: s.src, labels: s.labels.slice(-3), tensors: s.tensors };
      check('upload', `${tag}: video file plays and is analysed`, s.src.kind === 'video' && s.inferences >= n0 + 3, s.src);
      check('upload', `${tag}: no horizontal scroll`, s.overflow <= 0, s.overflow);
      if (tag === 'stub') {   // a portrait photo: the screen takes its 3:4 shape, capped by the window height
        const n1 = s.inferences;
        await page.setInputFiles('#file', path.join(PAGE_DIR, '..', 'testdata', 'people', 'images', 'd96e2962d8db7d41.jpg'));
        await waitInf(page, n1 + 1);
        await sleep(400);
        const p2 = await page.evaluate(() => ({ fit: window.PSCamera.state.fit, src: [window.PSCamera.state.src.w, window.PSCamera.state.src.h],
          labels: window.PSCamera.state.stats.labels.slice(-1) }));
        await shot(page, 'upload_portrait_photo.png', false);
        r[tag].portrait = p2;
        check('upload', 'portrait photo: kept 3:4 inside the screen, person box drawn', p2.src[0] < p2.src[1] && Math.abs(p2.fit.w / p2.fit.h - p2.src[0] / p2.src[1]) < 0.01 && /M/.test(p2.labels[0]), p2);
      }
      check('upload', `${tag}: no errors, no network`, !rec.external.length && !rec.errors.length, rec);
      await ctx.close();
    }
  } finally { await browser.close(); }
  return r;
};

SCN.denied = async () => {
  const r = {};
  const hostAllowNone = path.join(OUT, 'host_camera_blocked.html');
  const hostSandbox = path.join(OUT, 'host_sandboxed.html');
  const frameStyle = 'style="display:block;width:100%;height:100vh;border:0"';
  fs.writeFileSync(hostAllowNone, `<!doctype html><meta name="viewport" content="width=device-width, initial-scale=1"><style>html,body{margin:0}</style><iframe allow="camera 'none'" ${frameStyle} src="${DIST}"></iframe>`);
  fs.writeFileSync(hostSandbox, `<!doctype html><meta name="viewport" content="width=device-width, initial-scale=1"><style>html,body{margin:0}</style><iframe sandbox="allow-scripts" ${frameStyle} src="${DIST}"></iframe>`);
  const CLAUDE_STUB = (mode) => `
    window.__saved = null;
    window.claude = { use: async (name) => {
      if (name !== 'downloads' || ${JSON.stringify(mode)} === 'null') return null;
      return Object.freeze({ save: async (req) => {
        if (${JSON.stringify(mode)} === 'declined') { const e = new Error('no'); e.code = 'declined'; throw e; }
        window.__saved = { filename: req.filename, data: req.data, type: typeof req.data }; return { status: 'saved' }; } });
    } };`;
  const browser = await launch('still_fire');   // a camera exists, but the embedding page blocks it
  try {
    // 1. iframe with camera blocked by permissions policy (NotAllowedError), claude downloads available
    for (const [vp, dark, tag] of [[{ width: 1280, height: 900 }, false, 'desktop_light'], [{ width: 390, height: 844 }, false, 'mobile_light'], [{ width: 390, height: 844 }, true, 'mobile_dark']]) {
      const { ctx, page, rec } = await newPage(browser, { viewport: vp, dark, dpr: vp.width < 500 ? 2 : 1, mobile: vp.width < 500, init: CLAUDE_STUB('ok') });
      await page.goto('file://' + hostAllowNone);
      const fr = page.frames()[1];
      await fr.waitForFunction(() => window.PSCamera, null, { timeout: 30000 });
      await fr.click('#start');
      await fr.waitForSelector('#notice:not([hidden])', { timeout: 20000 });
      await sleep(400);
      const info = await fr.evaluate(() => ({ title: document.getElementById('notice-title').textContent, text: document.getElementById('notice-text').textContent,
        saveVisible: !document.getElementById('n-save').hidden, overflow: document.documentElement.scrollWidth - window.innerWidth,
        err: window.PSCamera.state.cameraError }));
      await shot(page, `denied_${tag}.png`, false);
      // the page itself (inside the iframe) in full length
      const fh = await fr.evaluate(() => document.documentElement.scrollHeight);
      r[tag] = info;
      check('denied', `${tag}: blocked camera -> clear message`, info.err === 'NotAllowedError' && /blocked/.test(info.title), info);
      check('denied', `${tag}: save button shown when claude downloads exists`, info.saveVisible, info.saveVisible);
      check('denied', `${tag}: no horizontal scroll`, info.overflow <= 0, info.overflow);
      if (tag === 'desktop_light') {
        await fr.click('#n-save');
        await fr.waitForFunction(() => window.__saved, null, { timeout: 20000 });
        const saved = await fr.evaluate(() => window.__saved);
        const dist = fs.readFileSync(DIST, 'utf8');
        fs.writeFileSync(path.join(OUT, 'saved_copy.html'), saved.data);
        r.saved = { filename: saved.filename, type: saved.type, bytes: Buffer.byteLength(saved.data), identical: saved.data === dist,
          message: await fr.textContent('#n-save-msg') };
        check('denied', 'save hands downloads.save the full standalone page (identical to dist file)', saved.filename === 'pyrosight_camera.html' && saved.type === 'string' && saved.data === dist,
          { filename: saved.filename, bytes: r.saved.bytes, identical: r.saved.identical });
      }
      check('denied', `${tag}: no errors, no network`, !rec.external.length && !rec.errors.length, rec);
      void fh;
      await ctx.close();
    }
    // 2. sandboxed iframe (like an artifact host): SecurityError; downloads capability absent -> no save button
    {
      const { ctx, page, rec } = await newPage(browser, { init: CLAUDE_STUB('null') });
      await page.goto('file://' + hostSandbox);
      const fr = page.frames()[1];
      await fr.waitForFunction(() => window.PSCamera, null, { timeout: 30000 });
      await fr.click('#start');
      await fr.waitForSelector('#notice:not([hidden])', { timeout: 20000 });
      await sleep(1500);
      const info = await fr.evaluate(() => ({ err: window.PSCamera.state.cameraError, title: document.getElementById('notice-title').textContent,
        saveVisible: !document.getElementById('n-save').hidden || !document.getElementById('save2').hidden }));
      r.sandboxed = info;
      check('denied', 'sandboxed iframe: camera refused, message shown, save hidden when use() gives null', /blocked/.test(info.title) && !info.saveVisible, info);
      // the file fallback works inside the sandbox
      await fr.setInputFiles('#file', path.join(OUT, 'person_fire.jpg'));
      await fr.waitForFunction(() => window.PSCamera.state.stats.inferences >= 1, null, { timeout: 300000 });
      await sleep(300);
      const s = await fr.evaluate(() => ({ labels: window.PSCamera.state.stats.labels, backend: window.PSCamera.state.backend }));
      r.sandboxed.photo = s;
      check('denied', 'sandboxed iframe: photo fallback analysed', /M/.test(s.labels[0] || ''), s);
      await shot(page, 'denied_sandboxed_photo.png', false);
      check('denied', 'sandboxed iframe: no errors, no network', !rec.external.length && !rec.errors.length, rec);
      await ctx.close();
    }
    // 3. viewer declines the save prompt: nothing happens, no message
    {
      const { ctx, page } = await newPage(browser, { init: CLAUDE_STUB('declined') });
      await page.goto('file://' + hostAllowNone);
      const fr = page.frames()[1];
      await fr.waitForFunction(() => window.PSCamera, null, { timeout: 30000 });
      await fr.click('#start');
      await fr.waitForSelector('#n-save:not([hidden])', { timeout: 20000 });
      await fr.click('#n-save');
      await sleep(800);
      const m = await fr.textContent('#n-save-msg');
      check('denied', 'declined save: no message, button stays', m === '' && await fr.isVisible('#n-save'), JSON.stringify(m));
      await ctx.close();
    }
    // 4. no window.claude at all, user denies (simulated NotAllowedError): page identical otherwise, no save button
    {
      const { ctx, page } = await newPage(browser, { init: `navigator.mediaDevices.getUserMedia = () => Promise.reject(new DOMException('Permission denied', 'NotAllowedError'));` });
      await page.goto('file://' + DIST);
      await page.click('#start');
      await page.waitForSelector('#notice:not([hidden])', { timeout: 20000 });
      const info = await page.evaluate(() => ({ hasClaude: 'claude' in window, saveVisible: !document.getElementById('n-save').hidden, title: document.getElementById('notice-title').textContent }));
      check('denied', 'no window.claude: denial message, no save button', !info.hasClaude && !info.saveVisible && /blocked/.test(info.title), info);
      await shot(page, 'denied_no_claude.png', false);
      await ctx.close();
    }
  } finally { await browser.close(); }
  // 5. the saved copy works on its own with a camera
  if (fs.existsSync(path.join(OUT, 'saved_copy.html'))) {
    const b2 = await launch('still_fire');
    try {
      const { page, rec } = await newPage(b2);
      await page.goto('file://' + path.join(OUT, 'saved_copy.html'));
      await page.click('#start');
      await waitInf(page, 2);
      const s = await S(page);
      check('denied', 'saved copy opened from file:// runs the camera and detectors', s.inferences >= 2 && /M/.test(s.labels[1]) && !rec.errors.length, s.labels);
    } finally { await b2.close(); }
  }
  return r;
};

// The fragment inside a host skeleton with a strict Content-Security-Policy: no eval, no
// WebAssembly, no network, and media only from data:/mediastream: (blob: video URLs blocked,
// which exercises the page's data: URL fallback for video files).
SCN.csp = async () => {
  const r = {};
  const frag = fs.readFileSync(path.join(PAGE_DIR, 'dist', 'pyrosight_camera.fragment.html'), 'utf8');
  const csp = "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data:; media-src data: mediastream:; connect-src 'none'; worker-src 'none'";
  const host = path.join(OUT, 'host_csp.html');
  fs.writeFileSync(host, '<!doctype html><html><head><meta charset=utf8><meta name=viewport content="width=device-width,initial-scale=1,viewport-fit=cover">' +
    '<meta http-equiv="Content-Security-Policy" content="' + csp + '"><style>html,body{margin:0}</style></head><body>' + frag + '</body></html>');
  const browser = await launch('still_fire');
  const { page, rec } = await newPage(browser, { init: "window.__csp = []; document.addEventListener('securitypolicyviolation', (e) => window.__csp.push(e.violatedDirective + ' ' + e.blockedURI));" });
  try {
    await page.goto('file://' + host);
    await page.click('#start');
    await waitInf(page, 3);
    let s = await S(page);
    r.camera = { backend: s.backend, labels: s.labels, tensors: s.tensors };
    check('csp', 'fragment in a strict-CSP host: detectors run on the camera', s.inferences >= 3 && !s.loadError && /M/.test(s.labels[2]), r.camera);
    const same = await page.evaluate(() => window.PSCamera.buildStandalone()) === fs.readFileSync(DIST, 'utf8');
    check('csp', 'save copy from the fragment equals the standalone page', same, same);
    const n0 = s.inferences;
    await page.setInputFiles('#file', path.join(OUT, 'mixed_640x480.webm'));
    await waitInf(page, n0 + 2);
    s = await S(page);
    const url = await page.evaluate(() => { const v = document.getElementById('file-video'); return v.currentSrc.slice(0, 16); });
    r.video = { src: s.src, url };
    check('csp', 'video file plays via the data: URL fallback when blob: media is blocked', s.src.kind === 'video' && url.startsWith('data:video'), r.video);
    const viol = await page.evaluate(() => window.__csp);
    r.violations = viol;
    // expected: the blocked blob: video URL, Long.js (inside tf.min.js) probing for a tiny
    // WebAssembly helper in a try/catch (it falls back to plain JS), and (since the detectors moved
    // to a Web Worker) the refused blob: worker, after which the page runs them on its own thread
    check('csp', 'only the expected CSP reports (blob: video; Long.js wasm probe; blob: worker and blob: scripts refused)', viol.every((v) => /^media-src blob|^script-src wasm-eval|^worker-src blob|^script-src(-elem)? blob/.test(v)), viol);
    const eng = await page.evaluate(() => ({ mode: window.PSCamera.state.engineMode, workerError: window.PSCamera.state.workerError }));
    r.engine = eng;
    check('csp', "worker-src 'none': detectors fall back to the page's own thread", eng.mode === 'main' && !!eng.workerError, eng);
    check('csp', 'no page errors, no network', !rec.external.length && !rec.errors.length, rec);
    await shot(page, 'csp_host_video.png');
  } finally { await browser.close(); }
  return r;
};

SCN.switch = async () => {
  const r = {};
  const browser = await launch('still_fire', { devices: 2 });
  const { page, rec } = await newPage(browser);
  try {
    await page.goto('file://' + DIST);
    await page.click('#start');
    await page.waitForSelector('#switch:not([hidden])', { timeout: 30000 });
    const d0 = await page.evaluate(() => window.PSCamera.state.src.deviceId);
    await page.click('#switch');
    await page.waitForFunction((d) => { const s = window.PSCamera.state.src; return s && s.deviceId && s.deviceId !== d; }, d0, { timeout: 30000 });
    const d1 = await page.evaluate(() => window.PSCamera.state.src.deviceId);
    const n = (await S(page)).inferences;
    await waitInf(page, n + 2);
    r.devices = [d0, d1];
    check('switch', 'two cameras: Switch camera shown and changes device; detection continues', d0 !== d1, r.devices);
    check('switch', 'no errors', !rec.errors.length, rec);
  } finally { await browser.close(); }
  return r;
};

// DeviceOrientation wiring: synthetic events (headless Chromium has no motion sensor).
SCN.sensor = async () => {
  const r = {};
  const browser = await launch('still_fire');
  const { page, rec } = await newPage(browser);
  try {
    await page.goto('file://' + DIST + '?mirror=0');
    await page.click('#start');
    await waitInf(page, 1);
    await page.click('#mark');
    // turn 60 degrees right according to the "sensor" (alpha decreases), picture unchanged
    for (let k = 0; k <= 60; k += 2) {
      await page.evaluate((a) => window.dispatchEvent(new DeviceOrientationEvent('deviceorientation', { alpha: a, beta: 90, gamma: 0 })), -k);
      await sleep(50);
    }
    for (let k = 0; k < 60; k++) { await page.evaluate(() => window.dispatchEvent(new DeviceOrientationEvent('deviceorientation', { alpha: -60, beta: 90, gamma: 0 }))); await sleep(50); }
    const t = await page.evaluate(() => ({ yaw: window.PSCamera.tracker.pose.yaw, source: window.PSCamera.tracker.source(performance.now()),
      track: document.getElementById('s-track').textContent, exit: document.getElementById('s-exit').textContent }));
    r.after = t;
    check('sensor', 'orientation events reach the tracker (source = picture + sensor; pose pulled toward the sensor)',
      t.source === 'camera+sensor' && t.yaw > 30 && /motion sensor/.test(t.track), t);
    check('sensor', 'no errors', !rec.errors.length, rec);
  } finally { await browser.close(); }
  return r;
};

// ------------------------------------------------------------------ navigation (demo)
// camera/nav (the device's navigation code as JavaScript) inside the page: no camera (as inside
// claude.ai), demo walk + camera, camera-turn heading, the relationship with "Mark way out".
const NAV_SHOTS = path.join(PAGE_DIR, '..', 'shots');
// viewport shots: scrolled so that the camera picture is at the top (the taps scrolled the page)
const navShot = async (page, name, full) => {
  if (!full) await page.evaluate(() => { const r = document.getElementById('screen').getBoundingClientRect(); window.scrollBy(0, r.top - 8); });
  await sleep(150);
  await page.screenshot({ path: path.join(NAV_SHOTS, name), fullPage: !!full });
};
const NS = (page) => page.evaluate(() => {
  const n = window.PSCamera.nav, u = window.PSCamera.navUi, g = n ? n.g : null;
  return { on: u.on, input: n && { mode: n.input.mode, status: n.input.status, label: n.input.label, detail: n.input.detail },
    g: g && { valid: g.valid, state: g.state, level: g.level, levelId: g.levelId, steps: g.steps, pos: g.pos, yawDeg: g.yaw * 180 / Math.PI,
      route: g.routeDistM, home: g.homeDistM, routeBrg: g.routeBearingDeg, homeBrg: g.homeBearingDeg, word: g.word, exitIsNext: g.exitIsNext, conf: g.confidence },
    walker: n && n.walker && { phase: n.walker.phase, result: n.walker.result }, overlay: u.overlay,
    log: window.PSCamera.state.log.map((l) => l.text), spoken: window.__spoken.slice(), mark: !!window.PSCamera.state.mark,
    markDisabled: document.getElementById('mark').disabled, hint: document.getElementById('hint').textContent,
    inLabel: document.getElementById('nav-in-label').textContent, say: document.getElementById('nav-say').textContent,
    sExit: document.getElementById('s-exit').textContent, overflow: document.documentElement.scrollWidth - window.innerWidth,
    simView: document.getElementById('screen').classList.contains('sim') };
});
// green (#28FF50) pixels of the picture canvas inside a CSS-pixel-free canvas rect (device px)
const greenIn = (page, r) => page.evaluate((r) => {
  const c = document.getElementById('view'), d = c.getContext('2d').getImageData(Math.max(0, Math.floor(r.x)), Math.max(0, Math.floor(r.y)),
    Math.max(1, Math.ceil(r.w)), Math.max(1, Math.ceil(r.h))).data;
  let n = 0;
  for (let i = 0; i < d.length; i += 4) if (d[i + 1] > 200 && d[i] < 120 && d[i + 2] < 150 && d[i + 3] > 200) n++;
  return n;
}, r);
async function holdEl(page, sel, ms) {
  const b = await (await page.$(sel)).boundingBox();
  await page.mouse.move(b.x + b.width / 2, b.y + b.height / 2);
  await page.mouse.down(); await sleep(ms); await page.mouse.up();
}
// every visible button / select at least 44 px tall
const smallTargets = (page) => page.evaluate(() => Array.from(document.querySelectorAll('button, select')).filter((b) => {
  const r = b.getBoundingClientRect(); return r.width > 0 && r.height > 0 && r.height < 44 - 0.5;
}).map((b) => b.id + ':' + b.getBoundingClientRect().height.toFixed(1)));

// No camera at all (NotFoundError, like a blocked camera inside claude.ai): demo walk, guide me
// out, voice, the simulated eyepiece view, auto demo, 390 px layout, iOS-style permission request.
SCN.nav = async () => {
  const r = {};
  const PERM = `window.__perm = []; for (const k of ['DeviceMotionEvent', 'DeviceOrientationEvent']) { if (window[k]) window[k].requestPermission = () => {
    window.__perm.push({ k, gesture: !!(navigator.userActivation && navigator.userActivation.isActive), t: performance.now() }); return Promise.resolve('granted'); }; }`;
  const browser = await launch(null);
  try {
    for (const dark of [false, true]) {
      const tag = dark ? 'dark' : 'light';
      const { ctx, page, rec } = await newPage(browser, { viewport: { width: 390, height: 844 }, dpr: 2, mobile: true, dark, init: PERM });
      await page.goto('file://' + DIST);
      await sleep(2000);
      const perm0 = await page.evaluate(() => window.__perm.length);
      check('nav', `${tag}: no motion permission request on page load`, perm0 === 0, perm0);
      await page.click('#start');                      // no camera here
      await page.waitForSelector('#notice:not([hidden])', { timeout: 20000 });
      await page.click('#voice');
      // Start here: asks for motion permission inside the tap, finds no sensor data, falls back to the demo walk
      await page.click('#nav-start');
      const perm1 = await page.evaluate(() => window.__perm.slice());
      check('nav', `${tag}: Start here asks for motion permission inside the tap (iOS flow)`, perm1.length >= 1 && perm1.every((p) => p.gesture), perm1);
      await page.waitForFunction(() => window.PSCamera.nav.input.mode === 'demo', null, { timeout: 10000 });
      let s = await NS(page);
      r[tag] = { fallback: { input: s.input, inLabel: s.inLabel } };
      check('nav', `${tag}: no sensor data and no camera -> demo walk, input indicator says so`, /Demo walk/.test(s.inLabel) && /Motion sensors not used/.test(s.input.detail) && s.g.valid, { inLabel: s.inLabel, detail: s.input.detail });
      const pos0 = s.g.pos;
      await holdEl(page, '#nav-walk', 2500);
      await sleep(400);
      s = await NS(page);
      const moved = Math.hypot(s.g.pos.x - pos0.x, s.g.pos.y - pos0.y);
      r[tag].walk = { steps: s.g.steps, moved, overlay: s.overlay };
      check('nav', `${tag}: hold to walk moves the position marker`, s.g.steps >= 3 && moved > 1.5, { steps: s.g.steps, movedM: +moved.toFixed(2) });
      check('nav', `${tag}: no camera -> simulated eyepiece view with the way-out marker`, s.simView && s.overlay && s.overlay.kind === 'edge' && /EXIT/.test(s.overlay.label), s.overlay);
      check('nav', `${tag}: navigation replaces Mark way out while it runs`, s.markDisabled && /Navigation is on/.test(s.hint) && /^Navigation:/.test(s.sExit), { markDisabled: s.markDisabled, sExit: s.sExit });
      if (!dark) await navShot(page, 'nav_mobile_demo_walk.png', true);
      // turn around (hold Right) until facing the door: EXIT box in the simulated view
      await page.evaluate(() => window.PSCamera.nav.turn(180));
      await sleep(1500);
      s = await NS(page);
      r[tag].facing = s.overlay;
      const boxG = s.overlay && s.overlay.rect ? await greenIn(page, { x: s.overlay.rect.x - 3, y: s.overlay.rect.y - 3, w: s.overlay.rect.w + 6, h: s.overlay.rect.h + 6 }) : 0;
      check('nav', `${tag}: facing the way out -> green EXIT box sized by distance`, s.overlay && s.overlay.kind === 'box' && /^EXIT \dM/.test(s.overlay.label) && boxG > 30, { overlay: s.overlay, greenPx: boxG });
      if (!dark) await navShot(page, 'nav_mobile_sim_exit_box.png');
      // Guide me out: arrow + device phrase in the log and the voice
      await page.evaluate(() => window.PSCamera.nav.turn(180));
      await sleep(800);
      await page.click('#nav-out');
      await sleep(600);
      s = await NS(page);
      const arrowPx = await page.evaluate(() => {
        const c = document.getElementById('nav-arrow'), d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
        let n = 0; for (let i = 0; i < d.length; i += 4) if (d[i + 1] > 200 && d[i] < 120 && d[i + 2] < 150) n++;
        return n;
      });
      r[tag].guide = { log: s.log.slice(0, 4), spoken: s.spoken, say: s.say, arrowPx };
      check('nav', `${tag}: Guide me out shows the arrow (green ring + arrow on the widget)`, arrowPx > 500 && /^Way out is/.test(s.say), { arrowPx, say: s.say });
      check('nav', `${tag}: voice log and speech get "Way out is ..."`, s.log.some((t) => /^Way out is /.test(t)) && s.spoken.some((t) => /^Way out is /.test(t)), { log: s.log.slice(0, 4), spoken: s.spoken });
      await page.waitForFunction(() => window.PSCamera.nav.walker.phase === 'done', null, { timeout: 60000 });
      s = await NS(page);
      r[tag].guideResult = s.walker.result;
      check('nav', `${tag}: the demo walker follows the arrow back to the door`, s.walker.result && s.walker.result.doorErrorM < 2.5, s.walker.result);
      check('nav', `${tag}: 390 px: no horizontal scroll`, s.overflow <= 0, s.overflow);
      const small = await smallTargets(page);
      check('nav', `${tag}: 390 px: every visible button at least 44 px tall`, small.length === 0, small);
      await navShot(page, `nav_mobile_${tag}.png`, true);
      // Auto demo at 8x: walk in, look around, ask, walk out; an EXIT box appears on the way out
      await page.click('#nav-auto');
      for (let i = 0; i < 4 && !/8x/.test(await page.textContent('#nav-speed')); i++) await page.click('#nav-speed');   // 1x, 2x, 4x, 8x
      const kinds = {};
      const t0 = Date.now();
      let shotTaken = false;
      while (Date.now() - t0 < 90000) {
        s = await NS(page);
        if (s.overlay) kinds[s.overlay.kind] = (kinds[s.overlay.kind] || 0) + 1;
        if (!dark && !shotTaken && s.walker.phase === 'outbound' && s.overlay && s.overlay.kind === 'box') { await navShot(page, 'nav_mobile_auto_demo_exit.png'); shotTaken = true; }
        if (s.walker.phase === 'done') break;
        await sleep(150);
      }
      r[tag].auto = { result: s.walker.result, kinds, seconds: (Date.now() - t0) / 1000, log: s.log.slice(0, 8) };
      check('nav', `${tag}: auto demo (8x) walks in and back out; EXIT box shown on the way out`, s.walker.phase === 'done' && s.walker.result.doorErrorM < 2.5 && kinds.box > 0, r[tag].auto);
      if (!dark) await navShot(page, 'nav_mobile_auto_demo_done.png', true);
      // Stop: navigation off, Mark way out back (no camera: stays disabled, the hint is back)
      await page.click('#nav-stop');
      await sleep(300);
      s = await NS(page);
      check('nav', `${tag}: Stop ends navigation; the camera's mark hint returns`, !s.on && !s.simView && /^Mark way out/.test(s.hint), { on: s.on, hint: s.hint });
      check('nav', `${tag}: no errors, no network, no dialogs`, !rec.external.length && !rec.errors.length && !(await page.evaluate(() => window.__dialogs)), rec);
      await ctx.close();
    }
  } finally { await browser.close(); }
  return r;
};

// With a camera: demo walk drawn over the camera picture, the camera-turn tracker as the heading,
// DEGRADED / UNRELIABLE styles, eyepiece ring, Mark way out before / during / after navigation.
SCN.navcam = async () => {
  const r = {};
  {   // still camera, demo walk: EXIT box in the camera picture when the walker faces the way out
    const browser = await launch('still_fire');
    const { page, rec } = await newPage(browser, { viewport: { width: 1280, height: 900 } });
    try {
      await page.goto('file://' + DIST);
      await page.click('#start');
      await page.waitForFunction(() => window.PSCamera.state.src && window.PSCamera.state.src.kind === 'camera', null, { timeout: 30000 });
      await page.click('#voice');
      // Mark way out works while navigation is off
      await page.click('#mark');
      await sleep(400);
      let s = await NS(page);
      check('navcam', 'Mark way out works with navigation off', s.mark && !s.markDisabled && /^Way out marked/.test(s.log[0]), s.log.slice(0, 2));
      await page.click('#nav-demo');
      await holdEl(page, '#nav-walk', 2500);
      await page.evaluate(() => window.PSCamera.nav.turn(180));
      await sleep(1500);
      s = await NS(page);
      const rc = s.overlay && s.overlay.rect;
      const gp = rc ? await greenIn(page, { x: rc.x - 3, y: rc.y - 3, w: rc.w + 6, h: rc.h + 6 }) : 0;
      r.demoCamera = { overlay: s.overlay, greenPx: gp, markKept: s.mark, markDisabled: s.markDisabled };
      check('navcam', 'demo walk + camera: facing the way out -> green EXIT box in the camera picture', s.overlay && s.overlay.kind === 'box' && gp > 30, r.demoCamera);
      check('navcam', 'while navigation runs the mark is paused, not lost; Mark way out disabled', s.mark && s.markDisabled, { mark: s.mark, markDisabled: s.markDisabled });
      await navShot(page, 'nav_desktop_camera_exit_box.png');
      // eyepiece: the device ring, top centre
      await page.click('#eyepiece');
      await sleep(1200);
      const ring = await page.evaluate(() => {
        const st = window.PSCamera.state, f = st.fit, c = document.getElementById('view'), k = f.h / 240;
        const d = c.getContext('2d').getImageData(Math.round(f.x + f.w / 2 - 30 * k), Math.round(f.y + 4 * k), Math.round(60 * k), Math.round(56 * k)).data;
        let n = 0; for (let i = 0; i < d.length; i += 4) if (d[i + 1] > 200 && d[i] < 120 && d[i + 2] < 150) n++;
        return n;
      });
      check('navcam', 'eyepiece view: device-style navigation ring and arrow at the top centre', ring > 100, ring);
      await navShot(page, 'nav_desktop_eyepiece.png');
      await page.click('#eyepiece');
      // DEGRADED style: shrink the device's confidence scale (confidence = 1 / (1 + (sigma / scale)^2))
      // so the confidence drops to ~0.5 (DEGRADED < 0.6), then ~0.2 (UNRELIABLE < 0.3)
      const scale0 = await page.evaluate(() => window.PSCamera.nav.core.getCfg('navConfScaleM'));
      await page.evaluate(() => { const n = window.PSCamera.nav; n.core.setCfg('navConfScaleM', n.g.posSigmaM / 1.0); });
      await sleep(1200);
      s = await NS(page);
      r.degraded = s.overlay;
      check('navcam', 'DEGRADED: dashed "EXIT? nM" marker (confidence lowered for the test)', s.g.level === 'DEGRADED' && s.overlay && /^EXIT\? \dM/.test(s.overlay.label) && s.overlay.unsure, { level: s.g.level, overlay: s.overlay });
      await navShot(page, 'nav_desktop_degraded.png');
      await page.evaluate(() => { const n = window.PSCamera.nav; n.core.setCfg('navConfScaleM', n.g.posSigmaM / 2.2); });
      await sleep(1500);
      s = await NS(page);
      r.unreliable = { overlay: s.overlay, log: s.log.slice(0, 3) };
      check('navcam', 'UNRELIABLE: no box, FOLLOW HOSE, spoken "Follow the hose line out"', s.g.level === 'UNRELIABLE' && s.overlay.kind === 'hose' && s.log.some((t) => /Follow the hose line out/.test(t)), r.unreliable);
      await navShot(page, 'nav_desktop_follow_hose.png');
      await page.evaluate((v) => { window.PSCamera.nav.core.setCfg('navConfScaleM', v); }, scale0);
      // Stop: the mark comes back
      await page.click('#nav-stop');
      await sleep(600);
      s = await NS(page);
      check('navcam', 'after Stop the mark is back and Mark way out works again', s.mark && !s.markDisabled && !s.overlay && /^(In view|\d+°)/.test(s.sExit), { sExit: s.sExit, markDisabled: s.markDisabled });
      check('navcam', 'no errors, no network (demo + camera)', !rec.external.length && !rec.errors.length, rec);
    } finally { await browser.close(); }
  }
  {   // front (mirrored) camera, camera-turn heading: the way out behind the viewer is in its picture
    const browser = await launch('still_fire');
    const { page, rec } = await newPage(browser, { viewport: { width: 390, height: 844 }, dpr: 2, mobile: true });
    try {
      await page.goto('file://' + DIST + '?mirror=1');
      await page.click('#start');
      await page.waitForFunction(() => window.PSCamera.state.src && window.PSCamera.state.src.kind === 'camera', null, { timeout: 30000 });
      await page.click('#nav-start');
      await page.waitForFunction(() => window.PSCamera.nav.input.mode === 'camera' && window.PSCamera.nav.g.valid, null, { timeout: 15000 });
      let s = await NS(page);
      r.cameraMode = { inLabel: s.inLabel, detail: s.input.detail };
      check('navcam', 'no motion sensors but a camera -> camera turn heading + Walk button, entry marked', /Camera turn/.test(s.inLabel) && s.g.valid, r.cameraMode);
      await holdEl(page, '#nav-walk', 2500);
      await sleep(800);
      s = await NS(page);
      r.cameraMode.after = { steps: s.g.steps, overlay: s.overlay, word: s.g.word };
      check('navcam', 'front camera, walked forward: the way out (behind you) shows as an EXIT box in its picture', s.g.steps >= 3 && s.overlay && s.overlay.kind === 'box', r.cameraMode.after);
      await navShot(page, 'nav_mobile_front_camera_exit.png');
      check('navcam', 'no errors, no network (camera heading)', !rec.external.length && !rec.errors.length, rec);
    } finally { await browser.close(); }
  }
  {   // panning clip: the navigation heading follows the camera-turn tracker
    const browser = await launch('pan');
    const { page, rec } = await newPage(browser);
    try {
      await page.goto('file://' + DIST + '?mirror=0');
      await page.click('#start');
      await waitInf(page, 1);
      await page.click('#nav-start');
      await page.waitForFunction(() => window.PSCamera.nav.input.mode === 'camera' && window.PSCamera.nav.g.valid, null, { timeout: 15000 });
      const rows = [];
      const t0 = Date.now();
      while (Date.now() - t0 < 16000) {
        rows.push(await page.evaluate(() => ({ nav: window.PSCamera.nav.g.yaw * 180 / Math.PI, trk: window.PSCamera.tracker.pose.yaw, st: window.PSCamera.tracker.state })));
        await sleep(100);
      }
      const ok = rows.filter((x) => x.st === 'ok');
      const navs = ok.map((x) => x.nav), span = Math.max(...navs) - Math.min(...navs);
      // nav yaw (counter-clockwise) = -(tracker yaw, clockwise) + const
      const c = ok.length ? ok[0].nav + ok[0].trk : 0;
      const err = ok.map((x) => Math.abs(((x.nav - (c - x.trk)) + 540) % 360 - 180));
      r.pan = { samples: rows.length, okSamples: ok.length, spanDeg: +span.toFixed(1), maxErrDeg: +Math.max(...err).toFixed(2), medianErrDeg: +median(err).toFixed(2) };
      check('navcam', 'camera-turn heading: navigation turns with the camera (120 degree pan)', span > 90 && median(err) < 3, r.pan);
      check('navcam', 'no errors, no network (pan)', !rec.external.length && !rec.errors.length, rec);
    } finally { await browser.close(); }
  }
  return r;
};

(async () => {
  const names = (process.env.ONLY || 'desktop,mobile,cpu,pan,firedoor,upload,denied,csp,switch,sensor,nav,navcam').split(',');
  const results = {};
  for (const n of names) {
    const t0 = Date.now();
    try { results[n] = await SCN[n](); } catch (e) { results[n] = { error: String(e && e.stack || e) }; check(n, 'scenario ran', false, String(e && e.message || e)); }
    results[n] = Object.assign(results[n] || {}, { seconds: (Date.now() - t0) / 1000 });
    console.log(`-- ${n} done in ${results[n].seconds.toFixed(0)} s`);
  }
  const out = process.env.OUT_JSON || path.join(OUT, 'browser_test' + (process.env.ONLY ? '_' + names.join('_') : '') + '.json');
  fs.writeFileSync(out, JSON.stringify({ when: new Date().toISOString(), checks, results }, null, 1));
  const failed = checks.filter((c) => !c.ok);
  console.log(`${checks.length - failed.length}/${checks.length} checks passed; wrote ${path.relative(process.cwd(), out)}`);
  process.exit(failed.length ? 1 : 0);
})();
