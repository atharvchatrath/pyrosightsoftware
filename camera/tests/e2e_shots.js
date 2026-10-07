#!/usr/bin/env node
// Key screenshots of the integrated page, one fake-camera clip per subject (a single image that
// drifts a few pixels, so the boxes on screen always belong to the picture on screen).
//   python3 tests/make_e2e_clips.py --clips-dir DIR --only still_face,still_group,still_fire,still_door,still_window
//   taskset -c 2,3 node tests/e2e_shots.js
// Writes camera/shots/key_*.png, camera/shots/window_key_*.png and tests/out/e2e_shots.json (what was boxed in each shot).
// ONLY=window (or face,group,fire,door) runs a subset.
const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');
const pw = require(path.join(execSync('npm root -g').toString().trim(), 'playwright'));
const CAMERA = path.join(__dirname, '..');
const DIST = path.join(CAMERA, 'dist', 'pyrosight_camera.html');
const MAN = JSON.parse(fs.readFileSync(path.join(__dirname, 'out', 'e2e_clips.json')));
const SHOTS = path.join(CAMERA, 'shots');
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function open(clipName) {
  const c = MAN.clips[clipName];
  const browser = await pw.chromium.launch({ args: ['--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream',
    '--use-file-for-fake-video-capture=' + path.join(c.dir || MAN.clips_dir, c.file), '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] });
  const ctx = await browser.newContext({ viewport: { width: 1280, height: 900 } });
  await ctx.addInitScript('try { speechSynthesis.speak = () => {}; } catch (e) {}');
  const page = await ctx.newPage();
  const ext = [];
  page.on('request', (r) => { if (!/^(file|data|blob):/.test(r.url())) ext.push(r.url()); });
  await page.goto('file://' + DIST);
  await page.click('#start');
  return { browser, page, ext };
}
// wait until the latest update has what we want, and two updates in a row agree
async function waitFor(page, pred, timeout) {
  const t0 = Date.now();
  let prevOk = false, last = null;
  while (Date.now() - t0 < (timeout || 120000)) {
    await sleep(400);
    const d = await page.evaluate(() => window.__psLastDetections || null);
    if (!d || (last && d.i === last.i)) continue;
    const ok = pred(d);
    if (ok && prevOk) return d;
    prevOk = ok; last = d;
  }
  return last;
}
const snap = (page, file) => page.locator('#screen').screenshot({ path: path.join(SHOTS, file) });
const brief = (d) => d && { people: d.people.map((b) => b.label + '(' + b.from + ')'), fire: d.fire.map((b) => b.score), door: d.door.map((b) => b.score),
  window: (d.window || []).map((b) => b.score) };
const ONLY = process.env.ONLY ? process.env.ONLY.split(',') : null;
const want = (k) => !ONLY || ONLY.includes(k);
async function tapBox(page, b) {
  const p = await page.evaluate(([nx, ny]) => {
    const st = window.PSCamera.state, c = document.getElementById('view'), r = c.getBoundingClientRect(), fit = st.fit;
    const x = st.src.mirror ? 1 - nx : nx;
    return { x: r.left + (fit.x + x * fit.w) * r.width / c.width, y: r.top + (fit.y + ny * fit.h) * r.height / c.height };
  }, [b.x + b.w / 2, b.y + b.h / 2]);
  await page.mouse.click(p.x, p.y);
}

(async () => {
  let out = {};
  try { out = JSON.parse(fs.readFileSync(path.join(__dirname, 'out', 'e2e_shots.json'))); } catch (e) { out = {}; }
  if (want('window') && MAN.clips.still_window) {  // window: green WINDOW, tap it -> green EXIT on the window; eyepiece (Ironbow)
    const { browser, page, ext } = await open('still_window');
    const d = await waitFor(page, (x) => (x.window || []).length > 0);
    await snap(page, 'window_key_green.png');
    let mark = null;
    if (d && (d.window || []).length) {
      await tapBox(page, d.window[0]);
      await sleep(2500);
      await snap(page, 'window_key_exit_marked.png');
      mark = await page.evaluate(() => { const m = window.PSCamera.state.mark; return { mark: m && { door: m.door, window: m.window, at: m.at }, exit: window.PSCamera.exitInfo(),
        status: document.getElementById('s-exit').textContent, log: window.PSCamera.state.log.slice(0, 3).map((l) => l.text) }; });
      mark.exit = mark.exit && { inView: mark.exit.inView, trusted: mark.exit.trusted };
    }
    await page.click('#eyepiece');
    await sleep(1500);
    await snap(page, 'window_key_eyepiece_ironbow.png');
    const pal = await page.evaluate(() => ({ palette: window.PSCamera.state.palette, button: document.getElementById('palette').textContent }));
    await page.setViewportSize({ width: 390, height: 844 });
    await sleep(1200);
    await page.screenshot({ path: path.join(SHOTS, 'window_page_390.png'), fullPage: false });
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
    out.window = { boxes: brief(d), mark, palette: pal, overflow390: overflow, external: ext.length };
    await browser.close();
  }
  if (want('face')) {  // face: white box with distance; then Mark way out (green EXIT); then eyepiece view
    const { browser, page, ext } = await open('still_face');
    const d = await waitFor(page, (x) => x.people.length > 0);
    await snap(page, 'key_face_white.png');
    await page.click('#mark');
    await sleep(800);
    await snap(page, 'key_exit_marked.png');
    const ex = await page.evaluate(() => window.PSCamera.exitInfo());
    await page.click('#eyepiece');
    await sleep(1500);
    await snap(page, 'key_eyepiece.png');
    out.face = { boxes: brief(d), exit: ex && { inView: ex.inView, trusted: ex.trusted }, external: ext.length };
    await browser.close();
  }
  if (want('group')) {  // group: several white boxes
    const { browser, page, ext } = await open('still_group');
    const d = await waitFor(page, (x) => x.people.length >= 2);
    await snap(page, 'key_group_white.png');
    out.group = { boxes: brief(d), external: ext.length };
    await browser.close();
  }
  if (want('fire')) {  // fire (FireNET test image with a cook): purple FIRE (+ white if the person is found)
    const { browser, page, ext } = await open('still_fire');
    const d = await waitFor(page, (x) => x.fire.length > 0 && x.people.length > 0, 90000);
    await snap(page, 'key_fire_purple.png');
    out.fire = { boxes: brief(d), external: ext.length };
    await browser.close();
  }
  if (want('door')) {  // door: green DOOR, then tap it -> green EXIT on the door
    const { browser, page, ext } = await open('still_door');
    const d = await waitFor(page, (x) => x.door.length > 0);
    await snap(page, 'key_door_green.png');
    let mark = null;
    if (d && d.door.length) {
      const b = d.door[0];
      const p = await page.evaluate(([nx, ny]) => {
        const st = window.PSCamera.state, c = document.getElementById('view'), r = c.getBoundingClientRect(), fit = st.fit;
        const x = st.src.mirror ? 1 - nx : nx;
        return { x: r.left + (fit.x + x * fit.w) * r.width / c.width, y: r.top + (fit.y + ny * fit.h) * r.height / c.height };
      }, [b.x + b.w / 2, b.y + b.h / 2]);
      await page.mouse.click(p.x, p.y);
      await sleep(2500);   // a couple of updates later: still marked, still tracked
      await snap(page, 'key_door_exit_marked.png');
      mark = await page.evaluate(() => ({ mark: window.PSCamera.state.mark && { door: window.PSCamera.state.mark.door }, exit: window.PSCamera.exitInfo(),
        status: document.getElementById('s-exit').textContent }));
      mark.exit = mark.exit && { inView: mark.exit.inView, trusted: mark.exit.trusted };
    }
    out.door = { boxes: brief(d), mark, external: ext.length };
    await browser.close();
  }
  fs.writeFileSync(path.join(__dirname, 'out', 'e2e_shots.json'), JSON.stringify(out, null, 1));
  console.log(JSON.stringify(out, null, 1));
})().catch((e) => { console.error(e); process.exit(1); });
