#!/usr/bin/env node
// Final screenshots of the PyroSight Camera page (camera/shots/final_*.png) and a pixel check of the
// box colours: person WHITE (255,255,255), fire PURPLE (200,80,255), exit / door GREEN (40,255,80).
//   node tests/final_shots.js [--page dist/pyrosight_camera.html] [--clips DIR] [--out shots]
// Phone 390x844 dark (touch, dpr 2) and desktop 1280x800 light: the navigation demo without a camera
// (simulated eyepiece view), then the camera on the still face clip with the navigation demo running.
// Fire and door colours come from the still fire / still door clips (desktop). Results: tests/out/final_shots.json
const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');
const pw = require(path.join(execSync('npm root -g').toString().trim(), 'playwright'));

const CAM = path.join(__dirname, '..');
const argv = process.argv.slice(2);
const opt = (k, d) => { const i = argv.indexOf(k); return i >= 0 ? argv[i + 1] : d; };
const PAGE = path.resolve(CAM, opt('--page', 'dist/pyrosight_camera.html'));
const CLIPS = opt('--clips', '/tmp/claude-0/-home-claude/069cc723-8434-5b03-8db8-df5ff0accd4e/scratchpad/clips');
const OUT = path.resolve(CAM, opt('--out', 'shots'));
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const COLORS = { person: [255, 255, 255], fire: [200, 80, 255], exit: [40, 255, 80] };

async function launch(clip) {
  const args = ['--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'];
  if (clip) args.push('--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream', '--use-file-for-fake-video-capture=' + path.join(CLIPS, clip + '.y4m'));
  return pw.chromium.launch({ args });
}
async function open(browser, phone) {
  const ctx = await browser.newContext(phone ?
    { viewport: { width: 390, height: 844 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true, colorScheme: 'dark' } :
    { viewport: { width: 1280, height: 800 }, deviceScaleFactor: 1, colorScheme: 'light' });
  await ctx.addInitScript(() => { try { speechSynthesis.speak = () => {}; } catch (e) { /* none */ } });
  const page = await ctx.newPage();
  const rec = { external: [], errors: [] };
  page.on('request', (r) => { if (!/^(file|data|blob):/.test(r.url())) rec.external.push(r.url()); });
  page.on('pageerror', (e) => rec.errors.push(e.message));
  await page.goto('file://' + PAGE);
  return { ctx, page, rec };
}
// exact-colour pixels drawn in the picture canvas (boxes and their labels), per colour
const canvasColours = (page) => page.evaluate((C) => {
  const c = document.getElementById('view'), d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
  const n = {};
  for (const k in C) n[k] = 0;
  for (let i = 0; i < d.length; i += 4) {
    if (d[i + 3] < 250) continue;
    for (const k in C) if (d[i] === C[k][0] && d[i + 1] === C[k][1] && d[i + 2] === C[k][2]) n[k]++;
  }
  return n;
}, COLORS);
// the same colours in the screenshot (what the viewer sees): pixels within 3 of each colour inside the
// picture area (decoded with Python / Pillow)
function shotColours(file, rect, dpr) {
  const py = `
import json, sys
from PIL import Image
im = Image.open(sys.argv[1]).convert('RGB'); r = json.loads(sys.argv[2]); d = float(sys.argv[3])
C = {'person': (255, 255, 255), 'fire': (200, 80, 255), 'exit': (40, 255, 80)}
x0, y0 = max(0, int(r['x'] * d)), max(0, int(r['y'] * d))
x1, y1 = min(im.width, int((r['x'] + r['width']) * d)), min(im.height, int((r['y'] + r['height']) * d))
n = {k: 0 for k in C}
px = im.load()
for y in range(y0, y1):
    for x in range(x0, x1):
        p = px[x, y]
        for k, c in C.items():
            if abs(p[0] - c[0]) <= 3 and abs(p[1] - c[1]) <= 3 and abs(p[2] - c[2]) <= 3: n[k] += 1
print(json.dumps(n))`;
  try { return JSON.parse(execSync('python3 -c ' + JSON.stringify(py) + ' ' + JSON.stringify(file) + " '" + JSON.stringify(rect) + "' " + dpr).toString()); } catch (e) { return null; }
}
async function scrollPicture(page) {
  await page.evaluate(() => { const r = document.getElementById('screen').getBoundingClientRect(); window.scrollBy(0, r.top - 8); });
  await sleep(300);
}
async function shot(page, name, dpr) {
  const file = path.join(OUT, name);
  await page.screenshot({ path: file });
  const rect = await page.evaluate(() => { const r = document.getElementById('screen').getBoundingClientRect(); return { x: r.x, y: r.y, width: r.width, height: r.height }; });
  return { file: path.relative(CAM, file), shotColours: shotColours(file, rect, dpr) };
}
const status = (page) => page.evaluate(() => {
  const s = window.PSCamera.state, n = window.PSCamera.nav, u = window.PSCamera.navUi, d = window.__psLastDetections;
  const r = (id) => { const b = document.getElementById(id).getBoundingClientRect(); return { top: Math.round(b.top), bottom: Math.round(b.bottom) }; };
  return { engine: s.engineMode, backend: s.backend, src: s.src && s.src.kind, nav: n && { mode: n.input.mode, phase: n.walker && n.walker.phase, level: n.g.level },
    overlay: u.overlay && { kind: u.overlay.kind, label: u.overlay.label }, last: d && { people: d.people.map((p) => p.label), fire: d.fire.length, door: d.door.length, ms: d.ms.total },
    screen: r('screen'), camPad: r('cam-pad'), overflow: document.documentElement.scrollWidth - innerWidth, vh: innerHeight };
});

(async () => {
  fs.mkdirSync(OUT, { recursive: true });
  const res = {};
  for (const phone of [true, false]) {
    const tag = phone ? 'mobile_dark' : 'desktop_light', dpr = phone ? 2 : 1;
    // 1. navigation demo without a camera (as inside claude.ai): simulated eyepiece view
    let b = await launch(null);
    let { page, rec } = await open(b, phone);
    await sleep(1500);
    await page.click('#start');                  // no camera on this browser: the notice
    await page.waitForSelector('#notice:not([hidden])', { timeout: 20000 });
    await page.click('#n-nav');                  // "Try the navigation demo": auto demo
    await page.evaluate(() => { window.PSCamera.nav.setSpeed(4); });
    await page.waitForFunction(() => { const o = window.PSCamera.navUi.overlay; return o && (o.kind === 'box' || o.kind === 'route') && window.PSCamera.nav.walker.phase === 'outbound'; }, null, { timeout: 120000 }).catch(() => {});
    await scrollPicture(page);
    let s = await status(page);
    res[tag + '_nav_demo'] = Object.assign({ status: s, canvas: await canvasColours(page) }, await shot(page, 'final_' + tag + '_nav_demo.png', dpr), { rec });
    await b.close();
    // 2. camera on the still face clip, navigation demo running
    b = await launch('e2e_still_face');
    ({ page, rec } = await open(b, phone));
    await page.click('#start');
    await page.waitForFunction(() => { const d = window.__psLastDetections; return d && d.people.length > 0; }, null, { timeout: 120000 });
    await page.click('#nav-demo');
    await page.evaluate(() => window.PSCamera.nav.turn(180));   // face the way in: the EXIT box
    await sleep(phone ? 2500 : 2500);
    await page.evaluate(() => window.PSCamera.nav.turn(180));   // and back: edge arrow
    await page.evaluate(() => window.PSCamera.nav.walk(true));
    await sleep(2500);
    await page.evaluate(() => { window.PSCamera.nav.walk(false); window.PSCamera.nav.turn(180); });
    await sleep(2000);
    await scrollPicture(page);
    s = await status(page);
    res[tag + '_camera_face'] = Object.assign({ status: s, canvas: await canvasColours(page) }, await shot(page, 'final_' + tag + '_camera_face_nav.png', dpr), { rec });
    await b.close();
  }
  // fire and door colours (desktop)
  for (const [clip, key] of [['e2e_still_fire', 'fire'], ['e2e_still_door', 'door']]) {
    const b = await launch(clip);
    const { page, rec } = await open(b, false);
    await page.click('#start');
    await page.waitForFunction((k) => { const d = window.__psLastDetections; return d && d[k].length > 0; }, key, { timeout: 120000 }).catch(() => {});
    await sleep(500);
    await scrollPicture(page);
    const s = await status(page);
    res['desktop_light_' + key] = Object.assign({ status: s, canvas: await canvasColours(page) }, await shot(page, 'final_desktop_light_' + key + '.png', 1), { rec });
    await b.close();
  }
  const ok = {
    personWhite: res.mobile_dark_camera_face.canvas.person > 100 && res.desktop_light_camera_face.canvas.person > 100,
    firePurple: res.desktop_light_fire.canvas.fire > 100,
    doorGreen: res.desktop_light_door.canvas.exit > 100,
    exitGreen: res.mobile_dark_nav_demo.canvas.exit > 100 && res.desktop_light_nav_demo.canvas.exit > 100,
    noRequests: Object.values(res).every((r) => !r.rec.external.length && !r.rec.errors.length),
  };
  res.ok = ok;
  fs.writeFileSync(path.join(CAM, 'tests', 'out', 'final_shots.json'), JSON.stringify(res, null, 1));
  console.log(JSON.stringify(res, (k, v) => (k === 'rec' ? { external: v.external.length, errors: v.errors } : v), 1));
  process.exit(Object.values(ok).every(Boolean) ? 0 : 1);
})().catch((e) => { console.error(e); process.exit(2); });
