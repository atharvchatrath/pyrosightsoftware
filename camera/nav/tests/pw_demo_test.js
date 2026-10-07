/*
 * Playwright: the standalone demo page (dist/nav_demo.html = demo.html with a
 * doctype wrapper) in Chromium at phone width.
 *
 *   node camera/nav/tests/pw_demo_test.js [--shots DIR]
 *
 * Checks: loads without errors, 390 px wide without horizontal scroll, touch
 * targets >= 44 px; with no sensor events the page says so and falls back to
 * the demo walk; Walk / Left / Right buttons and keyboard (arrows, WASD)
 * move the walker; auto demo runs to the end; heading-up toggle; emulated
 * deviceorientation + devicemotion drive steps and turns; a 5 s main-thread
 * freeze does not collapse the estimate; iOS-style permission flow (denied /
 * granted); dark theme. No real sensors, phone or GPU here.
 */
'use strict';
const path = require('path');
const fs = require('fs');
const pw = require(require('child_process').execSync('npm root -g').toString().trim() + '/playwright');

const PAGE = 'file://' + path.join(__dirname, '..', 'dist', 'nav_demo.html');
const si = process.argv.indexOf('--shots');
const SHOTS = si > 0 ? process.argv[si + 1] : null;
if (SHOTS) fs.mkdirSync(SHOTS, { recursive: true });

let failures = 0, checks = 0;
function check(label, cond, info) {
  checks++;
  console.log((cond ? 'ok   ' : 'FAIL ') + label + (info !== undefined ? '  [' + (typeof info === 'string' ? info : JSON.stringify(info)) + ']' : ''));
  if (!cond) failures++;
}
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function newPage(browser, opts) {
  opts = opts || {};
  const ctx = await browser.newContext({ viewport: { width: 390, height: 844 }, deviceScaleFactor: 2, colorScheme: opts.dark ? 'dark' : 'light' });
  if (opts.init) await ctx.addInitScript(opts.init);
  const page = await ctx.newPage();
  const errors = [];
  page.on('pageerror', (e) => errors.push(String(e)));
  page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });
  page.on('request', (r) => { if (!r.url().startsWith('file:') && !r.url().startsWith('data:')) errors.push('request ' + r.url()); });
  await page.goto(PAGE);
  return { ctx, page, errors };
}

// In-page emulated phone: deviceorientation + devicemotion at 60 Hz. plan = [{dur, walkHz, turnDps}]
const SENSOR_SCRIPT = (plan) => {
  return new Promise((resolve) => {
    let alpha = 10, seg = 0, segT0 = performance.now(), last = performance.now(), phase = 0;
    const G = 9.81;
    const id = setInterval(() => {
      const now = performance.now(), dt = (now - last) / 1000; last = now;
      while (seg < plan.length && (now - segT0) / 1000 > plan[seg].dur) { segT0 += plan[seg].dur * 1000; seg++; }
      if (seg >= plan.length) { clearInterval(id); resolve(alpha); return; }
      const p = plan[seg];
      alpha = (alpha + (p.turnDps || 0) * dt + 360) % 360;
      window.dispatchEvent(new DeviceOrientationEvent('deviceorientation', { alpha, beta: 80, gamma: 2, absolute: false }));
      let vert = 0;
      if (p.walkHz) { phase += 2 * Math.PI * p.walkHz * dt; vert = 2.2 * Math.sin(phase) + 0.7 * Math.sin(2 * phase + 0.3) + 0.15 * (Math.random() - 0.5); }
      else vert = 0.08 * (Math.random() - 0.5);
      // phone upright (beta 80): gravity mostly along device +y
      const b = 80 * Math.PI / 180;
      window.dispatchEvent(new DeviceMotionEvent('devicemotion', {
        accelerationIncludingGravity: { x: 0.1 * (Math.random() - 0.5), y: (G + vert) * Math.sin(b), z: (G + vert) * Math.cos(b) },
        acceleration: { x: 0, y: vert * Math.sin(b), z: vert * Math.cos(b) },
        rotationRate: { alpha: 0, beta: 0, gamma: 0 }, interval: 16,
      }));
    }, 16);
  });
};

(async () => {
  const browser = await pw.chromium.launch();
  const t0 = Date.now();

  // ---------------------------------------------------------------- A/B: load, layout, no sensors
  {
    const { ctx, page, errors } = await newPage(browser);
    check('title', (await page.title()) === 'PyroSight Navigation Demo');
    const tStart = Date.now();
    await page.waitForFunction(() => window.__psNav && window.__psNav.input.mode === 'demo', null, { timeout: 8000 });
    const fallbackMs = Date.now() - tStart;
    const inp = await page.evaluate(() => window.__psNav.input);
    check('no sensor events -> demo walk chosen automatically', inp.mode === 'demo' && inp.status === 'ok', fallbackMs + ' ms');
    const detail = await page.textContent('#inDetail');
    check('sensor-absent message shown', /No motion sensor data arrived/.test(detail), detail.slice(0, 90));
    check('indicator says demo walk', /Demo walk/.test(await page.textContent('#inLabel')));
    const lay = await page.evaluate(() => {
      const small = [];
      for (const el of document.querySelectorAll('button, select, label.inline')) {
        const r = el.getBoundingClientRect();
        if (r.height < 44 - 0.5) small.push(el.id || el.textContent.trim().slice(0, 20) + ':' + r.height);
      }
      return { sw: document.documentElement.scrollWidth, cw: document.documentElement.clientWidth, small };
    });
    check('390 px: no horizontal scroll', lay.sw <= 390, lay);
    check('touch targets >= 44 px', lay.small.length === 0, lay.small);
    const entry = await page.evaluate(() => window.__psNav.g.state);
    check('demo walk starts with the entry marked', entry === 'tracking');

    // ---------------------------------------------------------------- C: buttons (manual demo runs in real time)
    const w180 = (a) => ((a + 540) % 360) - 180;
    const st = () => page.evaluate(() => ({ steps: __psNav.g.steps, yaw: __psNav.g.yaw * 180 / Math.PI, wyaw: __psNav.walker.yaw * 180 / Math.PI, x: __psNav.walker.x, y: __psNav.walker.y }));
    const holdBtn = async (sel, ms) => {
      await page.locator(sel).scrollIntoViewIfNeeded();
      const b = await page.locator(sel).boundingBox();
      await page.mouse.move(b.x + b.width / 2, b.y + b.height / 2);
      await page.mouse.down(); await sleep(ms); await page.mouse.up();
    };
    await page.click('#btnDemo');        // fresh walker at the door, facing down corridor A
    let s0 = await st();
    await holdBtn('#btnWalk', 2000); await sleep(300);
    let s1 = await st();
    check('hold "Hold to walk" 2 s -> steps counted, walker moved', s1.steps - s0.steps >= 3 && s1.x - s0.x > 1.5, { steps: s1.steps - s0.steps, moved: +(s1.x - s0.x).toFixed(2) });
    await page.click('#btnLeft'); await sleep(800);
    let s2 = await st();
    check('tap Left -> turns 15 deg left (true and estimated yaw)', Math.abs(w180(s2.wyaw - s1.wyaw) - 15) < 1.5 && Math.abs(w180(s2.yaw - s1.yaw) - 15) < 3, { true: +w180(s2.wyaw - s1.wyaw).toFixed(1), est: +w180(s2.yaw - s1.yaw).toFixed(1) });
    await holdBtn('#btnRight', 1300); await sleep(500);
    let s3 = await st();
    const turned = -w180(s3.wyaw - s2.wyaw);
    check('hold Right 1.3 s -> keeps turning right (15 deg tap + 90 deg/s)', turned > 80 && turned < 130, +turned.toFixed(1));

    // ---------------------------------------------------------------- D: keyboard
    await page.click('#btnDemo');
    await page.focus('body');
    s0 = await st();
    await page.keyboard.down('ArrowUp'); await sleep(1500); await page.keyboard.up('ArrowUp'); await sleep(300);
    s1 = await st();
    check('keyboard ArrowUp held -> walks', s1.steps - s0.steps >= 2 && s1.x - s0.x > 1, s1.steps - s0.steps);
    await page.keyboard.press('a'); await sleep(700);
    s2 = await st();
    check('keyboard A tap -> 15 deg left', Math.abs(w180(s2.wyaw - s1.wyaw) - 15) < 1.5, +w180(s2.wyaw - s1.wyaw).toFixed(1));
    await page.keyboard.down('d'); await sleep(1000); await page.keyboard.up('d'); await sleep(500);
    s3 = await st();
    check('keyboard D held -> continuous right turn', -w180(s3.wyaw - s2.wyaw) > 50, +(-w180(s3.wyaw - s2.wyaw)).toFixed(1));
    await page.keyboard.press('ArrowLeft'); await page.keyboard.press('ArrowLeft'); await page.keyboard.press('ArrowLeft'); await sleep(800);
    s0 = await st();
    await page.keyboard.down('w'); await sleep(1200); await page.keyboard.up('w'); await sleep(300);
    s1 = await st();
    check('keyboard W held -> walks', s1.steps - s0.steps >= 2, s1.steps - s0.steps);

    // ---------------------------------------------------------------- F: heading-up toggle
    await page.click('#btnUp');
    check('map heading-up toggle', (await page.evaluate(() => window.__psNavUi.map.o.up)) === 'heading' && /Entry direction up/.test(await page.textContent('#btnUp')));
    await page.click('#btnUp');

    // ---------------------------------------------------------------- E: auto demo
    await page.selectOption('#speed', '8');
    await page.evaluate(() => { window.__kinds = {}; setInterval(() => { const b = window.__psNavUi.lastBox; if (b) window.__kinds[b.kind] = (window.__kinds[b.kind] || 0) + 1; }, 50); });
    await page.click('#btnAuto');
    const tA = Date.now();
    await page.waitForFunction(() => __psNav.walker && __psNav.walker.phase === 'done', null, { timeout: 60000 });
    const auto = await page.evaluate(() => ({ r: __psNav.walker.result, stats: __psNav.stats, text: document.getElementById('result').textContent,
      alerts: __psNav.alertLog.map((a) => a.text), kinds: window.__kinds, simS: __psNav.coreT / 1000 }));
    check('auto demo runs to the end', !!auto.r && /from the real door/.test(auto.text), auto.text + ' (' + ((Date.now() - tA) / 1000).toFixed(1) + ' s wall, ' + auto.simS.toFixed(0) + ' s simulated)');
    check('auto demo spoke "Way out is ..."', auto.alerts.some((a) => /^Way out is/.test(a)), auto.alerts.slice(-4));
    check('camera picture showed an EXIT box during the walk out', (auto.kinds.box || 0) > 0, auto.kinds);
    if (SHOTS) await page.screenshot({ path: path.join(SHOTS, 'demo_light_after_auto.png'), fullPage: true });

    // frame cost (map + arrow + camera mock + update)
    const cost = await page.evaluate(() => {
      const ui = window.__psNavUi, n = 120, t = performance.now();
      for (let i = 0; i < n; i++) { __psNav.update(); ui.map.draw(__psNav.snapshot()); ui.arrow.draw(__psNav.g); }
      return (performance.now() - t) / n;
    });
    check('per-frame cost (update + map + arrow) measured', cost < 50, cost.toFixed(2) + ' ms/frame');
    check('no page errors / requests', errors.length === 0, errors);
    await ctx.close();
  }

  // ---------------------------------------------------------------- G: emulated phone sensors (Android-like)
  {
    const { ctx, page, errors } = await newPage(browser);
    await page.waitForFunction(() => window.__psNav && window.__psNav.input.mode === 'demo', null, { timeout: 8000 });
    await page.click('#btnSensors');
    const plan = [{ dur: 1.0 }, { dur: 5.0, walkHz: 1.8 }, { dur: 1.0 }, { dur: 2.0, turnDps: 45 }, { dur: 1.0 }, { dur: 5.0, walkHz: 1.8 }, { dur: 1.0 }];
    const done = page.evaluate(SENSOR_SCRIPT, plan);
    await page.waitForFunction(() => __psNav.input.mode === 'sensors' && __psNav.input.status === 'ok', null, { timeout: 4000 });
    const inp = await page.evaluate(() => __psNav.input);
    check('emulated sensors -> input "Phone motion sensors" ok', /Phone motion sensors/.test(await page.textContent('#inLabel')), inp.detail);
    await done;
    await sleep(200);
    const g = await page.evaluate(() => ({ steps: __psNav.g.steps, yaw: __psNav.g.yaw * 180 / Math.PI, word: __psNav.g.word, home: __psNav.g.homeWord,
      state: __psNav.g.state, rb: __psNav.g.routeBearingDeg, hb: __psNav.g.homeBearingDeg, det: __psNav.sensors.stats }));
    check('emulated walking 2 x 5 s at 1.8 Hz -> ~18 steps', Math.abs(g.steps - 18) <= 2, g.steps);
    check('emulated alpha +90 (left turn) -> estimated heading +90', Math.abs(g.yaw - 90) < 6, +g.yaw.toFixed(1));
    check('walk, turn left, walk -> route "behind", straight line "behind-left"', g.word === 'behind' && g.home === 'behind-left', { route: g.rb.toFixed(1), home: g.hb.toFixed(1), word: g.word, homeWord: g.home });
    // a 5 s main-thread freeze is not a sensor outage
    const before = await page.evaluate(() => ({ c: __psNav.g.confidence, s: __psNav.g.state }));
    const feeder = page.evaluate(SENSOR_SCRIPT, [{ dur: 9.0 }]);
    await sleep(500);
    await page.evaluate(() => { const t = performance.now(); while (performance.now() - t < 5000) { /* busy */ } });
    await sleep(1500);
    const after = await page.evaluate(() => ({ c: __psNav.g.confidence, s: __psNav.g.state, stalls: __psNav.stalls, input: __psNav.input.status }));
    check('5 s page freeze: still tracking, confidence kept, sensors still ok', after.s === 'tracking' && before.c - after.c < 0.05 && after.input === 'ok' && after.stalls >= 1, { before, after });
    await feeder;
    // silence -> lost
    await sleep(2200);
    const lost = await page.evaluate(() => ({ st: __psNav.input.status, g: __psNav.g.state, msg: __psNav.input.detail }));
    check('sensor events stop -> "lost" shown and the device state goes LOST', lost.st === 'lost' && lost.g === 'lost', lost);
    if (SHOTS) await page.screenshot({ path: path.join(SHOTS, 'sensors_light.png'), fullPage: true });
    check('no page errors (sensor run)', errors.length === 0, errors);
    await ctx.close();
  }

  // ---------------------------------------------------------------- H: explicit "Use phone sensors" with nothing arriving
  {
    const { ctx, page } = await newPage(browser);
    await page.waitForFunction(() => window.__psNav && window.__psNav.input.mode === 'demo', null, { timeout: 8000 });
    await page.click('#btnSensors');
    await page.waitForFunction(() => __psNav.input.mode === 'sensors' && __psNav.input.status === 'blocked', null, { timeout: 4000 });
    const t = await page.textContent('#inDetail');
    check('"Use phone sensors" with no data -> plain message after ~1.5 s', /No motion sensor data arrived/.test(t), t.slice(0, 80));
    await ctx.close();
  }

  // ---------------------------------------------------------------- I: iOS-style permission
  for (const answer of ['denied', 'granted']) {
    const init = `(() => { const a = ${JSON.stringify(answer)};
      window.DeviceMotionEvent.requestPermission = () => Promise.resolve(a);
      window.DeviceOrientationEvent.requestPermission = () => Promise.resolve(a); })();`;
    const { ctx, page, errors } = await newPage(browser, { init });
    await page.waitForFunction(() => window.__psNav && window.__psNav.input.mode === 'demo', null, { timeout: 4000 });
    const d0 = await page.textContent('#inDetail');
    if (answer === 'denied') check('iOS-style: load starts the demo and says to tap for sensors', /Tap "Use phone sensors"/.test(d0), d0.slice(0, 90));
    await page.click('#btnSensors');
    if (answer === 'denied') {
      await page.waitForFunction(() => __psNav.input.status === 'denied', null, { timeout: 3000 });
      check('iOS-style permission denied -> message', /refused/.test(await page.textContent('#inDetail')));
    } else {
      const done = page.evaluate(SENSOR_SCRIPT, [{ dur: 1.0 }, { dur: 3.0, walkHz: 1.8 }]);
      await page.waitForFunction(() => __psNav.input.mode === 'sensors' && __psNav.input.status === 'ok', null, { timeout: 4000 });
      await done;
      const steps = await page.evaluate(() => __psNav.g.steps);
      check('iOS-style permission granted -> sensors drive steps', Math.abs(steps - 5.4) <= 2, steps);
    }
    check('no page errors (permission ' + answer + ')', errors.length === 0, errors);
    await ctx.close();
  }

  // ---------------------------------------------------------------- J: dark theme
  {
    const { ctx, page, errors } = await newPage(browser, { dark: true });
    await page.waitForFunction(() => window.__psNav && window.__psNav.input.mode === 'demo', null, { timeout: 8000 });
    await page.selectOption('#speed', '8');
    await page.click('#btnAuto');
    await sleep(4000);
    const bg = await page.evaluate(() => getComputedStyle(document.body).backgroundColor);
    check('dark theme applies (body background dark)', /rgb\((\d+), (\d+), (\d+)\)/.test(bg) && +bg.match(/\d+/)[0] < 40, bg);
    if (SHOTS) await page.screenshot({ path: path.join(SHOTS, 'demo_dark_inbound.png'), fullPage: true });
    check('no page errors (dark)', errors.length === 0, errors);
    await ctx.close();
  }

  await browser.close();
  console.log(`${checks - failures}/${checks} checks passed in ${((Date.now() - t0) / 1000).toFixed(0)} s`);
  console.log(failures ? 'FAIL playwright demo page' : 'PASS playwright demo page');
  process.exit(failures ? 1 : 0);
})().catch((e) => { console.error(e); process.exit(1); });
