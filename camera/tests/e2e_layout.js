#!/usr/bin/env node
// Layout / theming check of the integrated page: phone width (390 px) in dark mode and a
// desktop light page, with the fake camera on the doors clip. Screenshots -> camera/shots.
//   taskset -c 2,3 node tests/e2e_layout.js
const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');
const pw = require(path.join(execSync('npm root -g').toString().trim(), 'playwright'));
const CAMERA = path.join(__dirname, '..');
const DIST = path.join(CAMERA, 'dist', 'pyrosight_camera.html');
const MAN = JSON.parse(fs.readFileSync(path.join(__dirname, 'out', 'e2e_clips.json')));
const SHOTS = path.join(CAMERA, 'shots');

(async () => {
  const out = {};
  for (const [tag, vp, dark, mobile, clip] of [['mobile_dark', { width: 390, height: 844 }, true, true, 'fire'], ['desktop_light', { width: 1280, height: 900 }, false, false, 'group']]) {
    const browser = await pw.chromium.launch({ args: ['--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream',
      '--use-file-for-fake-video-capture=' + path.join(MAN.clips[clip].dir || MAN.clips_dir, MAN.clips[clip].file), '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] });
    const ctx = await browser.newContext({ viewport: vp, deviceScaleFactor: mobile ? 2 : 1, isMobile: mobile, hasTouch: mobile, colorScheme: dark ? 'dark' : 'light' });
    const page = await ctx.newPage();
    const ext = [], errs = [];
    page.on('request', (r) => { if (!/^(file|data|blob):/.test(r.url())) ext.push(r.url()); });
    page.on('pageerror', (e) => errs.push(e.message));
    await page.goto('file://' + DIST);
    await page.screenshot({ path: path.join(SHOTS, 'e2e_' + tag + '_before_start.png'), fullPage: true });
    await page.click('#start');
    await page.waitForFunction(() => window.__psLastDetections && window.__psLastDetections.i >= 3, null, { timeout: 300000 });
    await page.waitForTimeout(500);
    await page.screenshot({ path: path.join(SHOTS, 'e2e_' + tag + '.png'), fullPage: true });
    out[tag] = await page.evaluate(() => ({
      overflowX: document.documentElement.scrollWidth - window.innerWidth,
      bg: getComputedStyle(document.body).backgroundColor,
      title: document.title,
      last: window.__psLastDetections && { people: window.__psLastDetections.people.length, fire: window.__psLastDetections.fire.length, door: window.__psLastDetections.door.length },
      fd: document.getElementById('s-fd').textContent,
      privacy: document.querySelector('.privacy').textContent,
    }));
    out[tag].external = ext; out[tag].errors = errs;
    console.log(tag, JSON.stringify(out[tag]));
    await browser.close();
  }
  fs.writeFileSync(path.join(__dirname, 'out', 'e2e_layout.json'), JSON.stringify(out, null, 1));
})().catch((e) => { console.error(e); process.exit(1); });
