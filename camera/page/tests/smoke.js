#!/usr/bin/env node
// Quick headless smoke run: node tests/smoke.js CLIP.y4m [page.html] [out.png]
const path = require('path');
const { execSync } = require('child_process');
const pw = require(path.join(execSync('npm root -g').toString().trim(), 'playwright'));
(async () => {
  const clip = path.resolve(process.argv[2]);
  const page_ = path.resolve(process.argv[3] || path.join(__dirname, '../dist/pyrosight_camera.html'));
  const out = process.argv[4] || path.join(__dirname, '../shots/smoke.png');
  const browser = await pw.chromium.launch({ args: ['--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream',
    '--use-file-for-fake-video-capture=' + clip, '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] });
  const ctx = await browser.newContext({ viewport: { width: 1280, height: 900 } });
  const page = await ctx.newPage();
  page.on('console', (m) => console.log('console.' + m.type() + ': ' + m.text().slice(0, 300)));
  page.on('pageerror', (e) => console.log('pageerror: ' + e.message));
  page.on('request', (r) => { if (!/^(file|data|blob):/.test(r.url())) console.log('EXTERNAL REQUEST ' + r.url()); });
  const t0 = Date.now();
  await page.goto('file://' + page_ + (process.env.Q || ''));
  await page.click('#start');
  await page.waitForFunction(() => window.PSCamera && (window.PSCamera.state.stats.inferences >= 3 || window.PSCamera.state.loadError), null, { timeout: 300000 });
  const st = await page.evaluate(() => {
    const s = window.PSCamera.state;
    return { backend: s.backend, info: s.backendInfo, loadMs: s.loadMs, inf: s.stats.inferences, tensors: s.stats.tensors, ms: s.stats.ms.slice(-3),
      labels: s.stats.labels, err: s.loadError && String(s.loadError), errors: s.stats.errors, src: s.src && { w: s.src.w, h: s.src.h, mirror: s.src.mirror, name: s.src.name },
      track: s.stats.trackStates, pose: window.PSCamera.tracker.pose, fd: s.fd && s.fd.name };
  });
  console.log(JSON.stringify(st, null, 1));
  console.log('elapsed', (Date.now() - t0) / 1000, 's');
  await page.screenshot({ path: out, fullPage: true });
  await browser.close();
})().catch((e) => { console.error(e); process.exit(1); });
