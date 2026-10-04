#!/usr/bin/env node
// Headless Chromium test of dist/people_demo.html (built by build_assets.py --demo):
//  * which backend comes up (WebGL via SwiftShader here, else CPU);
//  * fake webcam (Y4M) -> detections, ms/frame, tf.memory().numTensors over N frames;
//  * WebGL vs node-CPU parity on fixed images (same raw RGB passed into the page);
//  * no network requests other than file:/data:/blob:.
//   node browser_test.js [--frames 100] [--json out.json]
//   env: ONLY=face,empty,cpu,parity  SWGL=1|0  FACE_FRAMES  EMPTY_FRAMES  CAM_TIMEOUT_MS  RGB_PREFIX
const path = require('path');
const fs = require('fs');
const { execSync } = require('child_process');
const pw = require(path.join(execSync('npm root -g').toString().trim(), 'playwright'));

const HERE = __dirname;
const PAGE = 'file://' + path.join(HERE, '../dist/people_demo.html');
const args = process.argv.slice(2);
const N = +(args[args.indexOf('--frames') + 1] || 100) || 100;
const jsonOut = args.includes('--json') ? args[args.indexOf('--json') + 1] : null;

async function session(clip, backend, fn) {
  const browser = await pw.chromium.launch({
    args: ['--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream',
      '--use-file-for-fake-video-capture=' + clip, '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'],
  });
  const ctx = await browser.newContext();
  const page = await ctx.newPage();
  const external = [];
  page.on('request', (r) => { const u = r.url(); if (!/^(file|data|blob):/.test(u)) external.push(u); });
  const logs = [];
  page.on('console', (m) => logs.push(m.type() + ': ' + m.text()));
  page.on('pageerror', (e) => logs.push('pageerror: ' + e.message));
  const t0 = Date.now();
  await page.goto(PAGE + '?swgl=' + (process.env.SWGL || '1') + (backend ? '&backend=' + backend : ''));
  await page.waitForFunction(() => window.__ps && (window.__ps.ready || window.__ps.error), null, { timeout: 180000 });
  const ready = await page.evaluate(() => ({ backend: window.__ps.backend, info: window.__ps.info, error: window.__ps.error,
    webgl: (() => { try { const c = document.createElement('canvas'); const g = c.getContext('webgl2') || c.getContext('webgl');
      const d = g && g.getExtension('WEBGL_debug_renderer_info'); return g ? (d ? g.getParameter(d.UNMASKED_RENDERER_WEBGL) : 'yes') : null; } catch (e) { return String(e); } })() }));
  ready.pageLoadMs = Date.now() - t0;
  let res;
  try { res = await fn(page, ready); } finally { await browser.close(); }
  return Object.assign({ clip: path.basename(clip), requestedBackend: backend || 'auto', external, logs: logs.slice(0, 20) }, ready, res);
}

async function camera(page, nFrames) {
  await page.click('#go');
  let timedOut = false;
  try {
    await page.waitForFunction((n) => window.__ps.frames >= n, nFrames || N, { timeout: +(process.env.CAM_TIMEOUT_MS || 1200000) });
  } catch (e) { timedOut = true; }   // report the frames done so far
  const r = await page.evaluate(() => {
    const s = window.__ps;
    const tot = s.ms.map((m) => m.total).sort((a, b) => a - b);
    const per = s.ms.map((m) => m.person).sort((a, b) => a - b);
    const fac = s.ms.map((m) => m.face).sort((a, b) => a - b);
    const med = (a) => a[Math.floor(a.length / 2)];
    const tens = s.tensors;
    return { frames: s.frames, msMedian: med(tot), personMsMedian: med(per), faceMsMedian: med(fac),
      tensorsFirst: tens[0], tensorsLast: tens[tens.length - 1], tensorsMin: Math.min(...tens), tensorsMax: Math.max(...tens),
      lastDisplay: s.last ? s.last.display.map((d) => ({ src: d.src, score: +d.score.toFixed(3), label: d.label,
        box: [d.x, d.y, d.w, d.h].map((v) => +v.toFixed(3)) })) : null,
      msAll: s.ms.map((m) => Math.round(m.total)),
      labelCounts: (s.labels || []).reduce((o, l) => { o[l || '(none)'] = (o[l || '(none)'] || 0) + 1; return o; }, {}) };
  });
  r.timedOut = timedOut;
  return r;
}

function loadRGB(file) {
  const S = process.env.RGB_PREFIX;
  const meta = JSON.parse(fs.readFileSync(S + '.json', 'utf8'));
  const im = meta.images.find((q) => q.file === file);
  const fd = fs.openSync(S + '.bin', 'r');
  const n = im.width * im.height * 3;
  const buf = Buffer.alloc(n);
  fs.readSync(fd, buf, 0, n, im.offset);
  return { width: im.width, height: im.height, data: Array.from(buf) };
}

async function nodeCpu(images) {
  const tf = require(path.join(HERE, '../../node_modules/@tensorflow/tfjs'));
  require('../oplist.js');
  require('../people.js');
  await tf.setBackend('cpu');
  const src = fs.readFileSync(path.join(HERE, '../dist/people_assets.js'), 'utf8');
  new Function(src)();
  await PSPeople.init(tf, globalThis.PS_PEOPLE_ASSETS);
  const out = [];
  for (const im of images) {
    const x = tf.tensor3d(Uint8Array.from(im.data), [im.height, im.width, 3], 'int32');
    const r = await PSPeople.detect(x);
    x.dispose();
    out.push(r.detections.map((d) => ({ cls: d.cls, score: d.score, x: d.x, y: d.y, w: d.w, h: d.h })));
  }
  return out;
}

function cmp(a, b) {
  const res = { n: [a.length, b.length], maxCoord: 0, maxScore: 0, unmatched: 0 };
  for (const d of a) {
    const m = b.find((q) => q.cls === d.cls && Math.abs(q.x - d.x) < 0.05 && Math.abs(q.y - d.y) < 0.05);
    if (!m) { res.unmatched++; continue; }
    res.maxCoord = Math.max(res.maxCoord, Math.abs(m.x - d.x), Math.abs(m.y - d.y), Math.abs(m.w - d.w), Math.abs(m.h - d.h));
    res.maxScore = Math.max(res.maxScore, Math.abs(m.score - d.score));
  }
  return res;
}

async function main() {
  const R = { tests: [] };
  const faceClip = path.join(HERE, 'out/face.y4m');
  const emptyClip = path.join(HERE, 'out/empty.y4m');
  const only = process.env.ONLY || 'face,empty,cpu,parity';
  if (only.includes('face')) R.tests.push(await session(faceClip, null, (p) => camera(p, +(process.env.FACE_FRAMES || N))));
  if (only.includes('empty')) R.tests.push(await session(emptyClip, null, (p) => camera(p, +(process.env.EMPTY_FRAMES || 30))));
  if (only.includes('cpu')) R.tests.push(await session(faceClip, 'cpu', (p) => camera(p, 30)));
  // WebGL vs node CPU on fixed images
  if (process.env.RGB_PREFIX && only.includes('parity')) {
    const files = ['mediapipe/portrait.jpg', 'mediapipe/man-woman-okay.jpg', 'mediapipe/face_stylizer_raw_face_demo.png', 'mediapipe/pose.jpg'];
    const imgs = files.map(loadRGB);
    const ref = await nodeCpu(imgs);
    const pr = await session(emptyClip, null, async (page) => {
      const got = [];
      for (const im of imgs) {
        got.push(await page.evaluate(async (im) => {
          const x = tf.tensor3d(Uint8Array.from(im.data), [im.height, im.width, 3], 'int32');
          const r = await PSPeople.detect(x);
          x.dispose();
          return r.detections.map((d) => ({ cls: d.cls, score: d.score, x: d.x, y: d.y, w: d.w, h: d.h }));
        }, im));
      }
      return { parity: files.map((f, i) => Object.assign({ file: f }, cmp(ref[i], got[i]))) };
    });
    R.tests.push(pr);
  }
  const s = JSON.stringify(R, null, 1);
  console.log(s);
  if (jsonOut) fs.writeFileSync(jsonOut, s);
}
main().catch((e) => { console.error(e); process.exit(1); });
