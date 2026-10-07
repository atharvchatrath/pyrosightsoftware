/*
 * PyroSight Camera detection engine (engine.js): runs the three detectors (person, face, fire/door/window)
 * and hands back plain box lists. Plain script, no bundler; defines globalThis.PSEngine. Needs TF.js,
 * oplist.js (PSOpList), people.js (PSPeople), the embedded models (PS_PEOPLE_ASSETS,
 * PS_OPLIST_ASSETS.firedoor) and firedoor/decode.js (FireDoorDecode) + PS_FIREDOOR_META.
 *
 * It runs in one of two places (page/app.js decides):
 *   - a dedicated Web Worker made from a Blob of the page's own script texts (normal case): the page's
 *     main thread never parses TF.js, never decodes the models and never waits for a detector, so taps
 *     are handled at once. PSEngine.serveWorker() is the worker's message loop (protocol below).
 *   - the page's main thread (fallback, when a Content-Security-Policy or the browser refuses the
 *     worker): PSEngine.create({coop: true}) gives the event loop a turn every few tens of
 *     milliseconds while loading and while detecting (op by op; the person graph in short stages).
 *
 * Worker protocol (all plain data; frames are transferred, not copied):
 *   page -> worker  {type: 'init', opts: {prefer, allowSoftwareWebGL, warmup}}
 *                   {type: 'detect', id, frame: {buf: ArrayBuffer RGBA, w, h}, opts: {hfovDeg, person, zoom, fd}}
 *                   {type: 'reset'}           (new picture source: forget the fire hysteresis)
 *                   {type: 'debug', what: 'lose-context' | 'hang', ms}   (tests only)
 *   worker -> page  {type: 'boot'}            (script parsed and running)
 *                   {type: 'backend', info}   (TF.js backend chosen; models still loading)
 *                   {type: 'ready', info} | {type: 'failed', error}
 *                   {type: 'result', id, res, buf} | {type: 'error', id, error, lost}   (buf: the frame, handed back)
 *                   {type: 'lost'}            (the WebGL context was lost: the page restarts the worker)
 *                   {type: 'beat'}            (every second while the worker's event loop runs: liveness)
 * res = {detections, display, width, height, ms: {person, face, firedoor, total}, fd: [boxes], numTensors}
 */
(function (root) {
  'use strict';

  const now = () => (typeof performance !== 'undefined' ? performance : Date).now();

  function iou(a, b) {
    const ix = Math.max(0, Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x));
    const iy = Math.max(0, Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y));
    const i = ix * iy, u = a.w * a.h + b.w * b.h - i;
    return u > 0 ? i / u : 0;
  }

  // ------------------------------------------------------------ cooperative yielding
  /**
   * makeYielder(budgetMs) -> async y(force): gives the event loop a turn (input, rendering, timers)
   * when budgetMs have passed since the last turn, or always with force: a MessageChannel message
   * (no 4 ms timer clamping), and at least every 50 ms a rendered frame (requestAnimationFrame,
   * then a task). Resolves true when it yielded.
   */
  function makeYielder(budgetMs) {
    let last = now(), lastFrame = now(), lastStack = null;
    // a plain task turn: a MessageChannel message (no 4 ms timer clamping). Not scheduler.yield():
    // its continuations outrank rendering in Chromium, so taps were handled but the screen did not
    // update for seconds (measured: click handled after 12 ms, next frame after 5.9 s).
    let msgTurn;
    if (typeof MessageChannel !== 'undefined') {
      const ch = new MessageChannel(), q = [];
      ch.port1.onmessage = () => { const r = q.shift(); if (r) r(); };
      msgTurn = () => new Promise((r) => { q.push(r); ch.port2.postMessage(0); });
    } else msgTurn = () => new Promise((r) => setTimeout(r, 0));
    // at least one rendered frame every FRAME_MS of work, so a tap's visible response is not held
    // back (rAF, then a task after the frame; a timer as well, for hidden pages where rAF stops)
    const FRAME_MS = 50;
    const frameTurn = typeof requestAnimationFrame === 'function' ? () => new Promise((r) => {
      let done = false;
      const fin = () => { if (!done) { done = true; r(); } };
      requestAnimationFrame(() => setTimeout(fin, 0));
      setTimeout(fin, 100);
    }) : msgTurn;
    const y = async (force) => {
      const t = now();
      // progress for the page's stall watch: when the engine last got here, and the longest stretch
      // between two calls while it worked (y.mark() at the start of a run excludes idle time)
      if (t - y.lastCall > y.maxGap) y.maxGap = t - y.lastCall;
      y.lastCall = t;
      if (!force && t - last < budgetMs) return false;
      if (root.PS_DEBUG_SLICES && t - last > 100) {
        const st = new Error().stack;
        root.PS_DEBUG_SLICES.push({ ms: Math.round(t - last), at: Math.round(t), from: lastStack, to: st });
      }
      if (t - lastFrame >= FRAME_MS) { await frameTurn(); lastFrame = now(); } else await msgTurn();
      last = now();
      y.lastCall = last;
      if (root.PS_DEBUG_SLICES) lastStack = new Error().stack;
      return true;
    };
    y.budgetMs = budgetMs;
    y.lastCall = now();
    y.maxGap = 0;
    y.mark = () => { y.lastCall = now(); };
    return y;
  }

  // ------------------------------------------------------------ fire / door
  const FIREDOOR_STUB = {
    name: 'Stub: no fire/door model in this build',
    stub: true,
    credits: '',
    init: async () => ({}),
    detect: async () => [],          // STUB: always finds nothing
    dispose() {},
  };

  function fireDoorAdapter(tf, yieldFn) {
    const asset = root.PS_OPLIST_ASSETS && root.PS_OPLIST_ASSETS.firedoor;
    const dec = root.FireDoorDecode;
    if (!asset || !dec || !root.PSOpList) return null;
    const meta = root.PS_FIREDOOR_META || {};
    let model = null, layout = 'NCHW', inH = 256, inW = 320;
    // FIRE hysteresis (firedoor/MODEL.md): a box turns on at score >= on and stays on while a
    // fire box overlapping the one shown last time scores >= keep. It holds real fires whose score
    // flickers around 0.5 (CCTV clip: FIRE in 34 of 37 updates instead of 29), and it equally holds
    // a false FIRE once one starts: on a deliberately fire-like clip (sunsets, LEDs, lamps) FIRE was
    // shown 41 % longer (38 instead of 27 of 76 updates), in fewer, longer episodes (9 instead of 15).
    const hy = meta.hysteresis || null;
    // Centre zoom pass (meta.zoom = {frac, on}): the model sees the middle frac x frac of the frame
    // at full input size as well, so a lighter or candle flame 2-3 % of the frame wide (under one
    // heat-map cell of the whole-frame pass) is found more often: on 180 held-out small-flame
    // frames 2 %: 8 -> 18 of 30, 3 %: 16 -> 24 of 30. Its boxes need zoom.on (stricter than the
    // whole-frame on) because the zoomed picture also magnifies lamps and LEDs: false FIRE on 871
    // Open Images non-fire photos 17 -> 24. Implemented by lowering zoom-pass scores by
    // (zoom.on - on), so the hysteresis and overlap rules below treat both passes alike.
    // Cost: a second model run. When one run is slow (over 200 ms, e.g. software WebGL here: 0.4 s,
    // which made an update 2.3 s instead of 1.8 s) the centre pass runs on every second update and
    // its boxes are reused once in between; on a fast GPU it runs on every update. They are reused
    // only while the centre still looks the same (a 16 x 12 grey thumbnail, mean removed, differs by
    // under THUMB_MAX grey levels on average): a cut or a turn drops them. Measured over 2 s gaps:
    // still photos with 8 px drift 8-13, a real CCTV fire video 10-17, cuts between photos 37-44.
    const zoom = meta.zoom || null;
    let prevZoom = null, passTimes = [], zoomTurn = 0;
    const THUMB_MAX = 25;
    async function thumb(px, y0, x0, ch, cw) {
      const fh = Math.max(1, Math.floor(ch / 12)), fw = Math.max(1, Math.floor(cw / 16));
      const t = tf.tidy(() => tf.avgPool(tf.mean(tf.cast(tf.slice(px, [y0, x0, 0], [12 * fh, 16 * fw, 3]), 'float32'), 2, true).expandDims(0), [fh, fw], [fh, fw], 'valid'));
      try {
        const d = await t.data();
        let m = 0; for (let i = 0; i < d.length; i++) m += d[i];
        m /= d.length;
        return Float32Array.from(d, (v) => v - m);
      } finally { t.dispose(); }
    }
    const thumbDiff = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += Math.abs(a[i] - b[i]); return s / a.length; };
    let prevFire = [], prevT = 0;
    function hysteresis(dets) {
      const tNow = now();
      const recent = tNow - prevT <= (hy.maxGapMs || 3000) ? prevFire : [];
      const out = dets.filter((d) => d.cls !== 'fire' || d.score >= hy.on ||
        (d.score >= hy.keep && recent.some((p) => iou(p, d) >= (hy.iou || 0.1))));
      prevFire = out.filter((d) => d.cls === 'fire');
      prevT = tNow;
      return out;
    }
    // one model run: like the training preprocessing (whole picture stretched to inW x inH,
    // OpenCV INTER_AREA, x / 127.5 - 1): bilinear to 2x then 2x2 average
    async function pass(px, thresholds) {
      const x = tf.tidy(() => {
        let r = tf.cast(px, 'float32').expandDims(0);
        r = tf.image.resizeBilinear(r, [2 * inH, 2 * inW], false, true);
        r = tf.avgPool(r, 2, 2, 'valid');
        return tf.sub(tf.div(r, 127.5), 1);
      });
      let out;
      try { out = yieldFn ? await model.runAsync(x, { yieldFn }) : model.run(x); } finally { x.dispose(); }
      try {
        // the window model (FireDoorWindowNet) adds wh_w / off_w: the size and offset of WINDOW boxes
        const [heat, wh, off, whW, offW] = await Promise.all([out.heat.data(), out.wh.data(), out.off.data(),
          out.wh_w ? out.wh_w.data() : null, out.off_w ? out.off_w.data() : null]);
        const o = Object.assign({ layout, inW, inH, gridW: inW / 8, gridH: inH / 8 }, meta.decode || {});
        o.thresholds = Object.assign({}, o.thresholds || {}, thresholds);
        if (whW && offW) { o.windowWh = whW; o.windowOff = offW; }
        return dec.decode(heat, wh, off, o);
      } finally {
        tf.dispose(Object.values(out));
      }
    }
    const inside = (a, b) => { const cx = a.x + a.w / 2, cy = a.y + a.h / 2; return cx >= b.x && cx <= b.x + b.w && cy >= b.y && cy <= b.y + b.h; };
    return {
      name: meta.name || 'FireDoorNet',
      stub: false,
      credits: meta.credits || '',
      hysteresis: hy,
      reset() { prevFire = []; prevT = 0; prevZoom = null; zoomTurn = 0; },
      async init(tfRef, warmup) {
        model = yieldFn ? await root.PSOpList.loadEmbeddedAsync(asset, { tf: tfRef, yieldFn }) : root.PSOpList.loadEmbedded(asset, { tf: tfRef });
        if (yieldFn) await yieldFn(true);
        const s = model.inputs[0].shape;          // [1, H, W, 3]
        inH = s[1]; inW = s[2];
        layout = model.outputs[0].layout === 'nhwc' ? 'NHWC' : 'NCHW';
        const wu = warmup || [inH, inW];
        const z = tf.zeros([wu[0], wu[1], 3]);
        await this.detect(z);
        z.dispose();
        passTimes = [];   // the warm-up run compiles shaders: not a speed sample
      },
      // opts.zoom: false skips the centre pass (the last centre-pass boxes are then used once more),
      // 'auto' runs it on every update, or every second update when one model run is slow;
      // anything else (true, or no opts) runs it
      async detect(px, opts) {
        const base = meta.decode && meta.decode.thresholds || {};
        const fireMin = hy ? Math.min(hy.keep, hy.on) : (base.fire === undefined ? 0.5 : base.fire);
        const t0 = now();
        let dets = await pass(px, Object.assign({}, base, { fire: fireMin }));
        if (yieldFn) await yieldFn(true);
        passTimes = passTimes.concat(now() - t0).slice(-5);
        const passMs = passTimes.slice().sort((a, b) => a - b)[passTimes.length >> 1];   // median: shader compiles spike
        if (zoom) {
          let zd = null;
          const want = opts ? opts.zoom : true;
          const runZoom = want === 'auto' ? (passMs <= 200 || zoomTurn++ % 2 === 0) : want !== false;
          const H = px.shape[0], W = px.shape[1];
          const ch = Math.max(1, Math.round(H * zoom.frac)), cw = Math.max(1, Math.round(W * zoom.frac));
          const y0 = Math.round((H - ch) / 2), x0 = Math.round((W - cw) / 2);
          if (runZoom) {
            const lift = zoom.on - (hy ? hy.on : fireMin);
            const crop = tf.slice(px, [y0, x0, 0], [ch, cw, 3]);
            try {
              zd = (await pass(crop, { fire: fireMin + lift, door: 2, window: 2 })).filter((d) => d.cls === 'fire').map((d) => Object.assign(d, {
                x: (x0 + d.x * cw) / W, y: (y0 + d.y * ch) / H, w: d.w * cw / W, h: d.h * ch / H,
                rawScore: d.score, score: d.score - lift, zoom: true }));
            } finally { crop.dispose(); }
            prevZoom = { dets: zd, t: now(), W, H, thumb: zd.length && ch >= 12 && cw >= 16 ? await thumb(px, y0, x0, ch, cw) : null };
          } else if (prevZoom && prevZoom.thumb && prevZoom.W === W && prevZoom.H === H && now() - prevZoom.t < 10000) {
            const same = thumbDiff(await thumb(px, y0, x0, ch, cw), prevZoom.thumb) < THUMB_MAX;
            if (same) zd = prevZoom.dets.map((d) => Object.assign({}, d, { carried: true }));
            prevZoom = null;
          } else prevZoom = null;
          if (zd && zd.length) {
            // one box per fire: where the passes overlap keep the stronger (after the lift)
            const all = dets.concat(zd).sort((a, b) => b.score - a.score), keep = [];
            for (const d of all) {
              if (d.cls === 'fire' && keep.some((k) => k.cls === 'fire' && (iou(k, d) >= 0.3 || inside(d, k) || inside(k, d)))) continue;
              keep.push(d);
            }
            dets = keep;
          }
        }
        return hy ? hysteresis(dets) : dets.filter((d) => d.cls !== 'fire' || d.score >= (base.fire === undefined ? 0.5 : base.fire));
      },
      dispose() { if (model) model.dispose(); model = null; },
    };
  }


  // ------------------------------------------------------------ engine
  /**
   * create({tf, coop, budgetMs, fdDetector}) -> engine
   *   engine.init({prefer, allowSoftwareWebGL, worker, warmup}) -> Promise<info>
   *   engine.detect(source, {hfovDeg, person, zoom}) -> Promise<res>  (source: tf.Tensor3D, canvas,
   *       ImageData or {data, width, height} RGBA)
   *   engine.reset(), engine.dispose(), engine.gpuLost(), engine.info()
   * fdDetector: a custom fire/door detector {init(tf), detect(tensor, opts), dispose()} (main thread only).
   */
  function create(cfg) {
    cfg = cfg || {};
    const tf = cfg.tf || root.tf;
    const yieldFn = cfg.coop ? makeYielder(cfg.budgetMs || 30) : null;
    let fd = null, fdError = null, backend = null, backendInfo = null, loadMs = 0, ready = false;
    const phases = {};
    const info = () => ({
      backend, backendInfo, loadMs: Math.round(loadMs), coop: !!yieldFn, phases,
      personStages: root.PSPeople ? root.PSPeople.personStages : 0,
      fd: fd ? { name: fd.name, stub: !!fd.stub, credits: fd.credits || '', hysteresis: fd.hysteresis || null } : null,
      fdError, numTensors: tf && tf.memory ? tf.memory().numTensors : null,
    });
    const eng = {
      yieldFn,
      get ready() { return ready; },
      get fd() { return fd; },
      info,
      async init(o) {
        o = o || {};
        const t0 = now();
        const missing = [!tf && 'TF.js', !root.PSPeople && 'people.js', !root.PS_PEOPLE_ASSETS && 'person/face models'].filter(Boolean);
        if (missing.length) throw new Error('detector code missing from the page: ' + missing.join(', '));
        backend = await root.PSPeople.setupBackend(tf, { prefer: o.prefer, allowSoftwareWebGL: o.allowSoftwareWebGL, worker: !!o.worker });
        backendInfo = root.PSPeople.backendInfo;
        phases.backendMs = Math.round(now() - t0);
        if (o.onBackend) { try { o.onBackend(info()); } catch (e) { /* ignore */ } }
        const popts = { yieldFn, warmup: o.warmup };
        try {
          await root.PSPeople.init(tf, root.PS_PEOPLE_ASSETS, popts);
        } catch (e) {
          if (backend === 'cpu') throw e;
          backend = await root.PSPeople.setupBackend(tf, { prefer: 'cpu', worker: !!o.worker });   // WebGL failed: CPU
          backendInfo = root.PSPeople.backendInfo;
          await root.PSPeople.init(tf, root.PS_PEOPLE_ASSETS, popts);
        }
        phases.peopleMs = Math.round(now() - t0 - phases.backendMs);
        if (yieldFn) await yieldFn(true);
        const t1 = now();
        const det = cfg.fdDetector || fireDoorAdapter(tf, yieldFn) || FIREDOOR_STUB;
        fd = det;
        try { await det.init(tf, o.warmup); } catch (e) { fdError = String(e && e.message || e); fd = FIREDOOR_STUB; }
        phases.fireDoorMs = Math.round(now() - t1);
        loadMs = now() - t0;
        ready = true;
        return info();
      },
      async setFireDoor(det) {
        if (fd && fd.dispose) { try { fd.dispose(); } catch (e) { /* ignore */ } }
        fd = det || FIREDOOR_STUB; fdError = null;
        try { await fd.init(tf); } catch (e) { fdError = String(e && e.message || e); fd = FIREDOOR_STUB; }
        return info();
      },
      // one update: people + faces, then fire/door (with its centre pass when opts.zoom says so)
      async detect(source, opts) {
        opts = opts || {};
        const t0 = now();
        if (yieldFn) yieldFn.mark();
        let px;
        if (source instanceof tf.Tensor) px = source;
        else if (source && source.data && !(typeof ImageData !== 'undefined' && source instanceof ImageData) && typeof source.getContext !== 'function') {
          px = tf.browser.fromPixels({ data: new Uint8Array(source.data.buffer || source.data), width: source.width, height: source.height });
        } else px = tf.browser.fromPixels(source);
        try {
          const r = await root.PSPeople.detect(px, { hfovDeg: opts.hfovDeg, person: opts.person !== false });
          if (yieldFn) await yieldFn(true);
          const t1 = now();
          let boxes = [];
          if (fd && opts.fd !== false) boxes = (await fd.detect(px, { zoom: opts.zoom })) || [];
          const fdMs = now() - t1;
          return {
            detections: r.detections, display: r.display, width: r.width, height: r.height,
            ms: { person: r.ms.person == null ? null : r.ms.person, face: r.ms.face, firedoor: fdMs, total: now() - t0 },
            fd: boxes, numTensors: null,
          };
        } finally {
          if (px !== source) px.dispose();
        }
      },
      reset() { if (fd && fd.reset) { try { fd.reset(); } catch (e) { /* ignore */ } } },
      gpuLost() {
        if (backend !== 'webgl') return false;
        try { const gl = tf.backend().gpgpu.gl; return !!(gl && gl.isContextLost()); } catch (e) { return false; }
      },
      glCanvas() { try { return backend === 'webgl' ? tf.backend().gpgpu.gl.canvas : null; } catch (e) { return null; } },
      dispose() {
        try { root.PSPeople.dispose(); } catch (e) { /* lost context */ }
        try { if (fd && fd.dispose) fd.dispose(); } catch (e) { /* lost context */ }
        fd = null; ready = false;
      },
    };
    return eng;
  }

  // ------------------------------------------------------------ worker side
  /** The worker's message loop (see the protocol at the top). One request at a time. */
  function serveWorker(scope) {
    let eng = null, lostSent = false;
    const post = (m, tr) => { try { scope.postMessage(m, tr || []); } catch (e) { /* page gone */ } };
    const errText = (e) => String(e && e.message || e);
    const lost = () => {
      if (lostSent) return;
      lostSent = true;
      post({ type: 'lost' });
    };
    scope.onmessage = async (ev) => {
      const m = ev.data || {};
      if (m.type === 'init') {
        try {
          eng = create({ tf: root.tf });
          const info = await eng.init(Object.assign({}, m.opts, { worker: true, onBackend: (i) => post({ type: 'backend', info: i }) }));
          const cv = eng.glCanvas();
          if (cv && cv.addEventListener) cv.addEventListener('webglcontextlost', (e) => { try { e.preventDefault(); } catch (x) { /* ignore */ } lost(); }, false);
          post({ type: 'ready', info });
        } catch (e) {
          post({ type: 'failed', error: errText(e) });
        }
      } else if (m.type === 'detect') {
        const f = m.frame;
        try {
          if (!eng || !eng.ready) throw new Error('detectors not ready');
          const img = { data: new Uint8Array(f.buf), width: f.w, height: f.h };
          const res = await eng.detect(img, m.opts);
          res.numTensors = root.tf.memory().numTensors;
          post({ type: 'result', id: m.id, res, buf: f.buf }, [f.buf]);
        } catch (e) {
          const gone = eng && eng.gpuLost();
          post({ type: 'error', id: m.id, error: errText(e), lost: gone });
          if (gone) lost();
        }
      } else if (m.type === 'reset') {
        if (eng) eng.reset();
      } else if (m.type === 'debug') {
        // test hooks (page: PSCamera.debugEngine(what)): simulate a lost graphics context or a hung run
        if (m.what === 'lose-context') { try { root.tf.backend().gpgpu.gl.getExtension('WEBGL_lose_context').loseContext(); } catch (e) { /* not WebGL */ } }
        else if (m.what === 'hang') { const t = now(); while (now() - t < (m.ms || 60000)) { /* busy */ } }
      }
    };
    post({ type: 'boot' });
    // liveness for the page: a beat every second while this thread's event loop runs (it stops when
    // the worker dies or a run blocks it; the page restarts the detectors after a long silence)
    setInterval(() => post({ type: 'beat' }), 1000);
  }

  root.PSEngine = { create, makeYielder, serveWorker, FIREDOOR_STUB, fireDoorAdapter, _iou: iou };
  // started as a worker from the page's own script texts: serve requests
  if (typeof WorkerGlobalScope !== 'undefined' && typeof self !== 'undefined' && self instanceof WorkerGlobalScope && root.PS_ENGINE_WORKER) serveWorker(self);
})(typeof globalThis !== 'undefined' ? globalThis : this);
