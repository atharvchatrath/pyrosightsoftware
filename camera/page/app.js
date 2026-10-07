/*
 * PyroSight Camera: page logic (app.js). Runs early (only the page shell, oplist.js, people.js,
 * the fire/door decoder, engine.js and motion.js come before it); TF.js and the models are kept
 * as text (<script type="text/plain">) and run in a Web Worker, so this thread only draws, tracks
 * camera turns, speaks and answers taps. See README.md and "Detection engine" below.
 *
 * Boxes (same colours as core/include/pyrosight/ps_display.h):
 *   person  WHITE  #FFFFFF  label = distance estimate, device style ("1.2M")
 *   fire    PURPLE #C850FF  "FIRE"
 *   exit    GREEN  #28FF50  "DOOR" and "WINDOW" (fire/door/window model) and "EXIT" (the viewer's
 *                           mark, or the navigation's estimate while Navigation runs: "EXIT 5M")
 *
 * Navigation (demo): the eyepiece's own way-out code (camera/nav, global PSNav, inlined right
 * after this script); see "navigation (demo)" below.
 *
 * Fire/door detector interface (injectable). The engine (runtime/engine.js) uses, in order:
 *   1. globalThis.PS_FIREDOOR_DETECTOR, if a script before this one set it (this runs the
 *      detectors on the main thread, where that detector lives);
 *   2. an adapter around PS_OPLIST_ASSETS.firedoor + FireDoorDecode, if both exist;
 *   3. a STUB (finds nothing), clearly reported as such on the page.
 * A detector is {name, stub?, credits?, init(tf) -> Promise, detect(pixels) ->
 * Promise<[{cls: 'fire'|'door'|'window', score, x, y, w, h}]>, dispose()}; pixels is a
 * tf.Tensor3D [H, W, 3] RGB 0..255 that the detector must NOT dispose; boxes are
 * normalised 0..1 (x, y = top-left) in that image. PSCamera.setFireDoorDetector()
 * swaps it at run time.
 */
(function () {
  'use strict';

  const $ = (id) => document.getElementById(id);
  const COL = { person: '#FFFFFF', fire: '#C850FF', exit: '#28FF50' };
  const FOV = 65;                 // assumed camera field of view across the long side of the frame
  const ALERT_COOLDOWN_MS = 6000; // person / fire call-outs (device: 10 s)
  const MONO = 'ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace';
  const M = window.PSMotion;
  let qs;
  try { qs = new URLSearchParams(location.search); } catch (e) { qs = new URLSearchParams(''); }

  // ------------------------------------------------------- self copy (save)
  // The build stores the page's non-script part in #ps-shell; scripts are
  // rebuilt from their own text, so the saved file equals dist/pyrosight_camera.html.
  function buildStandalone() {
    const shellEl = $('ps-shell');
    if (!shellEl) throw new Error('page shell missing');
    const shell = JSON.parse(shellEl.textContent);
    const parts = [];
    document.querySelectorAll('script[data-ps-part]').forEach((el) => {
      let attrs = '';
      for (const a of Array.from(el.attributes)) {
        attrs += a.value === '' ? ' ' + a.name : ' ' + a.name + '="' + a.value.replace(/&/g, '&amp;').replace(/"/g, '&quot;') + '"';
      }
      parts.push('<script' + attrs + '>' + el.textContent + '</' + 'script>');
    });
    return shell.head + shell.shell + parts.join('\n') + '\n' + shell.tail;
  }

  // ------------------------------------------------------------ state
  const state = {
    backend: null, backendInfo: null, modelsReady: false, loadError: null, loadMs: 0,
    fd: null, fdReady: false, fdError: null,
    engine: null, engineMode: null, engineStage: null, engineInfo: null, workerError: null, workerTimes: null, numTensors: null, lastBusyMs: 0,
    src: null,              // {kind: 'camera'|'video'|'image', el, w, h, mirror, live, name}
    stream: null, devices: [], facing: null, mirrorOverride: qs.get('mirror'),
    busy: false, results: null, lastPersons: null, cycle: 0, lastInferEnd: 0, lastInferVideoTime: -1,
    inferredSrc: null, inferError: null,
    eyepiece: false, palette: 1, voice: false,    // palette 1 = Ironbow (EYE_PALETTES): thermal colours by default
    mark: null, markLostSaid: false,
    exitSector: null, exitSectorSince: 0, lastExitSay: -1e9,
    al: {}, log: [], spoken: new Map(),
    dpr: 1, fit: null,
    lastVideoTime: -1, lastTrackT: 0, trackMs: 0, newFrame: false,
    fpsCount: 0, fpsT0: 0, fps: 0, ups: 0, upsTimes: [],
    lastStatusT: 0, startT: performance.now(),
    camReq: 0, camPending: false, camMuted: false,
    inferGen: 0, inferT0: 0, recovering: false, gpuLosses: 0, stalls: 0, lastCheckT: 0,
    incidentStreak: 0,      // graphics resets / stalls with no completed WebGL run in between (3 -> CPU)
    runsSinceStart: 0,      // completed runs since the detectors were (re)started
    lastRecovery: null, paused: false, slowSaid: false, trackAfterT: 0,
    stats: { inferences: 0, tensors: [], ms: [], errors: [], trackStates: {}, labels: [], recoveries: [] },
  };
  // navigation (demo), see "navigation (demo)" below
  const navUi = {
    nav: null, on: false, pending: false, map: null, arrow: null, visible: true, dirty: true,
    lastArrowT: -1e9, lastMapT: -1e9, lastTextT: -1e9, arrowKey: '', mapKey: '', lastMapDrawT: -1e9,
    unbindKeys: null, overlay: null, simShown: false, note: '',
    yawEpoch: -1, yawOffset: 0, lastFedYaw: null, lastFeedT: -1e9, asked: -1e9, speedIdx: 1,
  };
  const tracker = new M.MotionTracker({ fovDeg: FOV });
  let trackerEpoch = 0;             // bumped on every tracker reset (navigation keeps its heading continuous)
  function resetTracker() { tracker.reset(); trackerEpoch++; }

  const view = $('view');
  const g = view.getContext('2d');
  const camVideo = $('cam-video'), fileVideo = $('file-video');
  const cap = document.createElement('canvas');            // frame handed to the detectors
  const capCtx = cap.getContext('2d', { willReadFrequently: true });   // read back by fromPixels on the CPU backend
  const trk = document.createElement('canvas');            // 128x128 grey copy for motion tracking
  trk.width = trk.height = tracker.o.n;
  const trkCtx = trk.getContext('2d', { willReadFrequently: true });
  const eye = document.createElement('canvas');            // low-res grey picture for eyepiece view
  const eyeCtx = eye.getContext('2d', { willReadFrequently: true });
  let eyeLo = 0, eyeHi = 255, eyeFrameKey = null;
  const grey = new Float32Array(tracker.o.n * tracker.o.n);

  // ------------------------------------------------------------ detection engine
  // Normal case: the three detectors run in a dedicated Web Worker built from this page's own script
  // texts (a Blob URL: no network), with TF.js on WebGL (OffscreenCanvas) or, failing that, its CPU
  // backend. One frame in flight at a time, sent as transferred RGBA pixels; plain boxes come back.
  // Fallback, when the worker cannot be made or does not start (a Content-Security-Policy without
  // blob: workers, no Worker/OffscreenCanvas, a custom fire/door detector): TF.js and the models are
  // run on this thread after the first paint, one script per turn (blob: scripts, which the browser
  // compiles off this thread, else inline scripts), and the detectors give the event
  // loop a turn every ~30 ms (op by op; the person graph in short stages), with an idle gap after
  // each update. Switch for tests: ?engine=main or #engine=main (or window.PS_ENGINE = 'main' set
  // before this script) forces the fallback; 'worker' forbids it.
  const FIREDOOR_STUB = window.PSEngine ? window.PSEngine.FIREDOOR_STUB : { name: 'Stub: no fire/door model in this build', stub: true, credits: '', init: async () => ({}), detect: async () => [], dispose() {} };
  const ENGINE_PARTS = ['ps-tf', 'ps-oplist', 'ps-people', 'ps-people-assets', 'ps-firedoor-model', 'ps-firedoor', 'ps-engine'];
  const HEAVY_PARTS = ['ps-tf', 'ps-people-assets', 'ps-firedoor-model'];     // kept as text unless the fallback runs here
  const APP_NONCE = (document.currentScript && document.currentScript.nonce) || '';
  const MAIN_IDLE_MS = 120;          // fallback: at least this long between two updates (taps, frames)
  // turn tracker: at most this share of this thread; its cost estimate starts slightly pessimistic,
  // rises at once (a slow phone's first runs were measured at 85 ms) and falls slowly; longest gap
  // 200 ms: with 500 ms gaps or a 40 ms start a 120-degree pan lost lock in 2 of 3 browser_test runs
  const TRACK_SHARE = 0.3, TRACK_MAX_GAP_MS = 200, TRACK_START_MS = 15, TRACK_SETTLE_MS = 400;
  let customFd = null;               // PSCamera.setFireDoorDetector() / PS_FIREDOOR_DETECTOR (main thread)

  function enginePref() {
    let h = '';
    try { h = (location.hash.match(/engine=(main|worker)/) || [])[1] || ''; } catch (e) { /* no location */ }
    const v = window.PS_ENGINE || qs.get('engine') || h;
    return v === 'main' || v === 'worker' ? v : 'auto';
  }

  // a turn of the event loop for this thread's own setup work (not the detectors)
  const turn = () => new Promise((r) => { const ch = new MessageChannel(); ch.port1.onmessage = () => r(); ch.port2.postMessage(0); });
  const afterPaint = () => new Promise((r) => requestAnimationFrame(() => setTimeout(r, 0)));
  // the model scripts come after this one in the page: wait until the parser has read all of them
  const domReady = () => new Promise((r) => {
    if (document.readyState !== 'loading') r();
    else document.addEventListener('DOMContentLoaded', () => r(), { once: true });
  });

  // A part's text: TF.js and the models come split over several elements (#id, then
  // script[data-ps-of=id]) so the HTML parser can pause between them.
  function partTexts(id) {
    const el = $(id);
    if (!el) return null;
    const out = [el.textContent];
    document.querySelectorAll('script[data-ps-of="' + id + '"]').forEach((c) => out.push(c.textContent));
    return out;
  }

  // The worker's script: the engine parts in order, joined in small slices (Blob of Blobs: nothing
  // is copied twice), a turn of the event loop between slices and a rendered frame at least every
  // 50 ms (MessageChannel turns alone let these tasks run back to back with no frame: a tap's
  // visible response waited 397 ms at 4x CPU throttling).
  let workerBlob = null;
  async function makeWorkerBlob() {
    if (workerBlob) return workerBlob;
    const SLICE = 1 << 19;
    const pause = window.PSEngine ? window.PSEngine.makeYielder(20) : null;
    const step = () => (pause ? pause(true) : turn());
    let blob = new Blob(['self.PS_ENGINE_WORKER = true;\n'], { type: 'text/javascript' });
    for (const id of ENGINE_PARTS) {
      const texts = partTexts(id);
      if (!texts) continue;            // e.g. no fire/door model in a stub build
      for (const text of texts) {
        for (let i = 0; i < text.length;) {
          let end = Math.min(text.length, i + SLICE);
          const c = text.charCodeAt(end - 1);
          if (end < text.length && c >= 0xD800 && c <= 0xDBFF) end--;     // never split a surrogate pair
          blob = new Blob([blob, text.slice(i, end)], { type: 'text/javascript' });
          i = end;
          await step();
        }
      }
      blob = new Blob([blob, '\n;\n'], { type: 'text/javascript' });
    }
    workerBlob = blob;
    return blob;
  }

  function workerClient() {
    let w = null, seq = 0, readyInfo = null;
    const pending = new Map();
    // liveness: the worker posts a beat every second while its event loop runs; the longest silence
    // seen while it was working (loading, detecting) sets how long a silence means it died or hangs
    let lastMsgT = 0, maxGap = 0;
    const rejectAll = (why) => { for (const p of pending.values()) p.reject(new Error(why)); pending.clear(); };
    return {
      mode: 'worker',
      async start(opts, hooks) {
        const t0 = performance.now();
        const blob = await makeWorkerBlob();
        const tBlob = performance.now();
        const url = URL.createObjectURL(blob);
        try { w = new Worker(url); } catch (e) { URL.revokeObjectURL(url); throw e; }   // CSP worker-src: SecurityError
        lastMsgT = performance.now();
        return new Promise((resolve, reject) => {
          let booted = false, settled = false;
          const fail = (why) => {
            if (settled) return;
            settled = true; clearTimeout(bootTimer); clearTimeout(initTimer);
            try { w.terminate(); } catch (e) { /* ignore */ }
            w = null;
            try { URL.revokeObjectURL(url); } catch (e) { /* ignore */ }
            reject(new Error(why));
          };
          const bootTimer = setTimeout(() => { if (!booted) fail('the detector worker did not start'); }, 20000);
          const initTimer = setTimeout(() => fail('the detector worker did not get ready in 3 minutes'), 180000);
          w.onerror = (e) => {
            try { e.preventDefault(); } catch (x) { /* ignore */ }
            if (!settled) fail('detector worker error: ' + ((e && e.message) || 'blocked'));
            else { rejectAll('worker error'); if (hooks.onLost) hooks.onLost('crash'); }
          };
          w.onmessage = (ev) => {
            const m = ev.data || {};
            const tm = performance.now();
            if (!readyInfo || pending.size) maxGap = Math.max(maxGap, tm - lastMsgT);
            lastMsgT = tm;
            if (m.type === 'beat') return;
            if (m.type === 'boot') {
              booted = true; clearTimeout(bootTimer);
              state.workerTimes = { blobMs: Math.round(tBlob - t0), bootMs: Math.round(performance.now() - tBlob) };
              try { URL.revokeObjectURL(url); } catch (e) { /* ignore */ }
              w.postMessage({ type: 'init', opts });
            } else if (m.type === 'backend') { if (hooks.onBackend) hooks.onBackend(m.info); }
            else if (m.type === 'ready') { settled = true; clearTimeout(initTimer); readyInfo = m.info; resolve(m.info); }
            else if (m.type === 'failed') fail(m.error || 'the detectors did not load in the worker');
            else if (m.type === 'result' || m.type === 'error') {
              const p = pending.get(m.id);
              if (!p) return;
              pending.delete(m.id);
              if (m.type === 'result') p.resolve(m.res);
              else p.reject(Object.assign(new Error(m.error), { lost: !!m.lost }));
            } else if (m.type === 'lost') { if (hooks.onLost) hooks.onLost('event'); }
          };
        });
      },
      detect(im, opts) {
        if (!w || !readyInfo) return Promise.reject(new Error('detectors not ready'));
        return new Promise((resolve, reject) => {
          const id = ++seq;
          pending.set(id, { resolve, reject });
          w.postMessage({ type: 'detect', id, frame: { buf: im.data.buffer, w: im.width, h: im.height }, opts }, [im.data.buffer]);
        });
      },
      reset() { if (w) w.postMessage({ type: 'reset' }); },
      debug(what, ms) { if (w) w.postMessage({ type: 'debug', what, ms }); },
      gpuLost() { return false; },      // the worker reports it ('lost')
      progressAt: () => lastMsgT,
      maxGap: () => maxGap,
      terminate() { if (w) { try { w.terminate(); } catch (e) { /* ignore */ } } w = null; readyInfo = null; rejectAll('stopped'); },
    };
  }

  // Fallback: run TF.js and the model scripts on this thread, one per turn. First as blob: scripts
  // (the browser parses and compiles those off this thread while it streams them in, so only
  // running them blocks here); where the host's CSP refuses blob: scripts, as inline scripts
  // (they need only what inline scripts already need; a nonce, if the host uses one, is copied).
  function runScriptText(text) {
    const el = document.createElement('script');
    if (APP_NONCE) el.nonce = APP_NONCE;
    el.textContent = text;
    (document.head || document.documentElement).appendChild(el);
    el.remove();
  }
  function runBlobScript(texts) {
    return new Promise((resolve) => {
      let url = null;
      try { url = URL.createObjectURL(new Blob(texts, { type: 'text/javascript' })); } catch (e) { resolve(false); return; }
      const s = document.createElement('script');
      s.async = true;
      s.onload = () => { URL.revokeObjectURL(url); s.remove(); resolve(true); };
      s.onerror = () => { URL.revokeObjectURL(url); s.remove(); resolve(false); };
      if (APP_NONCE) s.nonce = APP_NONCE;
      s.src = url;
      (document.head || document.documentElement).appendChild(s);
    });
  }
  // The fallback's TF.js / model scripts each block this thread while they run (0.3-0.5 s on a
  // slow phone): start each one only after a second without taps or keys (at most 8 s of waiting).
  let lastInputT = -1e9;
  for (const ev of ['pointerdown', 'keydown']) window.addEventListener(ev, () => { lastInputT = performance.now(); }, { capture: true, passive: true });
  async function inputQuiet(ms, maxMs) {
    const t0 = performance.now();
    while (performance.now() - lastInputT < ms && performance.now() - t0 < maxMs) await new Promise((r) => setTimeout(r, 100));
  }
  async function runHeavyParts() {
    const need = { 'ps-tf': () => !!window.tf, 'ps-people-assets': () => !!window.PS_PEOPLE_ASSETS,
      'ps-firedoor-model': () => !!(window.PS_OPLIST_ASSETS && window.PS_OPLIST_ASSETS.firedoor) };
    let blobOk = true;
    for (const id of HEAVY_PARTS) {
      const texts = partTexts(id);
      if (!texts || need[id]()) continue;
      await inputQuiet(1000, 8000);
      await turn();
      if (blobOk) blobOk = (await runBlobScript(texts)) && need[id]();
      if (!need[id]()) { await inputQuiet(1000, 8000); await turn(); runScriptText(texts.join('')); }
      if (!need[id]()) throw new Error('this page’s security settings do not let the detectors start');
    }
  }

  function mainClient() {
    let eng = null, lostCb = null;
    return {
      mode: 'main',
      async start(opts, hooks) {
        await runHeavyParts();
        await turn();
        eng = window.PSEngine.create({ tf: window.tf, coop: true, budgetMs: 30, fdDetector: customFd });
        const info = await eng.init(Object.assign({}, opts, { onBackend: hooks.onBackend }));
        const cv = eng.glCanvas();
        if (cv && cv.addEventListener && !cv.__psWatched) {
          cv.__psWatched = true;
          lostCb = () => { if (hooks.onLost) setTimeout(() => hooks.onLost('event'), 0); };
          cv.addEventListener('webglcontextlost', () => { if (eng && eng.glCanvas() === cv && lostCb) lostCb(); }, false);
        }
        return info;
      },
      async detect(canvas, opts) {
        const r = await eng.detect(canvas, opts);
        r.numTensors = window.tf.memory().numTensors;
        return r;
      },
      reset() { if (eng) eng.reset(); },
      debug(what) { if (what === 'lose-context') { try { window.tf.backend().gpgpu.gl.getExtension('WEBGL_lose_context').loseContext(); } catch (e) { /* not WebGL */ } } },
      gpuLost() { return !!(eng && eng.gpuLost()); },
      progressAt: () => (eng && eng.yieldFn ? eng.yieldFn.lastCall : 0),
      maxGap: () => (eng && eng.yieldFn ? eng.yieldFn.maxGap : 0),
      engine: () => eng,
      terminate(lost) {
        const tf = window.tf;
        const wasGl = eng && eng.info().backend === 'webgl';
        // TF.js polls a fence on a lost context forever (a busy loop with console warnings): drop it
        if (lost && tf) { try { const gp = tf.backend().gpgpu; if (gp && Array.isArray(gp.itemsToPoll)) gp.itemsToPoll = []; } catch (e) { /* ignore */ } }
        if (eng) eng.dispose();
        eng = null; lostCb = null;
        if (lost && wasGl && tf) {
          // a fresh WebGL backend (TF.js forgets the lost context and makes a new one)
          const fac = tf.findBackendFactory && tf.findBackendFactory('webgl');
          try { tf.removeBackend('webgl'); } catch (e) { /* ignore */ }
          if (fac && !tf.findBackendFactory('webgl')) { try { tf.registerBackend('webgl', fac, 2); } catch (e) { /* CPU next time */ } }
        }
      },
    };
  }

  // the frame size the detectors will see (warm-up at that size compiles the right WebGL shaders)
  function warmupSize() {
    const src = state.src;
    const W = src && src.w ? src.w : 640, H = src && src.h ? src.h : 480;
    const k = Math.min(1, 640 / Math.max(W, H));
    return [Math.max(1, Math.round(H * k)), Math.max(1, Math.round(W * k))];
  }

  function applyEngineInfo(info) {
    state.engineInfo = info;
    state.backend = info.backend;
    state.backendInfo = info.backendInfo;
    state.loadMs = info.loadMs;
    state.fd = info.fd ? Object.assign({}, info.fd) : FIREDOOR_STUB;
    state.fdError = info.fdError ? new Error(info.fdError) : null;
    state.fdReady = true;
    state.numTensors = info.numTensors;
  }

  // start (or restart) the detectors; o.prefer = 'cpu' after repeated graphics resets
  async function startEngine(o) {
    o = o || {};
    const pref = o.mode || enginePref();
    const opts = { prefer: o.prefer || (qs.get('backend') === 'cpu' ? 'cpu' : undefined), allowSoftwareWebGL: qs.get('swgl') !== '0', warmup: warmupSize() };
    const hooks = {
      onBackend: (i) => { state.backend = i.backend; state.backendInfo = i.backendInfo; updateStatus(true); },
      onLost: (why) => { onGpuLost(why === 'crash' ? 'stall' : 'event'); },
    };
    let client = null, info = null;
    const canWorker = typeof Worker !== 'undefined' && typeof Blob !== 'undefined' && !!(window.URL && URL.createObjectURL) && !!window.PSEngine;
    if (pref !== 'main' && !customFd && canWorker) {
      state.engineStage = 'worker';
      updateStatus(true);
      try { client = workerClient(); info = await client.start(opts, hooks); } catch (e) {
        state.workerError = String(e && e.message || e);
        if (client) client.terminate();
        client = null;
        if (pref === 'worker') throw e;
        console.warn('PyroSight Camera: detector worker unavailable, running the detectors on the page (slower to answer taps): ' + state.workerError);
      }
    }
    if (!client) {
      if (!window.PSEngine) throw new Error('detector code missing from the page');
      state.engineStage = 'main';
      updateStatus(true);
      client = mainClient();
      info = await client.start(opts, hooks);
    }
    state.engine = client;
    state.engineMode = client.mode;
    state.runsSinceStart = 0;
    state.paused = false;
    applyEngineInfo(info);
  }

  function engineReset() { if (state.engine) { try { state.engine.reset(); } catch (e) { /* ignore */ } } }

  async function setFireDoorDetector(det) {
    customFd = det || FIREDOOR_STUB;     // a detector object lives on this thread: the engine runs here with it
    if (!state.engine) { updateStatus(true); return; }   // picked up when the engine starts
    await restartEngine('main');
    $('credits').textContent = creditsText();
    updateStatus(true);
  }

  async function restartEngine(mode) {
    state.inferGen++; state.busy = false;
    state.modelsReady = false;
    const prev = state.engine;
    state.engine = null;
    if (prev) { try { prev.terminate(false); } catch (e) { /* ignore */ } }
    try {
      await startEngine({ mode });
      state.modelsReady = true;
      state.loadError = null;
    } catch (e) {
      state.loadError = e;
    }
  }

  function creditsText() {
    let s = 'People and faces: COCO-SSD (TensorFlow.js models) and MediaPipe BlazeFace, Apache-2.0. TensorFlow.js, Apache-2.0.';
    if (state.fd && state.fd.credits) s += ' ' + state.fd.credits;
    return s;
  }

  async function loadModels() {
    const t0 = performance.now();
    try {
      await domReady();
      await afterPaint();
      if (window.PS_FIREDOOR_DETECTOR) customFd = window.PS_FIREDOOR_DETECTOR;
      await startEngine();
      state.modelsReady = true;
      state.loadMs = performance.now() - t0;
    } catch (e) {
      state.loadError = e;
      console.error('PyroSight Camera: detectors failed to load', e);
    }
    $('credits').textContent = creditsText();
    updateStatus(true);
  }

  // ------------------------------------------------------------ sources
  function setSource(src) {
    state.src = src;
    state.results = null; state.lastPersons = null; state.inferredSrc = null; state.lastInferVideoTime = -1;
    state.lastVideoTime = -1; eyeFrameKey = null;
    state.trackAfterT = performance.now() + TRACK_SETTLE_MS;
    state.paused = false;
    clearMark(true);
    resetTracker();
    state.al = {};
    engineReset();
    if (src && src.w && src.h) $('screen').style.aspectRatio = src.w + ' / ' + src.h;
    if (src) state.cameraError = null;
    $('overlay').hidden = !!src;
    $('mark').disabled = !src || navUi.on;           // navigation replaces the mark while it runs
    $('whereout').disabled = !src && !navUi.on;
    state.camMuted = false;
    setStartButton();
    updateStatus(true);
  }

  function stopTracks(stream) { if (stream) { try { stream.getTracks().forEach((t) => t.stop()); } catch (e) { /* ignore */ } } }

  function stopSource(keepRequest) {
    const s = state.src;
    if (!keepRequest) {
      state.camReq++;               // a camera request still waiting for an answer is now stale
      state.camPending = false;
    }
    if (state.stream) { stopTracks(state.stream); state.stream = null; }
    camVideo.srcObject = null;
    if (s && s.kind === 'video') {
      fileVideo.pause();
      if (s.url) { try { URL.revokeObjectURL(s.url); } catch (e) { /* ignore */ } }
      fileVideo.removeAttribute('src'); fileVideo.load();
    }
    if (s && s.kind === 'image' && s.el && s.el.close) { try { s.el.close(); } catch (e) { /* ignore */ } }
    state.src = null;
    $('switch').hidden = true;
  }

  function showOverlay(title, text, buttons) {
    $('ov-title').textContent = title;
    $('ov-text').textContent = text;
    $('ov-start').hidden = !buttons;
    $('ov-file').hidden = !buttons;
    $('overlay').hidden = false;
  }

  function mobile() { return /Android|iPhone|iPad|iPod|Mobile/i.test(navigator.userAgent || ''); }

  function facingOf(track) {
    const s = track && track.getSettings ? track.getSettings() : {};
    if (s.facingMode === 'environment' || s.facingMode === 'user') return s.facingMode;
    const l = (track && track.label || '').toLowerCase();
    if (/back|rear|environment|world/.test(l)) return 'environment';
    if (/front|user|face|selfie/.test(l)) return 'user';
    return mobile() ? 'environment' : 'user';   // laptop and desktop webcams face the user
  }

  function requestOrientation() {
    try {
      const D = window.DeviceOrientationEvent;
      if (D && typeof D.requestPermission === 'function' && !state.orientAsked) {
        state.orientAsked = true;   // iOS: must be called from a tap
        D.requestPermission().then((r) => { state.orientPerm = r; }, () => { state.orientPerm = 'error'; });
      }
    } catch (e) { /* not available */ }
  }

  // One camera request at a time. Every start, stop or file opened bumps state.camReq; an answer
  // that arrives for an older request (a double click, Stop or a photo opened while the permission
  // prompt was up) is stopped at once, so no camera is left running unseen.
  async function startCamera(deviceId, prevDeviceId) {
    if (state.camPending) return;
    hideNotice();
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) { cameraError({ name: 'NoMediaDevices' }); return; }
    // a photo or video being analysed stays until the camera really starts (inside a page that may
    // not use the camera it would otherwise be thrown away for nothing); a camera is stopped first
    // (phones cannot open two)
    const keep = !!(state.src && state.src.kind !== 'camera');
    if (!keep) { stopSource(); setSource(null); }
    const req = ++state.camReq;
    state.camPending = true;
    const live = () => req === state.camReq;
    setStartButton();
    if (!keep) showOverlay('Waiting for the camera…', 'Allow camera access if your browser asks. Video never leaves your device.', false);
    const constraints = (id) => {
      const video = { width: { ideal: 1280 }, height: { ideal: 720 } };
      if (id) video.deviceId = { exact: id };
      else video.facingMode = { ideal: 'environment' };
      return { video, audio: false };
    };
    let stream = null, err = null, switchFailed = false;
    const ask = async (c) => {
      try { const s = await navigator.mediaDevices.getUserMedia(c); if (!live()) { stopTracks(s); return null; } return s; } catch (e) { err = err || e; return null; }
    };
    try {
      stream = await ask(constraints(deviceId));
      if (!stream && live() && deviceId && prevDeviceId && prevDeviceId !== deviceId) {
        stream = await ask(constraints(prevDeviceId));     // the other camera would not open: go back to the one that worked
        switchFailed = !!stream;
      }
      if (!stream && live() && deviceId) stream = await ask({ video: true, audio: false });
      if (!live()) { stopTracks(stream); return; }
      if (!stream) { state.camPending = false; cameraError(err); return; }
      if (state.src) { stopSource(true); setSource(null); }      // the kept photo / video: the camera replaces it now
      state.stream = stream;
      camVideo.srcObject = stream;
      camVideo.muted = true;
      // not awaited: play() never settles for a camera that sends no frames
      try { const pp = camVideo.play(); if (pp && pp.catch) pp.catch(() => { /* autoplay attribute and the frame loop retry */ }); } catch (e) { /* ignore */ }
      const ok = await waitFor(() => camVideo.videoWidth > 0 || !live(), 8000);
      if (!live()) { stopTracks(stream); return; }
      if (!ok) { stopSource(true); setSource(null); cameraError({ name: 'NoFrames' }); return; }
      state.camPending = false;
      const track = stream.getVideoTracks()[0];
      state.facing = facingOf(track);
      const mirror = state.mirrorOverride === '1' ? true : state.mirrorOverride === '0' ? false : state.facing === 'user';
      setSource({ kind: 'camera', el: camVideo, w: camVideo.videoWidth, h: camVideo.videoHeight, mirror, live: true,
        name: (state.facing === 'environment' ? 'Back camera' : 'Front camera') + (track && track.label ? ' (' + track.label + ')' : ''),
        deviceId: track && track.getSettings ? track.getSettings().deviceId : null });
      if (switchFailed) say('Could not switch to the other camera. Staying on this one.', { minGap: 0, speak: false });
      if (track) {
        track.addEventListener('ended', () => { if (state.stream === stream) cameraEnded(); });
        track.addEventListener('mute', () => { if (state.stream === stream) { state.camMuted = true; updateStatus(true); } });
        track.addEventListener('unmute', () => { if (state.stream === stream) { state.camMuted = false; updateStatus(true); } });
      }
      try {
        const devs = await navigator.mediaDevices.enumerateDevices();
        if (!live()) return;
        state.devices = devs.filter((d) => d.kind === 'videoinput');
        $('switch').hidden = state.devices.length < 2;
      } catch (e) { $('switch').hidden = true; }
    } finally {
      if (live()) state.camPending = false;
      setStartButton();
    }
  }

  function cameraEnded() {
    stopSource(); setSource(null); cameraError({ name: 'Ended' });
  }

  // The Start button keeps one width whatever it says (a wider label wrapped the row and moved every
  // button below it while the camera permission prompt was open).
  const START_LABELS = ['Start camera', 'Stop camera', 'Starting…'];
  function fixStartWidth() {
    const b = $('start'), t0 = b.textContent;
    let w = 0;
    for (const t of START_LABELS) { b.textContent = t; w = Math.max(w, b.getBoundingClientRect().width); }
    b.textContent = t0;
    if (w > 0) b.style.minWidth = Math.ceil(w) + 'px';
  }
  function setStartButton() {
    const b = $('start');
    const t = state.camPending ? START_LABELS[2] : state.src && state.src.kind === 'camera' ? START_LABELS[1] : START_LABELS[0];
    if (b.textContent !== t) b.textContent = t;
    b.disabled = state.camPending;
    $('ov-start').disabled = state.camPending;
  }

  function waitFor(cond, ms) {
    return new Promise((resolve) => {
      const t0 = performance.now();
      (function poll() {
        if (cond()) resolve(true);
        else if (performance.now() - t0 > ms) resolve(false);
        else setTimeout(poll, 50);
      })();
    });
  }

  function cameraError(e) {
    const name = (e && e.name) || '';
    let title = 'Camera not available', text;
    if (name === 'NoMediaDevices') {
      text = 'This browser does not let this page use a camera. Camera access needs a secure (https) page or a file opened from your own computer.';
    } else if (name === 'NotAllowedError' || name === 'SecurityError' || name === 'PermissionDeniedError' || name === 'NotSupportedError') {
      title = 'Camera access is blocked';
      text = 'If your browser asked and the answer was no, allow the camera for this page in the browser’s site settings and press Start camera again. Pages shown inside another page, such as a chat preview, are often not allowed to use the camera at all.';
    } else if (name === 'NotFoundError' || name === 'DevicesNotFoundError' || name === 'OverconstrainedError') {
      text = 'No camera was found on this device.';
    } else if (name === 'NotReadableError' || name === 'TrackStartError' || name === 'AbortError') {
      text = 'The camera could not be started. Another app may be using it.';
    } else if (name === 'NoFrames') {
      text = 'The camera started but sent no picture.';
    } else if (name === 'Ended') {
      title = 'Camera stopped';
      text = 'The camera was switched off or disconnected.';
    } else {
      text = 'The camera could not be started (' + (name || String(e)) + ').';
    }
    if (!state.src) showOverlay(title, 'See below for other ways to try it.', true);   // a kept photo / video stays visible
    $('ov-start').textContent = 'Try again';
    $('notice-title').textContent = title;
    $('notice-text').textContent = text;
    $('notice').hidden = false;
    setStartButton();
    state.cameraError = name || 'error';
    updateStatus(true);
  }

  function hideNotice() { $('notice').hidden = true; }

  async function openFile(file) {
    if (!file) return;
    hideNotice();
    stopSource();
    setSource(null);
    const isImage = (file.type || '').startsWith('image/') || /\.(jpe?g|png|gif|webp|bmp|avif)$/i.test(file.name || '');
    try {
      if (isImage) await openImage(file);
      else await openVideo(file);
    } catch (e) {
      showOverlay('Could not open that file', (e && e.message) || 'This browser could not read it.', true);
    }
  }

  async function openImage(file) {
    let bmp = null;
    try { bmp = await createImageBitmap(file, { imageOrientation: 'from-image' }); } catch (e) {
      try { bmp = await createImageBitmap(file); } catch (e2) { bmp = await imgFromDataUrl(file); }
    }
    setSource({ kind: 'image', el: bmp, w: bmp.width || bmp.naturalWidth, h: bmp.height || bmp.naturalHeight,
      mirror: false, live: false, name: file.name || 'photo' });
  }

  function readDataUrl(file) {
    return new Promise((resolve, reject) => {
      const fr = new FileReader();
      fr.onload = () => resolve(fr.result);
      fr.onerror = () => reject(fr.error || new Error('could not read the file'));
      fr.readAsDataURL(file);
    });
  }

  async function imgFromDataUrl(file) {
    const url = await readDataUrl(file);
    const img = new Image();
    await new Promise((resolve, reject) => { img.onload = resolve; img.onerror = () => reject(new Error('not an image this browser can show')); img.src = url; });
    return img;
  }

  function loadVideo(url) {
    return new Promise((resolve, reject) => {
      const done = (ok) => { fileVideo.onloadeddata = null; fileVideo.onerror = null; ok ? resolve() : reject(new Error('this browser cannot play that video here')); };
      fileVideo.onloadeddata = () => done(true);
      fileVideo.onerror = () => done(false);
      fileVideo.src = url;
      fileVideo.load();
    });
  }

  async function openVideo(file) {
    let url = null;
    try {
      url = URL.createObjectURL(file);
      await loadVideo(url);
    } catch (e) {
      if (url) { try { URL.revokeObjectURL(url); } catch (e2) { /* ignore */ } url = null; }
      if (file.size > 100e6) throw new Error('This page could not play the video, and it is too large to load another way. Try a shorter clip or a photo.');
      await loadVideo(await readDataUrl(file));    // blob: URLs may be blocked inside an embedded page
    }
    fileVideo.loop = true; fileVideo.muted = true;
    // not awaited for long: a browser that blocks autoplay leaves it paused; a tap on the picture plays it
    try { await withTimeout(fileVideo.play(), 3000); } catch (e) { /* plays on the next tap */ }
    setSource({ kind: 'video', el: fileVideo, w: fileVideo.videoWidth, h: fileVideo.videoHeight, mirror: false, live: true,
      name: file.name || 'video', url });
  }

  // ------------------------------------------------------------ canvas
  let needResize = true;
  if (window.ResizeObserver) new ResizeObserver(() => { needResize = true; }).observe(view);
  window.addEventListener('resize', () => { needResize = true; navUi.dirty = true; });

  function resizeCanvas() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2.5);
    if (!needResize && dpr === state.dpr) return;
    needResize = !window.ResizeObserver;
    const r = view.getBoundingClientRect();
    const w = Math.max(1, Math.round(r.width * dpr)), h = Math.max(1, Math.round(r.height * dpr));
    if (view.width !== w || view.height !== h) { view.width = w; view.height = h; }
    state.dpr = dpr;
  }

  function contain(sw, sh, cw, ch) {
    const s = Math.min(cw / sw, ch / sh);
    const w = sw * s, h = sh * s;
    return { x: (cw - w) / 2, y: (ch - h) / 2, w, h };
  }

  // normalised raw-frame box -> canvas px rect (mirrors x for a front camera)
  function toCanvas(b, fit, mirror) {
    const x = mirror ? 1 - b.x - b.w : b.x;
    return { x: fit.x + x * fit.w, y: fit.y + b.y * fit.h, w: b.w * fit.w, h: b.h * fit.h };
  }

  // ------------------------------------------------------------ frame loop
  function frame(ts) {
    requestAnimationFrame(frame);
    if (document.hidden) return;
    resizeCanvas();
    const src = state.src;
    state.newFrame = false;
    if (src && src.kind === 'camera') {
      // a camera track can end without an 'ended' event (stopped elsewhere, unplugged on some
      // systems), or be muted (taken by another app, privacy shutter): check it every frame
      const tr = state.stream && state.stream.getVideoTracks()[0];
      if (!tr || tr.readyState === 'ended') { cameraEnded(); draw(ts); return; }
      if (tr.muted !== state.camMuted) { state.camMuted = tr.muted; updateStatus(true); }
    }
    watchDetection(ts);
    if (src && src.live) {
      const el = src.el;
      if (el.readyState >= 2 && el.videoWidth) {
        if (el.videoWidth !== src.w || el.videoHeight !== src.h) {   // phone rotated, or camera changed size
          src.w = el.videoWidth; src.h = el.videoHeight;
          $('screen').style.aspectRatio = src.w + ' / ' + src.h;
          state.results = null; state.lastPersons = null;
          resetTracker();
          // the turn tracker cannot follow a rotation of the picture itself: say so instead of
          // dropping the mark silently
          if (state.mark) { clearMark(true); say('Way out mark cleared: the picture turned. Mark it again.', { minGap: 0 }); }
        }
        if (el.currentTime !== state.lastVideoTime) {
          state.lastVideoTime = el.currentTime;
          state.newFrame = true;
          state.fpsCount++;
          // the turn tracker gets at most ~30 % of this thread: on a slow CPU it runs less often
          // (down to 2 per second) instead of leaving no time for taps and frames; not at all in the
          // first 0.4 s of a new picture source (the camera's start-up work and taps come first)
          const gap = Math.min(TRACK_MAX_GAP_MS, Math.max(25, Math.max(state.trackMs, TRACK_START_MS * !state.trackRuns) / TRACK_SHARE));
          if (ts - state.lastTrackT >= gap && performance.now() >= state.trackAfterT) {
            state.lastTrackT = ts;
            const t0 = performance.now();
            track(src, ts);
            const dt = performance.now() - t0;
            state.trackRuns = (state.trackRuns || 0) + 1;
            state.trackMs = state.trackRuns === 1 ? dt : dt > state.trackMs ? 0.5 * (state.trackMs + dt) : 0.85 * state.trackMs + 0.15 * dt;
          }
        }
      }
    }
    if (ts - state.fpsT0 >= 1000) {
      // frames the page saw, or (when the <video> is shown directly) frames the browser presented,
      // which keep coming while a slow CPU detector blocks this thread
      let fps = state.fpsCount * 1000 / (ts - state.fpsT0);
      const q = src && src.live && src.el.getVideoPlaybackQuality ? src.el.getVideoPlaybackQuality() : null;
      const shown = q ? q.totalVideoFrames : 0;
      if (q && shown > 0 && state.fpsShownEl === src.el && state.fpsT0 > 0) fps = Math.max(fps, (shown - state.fpsShownFrames) * 1000 / (ts - state.fpsT0));
      state.fpsShownEl = src && src.el; state.fpsShownFrames = shown;
      state.fps = fps; state.fpsCount = 0; state.fpsT0 = ts;
    }
    draw(ts);
    navFrame(ts);
    exitAlerts(ts);
    pumpSpeech();
    maybeInfer(ts);
    if (ts - state.lastStatusT > 300) updateStatus();
  }

  function track(src, ts) {
    const n = tracker.o.n;
    trkCtx.drawImage(src.el, 0, 0, n, n);
    let d;
    try { d = trkCtx.getImageData(0, 0, n, n).data; } catch (e) { return; }
    M.greyFromRGBA(d, n, grey);
    const r = tracker.update(grey, src.w, src.h, ts);
    state.stats.trackStates[r.state] = (state.stats.trackStates[r.state] || 0) + 1;
    navFeedTracker(r);
    if (window.__psTrackHook) { try { window.__psTrackHook(src.el, tracker.pose, r); } catch (e) { /* test hook */ } }
  }

  // ------------------------------------------------------------ inference
  function maybeInfer(ts) {
    const src = state.src;
    if (!state.modelsReady || !state.engine || state.busy || !src || document.hidden || state.recovering) return;
    if (src.kind === 'image') {
      if (state.inferredSrc === src && state.inferredFd === state.fd) return;
    } else {
      const el = src.el;
      if (el.readyState < 2 || !el.videoWidth) return;
      if (src.kind === 'camera' && state.camMuted) return;          // the camera sends no picture
      if (el.paused && el.currentTime === state.lastInferVideoTime) return;
      if (src.blankUntil && ts < src.blankUntil) return;           // last frame was blank: retry shortly
      // CPU backend on this thread: a pause between updates keeps the page responsive (the worker needs none)
      if (state.backend === 'cpu' && state.engineMode !== 'worker' && ts - state.lastInferEnd < 250) return;
    }
    // fallback engine on this thread: an idle gap after every update, so taps and frames get through
    if (state.engineMode === 'main' && ts - state.lastInferEnd < Math.max(MAIN_IDLE_MS, 0.25 * (state.lastBusyMs || 0))) return;
    infer(src);
  }


  function withTimeout(p, ms) {
    return new Promise((resolve, reject) => {
      const t = setTimeout(() => reject(new Error('timed out after ' + Math.round(ms / 1000) + ' s')), ms);
      Promise.resolve(p).then((v) => { clearTimeout(t); resolve(v); }, (e) => { clearTimeout(t); reject(e); });
    });
  }

  function effectiveHfov(W, H) {
    const f = M.focalPx(W, H, FOV);
    return 2 * Math.atan((W / 2) / f) * 180 / Math.PI;
  }

  // An all-black capture: a video or camera whose first frame has not been presented yet (or a
  // lens that is covered). Sampled sparsely; only checked until a real picture has been seen.
  function blankPixels(d) {
    const step = 4 * 97;
    for (let i = 0; i < d.length; i += step) if (d[i] > 6 || d[i + 1] > 6 || d[i + 2] > 6) return false;
    return true;
  }

  // Person boxes kept from the last person run (CPU: the person model runs every second update),
  // moved by how far the camera has turned since, so they stay on the person.
  function carriedPersons(capPose, W, H) {
    const lp = state.lastPersons;
    if (!lp || performance.now() - lp.t > Math.max(8000, 3 * (state.lastCycleMs || 0))) return [];
    const dyaw = M.wrap180(capPose.yaw - lp.pose.yaw), dp = capPose.pitch - lp.pose.pitch;
    if (Math.abs(dyaw) > 30 || Math.abs(dp) > 30) return [];
    const f = M.focalPx(W, H, FOV);
    const dx = -f * Math.tan(dyaw * Math.PI / 180) / W, dy = f * Math.tan(dp * Math.PI / 180) / H;
    return lp.dets.map((d) => Object.assign({}, d, { x: d.x + dx, y: d.y + dy }))
      .filter((d) => d.x + d.w > 0.02 && d.x < 0.98 && d.y + d.h > 0.02 && d.y < 0.98);
  }

  async function infer(src) {
    const gen = ++state.inferGen;      // a stalled run (lost graphics context) is abandoned by bumping this
    const current = () => gen === state.inferGen;
    const client = state.engine;
    state.busy = true;
    const t0 = performance.now();
    state.inferT0 = t0;
    const capPose = { yaw: tracker.pose.yaw, pitch: tracker.pose.pitch };
    const W = src.w, H = src.h;
    const scale = Math.min(1, 640 / Math.max(W, H));
    const cw = Math.max(1, Math.round(W * scale)), ch = Math.max(1, Math.round(H * scale));
    if (cap.width !== cw || cap.height !== ch) { cap.width = cw; cap.height = ch; }
    let capInfo = null;
    try {
      capCtx.drawImage(src.el, 0, 0, cw, ch);
      let im = null;
      if (src.kind !== 'image' && !src.seenPicture) {
        try { im = capCtx.getImageData(0, 0, cw, ch); } catch (e) { im = null; }
        if (im && blankPixels(im.data)) { src.blankUntil = performance.now() + 300; return; }
        src.seenPicture = true;
      }
      if (src.kind !== 'image') state.lastInferVideoTime = src.el.currentTime;
      if (window.__psCaptureHook) { try { capInfo = window.__psCaptureHook(cap, capCtx); } catch (e) { /* test hook */ } }
      const hfov = effectiveHfov(W, H);
      const cpu = state.backend === 'cpu';
      const runPerson = !cpu || state.cycle % 2 === 0 || !state.lastPersons || src.kind === 'image';
      state.cycle++;
      // the fire/door centre pass: every update, except on the CPU backend only when the person model rests
      const opts = { hfovDeg: hfov, person: runPerson, zoom: src.kind === 'image' || (cpu ? !runPerson : 'auto'), fd: !!state.fdReady };
      // worker: the pixels (transferred, not copied); this thread: the canvas itself
      const frame = client.mode === 'worker' ? (im || capCtx.getImageData(0, 0, cw, ch)) : cap;
      const r = await client.detect(frame, opts);
      if (!current()) return;
      let people;
      if (runPerson) {
        people = r.display;
        state.lastPersons = { dets: r.detections.filter((d) => d.cls === 'person'), t: t0, pose: capPose };
      } else {
        const faces = r.detections.filter((d) => d.cls === 'face');
        people = PSPeople.merge(carriedPersons(capPose, W, H).concat(faces), r.width, r.height, { hfovDeg: hfov });
      }
      const fd = r.fd || [];
      if (state.src !== src) return;     // source changed meanwhile
      const door = fd.filter((d) => d.cls === 'door');
      const res = {
        people, fire: fd.filter((d) => d.cls === 'fire'), door,
        window: windowsShown(fd.filter((d) => d.cls === 'window'), door),
        pose: capPose, t: t0, w: W, h: H,
        ms: { person: runPerson ? r.ms.person : null, face: r.ms.face, firedoor: r.ms.firedoor, total: performance.now() - t0 },
      };
      state.results = res;
      state.numTensors = r.numTensors;
      state.runsSinceStart++;
      state.paused = false; state.slowSaid = false;
      if (state.backend === 'webgl') state.incidentStreak = 0;     // graphics work again: forget earlier resets
      state.lastCycleMs = state.upsTimes.length ? performance.now() - state.upsTimes[state.upsTimes.length - 1] : res.ms.total;
      if (src.kind === 'image') { state.inferredSrc = src; state.inferredFd = state.fd; }
      state.stats.inferences++;
      state.stats.tensors.push(r.numTensors);
      if (state.stats.tensors.length > 2000) state.stats.tensors.splice(0, 1000);
      state.stats.ms.push(res.ms);
      if (state.stats.ms.length > 500) state.stats.ms.splice(0, 250);
      state.stats.labels.push(people.map((d) => d.label || '?').concat(res.fire.map(() => 'FIRE'), res.door.map(() => 'DOOR'), res.window.map(() => 'WINDOW')).join(' '));
      if (state.stats.labels.length > 500) state.stats.labels.splice(0, 250);
      publishDetections(res, src, capInfo);
      state.upsTimes.push(performance.now());
      while (state.upsTimes.length > 1 && performance.now() - state.upsTimes[0] > 5000) state.upsTimes.shift();
      detectionAlerts(res, src);
      updateStatus(true);
    } catch (e) {
      if (!current()) return;
      state.inferError = e;
      state.stats.errors.push(String(e && e.message || e));
      console.warn('PyroSight Camera: detection failed', e);
      if ((e && e.lost) || gpuLost()) onGpuLost('error');
    } finally {
      if (current()) {
        state.busy = false;
        state.lastInferEnd = performance.now();
        state.lastBusyMs = state.lastInferEnd - t0;
      }
    }
  }

  // ------------------------------------------------------------ detector health
  // Phones drop the WebGL context under memory pressure or after the page was in the background.
  // TF.js then never finishes the pending read (busy for good). Detect it (the worker's 'lost'
  // message or the context-lost event, isContextLost(), or a run that takes far longer than usual),
  // drop the old boxes, tell the viewer and restart the detectors: a new worker (or, on this thread,
  // a new WebGL backend), on the CPU after repeated losses.
  function gpuLost() {
    try { return !!(state.engine && state.engine.gpuLost()); } catch (e) { return false; }
  }

  function typicalRunMs() {
    const m = state.stats.ms.slice(-20).map((x) => x.total).sort((a, b) => a - b);
    return m.length ? m[m.length >> 1] : 0;
  }

  // How long the detectors may be silent (no beat from the worker, no step of the engine on this
  // thread) before they count as dead or hung: generous on the CPU, and at least 3x the longest
  // silence seen while they were working normally (a slow phone's long steps).
  function quietLimitMs(cl) {
    const base = cl && cl.mode === 'worker' ? (state.backend === 'cpu' ? 30000 : 8000) : 20000;
    return Math.max(base, 3 * (cl && cl.maxGap ? cl.maxGap() : 0));
  }
  // How long one run may take in the worker (which keeps beating while a WebGL run hangs): the
  // first run after a (re)start is not judged by a fixed 20 s (a slow CPU needs longer).
  function runLimitMs() {
    if (!state.runsSinceStart) return state.backend === 'cpu' ? 180000 : 60000;
    return Math.max(20000, 8 * typicalRunMs());
  }

  function watchDetection(ts) {
    if (!state.modelsReady || state.recovering || ts - state.lastCheckT < 500) return;
    state.lastCheckT = ts;
    if (gpuLost()) { onGpuLost('poll'); return; }
    if (!state.busy) return;
    const now = performance.now(), runMs = now - state.inferT0, cl = state.engine, src = state.src;
    // much longer than usual: the boxes on screen are out of date; drop them and say so
    if (!state.paused && src && src.live && runMs > Math.max(5000, 3 * typicalRunMs())) {
      state.paused = true; state.results = null; state.lastPersons = null;
      updateStatus(true);
    }
    // nothing heard from the detectors: a worker that died or hangs, a run that stopped moving
    const quiet = now - Math.max(state.inferT0, cl && cl.progressAt ? cl.progressAt() : 0);
    if (quiet > quietLimitMs(cl)) { onGpuLost('stall'); return; }
    // the worker still answers but this run takes far too long (a WebGL run that never finishes)
    if (cl && cl.mode === 'worker' && runMs > runLimitMs()) {
      if (state.lastRecovery === 'stall' && !state.runsSinceStart) {
        // restarted for this already and no run has finished since: this device is just slow.
        // Restarting again would reload the models and time out again, forever: keep waiting.
        if (!state.slowSaid) { state.slowSaid = true; say('Detection is very slow on this device. Still trying.', { minGap: 0, speak: false }); }
        return;
      }
      onGpuLost('stall');
    }
  }

  async function onGpuLost(why) {
    if (state.recovering || !state.modelsReady) return;
    state.recovering = true;
    state.modelsReady = false;
    state.inferGen++; state.busy = false;              // abandon the run that will never finish
    state.results = null; state.lastPersons = null;    // its boxes are stale: stop drawing them
    if (why !== 'stall') state.gpuLosses++; else state.stalls++;
    state.incidentStreak++;
    state.lastRecovery = why === 'stall' ? 'stall' : 'lost';
    state.paused = false;
    state.stats.recoveries.push({ why, t: Math.round(performance.now()) });
    say(why === 'stall' ? 'Detection stalled. Restarting.' : 'Detection stopped: the graphics chip was reset. Restarting.', { minGap: 0 });
    updateStatus(true);
    const prev = state.engine;
    state.engine = null;
    try { if (prev) prev.terminate(true); } catch (e) { /* lost context */ }
    await new Promise((r) => setTimeout(r, 300));
    const mode = prev ? prev.mode : undefined;
    try {
      try {
        // keeps failing (3 resets or stalls with no completed WebGL run in between): the CPU from now on
        await withTimeout(startEngine({ mode, prefer: state.incidentStreak >= 3 ? 'cpu' : undefined }), 180000);
      } catch (e) {
        if (state.backend === 'cpu') throw e;
        if (state.engine) { try { state.engine.terminate(true); } catch (x) { /* ignore */ } state.engine = null; }
        await withTimeout(startEngine({ mode, prefer: 'cpu' }), 180000);
      }
      state.modelsReady = true;
      state.loadError = null;
      say('Detection running again' + (state.backend === 'cpu' ? ', slower, on the processor.' : '.'), { minGap: 0, speak: false });
    } catch (e) {
      state.loadError = e;
      console.error('PyroSight Camera: detectors could not be restarted', e);
      say('Detection could not be restarted. Reload the page.', { minGap: 0 });
    }
    state.recovering = false;
    updateStatus(true);
  }

  // Test / integration hook: the latest boxes (and a bounded history) as plain data,
  // normalised 0..1 in the camera frame (not mirrored), x, y = top-left.
  const detLog = [];
  function publishDetections(res, src, capInfo) {
    const box = (d, extra) => Object.assign({ x: +d.x.toFixed(4), y: +d.y.toFixed(4), w: +d.w.toFixed(4), h: +d.h.toFixed(4), score: +d.score.toFixed(4) }, extra);
    const rec = {
      i: state.stats.inferences, t: Math.round(res.t), src: src.kind, w: res.w, h: res.h, mirror: !!src.mirror,
      videoTime: src.kind === 'image' ? null : src.el.currentTime,
      backend: state.backend, software: !!(state.backendInfo && state.backendInfo.software),
      ms: { person: res.ms.person == null ? null : Math.round(res.ms.person), face: Math.round(res.ms.face),
        firedoor: Math.round(res.ms.firedoor), total: Math.round(res.ms.total) },
      tensors: state.numTensors,
      engine: state.engineMode,
      people: res.people.map((d) => box(d, { label: d.label || '', from: d.src, dist: d.dist == null ? null : +d.dist.toFixed(2) })),
      fire: res.fire.map((d) => box(d, { label: 'FIRE' })),
      door: res.door.map((d) => box(d, { label: 'DOOR' })),
      window: (res.window || []).map((d) => box(d, { label: 'WINDOW' })),
      capture: capInfo || null,
    };
    window.__psLastDetections = rec;
    detLog.push(rec);
    if (detLog.length > 1000) detLog.splice(0, 500);
    window.__psDetections = detLog;
  }

  // ------------------------------------------------------------ drawing
  function shiftSince(res, src) {
    // Move boxes by how far the camera has turned since their frame was captured
    // (the detectors can take a second on slow devices). Static objects stay put.
    if (!res || !src.live || tracker.state !== 'ok') return { dx: 0, dy: 0 };
    const dyaw = M.wrap180(tracker.pose.yaw - res.pose.yaw), dp = tracker.pose.pitch - res.pose.pitch;
    if (Math.abs(dyaw) > 30 || Math.abs(dp) > 30) return { dx: 0, dy: 0 };
    const f = M.focalPx(src.w, src.h, FOV);
    return { dx: -f * Math.tan(dyaw * Math.PI / 180) / src.w, dy: f * Math.tan(dp * Math.PI / 180) / src.h };
  }

  // Live video is shown by the <video> element itself under a transparent canvas (boxes only):
  // the browser keeps presenting camera frames even while a slow (CPU) detector blocks this
  // thread. Photos and the eyepiece view are drawn into the canvas.
  const videoShown = new Map();
  function showVideo(src) {
    for (const el of [camVideo, fileVideo]) {
      const on = !!(src && src.el === el), key = on ? (src.mirror ? 'm' : 'n') : 'off';
      if (videoShown.get(el) === key) continue;
      videoShown.set(el, key);
      el.style.opacity = on ? '1' : '0';
      el.style.transform = on && src.mirror ? 'scaleX(-1)' : '';
    }
  }

  function draw(ts) {
    const cw = view.width, ch = view.height, s = state.dpr;
    const src = state.src;
    const direct = !!(src && src.live && !state.eyepiece && src.w && src.h);
    showVideo(direct ? src : null);
    g.setTransform(1, 0, 0, 1, 0, 0);
    if (direct) g.clearRect(0, 0, cw, ch);
    else {
      g.fillStyle = '#0a0d12';
      g.fillRect(0, 0, cw, ch);
    }
    const sim = navSimView();
    if (sim !== navUi.simShown) { navUi.simShown = sim; $('screen').classList.toggle('sim', sim); }
    if (!src || !src.w || !src.h) {
      state.fit = null;
      if (sim) drawSimView(ts); else navUi.overlay = null;
      return;
    }
    const fit = contain(src.w, src.h, cw, ch);
    state.fit = fit;
    if (state.eyepiece) drawEyepiece(src, fit);
    else if (!direct) {
      g.imageSmoothingEnabled = true;
      g.save();
      if (src.mirror) { g.translate(fit.x + fit.w, fit.y); g.scale(-1, 1); g.drawImage(src.el, 0, 0, fit.w, fit.h); }
      else g.drawImage(src.el, fit.x, fit.y, fit.w, fit.h);
      g.restore();
    }
    const res = resultsToDraw(src);
    const ex = navUi.on ? null : exitInfo(src);        // while navigation runs its marker replaces the mark
    const exitBox = ex && ex.trusted && ex.inView ? ex.box : null;
    // boxes first, labels last (FIRE on top), so no box edge runs through a label
    const labels = [];
    if (res) {
      const sh = shiftSince(res, src);
      const mv = (b) => ({ x: b.x + sh.dx, y: b.y + sh.dy, w: b.w, h: b.h });
      for (const [list, word] of [[res.door, 'DOOR'], [res.window || [], 'WINDOW']]) {
        for (const d of list) {
          if (exitBox && iou(mv(d), exitBox) > 0.3) continue;   // the EXIT mark is on this door or window: one green box
          drawBox(toCanvas(mv(d), fit, src.mirror), COL.exit, word, fit, s, labels, 1);
        }
      }
      for (const d of res.people) drawBox(toCanvas(mv(d), fit, src.mirror), COL.person, d.label || '', fit, s, labels, 2);
      for (const d of res.fire) drawBox(toCanvas(mv(d), fit, src.mirror), COL.fire, 'FIRE', fit, s, labels, 3);
    }
    if (navUi.on) drawNavExit(src, fit, s, ts, labels, state.eyepiece); else { navUi.overlay = null; drawExit(src, fit, s, ts, labels); }
    const ring = navUi.on && state.eyepiece;          // eyepiece view: the device's navigation ring, top centre
    drawLabels(labels, fit, s, ring ? [navRingRect(fit)] : null);
    if (ring) drawNavRing(fit, s, ts);
    drawHud(src, fit, s);
  }

  // Boxes are dropped once they are clearly out of date on a moving picture (detection stopped or
  // stalled), instead of being left on whatever the camera shows now.
  function resultsToDraw(src) {
    const res = state.results;
    if (!res || !src || !src.live) return res;
    if (src.kind === 'video' && src.el.paused) return res;
    const age = performance.now() - res.t;
    return age > Math.max(6000, 4 * typicalRunMs()) ? null : res;
  }

  function iou(a, b) {
    const ix = Math.max(0, Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x));
    const iy = Math.max(0, Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y));
    const i = ix * iy, u = a.w * a.h + b.w * b.h - i;
    return u > 0 ? i / u : 0;
  }

  // WINDOW boxes shown: none where a DOOR box is (a glazed door is a door), and only the strongest
  // MAX_WINDOWS (a building front can hold dozens of windows; a few green boxes already show the way)
  const MAX_WINDOWS = 4;
  function windowsShown(wins, doors) {
    const onDoor = (w, d) => {
      const ix = Math.max(0, Math.min(w.x + w.w, d.x + d.w) - Math.max(w.x, d.x));
      const iy = Math.max(0, Math.min(w.y + w.h, d.y + d.h) - Math.max(w.y, d.y));
      return iou(w, d) >= 0.3 || ix * iy >= 0.6 * w.w * w.h;
    };
    return wins.filter((w) => !doors.some((d) => onDoor(w, d))).sort((a, b) => b.score - a.score).slice(0, MAX_WINDOWS);
  }

  function setFont(px) { g.font = '700 ' + Math.round(px) + 'px ' + MONO; }

  function outlinedText(text, x, y, color, s) {
    g.lineJoin = 'round';
    g.lineWidth = 3.5 * s;
    g.strokeStyle = 'rgba(0,0,0,0.9)';
    g.strokeText(text, x, y);
    g.fillStyle = color;
    g.fillText(text, x, y);
  }

  function drawBox(r, color, label, fit, s, labels, pri, dashed) {
    // clip to the picture
    const x0 = Math.max(fit.x, r.x), y0 = Math.max(fit.y, r.y);
    const x1 = Math.min(fit.x + fit.w, r.x + r.w), y1 = Math.min(fit.y + fit.h, r.y + r.h);
    if (x1 - x0 < 2 || y1 - y0 < 2) return;
    const lw = Math.max(2, 2.5 * s);
    g.lineJoin = 'miter';
    g.setLineDash([]);
    g.lineWidth = lw + 2.5 * s;
    g.strokeStyle = 'rgba(0,0,0,0.75)';
    g.strokeRect(x0 + lw / 2, y0 + lw / 2, x1 - x0 - lw, y1 - y0 - lw);
    if (dashed) g.setLineDash([8 * s, 6 * s]);
    g.lineWidth = lw;
    g.strokeStyle = color;
    g.strokeRect(x0 + lw / 2, y0 + lw / 2, x1 - x0 - lw, y1 - y0 - lw);
    g.setLineDash([]);
    if (label) labels.push({ text: label, color, x0, y0, x1, y1, lw, pri });
  }

  // Labels above their box like the device; if that spot is taken by another label (or off the
  // picture), just inside the top edge, then below the box. Higher pri is placed first (FIRE).
  function drawLabels(labels, fit, s, reserved) {
    const fs = 14 * s;
    setFont(fs);
    g.textBaseline = 'alphabetic';
    g.textAlign = 'left';
    const placed = reserved ? reserved.slice() : [];
    const hit = (a) => placed.some((b) => a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.h && b.y < a.y + a.h);
    labels.sort((a, b) => b.pri - a.pri);
    const out = [];
    for (const L of labels) {
      const tw = g.measureText(L.text).width;
      const lx = Math.min(Math.max(L.x0, fit.x + 3 * s), fit.x + fit.w - tw - 3 * s);
      const spots = [L.y0 - 5 * s, L.y0 + L.lw + fs + 2 * s, L.y1 + fs + 2 * s, L.y1 - L.lw - 4 * s]
        .filter((y) => y - fs >= fit.y - 0.5 && y + 4 * s <= fit.y + fit.h + 0.5);
      let ly = spots.find((y) => !hit({ x: lx - 2 * s, y: y - fs, w: tw + 4 * s, h: fs + 4 * s }));
      if (ly === undefined) ly = spots.length ? spots[0] : L.y0 + L.lw + fs + 2 * s;
      placed.push({ x: lx - 2 * s, y: ly - fs, w: tw + 4 * s, h: fs + 4 * s });
      out.push([L.text, lx, ly, L.color]);
    }
    for (let i = out.length - 1; i >= 0; i--) outlinedText(out[i][0], out[i][1], out[i][2], out[i][3], s);   // FIRE drawn last, on top
  }

  /*
   * The device's own three palettes, ported value for value from
   * core/src/ps_display.c (build_luts). Showing an invented colour ramp here
   * would misrepresent what the eyepiece looks like, which is the one thing
   * this view exists to show.
   *
   * The data behind them is still camera brightness, not temperature — see
   * drawEyepiece. A bright white shirt reads "hot" and a black radiator reads
   * "cold"; that is the honest limit of doing this with an RGB camera, and
   * the page says so on screen.
   */
  const EYE_PALETTES = (function () {
    const iron = [[0, 0, 0], [80, 0, 140], [200, 30, 60], [250, 140, 0], [255, 255, 220]];
    const whiteHot = new Uint8Array(256 * 3), ironLut = new Uint8Array(256 * 3), night = new Uint8Array(256 * 3);
    for (let v = 0; v < 256; v++) {
      whiteHot[v * 3] = whiteHot[v * 3 + 1] = whiteHot[v * 3 + 2] = v;
      // Math.fround keeps the interpolation in float32, which is what the
      // device computes in. In double precision six of the 256 ironbow entries
      // land one level off (an exact boundary such as v=68 falling to 1.9999
      // instead of 2.0), and the point of this view is to show the device's
      // picture, not one that is nearly it.
      const fr = Math.fround;
      const t = fr(fr(v / 255) * 4), k = Math.min(3, Math.max(0, Math.floor(t))), f = fr(t - k);
      for (let c = 0; c < 3; c++) ironLut[v * 3 + c] = fr(iron[k][c] + fr((iron[k + 1][c] - iron[k][c]) * f));
      // Integer division, truncating, exactly as the C does: Math.round here
      // put the amber ramp one level above the device's on 190 of 256 values.
      const a = (v * 205 / 255) | 0;                       // amber at ~80% luminance
      night[v * 3] = a; night[v * 3 + 1] = (a * 140 / 255) | 0; night[v * 3 + 2] = 0;
    }
    return [
      { id: 'white', name: 'White-hot', lut: whiteHot },
      { id: 'iron', name: 'Ironbow', lut: ironLut },
      { id: 'night', name: 'Amber night', lut: night },
    ];
  })();

  function drawEyepiece(src, fit) {
    // Low resolution like the 160 x 120 thermal sensor, contrast stretched,
    // then through the device's palette. It shows brightness, not heat.
    const ew = 160, eh = Math.max(1, Math.round(160 * src.h / src.w));
    if (eye.width !== ew || eye.height !== eh) { eye.width = ew; eye.height = eh; eyeFrameKey = null; }
    const key = src.kind === 'image' ? src : state.lastVideoTime;
    if (key !== eyeFrameKey || state.newFrame) {
      eyeFrameKey = key;
      eyeCtx.drawImage(src.el, 0, 0, ew, eh);
      let im;
      try { im = eyeCtx.getImageData(0, 0, ew, eh); } catch (e) { im = null; }
      if (im) {
        const d = im.data, n = ew * eh, hist = new Uint32Array(256), Y = new Uint8Array(n);
        for (let i = 0, j = 0; i < n; i++, j += 4) { const v = (77 * d[j] + 150 * d[j + 1] + 29 * d[j + 2]) >> 8; Y[i] = v; hist[v]++; }
        let acc = 0, lo = 0, hi = 255;
        for (let v = 0; v < 256; v++) { acc += hist[v]; if (acc >= n * 0.02) { lo = v; break; } }
        acc = 0;
        for (let v = 255; v >= 0; v--) { acc += hist[v]; if (acc >= n * 0.02) { hi = v; break; } }
        const k = src.kind === 'image' ? 1 : 0.2;
        eyeLo += k * (lo - eyeLo); eyeHi += k * (hi - eyeHi);
        const span = Math.max(24, eyeHi - eyeLo);
        for (let i = 0, j = 0; i < n; i++, j += 4) {
          let v = (Y[i] - eyeLo) / span;
          v = v < 0 ? 0 : v > 1 ? 1 : v;
          v = Math.round(255 * Math.pow(v, 1.15));
          const lut = EYE_PALETTES[state.palette % EYE_PALETTES.length].lut, o = v * 3;
          d[j] = lut[o]; d[j + 1] = lut[o + 1]; d[j + 2] = lut[o + 2]; d[j + 3] = 255;
        }
        eyeCtx.putImageData(im, 0, 0);
      }
    }
    g.imageSmoothingEnabled = false;
    g.save();
    if (src.mirror) { g.translate(fit.x + fit.w, fit.y); g.scale(-1, 1); g.drawImage(eye, 0, 0, fit.w, fit.h); }
    else g.drawImage(eye, fit.x, fit.y, fit.w, fit.h);
    g.restore();
    g.imageSmoothingEnabled = true;
  }

  function exitInfo(src) {
    if (!state.mark || !src) return null;
    const p = M.project(state.mark, tracker.pose, src.w, src.h, FOV);
    // bearing for the viewer, + = left (device convention); mirrored view: screen left = your left
    p.bearing = src.mirror ? p.rel.yaw : -p.rel.yaw;
    const sens = tracker.sensorActive(performance.now());
    p.trusted = !(tracker.state === 'lost' && !sens);
    // 'weak': the picture did not match just now (something large moved, or a fast turn while the
    // page was busy): the position is the last good one and may be out of date
    p.unsure = src.live && tracker.state === 'weak' && !sens;
    return p;
  }

  function drawExit(src, fit, s, ts, labels) {
    const p = exitInfo(src);
    if (!p) return;
    if (!p.trusted) {
      setFont(13 * s); g.textAlign = 'center'; g.textBaseline = 'top';
      outlinedText('WAY OUT LOST', fit.x + fit.w / 2, fit.y + 8 * s, '#FFD84A', s);
      g.textAlign = 'left'; g.textBaseline = 'alphabetic';
      return;
    }
    if (p.inView && p.box) {
      drawBox(toCanvas(p.box, fit, src.mirror), COL.exit, p.unsure ? 'EXIT?' : 'EXIT', fit, s, labels, 1.5, p.unsure);
      return;
    }
    // off screen: arrow at the edge, pointing toward it
    const sign = src.mirror ? -1 : 1;
    let vx = sign * p.rel.yaw, vy = -p.rel.pitch;
    if (Math.abs(p.rel.yaw) > 90) vy *= 0.25;
    const turn = Math.round(Math.min(180, Math.hypot(p.rel.yaw, p.rel.pitch)));
    drawEdgeArrow(fit, s, vx, vy, 'EXIT ' + turn + '°' + (p.unsure ? '?' : ''));
  }

  // green arrow at the edge of the picture pointing along (vx, vy) (screen axes), with a label
  function drawEdgeArrow(fit, s, vx, vy, text) {
    const len = Math.hypot(vx, vy) || 1;
    vx /= len; vy /= len;
    const m = 30 * s, cx = fit.x + fit.w / 2, cy = fit.y + fit.h / 2;
    const tx = Math.abs(vx) > 1e-6 ? (fit.w / 2 - m) / Math.abs(vx) : Infinity;
    const ty = Math.abs(vy) > 1e-6 ? (fit.h / 2 - m) / Math.abs(vy) : Infinity;
    const t = Math.max(0, Math.min(tx, ty));
    const ax = cx + vx * t, ay = cy + vy * t;
    const r = 22 * s;
    const px = -vy, py = vx;
    const tip = [ax + vx * r, ay + vy * r], l = [ax - vx * r * 0.6 + px * r * 0.75, ay - vy * r * 0.6 + py * r * 0.75];
    const rr = [ax - vx * r * 0.6 - px * r * 0.75, ay - vy * r * 0.6 - py * r * 0.75], notch = [ax - vx * r * 0.15, ay - vy * r * 0.15];
    g.beginPath(); g.moveTo(tip[0], tip[1]); g.lineTo(l[0], l[1]); g.lineTo(notch[0], notch[1]); g.lineTo(rr[0], rr[1]); g.closePath();
    g.lineJoin = 'round'; g.lineWidth = 4 * s; g.strokeStyle = 'rgba(0,0,0,0.9)'; g.stroke();
    g.fillStyle = COL.exit; g.fill();
    setFont(13 * s);
    const tw = g.measureText(text).width;
    // beside the arrow's tail (centred behind it, a long label ran over the arrow itself)
    let lx = ax - vx * (r + 8 * s) - tw / 2, ly = ay - vy * (r + 8 * s) + 5 * s;
    if (Math.abs(vx) > 0.3) { lx = vx < 0 ? ax + r * 0.95 + 6 * s : ax - r * 0.95 - 6 * s - tw; ly = ay + 5 * s; }
    lx = Math.min(Math.max(lx, fit.x + 4 * s), fit.x + fit.w - tw - 4 * s);
    ly = Math.min(Math.max(ly, fit.y + 16 * s), fit.y + fit.h - 6 * s);
    g.textAlign = 'left'; g.textBaseline = 'alphabetic';
    outlinedText(text, lx, ly, COL.exit, s);
    return { x: ax, y: ay };
  }

  function hudText(text, x, y, color, s) {
    const w = g.measureText(text).width, h = 12 * s, pad = 3 * s;
    const x0 = g.textAlign === 'right' ? x - w : g.textAlign === 'center' ? x - w / 2 : x;
    g.fillStyle = 'rgba(0,0,0,0.6)';
    g.fillRect(x0 - pad, y - h - pad + 2 * s, w + 2 * pad, h + 2 * pad);
    outlinedText(text, x, y, color, s);
  }

  function drawHud(src, fit, s) {
    setFont(12 * s);
    g.textBaseline = 'alphabetic';
    const bottom = fit.y + fit.h - 8 * s;
    if (!state.modelsReady) {
      g.textAlign = 'left';
      hudText(state.loadError ? 'DETECTORS FAILED' : state.recovering ? 'DETECTION STOPPED · RESTARTING…' : 'LOADING DETECTORS…',
        fit.x + 8 * s, fit.y + 20 * s, state.loadError ? '#FF6B6B' : state.recovering ? '#FFD84A' : '#B4B4B4', s);
    } else if (src.live && state.results && !resultsToDraw(src)) {
      g.textAlign = 'left';
      hudText('DETECTION PAUSED', fit.x + 8 * s, fit.y + 20 * s, '#FFD84A', s);
    }
    if (src.kind === 'camera' && state.camMuted) {
      g.textAlign = 'center';
      hudText('NO PICTURE FROM THE CAMERA', fit.x + fit.w / 2, fit.y + fit.h / 2, '#FFD84A', s);
      g.textAlign = 'left';
    }
    if (state.eyepiece) {
      g.textAlign = 'left';
      hudText(state.modelsReady ? 'AI' : '--', fit.x + 8 * s, bottom, '#B4B4B4', s);
      g.textAlign = 'right';
      hudText('LOOK-ALIKE, NOT THERMAL', fit.x + fit.w - 8 * s, bottom, '#B4B4B4', s);
      if (!state.mark && !navUi.on) { g.textAlign = 'center'; hudText('MARK WAY OUT', fit.x + fit.w / 2, fit.y + 20 * s, '#FFFFFF', s); }
      g.textAlign = 'left';
    }
    if (src.kind === 'video' && src.el.paused) {
      g.textAlign = 'right';
      hudText('PAUSED · TAP TO PLAY', fit.x + fit.w - 8 * s, fit.y + 20 * s, '#FFFFFF', s);
      g.textAlign = 'left';
    }
  }

  // ------------------------------------------------------------ way out
  function markWayOut(nx, ny, how) {
    const src = state.src;
    if (!src || !src.w) return false;
    let cx = nx, cy = ny, size = { w: 14, h: 28 }, door = null, at = null;
    const res = state.results;
    const ways = res ? res.door.map((d) => [d, 'door']).concat((res.window || []).map((d) => [d, 'window'])) : [];
    if (ways.length) {
      // snap to a detected door or window at the tapped point (or near the centre for the button)
      const sh = shiftSince(res, src);
      let best = null, bestD = how === 'tap' ? 0 : 0.2;
      for (const [d, kind] of ways) {
        const b = { x: d.x + sh.dx, y: d.y + sh.dy, w: d.w, h: d.h, kind };
        const inside = nx >= b.x && nx <= b.x + b.w && ny >= b.y && ny <= b.y + b.h;
        const dist = Math.hypot(b.x + b.w / 2 - nx, b.y + b.h / 2 - ny);
        if (inside && (!best || dist < bestD)) { best = b; bestD = dist; }
        else if (!inside && how !== 'tap' && dist < bestD) { best = b; bestD = dist; }
      }
      if (best) {
        at = best.kind; door = best.kind === 'door' ? best : null;
        cx = best.x + best.w / 2; cy = best.y + best.h / 2;
        size = M.boxAngles(best, src.w, src.h, FOV);
      }
    }
    if (src.live) tracker.reanchor();   // a mark made while the picture is not matching starts a fresh key frame
    const dir = M.pointToDirection(cx, cy, tracker.pose, src.w, src.h, FOV);
    state.mark = { yaw: dir.yaw, pitch: dir.pitch, w: Math.max(4, size.w), h: Math.max(6, size.h), t: performance.now(), door: !!door, window: at === 'window', at, how };
    tracker.hasMark = true;
    state.markLostSaid = false; state.exitSector = null; state.exitOutSince = 0; state.lastExitSay = performance.now();
    $('clear-mark').hidden = false;
    say(at ? 'Way out marked at the ' + at + '.' : 'Way out marked.', { minGap: 0 });
    updateStatus(true);
    return true;
  }

  function clearMark(silent) {
    const had = !!state.mark;
    state.mark = null;
    tracker.hasMark = false;
    $('clear-mark').hidden = true;
    if (had && !silent) say('Way out mark cleared.', { minGap: 0, speak: false });
  }

  function whereOut() {
    const src = state.src, p = exitInfo(src);
    if (!p) { say('No way out marked.', { minGap: 0 }); return; }
    if (!p.trusted) { say('Way out tracking lost.', { minGap: 0 }); return; }
    say('Way out is ' + M.directionPhrase(p.bearing), { minGap: 0 });
    state.lastExitSay = performance.now();
  }

  // "lost" is said once it has lasted LOST_SAY_MS (a hand over the lens or someone walking past
  // for a moment is not worth a word), and not more than every LOST_GAP_MS; "restored" only after
  // a "lost" was said and tracking has held for RESTORED_SAY_MS.
  const LOST_SAY_MS = 2000, LOST_GAP_MS = 20000, RESTORED_SAY_MS = 2000;
  function exitAlerts(ts) {
    if (navUi.on) { state.exitLostSince = 0; state.exitOkSince = 0; state.exitOutSince = 0; return; }   // navigation speaks instead
    const src = state.src, p = exitInfo(src);
    if (!p) { state.exitLostSince = 0; state.exitOkSince = 0; return; }
    if (!p.trusted) {
      state.exitOkSince = 0;
      if (!state.exitLostSince) state.exitLostSince = ts;
      if (!state.markLostSaid && ts - state.exitLostSince >= LOST_SAY_MS && ts - (state.lastLostSay || -1e9) >= LOST_GAP_MS) {
        say('Way out tracking lost.', { minGap: 0 }); state.markLostSaid = true; state.lastLostSay = ts;
      }
      return;
    }
    state.exitLostSince = 0;
    if (!state.exitOkSince) state.exitOkSince = ts;
    if (state.markLostSaid) {
      if (ts - state.exitOkSince < RESTORED_SAY_MS) return;
      say('Way out tracking restored.', { minGap: 0 }); state.markLostSaid = false;
    }
    if (p.inView) { state.exitOutSince = 0; state.exitSector = null; return; }
    // out of view: say where it is once it has been out for a moment, and again
    // when the direction settles in a new sector (not more than every 4 s)
    if (!state.exitOutSince) { state.exitOutSince = ts; state.exitSaidOut = false; }
    const sec = M.directionSector(p.bearing);
    if (sec !== state.exitSector) { state.exitSector = sec; state.exitSectorSince = ts; state.exitSectorSaid = false; }
    const firstOut = !state.exitSaidOut && ts - state.exitOutSince > 800;
    const settled = !state.exitSectorSaid && ts - state.exitSectorSince > 1200 && ts - state.lastExitSay > 4000;
    if (firstOut || settled) {
      state.exitSaidOut = true; state.exitSectorSaid = true; state.lastExitSay = ts;
      say('Way out is ' + M.DIR_PHRASES[sec]);
    }
  }

  // ------------------------------------------------------------ alerts
  // Spoken alerts go through a small queue: nothing already being spoken is cut off; when the
  // speech engine is free the most important waiting phrase goes next (fire, then people, then the
  // rest), and phrases that waited too long to still be true are dropped.
  const speech = { q: [], cur: null, curT: 0, pumpQueued: false };
  function priorityOf(text) { return /^Fire/.test(text) ? 3 : /^Person/.test(text) ? 2 : /^(Detection|Way out tracking lost)/.test(text) ? 1.5 : 1; }

  function say(text, opts) {
    opts = opts || {};
    const now = performance.now();
    const minGap = opts.minGap === undefined ? 5000 : opts.minGap;
    const last = state.spoken.get(text);
    if (last !== undefined && now - last < minGap) return false;
    state.spoken.set(text, now);
    state.log.unshift({ t: now, text });
    if (state.log.length > 50) state.log.pop();
    renderLog();
    if (state.voice && opts.speak !== false && 'speechSynthesis' in window) {
      speech.q = speech.q.filter((x) => x.text !== text);
      speech.q.push({ text, pri: opts.pri || priorityOf(text), t: now, ttl: opts.ttl || 8000 });
      // pick after the current task, so phrases raised together (a person and a fire in one
      // update) are ordered by importance rather than by which was raised first
      if (!speech.pumpQueued) { speech.pumpQueued = true; Promise.resolve().then(() => { speech.pumpQueued = false; pumpSpeech(); }); }
    }
    return true;
  }

  function pumpSpeech() {
    if (!speech.q.length && !speech.cur) return;
    let ss;
    try { ss = window.speechSynthesis; } catch (e) { ss = null; }
    if (!ss) { speech.q = []; return; }
    const now = performance.now();
    if (ss.speaking || ss.pending) {
      if (!(speech.cur && now - speech.curT > 15000)) return;   // stuck engine (seen in some browsers): reset it
      try { ss.cancel(); } catch (e) { /* ignore */ }
    }
    speech.cur = null;
    speech.q = speech.q.filter((x) => now - x.t < x.ttl);
    if (!speech.q.length) return;
    speech.q.sort((a, b) => b.pri - a.pri || a.t - b.t);
    const it = speech.q.shift();
    try {
      const u = new SpeechSynthesisUtterance(it.text);
      u.rate = 1.05;
      u.onend = u.onerror = () => { if (speech.cur === u) { speech.cur = null; pumpSpeech(); } };
      speech.cur = u; speech.curT = now;
      ss.speak(u);
    } catch (e) { speech.cur = null; }
  }

  function stopSpeech() {
    speech.q = []; speech.cur = null;
    try { window.speechSynthesis.cancel(); } catch (e) { /* ignore */ }
  }

  function renderLog() {
    const ol = $('log');
    ol.textContent = '';
    if (!state.log.length) {
      const li = document.createElement('li'); li.className = 'empty'; li.textContent = 'Nothing yet.'; ol.appendChild(li);
      return;
    }
    state.log.slice(0, 20).forEach((a) => {
      const li = document.createElement('li'), tm = document.createElement('time'), sp = document.createElement('span');
      const sec = Math.floor((a.t - state.startT) / 1000);
      tm.textContent = Math.floor(sec / 60) + ':' + String(sec % 60).padStart(2, '0');
      sp.textContent = a.text;
      li.appendChild(tm); li.appendChild(sp); ol.appendChild(li);
    });
  }

  // Like ps_alerts_update_detections: call out when something (re)appears
  // after a gap, at most once per cooldown, with its side of the picture.
  function detectionAlerts(res, src) {
    const now = performance.now();
    const hfov = effectiveHfov(src.w, src.h);
    const pick = (list, min) => list.filter((d) => d.score >= min).sort((a, b) => b.score - a.score)[0];
    const one = (kind, det, word) => {
      const st = state.al[kind] || (state.al[kind] = { seen: false, lastSeen: 0, lastAlert: -1e9 });
      if (!det) return;
      const afterGap = !st.seen || now - st.lastSeen > ALERT_COOLDOWN_MS;
      const cooled = now - st.lastAlert >= ALERT_COOLDOWN_MS;
      st.seen = true; st.lastSeen = now;
      if (!afterGap || !cooled) return;
      const cx = det.x + det.w / 2;
      say(word + ' ' + M.sidePhrase(src.mirror ? 1 - cx : cx, hfov), { minGap: 3000 });
      st.lastAlert = now;
    };
    one('fire', pick(res.fire, 0), 'Fire');
    one('person', pick(res.people, 0.5), 'Person');
  }

  // ------------------------------------------------------------ status
  function plural(n, one, many) { return n + ' ' + (n === 1 ? one : many); }

  function setV(id, text, cls) {
    const el = $(id);
    if (el.textContent !== text) el.textContent = text;
    el.className = 'v' + (cls ? ' ' + cls : '');
  }

  function updateStatus(force) {
    const now = performance.now();
    if (!force && now - state.lastStatusT < 300) return;
    state.lastStatusT = now;
    const src = state.src, res = state.results;
    // engine
    let eng, engCls = '';
    if (state.loadError) { eng = 'Detectors failed to load: ' + (state.loadError.message || state.loadError); engCls = 'bad'; }
    else if (!state.backend) eng = 'Loading…';
    else if (state.backend === 'webgl') {
      const soft = state.backendInfo && state.backendInfo.software;
      eng = soft ? 'WebGL on a software renderer (slow, no graphics chip in use)' : 'WebGL (graphics chip)';
      if (soft) engCls = 'warn';
    } else { eng = 'CPU (slow: no usable WebGL here; the person check runs every second update)'; engCls = 'warn'; }
    if (state.backend && state.engineMode === 'main') eng += ' · on the page itself (no background worker here: taps may lag)';
    else if (state.backend && state.engineMode === 'worker') eng += ' · in a background worker';
    if (state.recovering) { eng += ' · detection stopped (graphics reset or stall), restarting…'; engCls = 'warn'; }
    else if (state.backend && !state.modelsReady && !state.loadError) eng += ' · loading detectors…';
    else if (state.gpuLosses || state.stalls) eng += ' · restarted ' + plural(state.gpuLosses + state.stalls, 'time', 'times');
    setV('s-engine', eng, engCls);
    $('engine').textContent = state.loadError ? 'Detectors failed' : state.recovering ? 'Restarting detectors…' : !state.modelsReady ? 'Loading detectors…' :
      state.paused ? 'Detection paused…' :
      (state.backend === 'webgl' ? 'WebGL' : 'CPU') + (res ? ' · ' + Math.round(res.ms.total) + ' ms' : '') +
      (src && src.live ? ' · ' + state.fps.toFixed(0) + ' fps' : '');
    // speed
    if (res) {
      const ups = state.upsTimes.length > 1 ? (state.upsTimes.length - 1) * 1000 / (state.upsTimes[state.upsTimes.length - 1] - state.upsTimes[0]) : 0;
      const parts = [];
      if (res.ms.person != null) parts.push('people ' + Math.round(res.ms.person) + ' ms');
      if (res.ms.face != null) parts.push('faces ' + Math.round(res.ms.face) + ' ms');
      if (state.fd && !state.fd.stub) parts.push('fire/doors/windows ' + Math.round(res.ms.firedoor) + ' ms');
      let txt = parts.join(', ');
      if (src && src.live) txt += ' · ' + ups.toFixed(1) + ' checks/s · video ' + state.fps.toFixed(0) + ' fps';
      setV('s-speed', txt);
    } else setV('s-speed', state.modelsReady ? 'Waiting for a picture' : '–');
    // tensors (stays flat when nothing leaks)
    $('tensors').textContent = state.modelsReady && state.numTensors != null ? 'tensors ' + state.numTensors : '';
    // seen
    if (src && src.kind === 'camera' && state.camMuted) setV('s-seen', 'No picture from the camera (another app may be using it, or it is covered)', 'warn');
    else if (state.recovering) setV('s-seen', 'Detection stopped: restarting the detectors…', 'warn');
    else if (state.paused) setV('s-seen', 'Detection paused: waiting for the detectors (this check is taking much longer than usual)…', 'warn');
    else if (res && src && !resultsToDraw(src)) setV('s-seen', 'Detection paused: no recent check', 'warn');
    else if (res && src) {
      const near = res.people.filter((d) => d.dist > 0).sort((a, b) => a.dist - b.dist)[0];
      const bits = [];
      if (res.people.length) bits.push(plural(res.people.length, 'person', 'people') + (near ? ' (nearest about ' + near.dist.toFixed(1) + ' m)' : ''));
      if (res.fire.length) bits.push(plural(res.fire.length, 'fire', 'fires'));
      if (res.door.length) bits.push(plural(res.door.length, 'door', 'doors'));
      if (res.window && res.window.length) bits.push(plural(res.window.length, 'window', 'windows'));
      setV('s-seen', bits.length ? bits.join(', ') : 'Nothing found');
    } else setV('s-seen', '–');
    // way out
    const p = navUi.on ? null : exitInfo(src);
    if (navUi.on) navStatusLine();
    else if (!p) setV('s-exit', 'Not marked');
    else if (!p.trusted) setV('s-exit', 'Tracking lost: point the camera back where it was, or mark it again', 'warn');
    else if (p.inView) setV('s-exit', 'In view' + (state.mark.at ? ' (marked at a ' + state.mark.at + ')' : '') + (p.unsure ? ' (unsure: picture not matching)' : ''), p.unsure ? 'warn' : '');
    else {
      const deg = Math.round(Math.abs(p.bearing));
      setV('s-exit', (Math.abs(p.bearing) > 135 ? 'Behind you (' + deg + '° turn)' : deg + '° to your ' + (p.bearing > 0 ? 'left' : 'right') +
        (Math.abs(p.rel.pitch) > 20 ? (p.rel.pitch > 0 ? ', up' : ', down') : '')) + (p.unsure ? ' (unsure)' : ''), p.unsure ? 'warn' : '');
    }
    // tracking
    if (!src) setV('s-track', '–');
    else if (!src.live) setV('s-track', 'Still photo: no turning to follow');
    else {
      const srcName = tracker.source(now);
      const label = srcName === 'sensor' ? 'Motion sensor only (picture not matching)' : srcName === 'camera+sensor' ? 'Picture motion + motion sensor' : 'Picture motion';
      const st = tracker.state === 'ok' ? '' : tracker.state === 'weak' ? ' · unsure' : tracker.state === 'lost' ? ' · lost' : '';
      setV('s-track', label + st, tracker.state === 'lost' ? 'warn' : '');
    }
    // source
    setV('s-source', !src ? (state.cameraError ? 'Camera unavailable' : 'None') :
      src.kind === 'camera' ? src.name + (src.mirror ? ', mirrored' : '') :
        (src.kind === 'image' ? 'Photo: ' : 'Video: ') + src.name);
    // fire / door
    if (!state.fd) setV('s-fd', state.loadError ? 'Not loaded' : 'Loading…');
    else if (state.fd.stub) setV('s-fd', 'Not in this build yet (stub finds nothing)' + (state.fdError ? '; model failed: ' + (state.fdError.message || state.fdError) : ''), 'warn');
    else setV('s-fd', state.fd.name + (state.fdReady ? '' : ' (loading)'));
  }

  // ------------------------------------------------------------ navigation (demo)
  // The eyepiece's own way-back-out navigation: core/src/ps_nav.c, ps_config.c and ps_alerts.c
  // compiled to plain JavaScript (camera/nav, global PSNav, inlined right after this script; see
  // camera/nav/README.md). Dead reckoning from steps and turns since the entry, breadcrumbs, the
  // device's confidence (GOOD / DEGRADED < 0.6 / UNRELIABLE < 0.3, with hysteresis) and its phrases.
  // "Start here" tries, in order: the phone's motion sensors (the iOS permission request is made
  // inside that tap), else the camera-turn tracker for the heading with the Walk button for steps
  // (when a camera or video is running), else the demo walk (a simulated firefighter with a
  // simulated motion sensor; works with no camera at all). While navigation runs, its marker
  // replaces "Mark way out" in the picture (the mark is kept and comes back after Stop): an EXIT
  // box sized by distance when the way out is in the field of view ("EXIT? 5M", dashed, when the
  // confidence is DEGRADED), else a green edge arrow, and no box but FOLLOW HOSE when UNRELIABLE.
  // Its phrases go through say() (voice toggle, priority queue below fire and people, alert log).
  const NAV_COL = { good: '#28FF50', warn: '#FFDC00', bad: '#FF2828' };   // ps_display.c draw_nav
  const NAV_SPEEDS = [1, 2, 4, 8];
  const HINT_MARK = 'Mark way out remembers where the camera points now; or tap the picture to mark that spot.';
  const HINT_NAV = 'Navigation is on: its green EXIT marker and the buttons above replace Mark way out. Press Stop under Navigation to mark by hand again.';
  const navLoaded = () => !!(window.PSNav && window.PSNav.Navigator);

  $('hint-size1').textContent = HINT_MARK;
  $('hint-size2').textContent = HINT_NAV;

  function navInitUi() {   // PSNav comes after this script in the page: called once it exists
    if (navUi.map || !navLoaded()) return;
    navUi.map = new PSNav.MapRenderer($('nav-map'), { up: 'entry' });
    navUi.arrow = new PSNav.ArrowWidget($('nav-arrow'));
    // the same Left / Hold to walk / Right in the navigation card and under the picture
    const hold = (id, down, up) => {
      const el = $(id);
      let downed = false;
      PSNav.holdButton(el, () => { if (el.disabled || !navUi.nav) return; downed = true; el.classList.add('on'); down(); },
        () => { el.classList.remove('on'); if (downed && navUi.nav) up(); downed = false; });
    };
    for (const pre of ['nav', 'cam']) {
      hold(pre + '-walk', () => navUi.nav.walk(true), () => navUi.nav.walk(false));
      hold(pre + '-left', () => navUi.nav.turnHold(1), () => navUi.nav.turnHold(0));
      hold(pre + '-right', () => navUi.nav.turnHold(-1), () => navUi.nav.turnHold(0));
    }
    try {
      window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
        navUi.map.refreshTheme(); navUi.arrow.refreshTheme(); navUi.dirty = true;
      });
    } catch (e) { /* old browser: the map re-reads its colours every 30 draws */ }
    if (window.IntersectionObserver) {
      new IntersectionObserver((es) => { navUi.visible = es[es.length - 1].isIntersecting; if (navUi.visible) navUi.dirty = true; }).observe($('nav-card'));
    }
    navUi.dirty = true;
  }

  function ensureNav() {
    if (navUi.nav) return navUi.nav;
    navInitUi();
    const nav = navUi.nav = new PSNav.Navigator({ mode: 'auto', demoSpeed: NAV_SPEEDS[navUi.speedIdx] });
    nav.on('alert', navAlert);
    nav.on('input', () => { navUi.dirty = true; navButtons(); });
    nav.on('demo', () => { navUi.dirty = true; });
    nav.on('entry', () => { navUi.dirty = true; });
    return nav;
  }

  function navAlert(a) {
    if (!navUi.on) return;
    // the device's queue already spaces its phrases (way out every 15 s while DEGRADED, the hose
    // every 20 s); one asked for with a button is said even if it was just said
    const asked = performance.now() - navUi.asked < 2000;
    const hose = a.names.indexOf('FOLLOW_HOSE') >= 0;
    say(a.text, { pri: hose ? 1.5 : 1, minGap: asked ? 0 : 3000 });
  }

  const cameraTurnAvailable = () => !!(state.src && state.src.live);
  const NAV_WHY = { absent: 'this device reports no motion sensor readings', blocked: 'this page may not read the motion sensors',
    denied: 'motion access was refused', lost: 'the motion sensors stopped' };
  // short form for the input indicator (the full reason stays in nav.input.detail)
  const NAV_WHY_SHORT = { absent: 'No usable motion sensors on this device.', blocked: 'Motion sensors are blocked here.',
    denied: 'Motion access was refused.', lost: 'The motion sensors stopped.' };
  function navWhy(status) {
    let w = NAV_WHY_SHORT[status] || '';
    // real sensors work in the phone's own browser: say how to get there only where Save exists
    if (w && downloads && (status === 'blocked' || status === 'absent')) w += ' Save this page to try them in your phone’s browser.';
    return w;
  }

  function navStartHere() {
    if (!navLoaded()) return;
    const nav = ensureNav();
    if (navUi.pending) return;
    navUi.asked = performance.now();
    navUi.note = '';
    const m = nav.input.mode, st = nav.input.status;
    if (navUi.on && (m === 'demo' || m === 'camera' || (m === 'sensors' && st === 'ok'))) {
      nav.markEntry();       // already running: the entry is here, now
      navRefresh();
      return;
    }
    navUi.on = true;
    navUi.pending = true;
    navUi.why = '';
    nav.run();
    const p = nav.useSensors();   // the motion permission request (iOS) is made inside this tap
    navAfterModeChange();
    p.then((status) => {
      navUi.pending = false;
      if (!navUi.on || nav.input.mode !== 'sensors') { navRefresh(); return; }   // stopped or switched meanwhile
      if (status !== 'ok') {
        navUi.why = navWhy(status);
        if (cameraTurnAvailable()) {
          nav.useCameraHeading('Motion sensors not used (' + (NAV_WHY[status] || status) + '). Heading: from how the camera picture turns. Steps: hold Walk (or W / up arrow).');
        } else nav.useDemo({ reason: nav.input.detail });
      }
      navAfterModeChange();
      updateStatus(true);
    }, () => { navUi.pending = false; navRefresh(); });
    updateStatus(true);
  }

  function navDemo(auto) {
    if (!navLoaded()) return;
    const nav = ensureNav();
    navUi.asked = performance.now();
    navUi.note = '';
    navUi.why = '';
    navUi.on = true;
    navUi.pending = false;
    if (auto) nav.autoDemo(); else nav.useDemo();
    nav.run();
    navAfterModeChange();
    updateStatus(true);
  }

  function navGuideOut() {
    const nav = navUi.nav;
    navUi.asked = performance.now();
    if (!navUi.on || !nav || !nav.g.valid) {
      navUi.note = navUi.pending ? 'Checking for motion sensors… try again in a moment.' :
        'Press Start here where you go in first (or Demo walk / Auto demo to try it).';
      navRefresh();
      return;
    }
    navUi.note = '';
    if (nav.input.mode === 'demo') nav.guideMeOut();   // the walker follows the arrow out by itself
    else nav.whereOut();                               // the device says the way
    navRefresh();
  }

  function navWhereOut() {
    const nav = navUi.nav;
    navUi.asked = performance.now();
    if (nav && nav.g.valid) nav.whereOut();
    else say('Entry not marked yet.', { minGap: 0 });
  }

  function navStop() {
    const nav = navUi.nav;
    if (!nav || !navUi.on) return;
    nav.stop();
    navUi.on = false; navUi.pending = false; navUi.note = '';
    for (const id of ['nav-walk', 'nav-left', 'nav-right', 'cam-walk', 'cam-left', 'cam-right']) $(id).classList.remove('on');
    say('Navigation stopped.', { minGap: 0, speak: false });
    navAfterModeChange();
    updateStatus(true);
  }

  function navAfterModeChange() {
    const nav = navUi.nav;
    if (navUi.unbindKeys) { navUi.unbindKeys(); navUi.unbindKeys = null; }
    // keys W/A/D and the arrows, only while the demo walker (or the camera-heading Walk) is in use
    if (nav && navUi.on && (nav.input.mode === 'demo' || nav.input.mode === 'camera')) navUi.unbindKeys = nav.bindKeys(window);
    navUi.yawEpoch = -1; navUi.yawOffset = 0; navUi.lastFedYaw = null;
    navRefresh();
  }

  // camera-turn tracker -> navigation heading ('camera' input). motion.js yaw is degrees clockwise.
  function navFeedTracker(r) {
    const nav = navUi.nav;
    if (!nav || !navUi.on || nav.input.mode !== 'camera') return;
    if (navUi.yawEpoch !== trackerEpoch) {
      // the tracker was reset (another source, the picture rotated): keep the heading continuous
      navUi.yawOffset = navUi.lastFedYaw == null ? 0 : navUi.lastFedYaw - r.pose.yaw;
      navUi.yawEpoch = trackerEpoch;
    }
    if (r.state === 'lost') {
      // the picture still comes but does not match: the navigator keeps the last good heading
      // (a blank wall or a fast turn is not a motion-sensor outage)
      if (navUi.lastFedYaw != null) nav.feedCameraTrackerYaw(navUi.lastFedYaw, 'lost');
      return;
    }
    const yaw = r.pose.yaw + navUi.yawOffset;
    navUi.lastFedYaw = yaw;
    navUi.lastFeedT = performance.now();
    nav.feedCameraTrackerYaw(yaw, r.state);
  }

  function navRefresh() { navUi.dirty = true; navButtons(); navText(); }

  function navButtons() {
    const nav = navUi.nav, on = navUi.on, mode = nav ? nav.input.mode : 'none';
    $('nav-stop').disabled = !on;
    // nothing appears or disappears here (that moved the buttons under a finger): unused ones are disabled
    $('nav-speed').disabled = !(on && mode === 'demo');
    $('nav-speed').textContent = 'Speed: ' + NAV_SPEEDS[navUi.speedIdx] + 'x';
    const walkOk = on && (mode === 'demo' || mode === 'camera'), turnOk = on && mode === 'demo';
    for (const pre of ['nav', 'cam']) {
      $(pre + '-walk').disabled = !walkOk;
      $(pre + '-left').disabled = $(pre + '-right').disabled = !turnOk;
    }
    // under the picture: the navigation's Left / Walk / Right / Guide me out replace Mark way out
    // while it runs (the same space: the row below does not move)
    $('mark-row').classList.toggle('off', on);
    $('cam-pad').classList.toggle('off', !on);
    $('nav-state').textContent = on ? (navUi.pending ? 'Starting…' : 'Running') : nav && nav.g.valid ? 'Stopped' : 'Off';
    // the camera card: navigation replaces Mark way out while it runs
    $('mark').disabled = !state.src || on;
    $('whereout').disabled = !state.src && !on;
    const hint = on ? HINT_NAV : HINT_MARK;
    if ($('hint').textContent !== hint) $('hint').textContent = hint;
  }

  function setText(id, t) { const el = $(id); if (el.textContent !== t) el.textContent = t; }

  function navText() {
    const nav = navUi.nav, inp = nav ? nav.input : null, g = nav ? nav.g : null;
    // which input is in use
    if (inp && inp.mode !== 'none') {
      const detail = navDetail(inp);
      setText('nav-in-label', inp.label + (inp.status && inp.status !== 'ok' ? ' (' + inp.status + ')' : '') + (navUi.on ? '' : ' · stopped'));
      setText('nav-in-detail', detail);
      $('nav-dot').className = 'ndot ' + (!navUi.on ? '' : inp.status === 'ok' ? 'ok' : inp.status === 'waiting' || inp.status === 'asking' ? 'wait' : inp.status === 'off' ? '' : 'bad');
    }
    // what the device would say, and its numbers
    let say1;
    if (!nav || (!navUi.on && !g.valid)) say1 = 'Not started.';
    else if (navUi.pending) say1 = 'Checking for motion sensors…';
    else if (!g.valid) say1 = inp.mode === 'camera' ? 'Waiting for the camera picture to mark the entry…' : 'Waiting to mark the entry…';
    else if (g.followHose) say1 = 'Navigation estimate unreliable. Follow the hose line out.';
    else if (g.atExit) say1 = 'You are at the way out.';
    else say1 = g.phrase;
    if (nav && !navUi.on && g.valid) say1 = 'Stopped. ' + say1;
    setText('nav-say', say1);
    const ok = !!(g && g.valid);
    setText('nav-route', ok ? g.routeDistM.toFixed(1) + ' m' : '–');
    setText('nav-home', ok ? g.homeDistM.toFixed(1) + ' m' : '–');
    setText('nav-conf', ok ? Math.round(g.confidence * 100) + ' % (±' + g.posSigmaM.toFixed(1) + ' m)' : '–');
    setText('nav-level', ok ? g.level : '–');
    $('nav-level').className = 'badge' + (ok ? ' ' + g.level : '');
    let note = navUi.note;
    if (!note && nav && navUi.on) {
      const w = nav.walker;
      if (inp.mode === 'demo' && w) {
        const r = w.result;
        note = w.phase === 'inbound' ? 'Auto demo: walking in…' : w.phase === 'scan' ? 'Auto demo: looking around the room…' :
          w.phase === 'outbound' ? 'Following the arrow out…' :
          w.phase === 'done' && r ? (r.how === 'hose' ? 'Out by the hose line' : r.how === 'exit' ? 'Arrow says EXIT' : r.how === 'outside' ? 'Walked out' : 'Stopped') +
            ': ' + r.doorErrorM.toFixed(1) + ' m from ' + (r.ref === 'mark' ? 'where the entry was marked' : 'the real door') + ' (simulation).' :
            'Hold to walk, Left / Right to turn (or keys W, A, D). Guide me out walks back by itself.';
      } else if (inp.mode === 'camera') note = 'Turn the camera to turn; hold Walk while you walk. Guide me out says the way.';
      else if (inp.mode === 'sensors') note = 'Walk with the phone held upright in front of you. Guide me out says the way.';
    }
    setText('nav-result', note || '');
  }

  // the input indicator's sentence: short (it sits above the readouts and the map), the full
  // reason stays in nav.input.detail
  const NAV_SENSOR_TEXT = { asking: 'Asking for permission to use the motion sensors…', waiting: 'Waiting for motion sensor data…',
    ok: 'Heading and steps from this phone’s motion sensors.', lost: 'Motion sensor data stopped.', off: '' };
  function navDetail(inp) {
    if (inp.mode === 'demo') return navUi.why ? navUi.why + ' Using the simulated walk instead.' : 'Its estimate drifts like a real motion sensor’s would.';
    if (inp.mode === 'camera') {
      if (navUi.on && performance.now() - navUi.lastFeedT > 2500) {
        return cameraTurnAvailable() ? 'The camera picture is not matching just now (turn slower, or point at something with detail).' :
          'No moving picture: start the camera (or open a video) to turn with it.';
      }
      return (navUi.why ? navUi.why + ' ' : '') + 'Heading from how the camera picture turns; hold Walk for steps.';
    }
    if (inp.mode === 'sensors') return NAV_SENSOR_TEXT[inp.status] != null ? NAV_SENSOR_TEXT[inp.status] : (navWhy(inp.status) || inp.detail || '');
    return inp.detail || '';
  }

  function navStatusLine() {
    const nav = navUi.nav, g = nav && nav.g;
    if (!g || !g.valid) setV('s-exit', navUi.pending ? 'Navigation: checking for motion sensors…' : 'Navigation: entry not marked yet');
    else if (g.followHose) setV('s-exit', 'Navigation unreliable: follow the hose line out', 'bad');
    else if (g.atExit) setV('s-exit', 'Navigation: you are at the way out');
    else setV('s-exit', 'Navigation: ' + g.word.replace('-', ' ') + ', ' + Math.round(g.routeDistM) + ' m (' + g.level.toLowerCase() + ', ' + Math.round(g.confidence * 100) + ' %)', g.levelId ? 'warn' : '');
  }

  function navFrame(ts) {
    if (!navUi.map) { navInitUi(); if (!navUi.map) return; }
    if (!navUi.dirty && (!navUi.on || !navUi.visible)) return;
    const nav = navUi.nav, d = navUi.dirty;
    // Redrawing these canvases is most of the card's cost on a slow CPU (measured with Chromium's
    // 4x throttling: about 7 % of this thread at the old 20 / 10 per second), so the ring is drawn
    // at the eyepiece's own rate (10 per second) and only when what it shows has changed, and the
    // map 5 times a second (also when unchanged, at least once a second, so its zoom settles).
    if (d || ts - navUi.lastArrowT >= 100) {
      navUi.lastArrowT = ts;
      const g = nav ? nav.g : null, k = navArrowKey(g);
      if (d || k !== navUi.arrowKey) { navUi.arrowKey = k; navUi.arrow.draw(g); }
    }
    if (d || ts - navUi.lastMapT >= 200) {
      navUi.lastMapT = ts;
      const k = navMapKey(nav);
      if (d || k !== navUi.mapKey || ts - navUi.lastMapDrawT >= 1000) {
        navUi.mapKey = k; navUi.lastMapDrawT = ts;
        navUi.map.draw(nav ? nav.snapshot() : { g: {}, trail: [], crumbs: [], demo: null });
      }
    }
    if (d || ts - navUi.lastTextT >= 250) { navUi.lastTextT = ts; navText(); }
    navUi.dirty = false;
  }

  // what PSNav.ArrowWidget draws, rounded to what can be seen (1 degree, 1 m, the blink phase)
  function navArrowKey(g) {
    if (!g || !g.valid) return 'idle';
    if (g.levelId === 2) return 'hose ' + Math.round(g.posSigmaM) + ' ' + ((Date.now() / 500 | 0) % 2);
    return g.levelId + ' ' + (g.routeDistM < 1 ? 'exit' : Math.round(g.routeBearingDeg)) + ' ' + Math.round(g.homeBearingDeg) +
      ' ' + Math.round(g.routeDistM) + ' ' + g.state;
  }
  // changes of the map's content (position to 2 cm, heading to 1 degree, uncertainty, path, breadcrumbs, demo walker)
  function navMapKey(nav) {
    if (!nav) return 'none';
    const g = nav.g, w = nav.walker;
    if (!g.valid) return 'invalid ' + nav.input.mode;
    return [Math.round(g.pos.x * 50), Math.round(g.pos.y * 50), Math.round(g.yaw * 57.3), Math.round(g.posSigmaM * 20), g.levelId,
      nav.trail.length, g.nCrumbs, g.returnTarget, w ? Math.round(w.x * 50) + ' ' + Math.round(w.y * 50) + ' ' + Math.round(w.yaw * 57.3) + ' ' + w.phase : ''].join(' ');
  }

  // ---- the navigation's way-out marker in the picture (device draw_exit / draw_nav)
  function navMarker(src) {
    const nav = navUi.nav;
    if (!navUi.on || !nav) return null;
    const g = nav.g;
    if (!g.valid) return { kind: 'idle', g };
    if (g.levelId === 2) return { kind: 'hose', g };
    // the demo walk (and the view without a camera) stands in for the eyepiece, which looks where
    // the walker faces; a real front camera (mirrored preview) looks back at the viewer
    const fwd = nav.input.mode === 'demo' || !!src.sim;
    const b = PSNav.exitBox(g, { hfovDeg: effectiveHfov(src.w, src.h), mirrored: fwd ? false : !!src.mirror,
      cameraLooksBack: fwd ? false : undefined, width: src.w, height: src.h });
    b.g = g;
    b.unsure = g.levelId === 1;
    return b;
  }

  function drawNavExit(src, fit, s, ts, labels, ringShown) {
    const b = navMarker(src);
    const o = navUi.overlay = b ? { kind: b.kind, label: null, unsure: !!b.unsure, level: b.g.level, side: b.side || null, behind: !!b.behind, rect: null, at: null } : null;
    if (!b) return;
    const q = (t) => (b.unsure ? t.replace(/^(\S+)/, '$1?') : t);     // DEGRADED: 'EXIT 5M' -> 'EXIT? 5M'
    if (b.kind === 'box') {
      o.label = q(b.label);
      o.rect = toCanvas(b, fit, false);           // exitBox gives display coordinates (already mirrored)
      drawBox(o.rect, COL.exit, o.label, fit, s, labels, 1.5, b.unsure);
    } else if (b.kind === 'edge') {
      o.label = q(b.label) + (b.behind ? ' BEHIND' : '');
      o.at = drawEdgeArrow(fit, s, b.side === 'left' ? -1 : 1, b.behind ? 0.45 : 0, o.label);
    } else if (b.kind === 'route') {
      // the next leg of the way out is in the picture (the door is further on): a chevron there
      o.label = q(b.label);
      const x = fit.x + b.x * fit.w, y = fit.y + fit.h * 0.62, r = 16 * s;
      o.at = { x, y };
      g.beginPath(); g.moveTo(x, y - r); g.lineTo(x + r, y + r * 0.6); g.lineTo(x, y); g.lineTo(x - r, y + r * 0.6); g.closePath();
      g.lineJoin = 'round'; g.lineWidth = 4 * s; g.strokeStyle = 'rgba(0,0,0,0.9)'; g.stroke();
      g.fillStyle = COL.exit; g.fill();
      setFont(13 * s); g.textAlign = 'center'; g.textBaseline = 'top';
      outlinedText(o.label, Math.min(Math.max(x, fit.x + 40 * s), fit.x + fit.w - 40 * s), y + r, COL.exit, s);
      g.textAlign = 'left'; g.textBaseline = 'alphabetic';
    } else if (b.kind === 'hose' && !ringShown) {
      // UNRELIABLE: no marker at all (it is not believed), only the device's instruction (steady
      // here, so it can be read on a busy picture; the eyepiece ring blinks it like the device)
      o.label = 'FOLLOW HOSE';
      const cx = fit.x + fit.w / 2, sub = 'NAV ±' + Math.round(b.g.posSigmaM) + 'M';
      setFont(20 * s);
      const tw = g.measureText('FOLLOW HOSE').width;
      g.fillStyle = 'rgba(0,0,0,0.6)';
      g.fillRect(cx - tw / 2 - 10 * s, fit.y + 6 * s, tw + 20 * s, 52 * s);
      g.textAlign = 'center'; g.textBaseline = 'top';
      outlinedText('FOLLOW HOSE', cx, fit.y + 12 * s, NAV_COL.bad, s);
      setFont(12 * s);
      outlinedText(sub, cx, fit.y + 38 * s, NAV_COL.bad, s);
      g.textAlign = 'left'; g.textBaseline = 'alphabetic';
    } else if (b.kind === 'none' && b.reason === 'at the door' && !ringShown) {
      o.label = 'EXIT';
      setFont(18 * s); g.textAlign = 'center'; g.textBaseline = 'top';
      outlinedText('EXIT', fit.x + fit.w / 2, fit.y + 12 * s, COL.exit, s);
      g.textAlign = 'left'; g.textBaseline = 'alphabetic';
    } else if (b.kind === 'idle' && !ringShown) {
      setFont(12 * s); g.textAlign = 'center';
      hudText('MARK ENTRY', fit.x + fit.w / 2, fit.y + 20 * s, '#FFFFFF', s);
      g.textAlign = 'left';
    }
  }

  // the device's ring (draw_nav), scaled from its 320 x 240 screen: top centre, radius 22 at y 30
  function navRingRect(fit) {
    const k = fit.h / 240, cx = fit.x + fit.w / 2;
    return { x: cx - 40 * k, y: fit.y, w: 80 * k, h: 66 * k };
  }

  function drawNavRing(fit, s, ts) {
    const nav = navUi.nav;
    if (!nav) return;
    const gd = nav.g, k = fit.h / 240, cx = fit.x + fit.w / 2, cy = fit.y + 30 * k, r = 22 * k;
    const font = (px) => { g.font = '700 ' + Math.round(Math.max(px * k, 9 * s)) + 'px ' + MONO; };
    g.textAlign = 'center'; g.textBaseline = 'middle';
    if (!gd.valid) {
      font(10); outlinedText('MARK ENTRY', cx, fit.y + 12 * k, '#FFFFFF', s);
    } else if (gd.levelId === 2) {
      if (Math.floor(ts / 500) % 2 === 0) { font(16); outlinedText('FOLLOW HOSE', cx, fit.y + 18 * k, NAV_COL.bad, s); }
      font(9); outlinedText('NAV ±' + Math.round(gd.posSigmaM) + 'M', cx, fit.y + 36 * k, NAV_COL.bad, s);
    } else {
      const col = gd.levelId === 0 ? NAV_COL.good : NAV_COL.warn;
      g.beginPath(); g.arc(cx, cy, r + 2 * k, 0, 2 * Math.PI);
      g.lineWidth = 3.2 * k; g.strokeStyle = '#000'; g.stroke();
      g.lineWidth = 1.4 * k; g.strokeStyle = col; g.stroke();
      if (gd.routeDistM < 1) { font(16); outlinedText('EXIT', cx, cy, col, s); }
      else {
        const a = gd.routeBearingDeg * Math.PI / 180, R = r - 2 * k;
        const fx = -Math.sin(a), fy = -Math.cos(a), px = -fy, py = fx;
        const bx = cx - fx * R * 0.55, by = cy - fy * R * 0.55;
        g.beginPath();
        g.moveTo(cx + fx * R, cy + fy * R);
        g.lineTo(bx + px * R * 0.6, by + py * R * 0.6);
        g.lineTo(cx - fx * R * 0.2, cy - fy * R * 0.2);
        g.lineTo(bx - px * R * 0.6, by - py * R * 0.6);
        g.closePath();
        g.lineJoin = 'round'; g.lineWidth = 2.5 * k; g.strokeStyle = '#000'; g.stroke();
        g.fillStyle = col; g.fill();
      }
      // straight line to the door: a small white square on the ring
      const h = gd.homeBearingDeg * Math.PI / 180, dx = cx - Math.sin(h) * (r + 2 * k), dy = cy - Math.cos(h) * (r + 2 * k);
      g.fillStyle = '#000'; g.fillRect(dx - 3.5 * k, dy - 3.5 * k, 7 * k, 7 * k);
      g.fillStyle = '#FFF'; g.fillRect(dx - 2.5 * k, dy - 2.5 * k, 5 * k, 5 * k);
      font(10); outlinedText(Math.round(gd.routeDistM) + 'M', cx, cy + r + 10 * k, col, s);
      if (gd.state === 'lost') { font(9); outlinedText('NO IMU', cx, cy + r + 22 * k, NAV_COL.bad, s); }
    }
    g.textAlign = 'left'; g.textBaseline = 'alphabetic';
  }

  // No camera (blocked, as inside claude.ai) while navigation runs: a plain simulated eyepiece
  // view, so the EXIT box and the ring can still be seen.
  const navSimView = () => !!(navUi.on && navUi.nav && !state.src);
  const SIM_SRC = { w: 640, h: 480, mirror: false, sim: true };
  function drawSimView(ts) {
    const cw = view.width, ch = view.height, s = state.dpr;
    const fit = contain(4, 3, cw, ch);
    g.setTransform(1, 0, 0, 1, 0, 0);
    const grd = g.createLinearGradient(0, fit.y, 0, fit.y + fit.h);
    grd.addColorStop(0, '#2b2f34'); grd.addColorStop(0.5, '#454a50'); grd.addColorStop(1, '#202327');
    g.fillStyle = grd; g.fillRect(fit.x, fit.y, fit.w, fit.h);
    // corridor edges toward the centre, eyepiece grey
    g.strokeStyle = 'rgba(255,255,255,0.10)'; g.lineWidth = Math.max(1, s);
    const vx = fit.x + fit.w / 2, vy = fit.y + fit.h / 2;
    g.beginPath();
    for (const [x, y] of [[fit.x, fit.y], [fit.x + fit.w, fit.y], [fit.x, fit.y + fit.h], [fit.x + fit.w, fit.y + fit.h]]) { g.moveTo(x, y); g.lineTo(vx + (x - vx) * 0.08, vy + (y - vy) * 0.08); }
    g.moveTo(fit.x, vy); g.lineTo(fit.x + fit.w, vy);
    g.stroke();
    const labels = [];
    drawNavExit(SIM_SRC, fit, s, ts, labels, true);
    drawLabels(labels, fit, s, [navRingRect(fit)]);
    drawNavRing(fit, s, ts);
    setFont(12 * s); g.textBaseline = 'alphabetic';
    g.textAlign = 'right';
    hudText('SIMULATED VIEW: NO CAMERA', fit.x + fit.w - 8 * s, fit.y + fit.h - 8 * s, '#B4B4B4', s);
    g.textAlign = 'left';
  }

  $('nav-start').addEventListener('click', navStartHere);
  $('nav-out').addEventListener('click', navGuideOut);
  $('cam-out').addEventListener('click', navGuideOut);
  // camera blocked (as inside claude.ai): the navigation demo works without one; it shows in the picture above
  $('n-nav').addEventListener('click', () => navDemo(true));
  $('nav-stop').addEventListener('click', navStop);
  $('nav-demo').addEventListener('click', () => navDemo(false));
  $('nav-auto').addEventListener('click', () => navDemo(true));
  $('nav-speed').addEventListener('click', () => {
    navUi.speedIdx = (navUi.speedIdx + 1) % NAV_SPEEDS.length;
    if (navUi.nav) navUi.nav.setSpeed(NAV_SPEEDS[navUi.speedIdx]);
    navButtons();
  });
  $('nav-up').addEventListener('click', () => {
    if (!navUi.map) return;
    const up = navUi.map.toggleUp();
    $('nav-up').textContent = up === 'heading' ? 'Entry direction up' : 'Heading up';
    navUi.dirty = true;
  });

  // ------------------------------------------------------------ downloads (claude.ai)
  let downloads = null;
  function refreshSaveButtons() { $('n-save').hidden = !downloads; $('save2').hidden = !downloads; }
  try {
    if (window.claude && typeof window.claude.use === 'function') {
      Promise.resolve(window.claude.use('downloads')).then((d) => { downloads = d || null; refreshSaveButtons(); }, () => { downloads = null; refreshSaveButtons(); });
    }
  } catch (e) { downloads = null; }

  function savePage(msgId) {
    const msg = (t) => { $(msgId).textContent = t; };
    if (!downloads) return;
    let html;
    try { html = buildStandalone(); } catch (e) { msg('Could not prepare the file: ' + (e && e.message || e)); return; }
    msg('Preparing the file…');
    let p;
    try { p = downloads.save({ filename: 'pyrosight_camera.html', data: html }); } catch (e) { p = Promise.reject(e); }
    Promise.resolve(p).then((r) => {
      msg(r && r.status === 'delivered' ? '' : 'Saved as pyrosight_camera.html. Open it from your downloads in Chrome, Edge, Firefox or Safari, press Start camera and allow the camera.');
    }, (e) => {
      const code = e && e.code;
      if (code === 'declined') { msg(''); return; }
      if (code === 'rate_limited') msg('A save prompt is already open.');
      else if (code === 'too_large') msg('The page is too large to save here.');
      else if (code === 'bad_request' || code === 'transform_error' || code === 'rejected_extension' || code === 'extension_not_enabled' || code === 'request_unknown') msg('Saving did not work here (' + code + ').');
      else { downloads = null; refreshSaveButtons(); msg('Saving is not available here.'); }
    });
  }

  // ------------------------------------------------------------ controls
  function onStart() {
    requestOrientation();
    if (state.camPending) return;   // a camera request is already waiting for an answer
    if (state.src && state.src.kind === 'camera') {
      stopSource(); setSource(null);
      showOverlay('Camera stopped', 'Press Start camera to begin again. Video never leaves your device.', true);
      $('ov-start').textContent = 'Start camera';
      return;
    }
    startCamera();
  }
  $('start').addEventListener('click', onStart);
  $('ov-start').addEventListener('click', onStart);
  $('switch').addEventListener('click', () => {
    if (state.devices.length < 2) return;
    const cur = state.src && state.src.deviceId;
    const i = state.devices.findIndex((d) => d.deviceId === cur);
    const next = state.devices[(i + 1) % state.devices.length];
    if (state.mark) say('Way out mark cleared.', { minGap: 0, speak: false });
    startCamera(next.deviceId, cur);
  });
  const pickFile = () => $('file').click();
  $('file-btn').addEventListener('click', pickFile);
  $('ov-file').addEventListener('click', pickFile);
  $('n-file').addEventListener('click', pickFile);
  $('file').addEventListener('change', (e) => {
    const f = e.target.files && e.target.files[0];
    openFile(f);
    e.target.value = '';
  });
  $('mark').addEventListener('click', () => { if (navUi.on) return; requestOrientation(); markWayOut(0.5, 0.5, 'centre'); });
  $('clear-mark').addEventListener('click', () => { clearMark(false); updateStatus(true); });
  $('whereout').addEventListener('click', () => { if (navUi.on) navWhereOut(); else whereOut(); });
  view.addEventListener('click', (e) => {
    const src = state.src, fit = state.fit;
    if (!src || !fit) return;
    if (src.kind === 'video' && src.el.paused) {   // a video the browser would not start by itself: a tap plays it
      try { const pp = src.el.play(); if (pp && pp.catch) pp.catch(() => {}); } catch (err) { /* ignore */ }
      return;
    }
    const r = view.getBoundingClientRect();
    const x = (e.clientX - r.left) * (view.width / r.width), y = (e.clientY - r.top) * (view.height / r.height);
    let nx = (x - fit.x) / fit.w, ny = (y - fit.y) / fit.h;
    if (nx < 0 || nx > 1 || ny < 0 || ny > 1) return;
    if (src.mirror) nx = 1 - nx;
    if (navUi.on) return;            // navigation's marker replaces the mark while it runs
    requestOrientation();
    markWayOut(nx, ny, 'tap');
  });
  $('voice').addEventListener('click', function () {
    if (!('speechSynthesis' in window)) { this.textContent = 'Voice not supported here'; this.disabled = true; return; }
    state.voice = !state.voice;
    this.textContent = state.voice ? 'Voice: on' : 'Voice: off';
    this.setAttribute('aria-pressed', state.voice ? 'true' : 'false');
    stopSpeech();
    if (state.voice) { speech.q.push({ text: 'Voice on.', pri: 4, t: performance.now(), ttl: 4000 }); pumpSpeech(); }
  });
  /*
   * The palette button only exists while the eyepiece view is on: in the
   * normal camera view there is nothing for it to change, and a dead control
   * in front of a room full of firefighters is a question you have to stop
   * and answer.
   */
  $('palette').addEventListener('click', function () {
    state.palette = (state.palette + 1) % EYE_PALETTES.length;
    this.textContent = 'Palette: ' + EYE_PALETTES[state.palette].name;
    eyeFrameKey = null;                       // recolour the frame already on screen
    draw(performance.now());
  });

  $('eyepiece').addEventListener('click', function () {
    state.eyepiece = !state.eyepiece;
    const pal = $('palette');
    pal.hidden = !state.eyepiece;
    pal.textContent = 'Palette: ' + EYE_PALETTES[state.palette].name;
    eyeFrameKey = null;
    this.textContent = state.eyepiece ? 'Eyepiece view: on' : 'Eyepiece view: off';
    this.setAttribute('aria-pressed', state.eyepiece ? 'true' : 'false');
  });
  $('n-save').addEventListener('click', () => savePage('n-save-msg'));
  $('save2').addEventListener('click', () => savePage('save2-msg'));

  document.addEventListener('visibilitychange', () => {
    const src = state.src;
    if (src && src.kind === 'video') {
      if (document.hidden) { state.resumeVideo = !src.el.paused; src.el.pause(); }
      else if (state.resumeVideo) { src.el.play().catch(() => {}); }
    }
    updateStatus(true);
  });
  window.addEventListener('deviceorientation', (e) => {
    if (e.alpha == null || e.beta == null || e.gamma == null) return;
    const yp = M.orientationToYawPitch(e.alpha, e.beta, e.gamma, state.facing === 'user');
    tracker.sensor(yp.yaw, yp.pitch, performance.now());
  });

  // ------------------------------------------------------------ test / integration hooks
  window.PSCamera = {
    get state() { return state; },
    get tracker() { return tracker; },
    get results() { return state.results; },
    setFireDoorDetector, markWayOut, clearMark, whereOut, buildStandalone, exitInfo: () => exitInfo(state.src),
    // tests: 'lose-context' (graphics reset) or 'hang' (a detector run that never ends) in the engine
    debugEngine: (what, ms) => { if (state.engine && state.engine.debug) state.engine.debug(what, ms); },
    FIREDOOR_STUB,
    // navigation (demo): the PSNav.Navigator (null until first used) and the page's view of it
    get nav() { return navUi.nav; },
    get navUi() { return navUi; },
    navStartHere, navGuideOut, navDemo, navStop,
  };

  // ------------------------------------------------------------ start
  fixStartWidth();
  updateStatus(true);
  renderLog();
  requestAnimationFrame(frame);
  loadModels();   // waits for the first paint, then starts the detector worker (or the fallback)
})();
