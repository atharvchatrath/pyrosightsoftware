/*
 * PyroSight Camera: page logic (app.js). Inlined last, after tf.min.js,
 * oplist.js (PSOpList), people.js (PSPeople), people_assets.js, the optional
 * fire/door model and motion.js (PSMotion). See README.md.
 *
 * Boxes (same colours as core/include/pyrosight/ps_display.h):
 *   person  WHITE  #FFFFFF  label = distance estimate, device style ("1.2M")
 *   fire    PURPLE #C850FF  "FIRE"
 *   exit    GREEN  #28FF50  "DOOR" (fire/door model) and "EXIT" (the viewer's mark)
 *
 * Fire/door detector interface (injectable). The page uses, in order:
 *   1. globalThis.PS_FIREDOOR_DETECTOR, if a script before this one set it;
 *   2. an adapter around PS_OPLIST_ASSETS.firedoor + FireDoorDecode, if both exist;
 *   3. the STUB below (finds nothing), clearly reported as such on the page.
 * A detector is {name, stub?, credits?, init(tf) -> Promise, detect(pixels) ->
 * Promise<[{cls: 'fire'|'door', score, x, y, w, h}]>, dispose()}; pixels is a
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
    src: null,              // {kind: 'camera'|'video'|'image', el, w, h, mirror, live, name}
    stream: null, devices: [], facing: null, mirrorOverride: qs.get('mirror'),
    busy: false, results: null, lastPersons: null, cycle: 0, lastInferEnd: 0, lastInferVideoTime: -1,
    inferredSrc: null, inferError: null,
    eyepiece: false, voice: false,
    mark: null, markLostSaid: false,
    exitSector: null, exitSectorSince: 0, lastExitSay: -1e9,
    al: {}, log: [], spoken: new Map(),
    dpr: 1, fit: null,
    lastVideoTime: -1, lastTrackT: 0, newFrame: false,
    fpsCount: 0, fpsT0: 0, fps: 0, ups: 0, upsTimes: [],
    lastStatusT: 0, startT: performance.now(),
    camReq: 0, camPending: false, camMuted: false,
    inferGen: 0, inferT0: 0, recovering: false, gpuLosses: 0, stalls: 0, lastCheckT: 0,
    stats: { inferences: 0, tensors: [], ms: [], errors: [], trackStates: {}, labels: [], recoveries: [] },
  };
  const tracker = new M.MotionTracker({ fovDeg: FOV });

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

  // ------------------------------------------------------------ fire / door
  const FIREDOOR_STUB = {
    name: 'Stub: no fire/door model in this build',
    stub: true,
    credits: '',
    init: async () => ({}),
    detect: async () => [],          // STUB: always finds nothing
    dispose() {},
  };

  function fireDoorAdapter() {
    const asset = window.PS_OPLIST_ASSETS && window.PS_OPLIST_ASSETS.firedoor;
    const dec = window.FireDoorDecode;
    if (!asset || !dec || !window.PSOpList) return null;
    const meta = window.PS_FIREDOOR_META || {};
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
      const now = performance.now();
      const recent = now - prevT <= (hy.maxGapMs || 3000) ? prevFire : [];
      const out = dets.filter((d) => d.cls !== 'fire' || d.score >= hy.on ||
        (d.score >= hy.keep && recent.some((p) => iou(p, d) >= (hy.iou || 0.1))));
      prevFire = out.filter((d) => d.cls === 'fire');
      prevT = now;
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
      const out = model.run(x);
      x.dispose();
      try {
        const [heat, wh, off] = await Promise.all([out.heat.data(), out.wh.data(), out.off.data()]);
        const o = Object.assign({ layout, inW, inH, gridW: inW / 8, gridH: inH / 8 }, meta.decode || {});
        o.thresholds = Object.assign({}, o.thresholds || {}, thresholds);
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
      async init(tfRef) {
        model = window.PSOpList.loadEmbedded(asset, { tf: tfRef });
        const s = model.inputs[0].shape;          // [1, H, W, 3]
        inH = s[1]; inW = s[2];
        layout = model.outputs[0].layout === 'nhwc' ? 'NHWC' : 'NCHW';
        const z = tf.zeros([inH, inW, 3]);
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
        const t0 = performance.now();
        let dets = await pass(px, Object.assign({}, base, { fire: fireMin }));
        passTimes = passTimes.concat(performance.now() - t0).slice(-5);
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
              zd = (await pass(crop, { fire: fireMin + lift, door: 2 })).filter((d) => d.cls === 'fire').map((d) => Object.assign(d, {
                x: (x0 + d.x * cw) / W, y: (y0 + d.y * ch) / H, w: d.w * cw / W, h: d.h * ch / H,
                rawScore: d.score, score: d.score - lift, zoom: true }));
            } finally { crop.dispose(); }
            prevZoom = { dets: zd, t: performance.now(), W, H, thumb: zd.length && ch >= 12 && cw >= 16 ? await thumb(px, y0, x0, ch, cw) : null };
          } else if (prevZoom && prevZoom.thumb && prevZoom.W === W && prevZoom.H === H && performance.now() - prevZoom.t < 10000) {
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

  async function setFireDoorDetector(det) {
    if (state.fd && state.fd.dispose) { try { state.fd.dispose(); } catch (e) { /* ignore */ } }
    state.fd = det || FIREDOOR_STUB;
    state.fdReady = false; state.fdError = null;
    if (state.modelsReady || state.backend) {
      try { await state.fd.init(tf); state.fdReady = true; } catch (e) { state.fdError = e; state.fd = FIREDOOR_STUB; state.fdReady = true; }
    }
    $('credits').textContent = creditsText();
    updateStatus(true);
  }

  function creditsText() {
    let s = 'People and faces: COCO-SSD (TensorFlow.js models) and MediaPipe BlazeFace, Apache-2.0. TensorFlow.js, Apache-2.0.';
    if (state.fd && state.fd.credits) s += ' ' + state.fd.credits;
    return s;
  }

  // ------------------------------------------------------------ models
  async function loadModels() {
    const t0 = performance.now();
    try {
      if (!window.tf || !window.PSPeople || !window.PS_PEOPLE_ASSETS) throw new Error('detector code missing from the page');
      const prefer = qs.get('backend') === 'cpu' ? 'cpu' : undefined;
      state.backend = await PSPeople.setupBackend(tf, { prefer, allowSoftwareWebGL: qs.get('swgl') !== '0' });
      state.backendInfo = PSPeople.backendInfo;
      updateStatus(true);
      try {
        await PSPeople.init(tf, window.PS_PEOPLE_ASSETS);
      } catch (e) {
        if (state.backend === 'cpu') throw e;
        state.backend = await PSPeople.setupBackend(tf, { prefer: 'cpu' });   // WebGL failed: CPU
        state.backendInfo = PSPeople.backendInfo;
        await PSPeople.init(tf, window.PS_PEOPLE_ASSETS);
      }
      const det = window.PS_FIREDOOR_DETECTOR || fireDoorAdapter() || FIREDOOR_STUB;
      state.fd = det;
      try { await det.init(tf); } catch (e) { state.fdError = e; state.fd = FIREDOOR_STUB; }
      state.fdReady = true;
      state.modelsReady = true;
      state.loadMs = performance.now() - t0;
      watchGpu();
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
    clearMark(true);
    tracker.reset();
    state.al = {};
    if (state.fd && state.fd.reset) { try { state.fd.reset(); } catch (e) { /* ignore */ } }
    if (src && src.w && src.h) $('screen').style.aspectRatio = src.w + ' / ' + src.h;
    if (src) state.cameraError = null;
    $('overlay').hidden = !!src;
    $('mark').disabled = !src;
    $('whereout').disabled = !src;
    state.camMuted = false;
    setStartButton();
    updateStatus(true);
  }

  function stopTracks(stream) { if (stream) { try { stream.getTracks().forEach((t) => t.stop()); } catch (e) { /* ignore */ } } }

  function stopSource() {
    const s = state.src;
    state.camReq++;                 // a camera request still waiting for an answer is now stale
    state.camPending = false;
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
    stopSource();
    setSource(null);
    const req = ++state.camReq;
    state.camPending = true;
    const live = () => req === state.camReq;
    setStartButton();
    showOverlay('Waiting for the camera…', 'Allow camera access if your browser asks. Video never leaves your device.', false);
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
      state.stream = stream;
      camVideo.srcObject = stream;
      camVideo.muted = true;
      // not awaited: play() never settles for a camera that sends no frames
      try { const pp = camVideo.play(); if (pp && pp.catch) pp.catch(() => { /* autoplay attribute and the frame loop retry */ }); } catch (e) { /* ignore */ }
      const ok = await waitFor(() => camVideo.videoWidth > 0 || !live(), 8000);
      if (!live()) { stopTracks(stream); return; }
      if (!ok) { stopSource(); setSource(null); cameraError({ name: 'NoFrames' }); return; }
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

  function setStartButton() {
    const b = $('start');
    const t = state.camPending ? 'Starting camera…' : state.src && state.src.kind === 'camera' ? 'Stop camera' : 'Start camera';
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
    showOverlay(title, 'See below for other ways to try it.', true);
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
  window.addEventListener('resize', () => { needResize = true; });

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
          tracker.reset();
          // the turn tracker cannot follow a rotation of the picture itself: say so instead of
          // dropping the mark silently
          if (state.mark) { clearMark(true); say('Way out mark cleared: the picture turned. Mark it again.', { minGap: 0 }); }
        }
        if (el.currentTime !== state.lastVideoTime) {
          state.lastVideoTime = el.currentTime;
          state.newFrame = true;
          state.fpsCount++;
          if (ts - state.lastTrackT >= 25) { state.lastTrackT = ts; track(src, ts); }
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
    if (window.__psTrackHook) { try { window.__psTrackHook(src.el, tracker.pose, r); } catch (e) { /* test hook */ } }
  }

  // ------------------------------------------------------------ inference
  function maybeInfer(ts) {
    const src = state.src;
    if (!state.modelsReady || state.busy || !src || document.hidden || state.recovering) return;
    if (src.kind === 'image') {
      if (state.inferredSrc === src && state.inferredFd === state.fd) return;
    } else {
      const el = src.el;
      if (el.readyState < 2 || !el.videoWidth) return;
      if (src.kind === 'camera' && state.camMuted) return;          // the camera sends no picture
      if (el.paused && el.currentTime === state.lastInferVideoTime) return;
      if (src.blankUntil && ts < src.blankUntil) return;           // last frame was blank: retry shortly
      if (state.backend === 'cpu' && ts - state.lastInferEnd < 250) return;   // keep the page responsive on CPU
    }
    infer(src);
  }

  function nextFrame() { return new Promise((r) => requestAnimationFrame(() => r())); }

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
  function blankCapture(cw, ch) {
    let d;
    try { d = capCtx.getImageData(0, 0, cw, ch).data; } catch (e) { return false; }
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
    state.busy = true;
    const t0 = performance.now();
    state.inferT0 = t0;
    const capPose = { yaw: tracker.pose.yaw, pitch: tracker.pose.pitch };
    const W = src.w, H = src.h;
    const scale = Math.min(1, 640 / Math.max(W, H));
    const cw = Math.max(1, Math.round(W * scale)), ch = Math.max(1, Math.round(H * scale));
    if (cap.width !== cw || cap.height !== ch) { cap.width = cw; cap.height = ch; }
    let px = null;
    let capInfo = null;
    try {
      capCtx.drawImage(src.el, 0, 0, cw, ch);
      if (src.kind !== 'image' && !src.seenPicture) {
        if (blankCapture(cw, ch)) { src.blankUntil = performance.now() + 300; return; }
        src.seenPicture = true;
      }
      if (src.kind !== 'image') state.lastInferVideoTime = src.el.currentTime;
      if (window.__psCaptureHook) { try { capInfo = window.__psCaptureHook(cap, capCtx); } catch (e) { /* test hook */ } }
      px = tf.browser.fromPixels(cap);
      const hfov = effectiveHfov(W, H);
      const cpu = state.backend === 'cpu';
      const runPerson = !cpu || state.cycle % 2 === 0 || !state.lastPersons || src.kind === 'image';
      state.cycle++;
      const r = await PSPeople.detect(px, { hfovDeg: hfov, person: runPerson });
      if (!current()) return;
      let people;
      if (runPerson) {
        people = r.display;
        state.lastPersons = { dets: r.detections.filter((d) => d.cls === 'person'), t: t0, pose: capPose };
      } else {
        const faces = r.detections.filter((d) => d.cls === 'face');
        people = PSPeople.merge(carriedPersons(capPose, W, H).concat(faces), r.width, r.height, { hfovDeg: hfov });
      }
      if (cpu) await nextFrame();
      if (!current()) return;
      const t1 = performance.now();
      let fd = [];
      // the fire/door centre pass: every update, except on the CPU backend only when the person model rests
      if (state.fd && state.fdReady && state.src === src) fd = (await state.fd.detect(px, { zoom: src.kind === 'image' || (cpu ? !runPerson : 'auto') })) || [];
      if (!current()) return;
      const fdMs = performance.now() - t1;
      if (state.src !== src) return;     // source changed meanwhile
      const res = {
        people, fire: fd.filter((d) => d.cls === 'fire'), door: fd.filter((d) => d.cls === 'door'),
        pose: capPose, t: t0, w: W, h: H,
        ms: { person: runPerson ? r.ms.person : null, face: r.ms.face, firedoor: fdMs, total: performance.now() - t0 },
      };
      state.results = res;
      state.lastCycleMs = state.upsTimes.length ? performance.now() - state.upsTimes[state.upsTimes.length - 1] : res.ms.total;
      if (src.kind === 'image') { state.inferredSrc = src; state.inferredFd = state.fd; }
      state.stats.inferences++;
      const nt = tf.memory().numTensors;
      state.stats.tensors.push(nt);
      if (state.stats.tensors.length > 2000) state.stats.tensors.splice(0, 1000);
      state.stats.ms.push(res.ms);
      if (state.stats.ms.length > 500) state.stats.ms.splice(0, 250);
      state.stats.labels.push(people.map((d) => d.label || '?').concat(res.fire.map(() => 'FIRE'), res.door.map(() => 'DOOR')).join(' '));
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
      if (gpuLost()) onGpuLost('error');
    } finally {
      if (px) { try { px.dispose(); } catch (e) { /* lost context */ } }
      if (current()) {
        state.busy = false;
        state.lastInferEnd = performance.now();
      }
    }
  }

  // ------------------------------------------------------------ detector health
  // Phones drop the WebGL context under memory pressure or after the page was in the background.
  // TF.js then never finishes the pending read (busy for good). Detect it (event, isContextLost(),
  // or a run that takes far longer than usual), drop the old boxes, tell the viewer and restart the
  // detectors: on a new WebGL context, or on the CPU after repeated losses.
  function gpuLost() {
    if (state.backend !== 'webgl') return false;
    try { const gl = tf.backend().gpgpu.gl; return !!(gl && gl.isContextLost()); } catch (e) { return false; }
  }

  function watchGpu() {
    if (state.backend !== 'webgl') return;
    try {
      const gl = tf.backend().gpgpu.gl, cv = gl && gl.canvas;
      if (cv && cv.addEventListener && !cv.__psWatched) {
        cv.__psWatched = true;
        cv.addEventListener('webglcontextlost', () => { if (gl === tf.backend().gpgpu.gl) setTimeout(() => onGpuLost('event'), 0); }, false);
      }
    } catch (e) { /* not WebGL */ }
  }

  function typicalRunMs() {
    const m = state.stats.ms.slice(-20).map((x) => x.total).sort((a, b) => a - b);
    return m.length ? m[m.length >> 1] : 0;
  }

  function watchDetection(ts) {
    if (!state.modelsReady || state.recovering || ts - state.lastCheckT < 500) return;
    state.lastCheckT = ts;
    if (gpuLost()) { onGpuLost('poll'); return; }
    if (state.busy && performance.now() - state.inferT0 > Math.max(20000, 8 * typicalRunMs())) onGpuLost('stall');
  }

  async function onGpuLost(why) {
    if (state.recovering || !state.modelsReady) return;
    state.recovering = true;
    state.modelsReady = false;
    state.inferGen++; state.busy = false;              // abandon the run that will never finish
    state.results = null; state.lastPersons = null;    // its boxes are stale: stop drawing them
    if (state.fd && state.fd.reset) { try { state.fd.reset(); } catch (e) { /* ignore */ } }
    if (why !== 'stall') state.gpuLosses++; else state.stalls++;
    state.stats.recoveries.push({ why, t: Math.round(performance.now()) });
    // TF.js polls a fence on the lost context forever (a busy loop with console warnings): drop it
    try { const gp = tf.backend().gpgpu; if (gp && Array.isArray(gp.itemsToPoll)) gp.itemsToPoll = []; } catch (e) { /* ignore */ }
    say(why === 'stall' ? 'Detection stalled. Restarting.' : 'Detection stopped: the graphics chip was reset. Restarting.', { minGap: 0 });
    updateStatus(true);
    await new Promise((r) => setTimeout(r, 300));
    const start = async (prefer) => {
      try { PSPeople.dispose(); } catch (e) { /* lost context */ }
      try { if (state.fd && state.fd.dispose) state.fd.dispose(); } catch (e) { /* lost context */ }
      state.fd = null; state.fdReady = false;
      if (state.backend === 'webgl') {
        // a fresh WebGL backend (TF.js forgets the lost context and makes a new one)
        const fac = tf.findBackendFactory && tf.findBackendFactory('webgl');
        try { tf.removeBackend('webgl'); } catch (e) { /* ignore */ }
        if (fac && !tf.findBackendFactory('webgl')) { try { tf.registerBackend('webgl', fac, 2); } catch (e) { prefer = 'cpu'; } }
        if (!fac) prefer = 'cpu';
      }
      state.backend = await withTimeout(PSPeople.setupBackend(tf, { prefer, allowSoftwareWebGL: qs.get('swgl') !== '0' }), 15000);
      state.backendInfo = PSPeople.backendInfo;
      await withTimeout(PSPeople.init(tf, window.PS_PEOPLE_ASSETS), 90000);
      const det = window.PS_FIREDOOR_DETECTOR || fireDoorAdapter() || FIREDOOR_STUB;
      state.fd = det;
      try { await withTimeout(det.init(tf), 60000); } catch (e) { state.fdError = e; state.fd = FIREDOOR_STUB; }
      state.fdReady = true;
    };
    try {
      try {
        await start(state.gpuLosses + state.stalls >= 3 ? 'cpu' : undefined);   // keeps failing: stay on the CPU
      } catch (e) {
        if (state.backend === 'cpu') throw e;
        await start('cpu');
      }
      state.modelsReady = true;
      state.loadError = null;
      watchGpu();
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
      tensors: tf.memory().numTensors,
      people: res.people.map((d) => box(d, { label: d.label || '', from: d.src, dist: d.dist == null ? null : +d.dist.toFixed(2) })),
      fire: res.fire.map((d) => box(d, { label: 'FIRE' })),
      door: res.door.map((d) => box(d, { label: 'DOOR' })),
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
    if (!src || !src.w || !src.h) { state.fit = null; return; }
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
    const ex = exitInfo(src);
    const exitBox = ex && ex.trusted && ex.inView ? ex.box : null;
    // boxes first, labels last (FIRE on top), so no box edge runs through a label
    const labels = [];
    if (res) {
      const sh = shiftSince(res, src);
      const mv = (b) => ({ x: b.x + sh.dx, y: b.y + sh.dy, w: b.w, h: b.h });
      for (const d of res.door) {
        if (exitBox && iou(mv(d), exitBox) > 0.3) continue;   // the EXIT mark is on this door: one green box
        drawBox(toCanvas(mv(d), fit, src.mirror), COL.exit, 'DOOR', fit, s, labels, 1);
      }
      for (const d of res.people) drawBox(toCanvas(mv(d), fit, src.mirror), COL.person, d.label || '', fit, s, labels, 2);
      for (const d of res.fire) drawBox(toCanvas(mv(d), fit, src.mirror), COL.fire, 'FIRE', fit, s, labels, 3);
    }
    drawExit(src, fit, s, ts, labels);
    drawLabels(labels, fit, s);
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
  function drawLabels(labels, fit, s) {
    const fs = 14 * s;
    setFont(fs);
    g.textBaseline = 'alphabetic';
    g.textAlign = 'left';
    const placed = [];
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

  function drawEyepiece(src, fit) {
    // Grey "white-hot" look-alike: low resolution like the 160 x 120 thermal
    // sensor, contrast stretched. It shows brightness, not heat.
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
          d[j] = d[j + 1] = d[j + 2] = v; d[j + 3] = 255;
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
    const turn = Math.round(Math.min(180, Math.hypot(p.rel.yaw, p.rel.pitch)));
    const text = 'EXIT ' + turn + '°' + (p.unsure ? '?' : '');
    setFont(13 * s);
    const tw = g.measureText(text).width;
    let lx = ax - vx * (r + 8 * s) - tw / 2, ly = ay - vy * (r + 8 * s) + 5 * s;
    lx = Math.min(Math.max(lx, fit.x + 4 * s), fit.x + fit.w - tw - 4 * s);
    ly = Math.min(Math.max(ly, fit.y + 16 * s), fit.y + fit.h - 6 * s);
    g.textAlign = 'left'; g.textBaseline = 'alphabetic';
    outlinedText(text, lx, ly, COL.exit, s);
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
      hudText(state.loadError ? 'DETECTORS FAILED' : state.recovering ? 'DETECTION STOPPED · RESTARTING…' : 'LOADING AI…',
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
      if (!state.mark) { g.textAlign = 'center'; hudText('MARK WAY OUT', fit.x + fit.w / 2, fit.y + 20 * s, '#FFFFFF', s); }
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
    let cx = nx, cy = ny, size = { w: 14, h: 28 }, door = null;
    const res = state.results;
    if (res && res.door.length) {
      // snap to a detected door at the tapped point (or near the centre for the button)
      const sh = shiftSince(res, src);
      let best = null, bestD = how === 'tap' ? 0 : 0.2;
      for (const d of res.door) {
        const b = { x: d.x + sh.dx, y: d.y + sh.dy, w: d.w, h: d.h };
        const inside = nx >= b.x && nx <= b.x + b.w && ny >= b.y && ny <= b.y + b.h;
        const dist = Math.hypot(b.x + b.w / 2 - nx, b.y + b.h / 2 - ny);
        if (inside && (!best || dist < bestD)) { best = b; bestD = dist; }
        else if (!inside && how !== 'tap' && dist < bestD) { best = b; bestD = dist; }
      }
      if (best) {
        door = best; cx = best.x + best.w / 2; cy = best.y + best.h / 2;
        size = M.boxAngles(best, src.w, src.h, FOV);
      }
    }
    if (src.live) tracker.reanchor();   // a mark made while the picture is not matching starts a fresh key frame
    const dir = M.pointToDirection(cx, cy, tracker.pose, src.w, src.h, FOV);
    state.mark = { yaw: dir.yaw, pitch: dir.pitch, w: Math.max(4, size.w), h: Math.max(6, size.h), t: performance.now(), door: !!door, how };
    tracker.hasMark = true;
    state.markLostSaid = false; state.exitSector = null; state.exitOutSince = 0; state.lastExitSay = performance.now();
    $('clear-mark').hidden = false;
    say(door ? 'Way out marked at the door.' : 'Way out marked.', { minGap: 0 });
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
    if (state.recovering) { eng += ' · detection stopped (graphics reset or stall), restarting…'; engCls = 'warn'; }
    else if (state.backend && !state.modelsReady && !state.loadError) eng += ' · loading detectors…';
    else if (state.gpuLosses || state.stalls) eng += ' · restarted ' + plural(state.gpuLosses + state.stalls, 'time', 'times');
    setV('s-engine', eng, engCls);
    $('engine').textContent = state.loadError ? 'Detectors failed' : state.recovering ? 'Restarting detectors…' : !state.modelsReady ? 'Loading detectors…' :
      (state.backend === 'webgl' ? 'WebGL' : 'CPU') + (res ? ' · ' + Math.round(res.ms.total) + ' ms' : '') +
      (src && src.live ? ' · ' + state.fps.toFixed(0) + ' fps' : '');
    // speed
    if (res) {
      const ups = state.upsTimes.length > 1 ? (state.upsTimes.length - 1) * 1000 / (state.upsTimes[state.upsTimes.length - 1] - state.upsTimes[0]) : 0;
      const parts = [];
      if (res.ms.person != null) parts.push('people ' + Math.round(res.ms.person) + ' ms');
      if (res.ms.face != null) parts.push('faces ' + Math.round(res.ms.face) + ' ms');
      if (state.fd && !state.fd.stub) parts.push('fire/doors ' + Math.round(res.ms.firedoor) + ' ms');
      let txt = parts.join(', ');
      if (src && src.live) txt += ' · ' + ups.toFixed(1) + ' checks/s · video ' + state.fps.toFixed(0) + ' fps';
      setV('s-speed', txt);
    } else setV('s-speed', state.modelsReady ? 'Waiting for a picture' : '–');
    // tensors (stays flat when nothing leaks)
    $('tensors').textContent = state.modelsReady ? 'tensors ' + tf.memory().numTensors : '';
    // seen
    if (src && src.kind === 'camera' && state.camMuted) setV('s-seen', 'No picture from the camera (another app may be using it, or it is covered)', 'warn');
    else if (state.recovering) setV('s-seen', 'Detection stopped: restarting the detectors…', 'warn');
    else if (res && src && !resultsToDraw(src)) setV('s-seen', 'Detection paused: no recent check', 'warn');
    else if (res && src) {
      const near = res.people.filter((d) => d.dist > 0).sort((a, b) => a.dist - b.dist)[0];
      const bits = [];
      if (res.people.length) bits.push(plural(res.people.length, 'person', 'people') + (near ? ' (nearest about ' + near.dist.toFixed(1) + ' m)' : ''));
      if (res.fire.length) bits.push(plural(res.fire.length, 'fire', 'fires'));
      if (res.door.length) bits.push(plural(res.door.length, 'door', 'doors'));
      setV('s-seen', bits.length ? bits.join(', ') : 'Nothing found');
    } else setV('s-seen', '–');
    // way out
    const p = exitInfo(src);
    if (!p) setV('s-exit', 'Not marked');
    else if (!p.trusted) setV('s-exit', 'Tracking lost: point the camera back where it was, or mark it again', 'warn');
    else if (p.inView) setV('s-exit', 'In view' + (state.mark.door ? ' (marked at a door)' : '') + (p.unsure ? ' (unsure: picture not matching)' : ''), p.unsure ? 'warn' : '');
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
  $('mark').addEventListener('click', () => { requestOrientation(); markWayOut(0.5, 0.5, 'centre'); });
  $('clear-mark').addEventListener('click', () => { clearMark(false); updateStatus(true); });
  $('whereout').addEventListener('click', whereOut);
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
  $('eyepiece').addEventListener('click', function () {
    state.eyepiece = !state.eyepiece;
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
    FIREDOOR_STUB,
  };

  // ------------------------------------------------------------ start
  updateStatus(true);
  renderLog();
  requestAnimationFrame(frame);
  setTimeout(loadModels, 30);   // after the first paint
})();
