/*
 * PyroSight Camera: people + face detector (people.js)
 *
 * Plain script, no bundler; defines globalThis.PSPeople. Needs TF.js and
 * oplist.js (PSOpList) loaded first. No network access: every model comes
 * from the embedded assets object passed to init().
 *
 *   await PSPeople.setupBackend(tf);                 // 'webgl', else 'cpu'
 *   await PSPeople.init(tf, PS_PEOPLE_ASSETS);       // {person, face}
 *   const r = await PSPeople.detect(videoOrCanvas);  // see below
 *   r.detections  raw boxes  [{cls:'person'|'face', score, x, y, w, h, ...}]
 *   r.display     merged     [{cls:'person', score, x, y, w, h, dist, label, src, ...}]
 *
 * Coordinates are normalised to the source image: x, y = top-left corner,
 * w, h = size, all in [0, 1].
 *
 * Models
 *   person  COCO-SSD lite_mobilenet_v2 (TF.js graph model, re-packed by
 *           repack_cocossd.py: preprocessor cut out, person-only class head,
 *           per-channel uint8 weights), 300x300 input.
 *   face    MediaPipe BlazeFace short-range (TFLite -> op list by
 *           export_oplist.py), 128x128 letterboxed input in [-1, 1],
 *           896 SSD anchors, MediaPipe-style decoding and weighted NMS.
 *
 * Distance (an ESTIMATE, pinhole model like the eyepiece firmware):
 *   f_px = (image_width_px / 2) / tan(hfov / 2)       (square pixels)
 *   face   : d = faceWidthM   * f_px / face_box_width_px
 *   person : d = personHeightM * f_px / max(box_w_px, box_h_px)
 *   A box touching the frame edge is 'truncated': the real person is bigger
 *   than the box, so the distance is an upper bound and the label gets '<'
 *   (same convention as ps_display.c). A person box that holds a face uses
 *   the face estimate, which is far better at webcam range.
 */
(function (root) {
  'use strict';

  const DEFAULTS = {
    personMinScore: 0.4,    // COCO-SSD person score (coco-ssd default 0.5; README has the eval)
    faceMinScore: 0.75,     // BlazeFace score for a face shown on its own (MediaPipe default 0.5)
    faceAttachMinScore: 0.75, // face score needed to give a person box its distance (may be < faceMinScore)
    faceDecodeMinScore: 0.5,  // MediaPipe's min_score_thresh: anchors blended by the weighted NMS
    personIoU: 0.5,         // NMS IoU, same as coco-ssd's default
    faceIoU: 0.3,           // MediaPipe weighted-NMS IoU
    maxPersons: 20,
    maxFaces: 10,
    hfovDeg: 65,            // horizontal field of view of the camera
    faceWidthM: 0.16,
    personHeightM: 1.7,
    faceInPerson: 0.6,      // fraction of the face box inside a person box to merge
    edgeMargin: 0.01,       // a box this close to the frame edge is cut off: its distance is an upper bound ('<')
    mirror: false,          // flip x (for a mirrored selfie preview)
    person: true,
    face: true,
  };

  let tf = null;
  let personModel = null, personInfo = null;
  let faceModel = null, faceAnchors = null;
  let opts0 = Object.assign({}, DEFAULTS);
  let backendInfo = null;

  // -------------------------------------------------------------- backend
  /**
   * setupBackend(tf, {prefer, allowSoftwareWebGL}) -> 'webgl' | 'cpu'
   * Tries WebGL first, then (only if allowSoftwareWebGL) WebGL on a software
   * rasteriser such as SwiftShader, which TF.js refuses by default, then the
   * CPU backend. Never uses the wasm backend (not in tf.min.js, and a
   * sandboxed artifact may block WebAssembly). A string second argument is
   * taken as `prefer`.
   */
  async function setupBackend(tfRef, o) {
    if (typeof o === 'string') o = { prefer: o };
    o = o || {};
    // TF.js caches its WebGL capability flags on the first attempt, so the
    // software-rasteriser permission has to be set before trying WebGL at all.
    if (o.allowSoftwareWebGL) tfRef.env().set('SOFTWARE_WEBGL_ENABLED', true);
    const tries = o.prefer === 'cpu' ? ['cpu'] : ['webgl', 'cpu'];
    for (const b of tries) {
      try {
        if (!(await tfRef.setBackend(b))) continue;
        await tfRef.ready();
        const t = tfRef.tidy(() => tfRef.add(tfRef.ones([2, 2]), 1).sum());
        const v = (await t.data())[0];
        t.dispose();
        if (v !== 8) continue;
        backendInfo = { backend: b, renderer: b === 'webgl' ? glRenderer(tfRef) : 'cpu' };
        backendInfo.software = /swiftshader|llvmpipe|software|basic render/i.test(backendInfo.renderer);
        return b;
      } catch (e) { /* try next */ }
    }
    throw new Error('PSPeople: no usable TF.js backend');
  }

  function glRenderer(tfRef) {
    try {
      const gl = tfRef.backend().gpgpu.gl;
      const ext = gl.getExtension('WEBGL_debug_renderer_info');
      return String(ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER));
    } catch (e) { return 'webgl'; }
  }

  // -------------------------------------------------------------- loading
  function assetBuffer(a) {
    if (a.weights) return a.weights;
    return root.PSOpList.base64ToArrayBuffer(a.weightsB64);
  }

  async function loadPerson(asset) {
    const doc = typeof asset.json === 'string' ? JSON.parse(asset.json) : asset.json;
    const vals = root.PSOpList.decodeWeights(doc.packed, assetBuffer(asset));
    let total = 0;
    for (const s of doc.weightSpecs) total += 4 * s.shape.reduce((a, b) => a * b, 1);
    const data = new Uint8Array(total);
    let off = 0;
    for (const s of doc.weightSpecs) {
      const v = vals[s.name];
      data.set(new Uint8Array(v.buffer, v.byteOffset, v.byteLength), off);
      off += v.byteLength;
    }
    const model = await tf.loadGraphModel(tf.io.fromMemory({
      modelTopology: doc.modelTopology, weightSpecs: doc.weightSpecs, weightData: data.buffer,
    }));
    return { model, doc };
  }

  /** MediaPipe SsdAnchorsCalculator, face_detection_short_range options. */
  function blazeFaceAnchors() {
    const strides = [8, 16, 16, 16];
    const numLayers = 4, size = 128, out = [];
    let layer = 0;
    while (layer < numLayers) {
      let last = layer, n = 0;
      while (last < numLayers && strides[last] === strides[layer]) { n += 2; last++; }  // ar 1.0 + interpolated
      const fm = Math.ceil(size / strides[layer]);
      for (let y = 0; y < fm; y++) {
        for (let x = 0; x < fm; x++) {
          for (let k = 0; k < n; k++) out.push([(x + 0.5) / fm, (y + 0.5) / fm]);  // fixed_anchor_size: w=h=1
        }
      }
      layer = last;
    }
    return out;   // 896 x [cx, cy]
  }

  async function init(tfRef, assets, options) {
    tf = tfRef;
    opts0 = Object.assign({}, DEFAULTS, options || {});
    if (!root.PSOpList) throw new Error('PSPeople.init: load oplist.js first');
    assets = assets || root.PS_PEOPLE_ASSETS;
    dispose();   // re-init frees the previous models
    const t0 = now();
    if (assets.person) {
      const p = await loadPerson(assets.person);
      personModel = p.model;
      personInfo = p.doc;
    }
    if (assets.face) {
      faceModel = root.PSOpList.loadEmbedded(assets.face, { tf });
      faceAnchors = blazeFaceAnchors();
      if (faceAnchors.length !== 896) throw new Error('anchor count ' + faceAnchors.length);
    }
    // warm-up (shader compilation on WebGL)
    const z = tf.zeros([240, 320, 3], 'int32');
    await detect(z);
    z.dispose();
    return { backend: tf.getBackend(), loadMs: now() - t0, person: !!personModel, face: !!faceModel,
      personClasses: personInfo ? personInfo.classes.map((c) => c.name) : [] };
  }

  function now() { return (typeof performance !== 'undefined' ? performance : Date).now(); }

  // -------------------------------------------------------------- person
  function personInput(pixels) {
    return tf.tidy(() => {
      const x = tf.image.resizeBilinear(tf.cast(pixels, 'float32'), [300, 300], false, false);
      return tf.expandDims(tf.sub(tf.mul(x, 2 / 255), 1), 0);
    });
  }

  async function runPerson(pixels, o) {
    const x = personInput(pixels);
    const [sc, bx] = personModel.execute({ [personInfo.input.name]: x },
      [personInfo.outputs.scores, personInfo.outputs.boxes]);
    x.dispose();
    const [scores, boxes] = await Promise.all([sc.data(), bx.data()]);
    const K = sc.shape[2];
    sc.dispose(); bx.dispose();
    return decodePerson(scores, boxes, K, o);
  }

  /** coco-ssd post-processing restricted to the person class. */
  function decodePerson(scores, boxes, K, o) {
    const n = boxes.length / 4;
    const personIdx = personInfo ? personInfo.classes.findIndex((c) => c.name === 'person') : 0;
    const cand = [];
    for (let i = 0; i < n; i++) {
      let best = -1, bi = -1;
      for (let k = 0; k < K; k++) { const v = scores[i * K + k]; if (v > best) { best = v; bi = k; } }
      if (bi === personIdx && best > o.personMinScore) cand.push(i);
    }
    cand.sort((a, b) => scores[b * K + personIdx] - scores[a * K + personIdx]);
    const sel = [];
    for (const i of cand) {
      if (sel.length >= o.maxPersons) break;
      let keep = true;
      for (const j of sel) { if (iouYX(boxes, i, j) >= o.personIoU) { keep = false; break; } }
      if (keep) sel.push(i);
    }
    return sel.map((i) => {
      const y0 = clamp01(Math.min(boxes[4 * i], boxes[4 * i + 2]));
      const x0 = clamp01(Math.min(boxes[4 * i + 1], boxes[4 * i + 3]));
      const y1 = clamp01(Math.max(boxes[4 * i], boxes[4 * i + 2]));
      const x1 = clamp01(Math.max(boxes[4 * i + 1], boxes[4 * i + 3]));
      return { cls: 'person', score: scores[i * K + personIdx], x: x0, y: y0, w: x1 - x0, h: y1 - y0 };
    });
  }

  function iouYX(b, i, j) {   // boxes as [ymin, xmin, ymax, xmax], like tf.image.nonMaxSuppression
    const ay0 = Math.min(b[4 * i], b[4 * i + 2]), ax0 = Math.min(b[4 * i + 1], b[4 * i + 3]);
    const ay1 = Math.max(b[4 * i], b[4 * i + 2]), ax1 = Math.max(b[4 * i + 1], b[4 * i + 3]);
    const by0 = Math.min(b[4 * j], b[4 * j + 2]), bx0 = Math.min(b[4 * j + 1], b[4 * j + 3]);
    const by1 = Math.max(b[4 * j], b[4 * j + 2]), bx1 = Math.max(b[4 * j + 1], b[4 * j + 3]);
    const aa = (ay1 - ay0) * (ax1 - ax0), ab = (by1 - by0) * (bx1 - bx0);
    if (aa <= 0 || ab <= 0) return 0;
    const iy = Math.max(Math.min(ay1, by1) - Math.max(ay0, by0), 0);
    const ix = Math.max(Math.min(ax1, bx1) - Math.max(ax0, bx0), 0);
    const inter = iy * ix;
    return inter / (aa + ab - inter);
  }

  // -------------------------------------------------------------- face
  /**
   * Letterbox to 128x128 the way MediaPipe's ImageToTensorCalculator does it
   * (keep aspect, centred, zero border, bilinear without half-pixel offset,
   * then [-1, 1]): pad to a centred square, resize, scale. Checked against
   * the official MediaPipe Tasks FaceDetector (tests/face_parity.js).
   */
  function faceInput(pixels) {
    const H = pixels.shape[0], W = pixels.shape[1];
    const S = Math.max(H, W);
    const py = Math.floor((S - H) / 2), px = Math.floor((S - W) / 2);
    const x = tf.tidy(() => {
      let r = tf.cast(pixels, 'float32');
      if (S !== H || S !== W) r = tf.pad(r, [[py, S - H - py], [px, S - W - px], [0, 0]]);
      r = tf.image.resizeBilinear(r, [128, 128], false, false);
      return tf.expandDims(tf.sub(tf.div(r, 127.5), 1), 0);
    });
    return { x, lb: { S, W, H, px, py } };
  }

  async function runFace(pixels, o) {
    const { x, lb } = faceInput(pixels);
    const out = faceModel.run(x);
    x.dispose();
    const [reg, cls] = await Promise.all([out.regressors.data(), out.classificators.data()]);
    tf.dispose(Object.values(out));
    return decodeFaces(reg, cls, o).map((d) => unletterbox(d, lb));
  }

  /**
   * TensorsToDetectionsCalculator (x/y/w/h scale 128, 6 keypoints, sigmoid,
   * clip 100) + NonMaxSuppressionCalculator (IoU, WEIGHTED).
   * Returns letterbox-space detections {score, x0,y0,x1,y1, kp:[[x,y]x6]}.
   */
  function decodeFaces(reg, cls, o) {
    const dets = [];
    const keep = Math.min(o.faceMinScore, o.faceAttachMinScore === undefined ? o.faceMinScore : o.faceAttachMinScore);
    const thr = Math.min(keep, o.faceDecodeMinScore === undefined ? 0.5 : o.faceDecodeMinScore);
    for (let i = 0; i < 896; i++) {
      let s = cls[i];
      s = s < -100 ? -100 : s > 100 ? 100 : s;
      s = 1 / (1 + Math.exp(-s));
      if (s < thr) continue;
      const a = faceAnchors[i], r = i * 16;
      const cx = reg[r] / 128 + a[0], cy = reg[r + 1] / 128 + a[1];
      const w = reg[r + 2] / 128, h = reg[r + 3] / 128;
      const kp = [];
      for (let k = 0; k < 6; k++) kp.push([reg[r + 4 + 2 * k] / 128 + a[0], reg[r + 5 + 2 * k] / 128 + a[1]]);
      dets.push({ score: s, x0: cx - w / 2, y0: cy - h / 2, x1: cx + w / 2, y1: cy + h / 2, kp });
    }
    return weightedNms(dets, o.faceIoU, o.maxFaces).filter((d) => d.score >= keep);
  }

  function iouRect(a, b) {
    const aa = (a.x1 - a.x0) * (a.y1 - a.y0), ab = (b.x1 - b.x0) * (b.y1 - b.y0);
    if (aa <= 0 || ab <= 0) return 0;
    const ix = Math.max(0, Math.min(a.x1, b.x1) - Math.max(a.x0, b.x0));
    const iy = Math.max(0, Math.min(a.y1, b.y1) - Math.max(a.y0, b.y0));
    const inter = ix * iy;
    return inter / (aa + ab - inter);
  }

  function weightedNms(dets, thr, maxN) {
    let rem = dets.slice().sort((a, b) => b.score - a.score);
    const out = [];
    while (rem.length && out.length < maxN) {
      const d = rem[0];
      const cand = [], rest = [];
      for (const e of rem) (iouRect(d, e) > thr ? cand : rest).push(e);
      let tot = 0;
      const acc = { x0: 0, y0: 0, x1: 0, y1: 0, kp: d.kp.map(() => [0, 0]) };
      for (const e of cand) {
        tot += e.score;
        acc.x0 += e.score * e.x0; acc.y0 += e.score * e.y0; acc.x1 += e.score * e.x1; acc.y1 += e.score * e.y1;
        e.kp.forEach((p, k) => { acc.kp[k][0] += e.score * p[0]; acc.kp[k][1] += e.score * p[1]; });
      }
      out.push({ score: d.score, x0: acc.x0 / tot, y0: acc.y0 / tot, x1: acc.x1 / tot, y1: acc.y1 / tot,
        kp: acc.kp.map((p) => [p[0] / tot, p[1] / tot]) });
      rem = rest;
    }
    return out;
  }

  function unletterbox(d, lb) {
    const fx = (v) => (v * lb.S - lb.px) / lb.W, fy = (v) => (v * lb.S - lb.py) / lb.H;
    const x0 = clamp01(fx(d.x0)), y0 = clamp01(fy(d.y0)), x1 = clamp01(fx(d.x1)), y1 = clamp01(fy(d.y1));
    return { cls: 'face', score: d.score, x: x0, y: y0, w: x1 - x0, h: y1 - y0,
      keypoints: d.kp.map((p) => [fx(p[0]), fy(p[1])]) };
  }

  // -------------------------------------------------------------- distance
  function focalPx(imgW, hfovDeg) { return (imgW / 2) / Math.tan(hfovDeg * Math.PI / 360); }

  function distLabel(d, truncated) {
    if (!(d > 0)) return '';
    const p = truncated ? '<' : '';
    if (d < 10) return p + Math.floor(d) + '.' + (Math.floor(d * 10) % 10) + 'M';
    return p + Math.round(d) + 'M';
  }

  function addDistance(det, imgW, imgH, o) {
    const f = focalPx(imgW, o.hfovDeg);
    // COCO-SSD and BlazeFace boxes of someone cut off by the frame edge stop about 0.2-1 % short of
    // it, so "touches the edge" means within 1 % (0.2 % missed 17 of 19 close faces at 1.2 m+)
    const e = o.edgeMargin;
    det.truncated = det.x <= e || det.y <= e || det.x + det.w >= 1 - e || det.y + det.h >= 1 - e;
    if (det.cls === 'face') {
      const wpx = Math.max(1, det.w * imgW);
      det.dist = o.faceWidthM * f / wpx;
      det.distMin = det.dist * 0.8;
      det.distMax = det.dist * 1.2;
      det.distSrc = 'face';
    } else {
      const px = Math.max(1, Math.max(det.w * imgW, det.h * imgH));
      det.dist = o.personHeightM * f / px;
      det.distMin = det.truncated ? 0.3 : det.dist * 0.55;
      det.distMax = det.dist * 1.15;
      det.distSrc = 'height';
    }
    det.distEstimate = true;
    det.label = distLabel(det.dist, det.cls !== 'face' && det.truncated);
    return det;
  }

  // -------------------------------------------------------------- merge
  /**
   * Display list: every person box; a face inside a person box (>= faceInPerson
   * of its area, score >= faceAttachMinScore) does not add a box but gives that
   * person its (better) distance; a face with no person around it
   * (score >= faceMinScore) is shown as a person.
   */
  function merge(dets, imgW, imgH, o) {
    o = Object.assign({}, opts0, o || {});
    const attach = o.faceAttachMinScore === undefined ? o.faceMinScore : Math.min(o.faceAttachMinScore, o.faceMinScore);
    const persons = dets.filter((d) => d.cls === 'person' && d.score >= o.personMinScore);
    const faces = dets.filter((d) => d.cls === 'face' && d.score >= attach).sort((a, b) => b.w * b.h - a.w * a.h);
    const out = persons.map((p) => Object.assign({}, p, { src: 'person' }));
    const taken = new Set();
    for (const f of faces) {
      let best = -1, bestArea = Infinity;
      out.forEach((p, i) => {
        if (taken.has(i) || p.src === 'face') return;
        const ix = Math.max(0, Math.min(f.x + f.w, p.x + p.w) - Math.max(f.x, p.x));
        const iy = Math.max(0, Math.min(f.y + f.h, p.y + p.h) - Math.max(f.y, p.y));
        const frac = (ix * iy) / Math.max(1e-9, f.w * f.h);
        const a = p.w * p.h;
        if (frac >= o.faceInPerson && f.w <= 1.0 * p.w + 1e-6 && a < bestArea) { best = i; bestArea = a; }
      });
      if (best >= 0) {
        const p = out[best];
        taken.add(best);
        p.face = f;
        p.src = 'person+face';
        p.dist = f.dist; p.distMin = f.distMin; p.distMax = f.distMax; p.distSrc = 'face';
        p.label = distLabel(p.dist, false);
        p.score = Math.max(p.score, f.score);
      } else if (f.score >= o.faceMinScore) {
        // a face alone: show it as a person (slightly padded head box)
        const pad = 0.15;
        const x0 = clamp01(f.x - pad * f.w), y0 = clamp01(f.y - pad * f.h);
        const x1 = clamp01(f.x + f.w * (1 + pad)), y1 = clamp01(f.y + f.h * (1 + pad));
        out.push({ cls: 'person', src: 'face', score: f.score, x: x0, y: y0, w: x1 - x0, h: y1 - y0,
          face: f, dist: f.dist, distMin: f.distMin, distMax: f.distMax, distSrc: 'face', distEstimate: true,
          truncated: false, label: distLabel(f.dist, false) });
      }
    }
    return out;
  }

  // -------------------------------------------------------------- detect
  /**
   * detect(source, opts) -> Promise<{detections, display, width, height, ms}>
   * source: HTMLVideoElement | HTMLCanvasElement | HTMLImageElement | ImageData |
   *         ImageBitmap | tf.Tensor3D [H, W, 3] (RGB, 0..255)
   */
  async function detect(source, options) {
    if (!tf) throw new Error('PSPeople.detect: call init() first');
    const o = Object.assign({}, opts0, options || {});
    const t0 = now();
    const own = !(source instanceof tf.Tensor);
    if (own && source && source.videoWidth === 0) {   // <video> without a frame yet
      return { detections: [], display: [], width: 0, height: 0, ms: { total: 0 } };
    }
    const pixels = own ? tf.browser.fromPixels(source) : source;
    const H = pixels.shape[0], W = pixels.shape[1];
    let persons = [], faces = [];
    const ms = {};
    try {
      if (personModel && o.person) {
        const t = now();
        persons = await runPerson(pixels, o);
        ms.person = now() - t;
      }
      if (faceModel && o.face) {
        const t = now();
        faces = await runFace(pixels, o);
        ms.face = now() - t;
      }
    } finally {
      if (own) pixels.dispose();
    }
    let dets = persons.concat(faces).filter((d) => d.w > 0 && d.h > 0);
    if (o.mirror) {
      dets = dets.map((d) => Object.assign(d, { x: 1 - d.x - d.w },
        d.keypoints ? { keypoints: d.keypoints.map((p) => [1 - p[0], p[1]]) } : {}));
    }
    dets.forEach((d) => addDistance(d, W, H, o));
    const display = merge(dets, W, H, o);
    ms.total = now() - t0;
    return { detections: dets, display, width: W, height: H, ms };
  }

  function clamp01(v) { return v < 0 ? 0 : v > 1 ? 1 : v; }

  function dispose() {
    if (personModel) personModel.dispose();
    if (faceModel) faceModel.dispose();
    personModel = faceModel = null;
  }

  root.PSPeople = {
    DEFAULTS, setupBackend, init, detect, merge, dispose,
    // exposed for tests / other modules
    _internals: { blazeFaceAnchors, decodeFaces, decodePerson, weightedNms, faceInput, personInput,
      addDistance, distLabel, focalPx, runFace, runPerson },
    get personModel() { return personModel; },
    get backendInfo() { return backendInfo; },
    get faceModel() { return faceModel; },
  };
})(typeof globalThis !== 'undefined' ? globalThis : this);
