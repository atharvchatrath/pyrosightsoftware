/*
 * PyroSight Camera: camera-motion tracking for the "way out" mark (motion.js)
 *
 * Plain script, no dependencies; defines globalThis.PSMotion (also a CommonJS
 * module for node tests). Nothing here touches the DOM.
 *
 * What it does: estimates how the camera has turned (yaw = left/right,
 * pitch = up/down, in degrees) from the video itself, by phase correlation of
 * a small grey copy of each frame against a key frame, and optionally fuses
 * that with the browser's DeviceOrientation angles. It does NOT know where
 * the camera is: walking forward or sideways is not measured. It is a
 * "which way is the camera pointing" tracker, not building navigation.
 *
 * Every match is checked against the whole picture before it is used: the
 * frame is cut into 4 x 4 tiles and each tile votes for the candidate turn
 * (best correlation peaks, or "no turn") its fine detail matches best. A
 * person walking close past a still camera wins only the tiles they cover,
 * so their motion is not taken for a turn. "No turn since the last good pose"
 * is always a candidate and needs fewer tiles; when only part of the picture
 * backs it, the pose is held exactly and the key frame kept, so the moving
 * part cannot pull the pose a fraction of a pixel at a time. A turn found
 * only against the previous frame must not be contradicted by the key frame
 * (someone filling most of the picture moves between two frames). When no
 * candidate is clearly backed, the tracker reports 'weak', then 'lost',
 * instead of drifting. Frames far apart in time (a busy page) need a clearer
 * majority, checked on coarser detail, because the turn in between is unknown
 * and a large turn leaves a pixel or two of perspective error.
 *
 * Conventions
 *   yaw   degrees, clockwise seen from above (turning right = +)
 *   pitch degrees, up = +
 *   Image coordinates are those of the raw (unmirrored) camera frame:
 *   a target to the right of the optical axis appears right of centre.
 *   f (focal length, px) = (max(W, H) / 2) / tan(fovDeg / 2): the assumed
 *   field of view applies to the long side of the frame.
 */
(function (root) {
  'use strict';

  const D2R = Math.PI / 180, R2D = 180 / Math.PI;

  function wrap180(a) {
    a = ((a + 180) % 360 + 360) % 360 - 180;
    return a === -180 ? 180 : a;
  }

  function focalPx(W, H, fovDeg) {
    return (Math.max(W, H) / 2) / Math.tan(fovDeg * D2R / 2);
  }

  // ------------------------------------------------------------------ FFT
  /** In-place radix-2 complex FFT of length n (power of two). */
  function makeFFT(n) {
    const levels = Math.round(Math.log2(n));
    if ((1 << levels) !== n) throw new Error('FFT size must be a power of two');
    const cosT = new Float64Array(n / 2), sinT = new Float64Array(n / 2);
    for (let i = 0; i < n / 2; i++) { cosT[i] = Math.cos(2 * Math.PI * i / n); sinT[i] = Math.sin(2 * Math.PI * i / n); }
    const rev = new Uint32Array(n);
    for (let i = 0; i < n; i++) {
      let r = 0;
      for (let b = 0; b < levels; b++) r |= ((i >>> b) & 1) << (levels - 1 - b);
      rev[i] = r;
    }
    return function fft(re, im, inverse) {
      for (let i = 0; i < n; i++) {
        const j = rev[i];
        if (j > i) {
          let t = re[i]; re[i] = re[j]; re[j] = t;
          t = im[i]; im[i] = im[j]; im[j] = t;
        }
      }
      const sgn = inverse ? 1 : -1;
      for (let size = 2; size <= n; size <<= 1) {
        const half = size >>> 1, step = n / size;
        for (let i = 0; i < n; i += size) {
          for (let j = i, k = 0; j < i + half; j++, k += step) {
            const wr = cosT[k], wi = sgn * sinT[k];
            const l = j + half;
            const tr = re[l] * wr - im[l] * wi;
            const ti = re[l] * wi + im[l] * wr;
            re[l] = re[j] - tr; im[l] = im[j] - ti;
            re[j] += tr; im[j] += ti;
          }
        }
      }
    };
  }

  // ------------------------------------------------------ phase correlation
  /**
   * PhaseCorrelator(n): works on n x n grey images (Float32Array, row-major).
   *   prepare(grey)   -> spectrum {re, im} (mean removed, Hann window)
   *   correlate(A, B) -> {dx, dy, peak, psr}: B's content is A's moved by
   *                      (+dx right, +dy down) pixels, sub-pixel.
   * psr = (peak - mean) / std of the correlation surface outside the peak:
   * about 4-6 for unrelated frames, 10+ for a clear match (see tests).
   */
  function PhaseCorrelator(n, opts) {
    opts = opts || {};
    this.n = n;
    this.fft = makeFFT(n);
    const win = new Float32Array(n * n);
    for (let y = 0; y < n; y++) {
      const wy = 0.5 - 0.5 * Math.cos(2 * Math.PI * (y + 0.5) / n);
      for (let x = 0; x < n; x++) win[y * n + x] = wy * (0.5 - 0.5 * Math.cos(2 * Math.PI * (x + 0.5) / n));
    }
    this.win = win;
    // Gaussian low-pass on the normalised cross-power spectrum: suppresses
    // the noisy high frequencies and makes the peak smooth enough for a
    // parabolic sub-pixel fit.
    const sigma = (opts.lowpass || 0.2) * n;
    const g = new Float32Array(n * n);
    for (let v = 0; v < n; v++) {
      const fv = v < n / 2 ? v : v - n;
      for (let u = 0; u < n; u++) {
        const fu = u < n / 2 ? u : u - n;
        g[v * n + u] = Math.exp(-(fu * fu + fv * fv) / (2 * sigma * sigma));
      }
    }
    g[0] = 0;   // DC carries no shift information
    this.lowpass = g;
    this.tmpRe = new Float32Array(n);
    this.tmpIm = new Float32Array(n);
    this.qRe = new Float32Array(n * n);
    this.qIm = new Float32Array(n * n);
  }

  PhaseCorrelator.prototype.fft2 = function (re, im, inverse) {
    const n = this.n, fft = this.fft, tr = this.tmpRe, ti = this.tmpIm;
    for (let y = 0; y < n; y++) fft(re.subarray(y * n, y * n + n), im.subarray(y * n, y * n + n), inverse);
    for (let x = 0; x < n; x++) {
      for (let y = 0; y < n; y++) { tr[y] = re[y * n + x]; ti[y] = im[y * n + x]; }
      fft(tr, ti, inverse);
      for (let y = 0; y < n; y++) { re[y * n + x] = tr[y]; im[y * n + x] = ti[y]; }
    }
  };

  PhaseCorrelator.prototype.prepare = function (grey) {
    const n = this.n, N = n * n;
    let mean = 0, sq = 0;
    for (let i = 0; i < N; i++) mean += grey[i];
    mean /= N;
    const re = new Float32Array(N), im = new Float32Array(N);
    for (let i = 0; i < N; i++) { const v = grey[i] - mean; re[i] = v * this.win[i]; sq += v * v; }
    this.fft2(re, im, false);
    // hp: 3 x 3 minus 9 x 9 (fine enough to tell a few pixels of misalignment apart, smooth enough
    // to survive resampling); hp2: 5 x 5 minus 17 x 17, for frames far apart in time, where a large
    // turn leaves a pixel or two of perspective error that the fine band does not survive
    return { re, im, std: Math.sqrt(sq / N), hp: bandPass(grey, n, 1, 4), hp2: bandPass(grey, n, 2, 8) };
  };

  /**
   * correlate(A, B, opts): opts.peaks = k also returns up to k local maxima of the correlation
   * surface (best first, at least 5 px apart) as .peaks [{dx, dy, psr}]; opts.at = [{dx, dy}]
   * also returns the correlation at those shifts (same PSR scale) as .at.
   */
  PhaseCorrelator.prototype.correlate = function (A, B, opts) {
    const n = this.n, N = n * n, qr = this.qRe, qi = this.qIm, g = this.lowpass;
    for (let i = 0; i < N; i++) {
      // B * conj(A)
      const r = B.re[i] * A.re[i] + B.im[i] * A.im[i];
      const m = B.im[i] * A.re[i] - B.re[i] * A.im[i];
      const mag = Math.sqrt(r * r + m * m) + 1e-9;
      qr[i] = g[i] * r / mag; qi[i] = g[i] * m / mag;
    }
    this.fft2(qr, qi, true);
    let best = -Infinity, bi = 0, sum = 0, sum2 = 0;
    for (let i = 0; i < N; i++) {
      const v = qr[i];
      sum += v; sum2 += v * v;
      if (v > best) { best = v; bi = i; }
    }
    const px = bi % n, py = (bi - px) / n;
    const at = (x, y) => qr[((y + n) % n) * n + ((x + n) % n)];
    // PSR: statistics outside an 11 x 11 window around the peak
    let ws = 0, ws2 = 0, wc = 0;
    for (let dy = -5; dy <= 5; dy++) for (let dx = -5; dx <= 5; dx++) { const v = at(px + dx, py + dy); ws += v; ws2 += v * v; wc++; }
    const cnt = N - wc, mean = (sum - ws) / cnt;
    const std = Math.sqrt(Math.max(1e-20, (sum2 - ws2) / cnt - mean * mean));
    const sub = (a, b, c) => { const d = a - 2 * b + c; return d < 0 ? 0.5 * (a - c) / d : 0; };
    let dx = px + sub(at(px - 1, py), best, at(px + 1, py));
    let dy = py + sub(at(px, py - 1), best, at(px, py + 1));
    if (dx > n / 2) dx -= n;
    if (dy > n / 2) dy -= n;
    // sum of the low-pass filter / N is the peak of a perfect match
    if (!this.peakNorm) { let s = 0; for (let i = 0; i < N; i++) s += g[i]; this.peakNorm = s / N; }
    const out = { dx, dy, peak: best / N / this.peakNorm, psr: (best - mean) / std };
    const wrapd = (v) => (v > n / 2 ? v - n : v);
    if (opts && opts.peaks > 1) {
      // further local maxima (3 x 3), at least 5 px (wrapped) from the ones already taken
      const taken = [[px, py]], peaks = [out];
      const far = (x, y) => taken.every(([tx, ty]) => {
        const ddx = Math.abs(x - tx), ddy = Math.abs(y - ty);
        return Math.min(ddx, n - ddx) > 4 || Math.min(ddy, n - ddy) > 4;
      });
      while (peaks.length < opts.peaks) {
        let b2 = -Infinity, bi2 = -1;
        for (let i = 0; i < N; i++) {
          const v = qr[i];
          if (v <= b2) continue;
          const x = i % n, y = (i - x) / n;
          if (v < at(x - 1, y) || v < at(x + 1, y) || v < at(x, y - 1) || v < at(x, y + 1)) continue;
          if (!far(x, y)) continue;
          b2 = v; bi2 = i;
        }
        if (bi2 < 0) break;
        const x = bi2 % n, y = (bi2 - x) / n;
        taken.push([x, y]);
        peaks.push({ dx: wrapd(x + sub(at(x - 1, y), b2, at(x + 1, y))), dy: wrapd(y + sub(at(x, y - 1), b2, at(x, y + 1))),
          psr: (b2 - mean) / std });
      }
      out.peaks = peaks;
    }
    if (opts && opts.at) {
      out.at = opts.at.map((p) => {
        const x = Math.round(p.dx), y = Math.round(p.dy);
        let v = -Infinity;
        for (let ddy = -1; ddy <= 1; ddy++) for (let ddx = -1; ddx <= 1; ddx++) v = Math.max(v, at(x + ddx, y + ddy));
        return { dx: p.dx, dy: p.dy, psr: (v - mean) / std };
      });
    }
    return out;
  };

  function boxBlur(grey, n, r) {
    const N = n * n, I = new Float64Array((n + 1) * (n + 1)), out = new Float32Array(N);
    for (let y = 0; y < n; y++) {
      let row = 0;
      for (let x = 0; x < n; x++) { row += grey[y * n + x]; I[(y + 1) * (n + 1) + x + 1] = I[y * (n + 1) + x + 1] + row; }
    }
    for (let y = 0; y < n; y++) {
      const y0 = Math.max(0, y - r), y1 = Math.min(n, y + r + 1);
      for (let x = 0; x < n; x++) {
        const x0 = Math.max(0, x - r), x1 = Math.min(n, x + r + 1);
        out[y * n + x] = (I[y1 * (n + 1) + x1] - I[y0 * (n + 1) + x1] - I[y1 * (n + 1) + x0] + I[y0 * (n + 1) + x0]) / ((y1 - y0) * (x1 - x0));
      }
    }
    return out;
  }
  /** Band-pass detail: (2 r1 + 1) box blur minus (2 r2 + 1) box blur. */
  function bandPass(grey, n, r1, r2) {
    const a = boxBlur(grey, n, r1), b = boxBlur(grey, n, r2);
    for (let i = 0; i < n * n; i++) a[i] -= b[i];
    return a;
  }

  /**
   * Per-tile agreement with a candidate camera turn: the current frame is cut into 4 x 4 tiles;
   * each tile's fine detail is compared (normalised cross-correlation) with the reference frame at
   * the place the turn (dYaw, dPitch, degrees, current - reference) predicts for that tile with the
   * pinhole model (edge tiles move more than the centre). Returns a Float32Array of 16 NCC values,
   * NaN where the tile is too plain or mostly outside the reference frame.
   */
  function tileNcc(cur, ref, dYaw, dPitch, W, H, f, n, o) {
    const T = n / 4, step = 2, sx = W / n, sy = H / n;
    const ca = Math.cos(dYaw * D2R), sa = Math.sin(dYaw * D2R), cp = Math.cos(dPitch * D2R), sp = Math.sin(dPitch * D2R);
    const out = new Float32Array(16).fill(NaN);
    for (let tj = 0; tj < 4; tj++) {
      for (let ti = 0; ti < 4; ti++) {
        const u1 = (ti + 0.5) * T, v1 = (tj + 0.5) * T;
        // the tile centre's viewing direction (x right, y down, z forward), turned into the reference
        // camera: d = Ry(dYaw) Rx(dPitch) d1, then projected (pinhole)
        const x1 = (u1 - n / 2) * sx, y1 = (v1 - n / 2) * sy;
        const y2 = y1 * cp - f * sp, z2 = y1 * sp + f * cp;
        const x3 = x1 * ca + z2 * sa, z3 = -x1 * sa + z2 * ca;
        if (z3 <= 0.2 * f) continue;
        const ox = f * x3 / z3 / sx + n / 2 - u1, oy = f * y2 / z3 / sy + n / 2 - v1;
        const ix = Math.floor(ox), iy = Math.floor(oy), fx = ox - ix, fy = oy - iy;
        const w00 = (1 - fx) * (1 - fy), w10 = fx * (1 - fy), w01 = (1 - fx) * fy, w11 = fx * fy;
        let k = 0, all = 0, cs = 0, cs2 = 0, rs = 0, rs2 = 0, cr = 0;
        for (let y = tj * T; y < (tj + 1) * T; y += step) {
          const ry = y + iy;
          for (let x = ti * T; x < (ti + 1) * T; x += step) {
            all++;
            const rx = x + ix;
            if (ry < 0 || ry >= n - 1 || rx < 0 || rx >= n - 1) continue;
            const q = ry * n + rx;
            const r = w00 * ref[q] + w10 * ref[q + 1] + w01 * ref[q + n] + w11 * ref[q + n + 1];
            const c = cur[y * n + x];
            cs += c; cs2 += c * c; rs += r; rs2 += r * r; cr += c * r; k++;
          }
        }
        if (k < all * 0.6) continue;
        const cm = cs / k, rm = rs / k, cv = cs2 / k - cm * cm, rv = rs2 / k - rm * rm;
        if (cv < o.texMin * o.texMin || rv < o.texMin * o.texMin) continue;
        out[tj * 4 + ti] = (cr / k - cm * rm) / Math.sqrt(cv * rv);
      }
    }
    return out;
  }

  /**
   * Grey n x n copy of an RGBA frame that was drawn (stretched) into an
   * n x n canvas. Returns Float32Array (0..255).
   */
  function greyFromRGBA(rgba, n, out) {
    out = out || new Float32Array(n * n);
    for (let i = 0, j = 0; i < n * n; i++, j += 4) out[i] = 0.299 * rgba[j] + 0.587 * rgba[j + 1] + 0.114 * rgba[j + 2];
    return out;
  }

  // ----------------------------------------------------- device orientation
  /**
   * W3C DeviceOrientation (alpha, beta, gamma in degrees; intrinsic Z-X'-Y'')
   * -> {yaw, pitch} of the camera's viewing direction. Back camera looks
   * along the device's -Z axis, front camera along +Z. alpha is often
   * relative to an arbitrary start heading: only differences are used.
   */
  function orientationToYawPitch(alpha, beta, gamma, front) {
    const a = alpha * D2R, b = beta * D2R, g = gamma * D2R;
    const ca = Math.cos(a), sa = Math.sin(a), cb = Math.cos(b), sb = Math.sin(b), cg = Math.cos(g), sg = Math.sin(g);
    // third column of R = Rz(a) Rx(b) Ry(g)
    const r02 = ca * sg + sa * sb * cg;
    const r12 = sa * sg - ca * sb * cg;
    const r22 = cb * cg;
    const s = front ? 1 : -1;       // direction = R * (0, 0, s)
    const dx = s * r02, dy = s * r12, dz = s * r22;
    return { yaw: Math.atan2(dx, dy) * R2D, pitch: Math.asin(Math.max(-1, Math.min(1, dz))) * R2D };
  }

  // ------------------------------------------------------------ tracker
  const TRACK_DEFAULTS = {
    n: 128,             // grey frame size for phase correlation
    fovDeg: 65,         // assumed field of view across the long side of the frame
    psrOk: 10,          // accept a match above this peak-to-sidelobe ratio
    psrRekey: 14,       // below this (but ok) refresh the key frame
    rekeyFrac: 0.125,   // refresh the key frame when the shift exceeds n * rekeyFrac
    lostMs: 1500,       // no acceptable match for this long -> 'lost'
    sensorGain: 0.03,   // per update: pull of the fused pose toward the motion sensor
    sensorStaleMs: 1000,
    // A match must also agree with most of the picture (see tileNcc / verified): a large object
    // moving past a still camera is then not taken for a camera turn.
    texMin: 3,          // grey-level std (band-passed) below which a 32 x 32 tile is too plain to vote
    nccMin: 0.5,        // a tile votes for the candidate turn it matches best, if its NCC reaches this
    supFrac: 0.5,       // the winner needs this share of the voting tiles,
    contraRatio: 2,     //   and this many times the votes of any other candidate
    rekeySupFrac: 0.6,  // a turn only part of the picture agrees with (something moved) starts a new key
                        // frame; "no turn" backed by less than this share holds the pose exactly
    candPsrMin: 5,      // weakest further correlation peak still tried as a candidate
    stillSupFrac: 0.35, // "no turn since the last good pose" needs only this share (and 3 tiles)
    gapMs: 300,         // frames further apart than this (page busy with a detector): the turn in
    gapSupFrac: 0.6,    //   between is unknown, so a match needs this share of the tiles,
    gapPlainPsr: 20,    //   or, in a picture too plain to check, this correlation
    gapPlainShift: 8,   //   and at most this shift (px of the 128-px frame)
  };

  /**
   * MotionTracker: feed it a grey n x n frame per video frame (update) and,
   * if available, DeviceOrientation readings (sensor). Read .pose.
   *   state: 'idle' | 'ok' | 'weak' | 'lost'
   *   source: 'camera' | 'camera+sensor' | 'sensor'
   */
  function MotionTracker(opts) {
    this.o = Object.assign({}, TRACK_DEFAULTS, opts || {});
    this.pc = new PhaseCorrelator(this.o.n);
    this.hasMark = false;    // set by the page: while a mark exists, a lost tracker holds its key
    this.reset();
  }

  MotionTracker.prototype.reset = function () {
    this.pose = { yaw: 0, pitch: 0 };
    this.vis = { yaw: 0, pitch: 0 };
    this.key = null; this.keyPose = null;
    this.prev = null; this.prevVis = null; this.prevValid = false;
    this.state = 'idle';
    this.badSince = 0;
    this.last = null;
    this.sensorOffset = null;
    this.prevSensor = null;
    this.updates = 0; this.rekeys = 0;
    this.lastT = null; this.lastCorr = null; this.rejected = 0;   // rejected: matches refused by the check
  };

  /**
   * Match F against a reference frame (key or previous) and check the match against the whole
   * picture. Candidates: the best correlation peaks and "no turn since the last frame"; the one
   * most of the picture agrees with wins. Returns the match ({dx, dy, psr, sup, voters, alt}) or
   * null (no candidate is clearly supported: something large moved, the view changed, or after a
   * long gap between frames the turn cannot be confirmed).
   */
  MotionTracker.prototype.verified = function (ref, refPose, F, W, H, f, gap) {
    const o = this.o, n = o.n, sx = W / n, sy = H / n;
    // always a candidate: "no turn since the last good pose" (held while frames do not match), so
    // the still background can outvote someone walking past even after a frame that did not match
    const pv = this.vis;
    const prior = pv ? {
      dx: -Math.tan((pv.yaw - refPose.yaw) * D2R) * f / sx,
      dy: Math.tan((pv.pitch - refPose.pitch) * D2R) * f / sy } : null;
    const m = this.pc.correlate(ref, F, { peaks: 3, at: prior ? [prior] : null });
    this.lastCorr = m;
    const cands = m.peaks.filter((c, i) => i === 0 || c.psr >= o.candPsrMin);
    if (prior && cands.every((c) => Math.hypot(c.dx - prior.dx, c.dy - prior.dy) > 1.5)) {
      cands.push({ dx: prior.dx, dy: prior.dy, psr: m.at[0].psr, prior: true });
    }
    // each tile votes for the candidate turn it agrees with best (if it agrees well enough)
    const scores = cands.map((c) => tileNcc(gap ? F.hp2 : F.hp, gap ? ref.hp2 : ref.hp, -Math.atan(c.dx * sx / f) * R2D, Math.atan(c.dy * sy / f) * R2D, W, H, f, n, o));
    cands.forEach((c) => { c.sup = 0; });
    let voters = 0;
    for (let i = 0; i < 16; i++) {
      let b = -1, bv = -Infinity;
      for (let k = 0; k < cands.length; k++) { const v = scores[k][i]; if (v === v && v > bv) { bv = v; b = k; } }
      if (b < 0) continue;
      voters++;
      if (bv >= o.nccMin) cands[b].sup++;
    }
    // ties go to "no turn since the last frame", then to the stronger correlation peak
    let best = null;
    for (const c of cands) if (!best || c.sup > best.sup || (c.sup === best.sup && (c.prior || (!best.prior && c.psr > best.psr)))) best = c;
    const main = m.peaks[0];
    const second = Math.max(0, ...cands.filter((c) => c !== best).map((c) => c.sup));
    let ok, hold = false;
    if (voters >= 3) {
      const need = (frac) => Math.max(2, Math.ceil(frac * voters - 1e-9));
      const clear = best.sup >= o.contraRatio * second && (best !== main || main.psr >= o.psrOk);
      ok = clear && best.sup >= need(gap ? o.gapSupFrac : o.supFrac);
      // "no turn" needs fewer tiles (the rest may be covered by someone moving, in this frame or in
      // the key frame). Backed by only part of the picture, it keeps the last good pose exactly:
      // the moving part pulls the correlation peak by a fraction of a pixel, and those fractions
      // would add up from one key frame to the next
      const still = prior && Math.hypot(best.dx - prior.dx, best.dy - prior.dy) <= 1.5;
      if (!ok && still && clear && best.sup >= Math.max(3, need(o.stillSupFrac))) ok = true;
      hold = ok && still && best.sup < o.rekeySupFrac * voters;
    } else if (!gap) {
      ok = best === main && main.psr >= o.psrOk;   // too plain to check: the correlation alone, as before
    } else {
      ok = best === main && main.psr >= o.gapPlainPsr && Math.abs(main.dx) <= o.gapPlainShift && Math.abs(main.dy) <= o.gapPlainShift;
    }
    if (!ok) { this.rejected++; return null; }
    const at = hold ? prior : best;
    return { dx: at.dx, dy: at.dy, psr: best.psr, peak: m.peak, sup: best.sup, voters, alt: best !== main, hold };
  };

  /**
   * A turn found only against the previous frame, small enough that the key frame still covers
   * it, must not be contradicted by the key frame. When the key frame's tiles favour "no turn
   * since the last good pose" over it, the previous frame was most likely matched on something
   * that moved between the two frames (someone walking past, filling most of the picture), and
   * the turn is refused.
   */
  MotionTracker.prototype.keyBacks = function (F, pose, W, H, f) {
    const o = this.o, n = o.n, kp = this.keyPose;
    const dYaw = wrap180(pose.yaw - kp.yaw), dPitch = pose.pitch - kp.pitch;
    const lim = Math.atan(n * o.rekeyFrac * Math.min(W, H) / n / f) * R2D;
    if (Math.abs(dYaw) > lim || Math.abs(dPitch) > lim) return true;   // beyond the key frame's reach
    const a = tileNcc(F.hp, this.key.hp, dYaw, dPitch, W, H, f, n, o);
    const b = tileNcc(F.hp, this.key.hp, wrap180(this.vis.yaw - kp.yaw), this.vis.pitch - kp.pitch, W, H, f, n, o);
    let forTurn = 0, forStill = 0;
    for (let i = 0; i < 16; i++) {
      const x = a[i] === a[i] ? a[i] : -Infinity, y = b[i] === b[i] ? b[i] : -Infinity;
      if (Math.max(x, y) < o.nccMin) continue;
      if (x >= y) forTurn++; else forStill++;
    }
    return forTurn >= forStill;
  };

  MotionTracker.prototype.sensor = function (yaw, pitch, t) {
    if (!(isFinite(yaw) && isFinite(pitch))) return;
    const s = this.sensorReading;
    this.sensorMoved = this.sensorMoved || (s && (Math.abs(wrap180(yaw - s.yaw)) > 0.01 || Math.abs(pitch - s.pitch) > 0.01));
    this.sensorReading = { yaw, pitch, t };
  };

  /**
   * The picture no longer matches the key frame (just after a fast turn or a cut) but the viewer
   * marks a direction now: that direction is defined by the current view, so take the next frame
   * as the key frame at the current pose instead of waiting for the old view to come back.
   */
  MotionTracker.prototype.reanchor = function () {
    if (this.state === 'ok' || this.state === 'idle') return false;
    this.vis = Object.assign({}, this.pose);
    this.key = null; this.keyPose = null;
    this.prev = null; this.prevValid = false;
    this.badSince = 0;
    this.state = 'ok';
    return true;
  };

  MotionTracker.prototype.sensorActive = function (t) {
    const s = this.sensorReading;
    return !!(s && this.sensorMoved && t - s.t < this.o.sensorStaleMs);
  };

  MotionTracker.prototype.source = function (t) {
    const sens = this.sensorActive(t);
    if (this.state === 'lost' && sens) return 'sensor';
    return sens ? 'camera+sensor' : 'camera';
  };

  /**
   * update(grey, W, H, t): grey = n x n Float32Array of the current frame
   * (stretched), W x H = real frame size in px, t = ms timestamp.
   * Returns {state, pose, match, valid}. state 'weak' = this frame could not be matched (the
   * pose is the last good one, unsure); 'lost' = no match for lostMs.
   */
  MotionTracker.prototype.update = function (grey, W, H, t) {
    const o = this.o, n = o.n;
    const f = focalPx(W, H, o.fovDeg), sx = W / n, sy = H / n;
    const F = this.pc.prepare(grey);
    this.updates++;
    let valid = false, match = null;
    const toPose = (base, m) => ({ yaw: base.yaw - Math.atan(m.dx * sx / f) * R2D,
      pitch: base.pitch + Math.atan(m.dy * sy / f) * R2D });
    const gap = this.lastT !== null && t - this.lastT > o.gapMs;
    this.lastT = t;
    if (!this.key || F.std < 1.0) {
      if (!this.key && F.std >= 1.0) {
        this.key = F; this.keyPose = Object.assign({}, this.vis);
        valid = true;
      }
    } else {
      match = this.verified(this.key, this.keyPose, F, W, H, f, gap);
      if (match) {
        this.vis = toPose(this.keyPose, match);
        valid = true;
      } else if (this.prev && this.prevValid && this.state === 'ok' && !gap) {
        // key frame too different; consecutive frames may still match
        const m2 = this.verified(this.prev, this.prevVis, F, W, H, f, gap);
        if (m2 && this.keyBacks(F, toPose(this.prevVis, m2), W, H, f)) { this.vis = toPose(this.prevVis, m2); valid = true; match = m2; this.key = null; }
      }
      if (!match) match = this.lastCorr;
      // (a held pose is not a measurement: keep the key frame it was held against)
      if (valid && !match.hold && (!this.key || Math.abs(match.dx) > n * o.rekeyFrac || Math.abs(match.dy) > n * o.rekeyFrac ||
                    match.psr < o.psrRekey || (match.voters >= 3 && match.sup < o.rekeySupFrac * match.voters))) {
        this.key = F; this.keyPose = Object.assign({}, this.vis); this.rekeys++;
      }
    }

    const sens = this.sensorActive(t);
    if (valid) {
      this.badSince = 0;
      this.state = 'ok';
    } else if (this.key) {
      if (!this.badSince) this.badSince = t;
      this.state = t - this.badSince > o.lostMs ? 'lost' : 'weak';
    }

    // ---- fuse into this.pose
    const sr = this.sensorReading;
    if (sens) {
      if (this.sensorOffset === null) this.sensorOffset = { yaw: wrap180(this.pose.yaw - sr.yaw), pitch: this.pose.pitch - sr.pitch };
      let dYaw = 0, dPitch = 0;
      if (valid && this.prevValid) {
        dYaw = wrap180(this.vis.yaw - this.prevVis.yaw); dPitch = this.vis.pitch - this.prevVis.pitch;
      } else if (this.prevSensor) {
        dYaw = wrap180(sr.yaw - this.prevSensor.yaw); dPitch = sr.pitch - this.prevSensor.pitch;
      }
      this.pose.yaw = wrap180(this.pose.yaw + dYaw);
      this.pose.pitch += dPitch;
      const k = o.sensorGain;
      this.pose.yaw = wrap180(this.pose.yaw + k * wrap180(sr.yaw + this.sensorOffset.yaw - this.pose.yaw));
      this.pose.pitch += k * (sr.pitch + this.sensorOffset.pitch - this.pose.pitch);
      this.prevSensor = { yaw: sr.yaw, pitch: sr.pitch };
      if (this.state === 'lost') {   // re-anchor the camera tracker on the sensor-carried pose
        this.vis = Object.assign({}, this.pose);
        this.key = F; this.keyPose = Object.assign({}, this.vis);
        this.state = 'ok'; this.badSince = 0; valid = true;
      }
    } else {
      this.prevSensor = null;
      if (this.state === 'lost' && !this.hasMark) {
        // nothing depends on the old direction: start again from here
        this.key = F; this.keyPose = Object.assign({}, this.vis);
        this.state = 'ok'; this.badSince = 0; valid = true;
      }
      if (valid) { this.pose.yaw = wrap180(this.vis.yaw); this.pose.pitch = this.vis.pitch; }
    }
    this.prev = F; this.prevVis = Object.assign({}, this.vis); this.prevValid = valid;
    this.last = match;
    return { state: this.state, pose: this.pose, match, valid };
  };

  // --------------------------------------------------------- geometry
  /**
   * Where a marked direction (yaw, pitch, angular size) falls in the frame.
   * Returns {inView, rel: {yaw, pitch}, box: {x, y, w, h} normalised raw-frame
   * coordinates, or null when it is behind or far outside}.
   */
  function project(mark, pose, W, H, fovDeg) {
    const f = focalPx(W, H, fovDeg);
    const ry = wrap180(mark.yaw - pose.yaw), rp = mark.pitch - pose.pitch;
    const out = { rel: { yaw: ry, pitch: rp }, inView: false, box: null };
    if (Math.abs(ry) >= 80 || Math.abs(rp) >= 80) return out;
    const X = (a) => W / 2 + f * Math.tan(a * D2R);
    const Y = (a) => H / 2 - f * Math.tan(a * D2R) / Math.cos(ry * D2R);
    const hw = (mark.w || 14) / 2, hh = (mark.h || 28) / 2;
    const x0 = X(Math.max(-85, ry - hw)), x1 = X(Math.min(85, ry + hw));
    const y0 = Y(Math.min(85, rp + hh)), y1 = Y(Math.max(-85, rp - hh));
    const cx = X(ry), cy = Y(rp);
    out.box = { x: x0 / W, y: y0 / H, w: (x1 - x0) / W, h: (y1 - y0) / H };
    out.centre = { x: cx / W, y: cy / H };
    out.inView = cx >= 0 && cx <= W && cy >= 0 && cy <= H;
    return out;
  }

  /** Normalised raw-frame point -> direction {yaw, pitch} for the current pose. */
  function pointToDirection(nx, ny, pose, W, H, fovDeg) {
    const f = focalPx(W, H, fovDeg);
    const yaw = wrap180(pose.yaw + Math.atan((nx - 0.5) * W / f) * R2D);
    const pitch = pose.pitch - Math.atan((ny - 0.5) * H / f) * R2D;
    return { yaw, pitch };
  }

  /** Angular size (degrees) of a normalised box in a W x H frame. */
  function boxAngles(b, W, H, fovDeg) {
    const f = focalPx(W, H, fovDeg);
    const a = (v) => Math.atan(v / f) * R2D;
    return { w: a((b.x + b.w - 0.5) * W) - a((b.x - 0.5) * W), h: a((0.5 - b.y) * H) - a((0.5 - b.y - b.h) * H) };
  }

  // --------------------------------------------------------- phrases
  // Same sectors and words as core/src/ps_alerts.c (ps_direction_phrase).
  const DIR_PHRASES = ['ahead.', 'ahead, to your left.', 'to your left.', 'behind you, to the left.',
    'behind you.', 'behind you, to the right.', 'to your right.', 'ahead, to your right.'];
  /** b: bearing in degrees, positive = to the LEFT (device convention). */
  function directionSector(b) {
    let s = Math.round(b / 45);
    return ((s % 8) + 8) % 8;
  }
  function directionPhrase(b) { return DIR_PHRASES[directionSector(b)]; }

  /**
   * Person / fire call-out side, like ps_alerts_update_detections: cx = box
   * centre in DISPLAY coordinates (0..1, as the viewer sees it).
   */
  function sidePhrase(cx, fovDeg) {
    const rel = (0.5 - cx) * fovDeg;     // + = left of centre
    return Math.abs(rel) < fovDeg / 6 ? 'ahead.' : rel > 0 ? 'ahead, to your left.' : 'ahead, to your right.';
  }

  const api = {
    wrap180, focalPx, makeFFT, PhaseCorrelator, greyFromRGBA, orientationToYawPitch,
    MotionTracker, TRACK_DEFAULTS, project, pointToDirection, boxAngles,
    DIR_PHRASES, directionSector, directionPhrase, sidePhrase,
  };
  root.PSMotion = api;
  if (typeof module === 'object' && module.exports) module.exports = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
