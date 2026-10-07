/*
 * PyroSight demo navigation: step detector for phone accelerometers.
 *
 * Input: accelerationIncludingGravity samples (m/s^2, any rate 20-200 Hz,
 * irregular is fine). The device has the BNO085's own step detector; a phone
 * page has to find steps itself:
 *   1. magnitude |a| (orientation-free), minus a slow gravity estimate (1 s);
 *   2. low-pass (two 3.5 Hz one-pole stages) -> band-passed vertical bounce;
 *   3. a peak is a rise above +minPeak after the signal went below -valley
 *      (hysteresis: one peak per bounce, double heel-strike bumps ignored);
 *   4. a step needs a regular rhythm: peaks 0.25-1.4 s apart, consecutive
 *      intervals within a factor of 2, and three peaks in a row before the
 *      first steps are released (the held ones are then emitted together);
 *   5. shakes are rejected: a peak sooner than 0.25 s after the previous one,
 *      a peak larger than maxPeak, or more energy above the step band (> 3.5 Hz:
 *      shaking, knocks) than in it cancels the bout and mutes steps for 0.6 s.
 * It also returns a linear-acceleration magnitude for the device's
 * crawl / untracked-motion path (ps_nav_on_linear_accel).
 */
(function (root) {
  'use strict';
  const PSNav = root.PSNav = root.PSNav || {};

  const DEFAULTS = {
    gravityTauS: 1.0,
    lowPassHz: 3.5,
    minPeak: 0.7,       // m/s^2 above the gravity estimate (band-passed)
    valley: 0.3,        // must dip below -valley between peaks
    maxPeak: 12.0,      // larger bounces are shakes / knocks, not steps
    minIntervalS: 0.25,
    maxIntervalS: 1.4,  // slow, careful walking in smoke: down to ~0.7 steps per second
    maxRatio: 2.0,      // consecutive intervals within this factor
    confirmPeaks: 3,
    shakeHoldS: 0.6,
    shakeHz: 4.5,       // "above the step band" high-pass corner
    shakeRatio: 1.5,    // energy above / in the step band that means shaking
    shakeMinEnergy: 1.5, // (m/s^2)^2
  };

  function StepDetector(opts) {
    this.o = Object.assign({}, DEFAULTS, opts || {});
    this.reset();
  }

  StepDetector.prototype.reset = function () {
    this.tPrev = null;
    this.g = null;
    this.lp1 = 0; this.lp2 = 0;
    this.armed = true;
    this.inPeak = false; this.peakV = 0; this.peakT = 0;
    this.lastPeakT = -1e9;
    this.bout = { n: 0, pending: [], lastDt: 0 };
    this.quietUntil = -1e9;
    this.linear = 0;
    this.ehf = 0; this.elf = 0;
    this.h1 = 0; this.h2 = 0; this.hpPrev = 0; this.h1Prev = 0;
    this.steps = 0;
    this.rejected = { shake: 0, irregular: 0, big: 0, hf: 0 };
  };

  /**
   * One sample. t in milliseconds. Returns an array of step times (ms) found
   * by this sample (usually empty; 1 per step; 3 when a walking bout is
   * confirmed). this.linear holds the linear-acceleration magnitude estimate.
   */
  StepDetector.prototype.push = function (ax, ay, az, t) {
    const o = this.o;
    if (!(isFinite(ax) && isFinite(ay) && isFinite(az))) return [];
    const m = Math.sqrt(ax * ax + ay * ay + az * az);
    if (this.tPrev === null) { this.tPrev = t; this.g = m; return []; }
    let dt = (t - this.tPrev) / 1000;
    this.tPrev = t;
    if (!(dt > 0)) return [];
    if (dt > 0.5) { this.g = m; this.lp1 = this.lp2 = 0; this.h1 = this.h2 = this.hpPrev = this.h1Prev = 0; this.armed = true; this.inPeak = false; return []; }
    this.g += (m - this.g) * (1 - Math.exp(-dt / o.gravityTauS));
    const hp = m - this.g;
    const k = 1 - Math.exp(-dt * 2 * Math.PI * o.lowPassHz);
    this.lp1 += (hp - this.lp1) * k;
    this.lp2 += (this.lp1 - this.lp2) * k;
    const s = this.lp2;
    this.linear += (Math.abs(hp) - this.linear) * (1 - Math.exp(-dt / 0.2));

    // Shake / knock detector: energy above the step band vs in it. Walking puts
    // most of its energy in the step band; shaking (4-10 Hz) and knocks do not.
    // (two one-pole high-pass stages at shakeHz: a plain "input minus low-pass"
    // residual would keep the step fundamental because of the low-pass phase lag)
    const rc = 1 / (2 * Math.PI * o.shakeHz), al = rc / (rc + dt);
    const h1 = al * (this.h1 + hp - this.hpPrev);
    const h2 = al * (this.h2 + h1 - this.h1Prev);
    this.hpPrev = hp; this.h1Prev = h1; this.h1 = h1; this.h2 = h2;
    const ke = 1 - Math.exp(-dt / 0.4);
    this.ehf += (h2 * h2 - this.ehf) * ke;
    this.elf += (s * s - this.elf) * ke;
    if (this.ehf > o.shakeMinEnergy && this.ehf > o.shakeRatio * this.elf) {
      if (t >= this.quietUntil) { this.rejected.hf++; this._resetBout(); }
      this.quietUntil = Math.max(this.quietUntil, t + o.shakeHoldS * 1000);
    }

    const out = [];
    if (this.inPeak) {
      if (s > this.peakV) { this.peakV = s; this.peakT = t; }
      if (s < this.peakV * 0.5 || s < o.minPeak * 0.5) {
        this.inPeak = false;
        this.armed = false;
        this._peak(this.peakT, this.peakV, out);
      }
    } else if (this.armed && s > o.minPeak) {
      this.inPeak = true; this.peakV = s; this.peakT = t;
    }
    if (!this.armed && !this.inPeak && s < -o.valley) this.armed = true;
    return out;
  };

  StepDetector.prototype._resetBout = function () { this.bout = { n: 0, pending: [], lastDt: 0 }; };

  StepDetector.prototype._peak = function (tp, v, out) {
    const o = this.o;
    const dt = (tp - this.lastPeakT) / 1000;
    this.lastPeakT = tp;
    if (v > o.maxPeak) { this.rejected.big++; this._resetBout(); this.quietUntil = tp + o.shakeHoldS * 1000; return; }
    if (dt < o.minIntervalS) { this.rejected.shake++; this._resetBout(); this.quietUntil = tp + o.shakeHoldS * 1000; return; }
    if (tp < this.quietUntil) { this._resetBout(); return; }
    const b = this.bout;
    if (b.n === 0 || dt > o.maxIntervalS) { this._resetBout(); this.bout.n = 1; this.bout.pending = [tp]; return; }
    if (b.n >= 2 && b.lastDt > 0 && (dt / b.lastDt > o.maxRatio || b.lastDt / dt > o.maxRatio)) {
      this.rejected.irregular++;
      this._resetBout(); this.bout.n = 1; this.bout.pending = [tp];
      return;
    }
    b.n++; b.lastDt = dt;
    if (b.n < o.confirmPeaks) { b.pending.push(tp); return; }
    if (b.n === o.confirmPeaks) { for (const p of b.pending) out.push(p); b.pending = []; }
    out.push(tp);
    this.steps += out.length;
  };

  StepDetector.DEFAULTS = DEFAULTS;
  PSNav.StepDetector = StepDetector;
})(typeof globalThis !== 'undefined' ? globalThis : this);
