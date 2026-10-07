/*
 * PyroSight demo navigation: the demo walker (a simulated firefighter with a
 * simulated BNO085), for laptops, sandboxed pages and anyone not walking.
 *
 * Ported from the device simulator (sim/sim_world.c): the same floor plan
 * (door at the origin, corridor A east, corridor B north, a room at the end),
 * the same inbound route as its "corridor" scenario (~35 m, five turns), the
 * same walk-out policy (follow ONLY the arrow; follow the hose line if the
 * device says its estimate is unreliable) and the same IMU error model:
 *   - stride: true stride = calibrated x (1 + bias) x (1 + 5% noise), bias per seed;
 *   - heading: game-rotation-vector yaw with gyro drift (deg/s per seed) +
 *     random walk, 1% scale-factor error on turns, 0.003 rad noise, and an
 *     arbitrary yaw reference;
 *   - 3% of steps missed by the step detector; shuffling against a wall is no step;
 *   - linear acceleration ~2.2 m/s^2 while walking, ~0.2 standing.
 * World frame: metres, +x into the building through the door, +y to the
 * left (north on the map), yaw counter-clockwise.
 */
(function (root) {
  'use strict';
  const PSNav = root.PSNav = root.PSNav || {};
  const D2R = Math.PI / 180;

  const PLAN = {
    walls: [
      [0, -1, 15, -1], [0, 1, 13, 1],          // corridor A
      [15, -1, 15, 12], [13, 1, 13, 12],       // corridor B
      [9, 12, 13, 12], [15, 12, 19, 12],       // room front wall with doorway
      [19, 12, 19, 20], [9, 20, 19, 20], [9, 12, 9, 20],
      [0, 1, 0, 4], [0, -1, 0, -4],            // outside wall either side of the door
    ],
    door: { x: 0, y: 0, halfWidth: 1 },
    start: { x: 0.3, y: 0, yaw: 0 },
    // Room legs are on the building axes or clearly diagonal: walking straight a few degrees
    // off an axis is where the device's heading-snap heuristic misleads itself.
    inbound: [[7, 0], [14, 0], [14, 13.5], [11.5, 15.5], [11.5, 18], [16.5, 18]],
    labels: [{ x: 7, y: 0, text: 'Corridor A' }, { x: 14, y: 6, text: 'Corridor B' }, { x: 14, y: 18.6, text: 'Room' }],
  };

  function mulberry32(a) {
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function wrapPi(a) { while (a > Math.PI) a -= 2 * Math.PI; while (a <= -Math.PI) a += 2 * Math.PI; return a; }

  function raySeg(ox, oy, dx, dy, w) {
    const ex = w[2] - w[0], ey = w[3] - w[1];
    const den = dx * ey - dy * ex;
    if (Math.abs(den) < 1e-9) return -1;
    const t = ((w[0] - ox) * ey - (w[1] - oy) * ex) / den;
    const u = ((w[0] - ox) * dy - (w[1] - oy) * dx) / den;
    if (t <= 1e-4 || u < 0 || u > 1) return -1;
    return t;
  }

  /** sim_collide(): axis-aligned walls stop the move along their normal (slide). */
  function collide(walls, ax, ay, b) {
    let dx = b.x - ax, dy = b.y - ay, len = Math.hypot(dx, dy);
    if (len < 1e-6) return false;
    const margin = 0.25;
    let hit = false;
    for (let i = 0; i < walls.length; i++) {
      const w = walls[i];
      const t = raySeg(ax, ay, dx / len, dy / len, w);
      if (t >= 0 && t < len + margin) {
        if (w[0] === w[2]) b.x = ax; else b.y = ay;
        dx = b.x - ax; dy = b.y - ay; len = Math.hypot(dx, dy);
        hit = true;
        if (len < 1e-6) return true;
      }
    }
    return hit;
  }

  function DemoWalker(opts) {
    this.o = Object.assign({
      seed: 1,
      plan: PLAN,
      stepLengthM: 0.62,      // the device's calibrated stride (ps_config.c)
      stepRateHz: 1.8,        // manual "Walk" and the auto demo
      turnRateDps: 90,        // continuous turn while "Turn" is held
      maxTurnDps: 120,        // body turn rate limit (simulator)
      errors: null,           // override the per-seed IMU errors
    }, opts || {});
    this.plan = this.o.plan;
    this.reset(this.o.seed);
  }

  /** Back to the doorway with fresh per-seed IMU errors. */
  DemoWalker.prototype.reset = function (seed) {
    if (seed !== undefined) this.o.seed = seed;
    const rnd = mulberry32((this.o.seed >>> 0) * 2654435761 + 12345);
    this.rand = rnd;
    const randn = () => { let u = 0; while (u === 0) u = rnd(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rnd()); };
    this.randn = randn;
    const sgn = () => (rnd() < 0.5 ? -1 : 1);
    this.err = Object.assign({
      strideBias: Math.max(-0.14, Math.min(0.06, -0.06 + 0.04 * randn())),
      strideNoise: 0.05,
      driftDps: sgn() * (0.02 + 0.03 * rnd()),   // deg/s; simulator scenarios use 0.02-0.06
      gyroNoise: 0.002,
      scaleErr: sgn() * (0.005 + 0.01 * rnd()),
      yawNoise: 0.003,
      stepMiss: 0.01 + 0.02 * rnd(),             // simulator: 0.03-0.05
    }, this.o.errors || {});
    const s = this.plan.start;
    this.t = 0;
    this.x = s.x; this.y = s.y; this.yaw = s.yaw;
    this.targetYaw = s.yaw;
    this.gyroErr = 0;
    this.yawRef = (rnd() * 2 - 1) * Math.PI;   // the IMU's arbitrary yaw reference
    this.stepPhase = 0;
    this.walkHeld = false; this.turnHeld = 0;
    this.phase = 'manual';     // 'manual' | 'inbound' | 'scan' | 'outbound' | 'done'
    this.leg = 0; this.scanTargets = null;
    this.slideSteps = 0; this.slideYaw = 0;
    this.hoseUsed = false; this.hoseIdx = 0; this.hoseProg = null; this.hoseSkips = 0; this.hoseLen = 0;
    this.entry = null;         // where the entry was marked (null = the plan's doorway point)
    this.result = null;
    this.trail = [{ x: this.x, y: this.y }];
    this.trueSteps = 0;
    this.bumps = 0;
    this.tOutStart = 0;
    this.onPhase = null;
  };

  DemoWalker.prototype._setPhase = function (p) { this.phase = p; if (this.onPhase) this.onPhase(p); };

  /* ---- controls (manual) ---- */
  DemoWalker.prototype.setWalk = function (on) { this.walkHeld = !!on; if (on && this.phase === 'done') this._setPhase('manual'); };
  /** Turn by deg (+ = left), e.g. 15 per tap. */
  DemoWalker.prototype.turnBy = function (deg) {
    if (this.phase !== 'manual') this.stopAuto();
    this.targetYaw = wrapPi(this.targetYaw + deg * D2R);
  };
  /** Continuous turn while held: dir +1 = left, -1 = right, 0 = stop. */
  DemoWalker.prototype.setTurn = function (dir) {
    if (dir && this.phase !== 'manual') this.stopAuto();
    // releasing a continuous turn stops it where the body is (a tap's 15 deg still completes)
    if (!dir && this.turnHeld) this.targetYaw = wrapPi(this.yaw + Math.max(-0.05, Math.min(0.05, wrapPi(this.targetYaw - this.yaw))));
    this.turnHeld = dir;
  };
  DemoWalker.prototype.stopAuto = function () { if (this.phase !== 'manual') { this._setPhase('manual'); this.targetYaw = this.yaw; } };

  /** Scripted walk in from the door (the walker must be at the door, entry marked). */
  DemoWalker.prototype.startAuto = function () { this.leg = 0; this.walkHeld = false; this.turnHeld = 0; this._setPhase('inbound'); };
  /** Follow the device's arrow back out from wherever the walker is. */
  DemoWalker.prototype.guideOut = function () {
    this.walkHeld = false; this.turnHeld = 0; this.hoseUsed = false; this.hoseProg = null; this.result = null;
    this.tOutStart = this.t;
    this.hoseLen = this.trail.length;   // the hose was laid on the way in: the trail so far
    this._setPhase('outbound');
  };

  /** The entry was marked here (Navigator.markEntry): results are measured from it. */
  DemoWalker.prototype.setEntry = function (x, y) { this.entry = { x, y }; };
  /** Is the marked entry the plan's real doorway (not a spot marked somewhere inside)? */
  DemoWalker.prototype.entryIsDoor = function () {
    const e = this.entry, s = this.plan.start;
    return !e || Math.hypot(e.x - s.x, e.y - s.y) < 0.5;
  };
  /** True distance from where the entry was marked (the doorway point unless it was marked again inside). */
  DemoWalker.prototype.doorDistance = function () {
    const e = this.entry || this.plan.start;
    return Math.hypot(this.x - e.x, this.y - e.y);
  };
  /** Clear line between the walker and p (no wall in between, with a little clearance). */
  DemoWalker.prototype._inSight = function (p) {
    const dx = p.x - this.x, dy = p.y - this.y, len = Math.hypot(dx, dy);
    if (len < 1e-6) return true;
    for (const w of this.plan.walls) {
      for (const off of [-0.2, 0, 0.2]) {   // the body is not a point: three parallel rays
        const ox = this.x - dy / len * off, oy = this.y + dx / len * off;
        const t = raySeg(ox, oy, dx / len, dy / len, w);
        if (t >= 0 && t < len) return false;
      }
    }
    return true;
  };

  DemoWalker.prototype._hosePoint = function (i) { return this.trail[Math.max(0, Math.min(this.trail.length - 1, i))]; };

  /**
   * Advance dtMs (10 ms recommended). g = the device guidance after the
   * previous tick (PSNav.Core#guidance) - only the walk-out reads it.
   * Returns the simulated IMU outputs: {yaw (rad, measured, arbitrary
   * reference), accel (m/s^2), step (bool), whereOut (bool, ask once)}.
   */
  DemoWalker.prototype.step = function (dtMs, g) {
    const dt = dtMs / 1000, plan = this.plan;
    let desired = this.yaw, wantWalk = false, whereOut = false;
    if (this.phase === 'inbound') {
      const p = plan.inbound[this.leg];
      if (Math.hypot(p[0] - this.x, p[1] - this.y) < 0.5) {
        this.leg++;
        if (this.leg >= plan.inbound.length) { this._setPhase('scan'); this.scanTargets = null; }
      } else { desired = Math.atan2(p[1] - this.y, p[0] - this.x); wantWalk = true; }
    } else if (this.phase === 'scan') {
      // Look around the room: sweep 90 deg left, 180 deg right, back to the start,
      // at up to 60 deg/s, then ask the device "where is out?".
      if (!this.scanTargets) { const y0 = this.yaw; this.scanTargets = [y0 + 90 * D2R, y0 - 90 * D2R, y0]; }
      const tgt = this.scanTargets[0], e = wrapPi(tgt - this.yaw);
      desired = this.yaw + Math.max(-60 * D2R * dt, Math.min(60 * D2R * dt, e));
      if (Math.abs(e) < 1 * D2R) this.scanTargets.shift();
      if (!this.scanTargets.length) { this.scanTargets = null; this.guideOut(); whereOut = true; }
    } else if (this.phase === 'outbound') {
      if (g && g.valid && g.levelId === 2 && !this.hoseUsed) {
        // "Navigation estimate unreliable. Follow the hose line out."
        this.hoseUsed = true;
        let best = 1e9;
        // nearest point of the hose (the way in), not of the path walked since "guide me out"
        const n = Math.min(this.trail.length, this.hoseLen || this.trail.length);
        for (let k = 0; k < n; k++) {
          const d = Math.hypot(this.trail[k].x - this.x, this.trail[k].y - this.y);
          if (d < best) { best = d; this.hoseIdx = k; }
        }
      }
      if (this.hoseUsed) {
        let p = this._hosePoint(this.hoseIdx);
        while (this.hoseIdx > 0 && Math.hypot(p.x - this.x, p.y - this.y) < 0.6) p = this._hosePoint(--this.hoseIdx);
        // Pinned against a corner (the next hose point is just round it): after 3 s without
        // getting closer, head for the furthest hose point within 8 m that is in plain sight,
        // or else back to the previous one, which was reached from here.
        const dp = Math.hypot(p.x - this.x, p.y - this.y), hp = this.hoseProg;
        if (!hp || this.hoseIdx < hp.idx || dp < hp.d - 0.2) this.hoseProg = { idx: this.hoseIdx, d: dp, t: this.t };
        else if (this.t - hp.t > 3000) {
          let k = this.hoseIdx - 1, best = -1;
          for (; k >= 0; k--) {
            const q = this._hosePoint(k);
            if (Math.hypot(q.x - this.x, q.y - this.y) > 8) break;
            if (this._inSight(q)) best = k;
          }
          this.hoseIdx = best >= 0 ? best : Math.min(this.trail.length - 1, this.hoseIdx + 2);
          this.hoseSkips++;
          this.slideSteps = 0;
          p = this._hosePoint(this.hoseIdx);
          this.hoseProg = { idx: this.hoseIdx, d: Math.hypot(p.x - this.x, p.y - this.y), t: this.t };
        }
        if (this.hoseIdx === 0 && Math.hypot(p.x - this.x, p.y - this.y) < 0.4) return this._finish('hose', dtMs);
        desired = Math.atan2(p.y - this.y, p.x - this.x);
        wantWalk = true;
      } else if (g && g.valid && g.routeDistM > 0.6) {
        desired = this.yaw + g.routeBearingDeg * D2R;
        wantWalk = true;
      } else if (g && g.valid) {
        return this._finish('exit', dtMs);   // the device says EXIT here
      }
      if (this.x < -0.3) return this._finish('outside', dtMs);
      if (this.t - this.tOutStart > 300000) return this._finish('timeout', dtMs);
    } else {
      // manual (and done): the controls
      if (this.turnHeld) this.targetYaw = wrapPi(this.targetYaw + this.turnHeld * this.o.turnRateDps * D2R * dt);
      desired = this.targetYaw;
      wantWalk = this.walkHeld;
    }
    return this._move(dtMs, desired, wantWalk, whereOut);
  };

  /** Axis direction (rad) of a corridor around the walker within 20 deg of dir, or null. */
  DemoWalker.prototype._corridorAxis = function (dir) {
    const walls = this.plan.walls;
    const a = Math.round(dir / (Math.PI / 2)) * (Math.PI / 2);
    if (Math.abs(wrapPi(dir - a)) > 20 * D2R) return null;
    const side = (ang) => {
      const dx = Math.cos(ang), dy = Math.sin(ang);
      for (const w of walls) { const t = raySeg(this.x, this.y, dx, dy, w); if (t >= 0 && t < 1.6) return true; }
      return false;
    };
    return side(a + Math.PI / 2) && side(a - Math.PI / 2) ? wrapPi(a) : null;
  };

  DemoWalker.prototype._finish = function (how, dtMs) {
    this.result = { how, doorErrorM: this.doorDistance(), tOutS: (this.t - this.tOutStart) / 1000, hose: this.hoseUsed,
      ref: this.entryIsDoor() ? 'door' : 'mark' };
    this._setPhase('done');
    return this._move(dtMs, this.yaw, false, false);
  };

  DemoWalker.prototype._move = function (dtMs, desired, wantWalk, whereOut) {
    const dt = dtMs / 1000, e = this.err, walls = this.plan.walls;
    const goal = desired;
    if (this.slideSteps > 0) desired = this.slideYaw;
    else if (wantWalk && this.phase !== 'manual') {
      // In a corridor people walk along it, not at a slant across it, even when
      // the arrow points a few degrees off its axis.
      const ax = this._corridorAxis(desired);
      if (ax !== null) desired = ax;
    }
    if (this.slideSteps === 0 && wantWalk) {
      // People walk along a wall they meet at a shallow angle instead of
      // bumping into it, so the body (and the IMU heading) turns along it.
      const probe = { x: this.x + 0.9 * Math.cos(desired), y: this.y + 0.9 * Math.sin(desired) };
      if (collide(walls, this.x, this.y, probe)) {
        const along = Math.atan2(probe.y - this.y, probe.x - this.x), mv = Math.hypot(probe.x - this.x, probe.y - this.y);
        if (mv > 0.9 * Math.cos(40 * D2R) - 1e-6 || (this.phase !== 'manual' && mv > 0.2)) desired = along;
        else if (this.phase === 'manual') wantWalk = false;   // head-on into a wall: no step
      }
    }
    const err = wrapPi(desired - this.yaw);
    const maxTurn = this.o.maxTurnDps * D2R * dt;
    const dyaw = Math.max(-maxTurn, Math.min(maxTurn, err));
    this.yaw = wrapPi(this.yaw + dyaw);
    let stepNow = false;
    let accel = 0.2 + 0.05 * this.randn();
    if (wantWalk && Math.abs(err) < 25 * D2R) {
      if (!this._walking) this.stepPhase = Math.max(this.stepPhase, 0.85); // first step comes at once
      this._walking = true;
      accel = 2.2 + 0.4 * this.randn();
      this.stepPhase += this.o.stepRateHz * dt;
      if (this.stepPhase >= 1) {
        this.stepPhase -= 1;
        const L = this.o.stepLengthM * (1 + e.strideBias) * (1 + e.strideNoise * this.randn());
        const b = { x: this.x + L * Math.cos(this.yaw), y: this.y + L * Math.sin(this.yaw) };
        const hit = collide(walls, this.x, this.y, b);
        const moved = Math.hypot(b.x - this.x, b.y - this.y);
        if (hit && moved < 0.5 * L && this.slideSteps === 0 && this.phase !== 'manual' && this.phase !== 'done') {
          // Blocked: turn along the wall toward the side the goal favours, follow it a while.
          if (b.x === this.x) this.slideYaw = (Math.sin(goal) > 0.15 || (Math.abs(Math.sin(goal)) <= 0.15 && this.rand() < 0.5)) ? Math.PI / 2 : -Math.PI / 2;
          else this.slideYaw = (Math.cos(goal) > 0.15 || (Math.abs(Math.cos(goal)) <= 0.15 && this.rand() < 0.5)) ? 0 : Math.PI;
          this.slideSteps = 14;
        } else {
          this.x = b.x; this.y = b.y;
          // A shuffle against a wall is not a counted stride (as in the simulator).
          stepNow = moved > 0.3 * L && this.rand() >= e.stepMiss;
          if (moved > 0.3 * L) this.trueSteps++;
          if (hit) this.bumps++;
          if (this.slideSteps > 0) {
            this.slideSteps--;
            const p2 = { x: this.x + Math.cos(goal), y: this.y + Math.sin(goal) };
            if (!collide(walls, this.x, this.y, p2)) this.slideSteps = 0;
          }
          const last = this.trail[this.trail.length - 1];
          if (Math.hypot(this.x - last.x, this.y - last.y) >= 0.25) this.trail.push({ x: this.x, y: this.y });
        }
      }
    } else {
      this._walking = false;
      if (!wantWalk) this.stepPhase = Math.min(this.stepPhase, 0.5);
    }
    // ---- BNO085 model (sim_world.c) ----
    this.gyroErr += (e.driftDps * D2R + e.gyroNoise * this.randn()) * dt;
    this.gyroErr += e.scaleErr * dyaw;
    const yaw = wrapPi(this.yaw + this.yawRef + this.gyroErr + e.yawNoise * this.randn());
    this.t += dtMs;
    return { yaw, accel: Math.abs(accel), step: stepNow, whereOut };
  };

  DemoWalker.PLAN = PLAN;
  DemoWalker.mulberry32 = mulberry32;
  PSNav.DemoWalker = DemoWalker;
})(typeof globalThis !== 'undefined' ? globalThis : this);
