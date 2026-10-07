/*
 * PyroSight demo navigation: PSNav.Navigator, the one object a page needs.
 *
 * It owns one instance of the device's navigation code (PSNav.Core) and
 * feeds it from exactly one input at a time:
 *   'sensors'  the phone's motion sensors (PSNav.MotionSensors): real walking;
 *   'demo'     the demo walker (PSNav.DemoWalker): simulated walking with a
 *              simulated IMU; on-screen / keyboard controls and an auto demo;
 *   'camera'   heading from an external source (e.g. the camera-turn tracker,
 *              motion.js) fed through feedHeading(); steps from walk(true);
 *   'none'     nothing yet.
 * start() (call it from a tap) picks automatically: sensors when they deliver
 * data, otherwise the demo walk, and says why in navigator.input.
 *
 * Clock. The C code runs on its own millisecond clock. Outside demo mode it
 * follows performance.now(), but one update never advances it by more than
 * maxGapMs (250 ms): a page that freezes for seconds (a long model run on a
 * slow phone) is not a motion-sensor outage, and the device code would
 * otherwise declare the IMU lost and collapse its confidence. In demo mode
 * the clock is the walker's simulated time (x demo speed), in 10 ms steps
 * like the device simulator.
 */
(function (root) {
  'use strict';
  const PSNav = root.PSNav = root.PSNav || {};
  const D2R = Math.PI / 180;

  const LABELS = {
    none: 'No motion input yet',
    sensors: 'Phone motion sensors',
    demo: 'Demo walk (simulated walker and motion sensor)',
    camera: 'Camera turn tracker (heading) + Walk button (steps)',
  };

  function Navigator(opts) {
    this.o = Object.assign({
      mode: 'auto',            // 'auto' | 'sensors' | 'demo' | 'camera'
      win: root,
      now: () => (root.performance ? root.performance.now() : Date.now()),
      maxGapMs: 250,
      sensorTimeoutMs: 1500,
      externalTimeoutMs: 2000,
      seed: 1,
      demoSpeed: 1,
      stepRateHz: 1.8,
      stepLengthM: null,       // null = the device default (0.62 m)
      headingSnap: null,       // null = the device default (on)
      updateMs: 33,            // run(): update interval
      autoMarkEntry: true,     // mark the entry where the input starts delivering data
    }, opts || {});
    this.core = new PSNav.Core();
    if (this.o.stepLengthM) this.core.setCfg('stepLengthM', this.o.stepLengthM);
    if (this.o.headingSnap !== null) this.core.setCfg('headingSnap', this.o.headingSnap);
    this.coreT = 0;
    this._lastReal = null;
    this.stalls = 0;
    this.input = { mode: 'none', status: 'off', label: LABELS.none, detail: '', heading: 'none', steps: 'none' };
    this.sensors = null;
    this.walker = null;
    this.speed = this.o.demoSpeed;
    this.seed = this.o.seed;
    this._ls = {};
    this.trail = [];
    this.markPose = null;
    this._ext = null;
    this._walkHeld = false; this._walkPhase = 0;
    this._turnTimer = null;
    this._simAcc = 0;
    this._timer = null;
    this.alertLog = [];
    this._resetStats();
    this.g = this.core.guidance();
  }

  /* ------------------------------------------------------------ events */
  /** Subscribe: 'guidance' (g), 'alert' ({text, parts, names, prio, t}), 'input' (input),
   *  'step' ({t, source}), 'entry' ({t}), 'demo' ({phase, result}). Returns an unsubscribe function. */
  Navigator.prototype.on = function (type, fn) {
    (this._ls[type] = this._ls[type] || []).push(fn);
    return () => this.off(type, fn);
  };
  Navigator.prototype.off = function (type, fn) {
    const a = this._ls[type];
    if (a) { const i = a.indexOf(fn); if (i >= 0) a.splice(i, 1); }
  };
  Navigator.prototype._emit = function (type, v) {
    const a = this._ls[type];
    if (!a) return;
    for (const fn of a.slice()) { try { fn(v); } catch (e) { if (root.console) console.error(e); } }
  };

  Navigator.prototype._setInput = function (mode, status, detail, extra) {
    this.input = Object.assign({ mode, status, label: LABELS[mode] || mode, detail: detail || '' }, extra || {});
    this._emit('input', this.input);
  };

  /* ------------------------------------------------------------ clock */
  Navigator.prototype._advance = function () {
    const r = this.o.now();
    if (this._lastReal === null) this._lastReal = r;
    let d = r - this._lastReal;
    this._lastReal = r;
    if (!(d > 0)) d = 0;
    if (d > this.o.maxGapMs) { this.stalls++; d = this.o.maxGapMs; }
    return d;
  };
  /** Core clock now (ms), advancing it in real-time modes. */
  Navigator.prototype.coreNow = function () {
    const d = this._advance();
    if (this.input.mode !== 'demo') this.coreT += d;
    else {
      // demo speed applies to the auto demo and "guide me out"; manual control is real time
      const manual = !this.walker || this.walker.phase === 'manual' || this.walker.phase === 'done';
      this._simAcc += d * (manual ? 1 : this.speed);
    }
    return Math.round(this.coreT);
  };

  /* ------------------------------------------------------------ inputs */
  /** Pick the input automatically (call from a tap): sensors, else the demo walk. */
  Navigator.prototype.start = function () {
    const m = this.o.mode;
    if (m === 'demo') { this.useDemo(); return Promise.resolve(this.input); }
    if (m === 'camera') { this.useCameraHeading(); return Promise.resolve(this.input); }
    return this.useSensors().then((st) => {
      if (st !== 'ok' && m === 'auto') this.useDemo({ reason: this.input.detail });
      return this.input;
    });
  };

  /**
   * Start without a tap (e.g. on page load). Browsers that ask before sharing
   * motion data (iOS: DeviceMotionEvent.requestPermission) only allow that
   * from a tap, so there the demo walk starts and input.detail says to tap
   * "use phone sensors" (useSensors() from a click handler). Elsewhere = start().
   */
  Navigator.prototype.autoStart = function () {
    const w = this.o.win;
    const DME = w.DeviceMotionEvent, DOE = w.DeviceOrientationEvent;
    const needsTap = (DME && typeof DME.requestPermission === 'function') || (DOE && typeof DOE.requestPermission === 'function');
    if (needsTap && this.o.mode === 'auto') {
      this.useDemo({ reason: 'this browser asks before sharing motion data. Tap "Use phone sensors" to allow it.' });
      return Promise.resolve(this.input);
    }
    return this.start();
  };

  /** Use the phone's motion sensors. Call from a tap (iOS permission). Resolves to the sensor status. */
  Navigator.prototype.useSensors = function () {
    this._leaveMode();
    const s = this.sensors = new PSNav.MotionSensors({
      win: this.o.win, timeoutMs: this.o.sensorTimeoutMs, now: this.o.now,
      onYaw: (yaw) => {
        if (this.input.mode !== 'sensors') return;
        this.core.onYaw(yaw, this.coreNow());
        if (this._pendingMark) { this._pendingMark = false; this.markEntry(); } // after a heading sample
      },
      onAccel: (mag) => { if (this.input.mode === 'sensors') this.core.onAccel(mag, this.coreNow()); },
      onStep: () => {
        if (this.input.mode !== 'sensors') return;
        const t = this.coreNow();
        this.core.onStep(t);
        this._emit('step', { t, source: 'sensors' });
      },
      onStatus: (st, msg) => {
        if (this.sensors !== s) return;
        this._setInput('sensors', st, msg, { heading: s.heading, steps: s.stats.motion ? 'accelerometer' : 'button' });
        if (st === 'ok' && this.o.autoMarkEntry && this.core.guidance().state === 'idle') this._pendingMark = true;
      },
    });
    this._setInput('sensors', 'asking', 'Starting motion sensors...');
    return s.start().then((st) => {
      if (st !== 'waiting') return st;
      return new Promise((resolve) => {
        const poll = setInterval(() => {
          if (s.status !== 'waiting' && s.status !== 'asking') { clearInterval(poll); resolve(s.status); }
        }, 100);
      });
    });
  };

  /** Simulated walking (works anywhere). The walker stands at the door and the entry is marked. */
  Navigator.prototype.useDemo = function (opt) {
    opt = opt || {};
    this._leaveMode();
    this.walker = new PSNav.DemoWalker({ seed: opt.seed !== undefined ? opt.seed : this.seed, stepRateHz: this.o.stepRateHz,
      stepLengthM: this.core.getCfg('stepLengthM') });
    this.walker.onPhase = (p) => this._emit('demo', { phase: p, result: this.walker.result });
    const why = opt.reason ? 'Motion sensors not used: ' + opt.reason : 'Simulated walking: the estimate drifts like a real motion sensor would.';
    this._setInput('demo', 'ok', why, { heading: 'simulated', steps: 'simulated' });
    this._simAcc = 0;
    this._feedWalker(0, false);
    this.markEntry();
  };

  /** Heading from an external source (feedHeading / feedCameraTrackerYaw), steps from walk(true). */
  Navigator.prototype.useCameraHeading = function (detail) {
    this._leaveMode();
    this._setInput('camera', this._ext ? 'ok' : 'waiting', detail || 'Turn the camera to turn; hold Walk to take steps.', { heading: 'camera', steps: 'button' });
  };

  Navigator.prototype._leaveMode = function () {
    // A track made with another input (other clock, other heading reference) is
    // meaningless now: back to "entry not marked".
    this.core.reset();
    this.g = this.core.guidance();
    this.trail = [];
    this.markPose = null;
    this._ext = null;
    this._pendingMark = false;
    if (this.sensors) { const s = this.sensors; this.sensors = null; s.stop(); }
    this.walker = null;
    this._walkHeld = false;
    this.turnHold(0);
  };

  /**
   * External heading hook. yawRad: counter-clockwise (+ = turned LEFT),
   * any fixed reference. Used in 'camera' mode; ignored otherwise.
   */
  Navigator.prototype.feedHeading = function (yawRad) {
    if (!isFinite(yawRad)) return;
    const t = this.coreNow();
    this._ext = { yaw: yawRad, t };
    if (this.input.mode === 'camera') {
      if (this.input.status !== 'ok') this._setInput('camera', 'ok', this.input.detail, { heading: 'camera', steps: 'button' });
      this.core.onYaw(yawRad, t);
      if (this.o.autoMarkEntry && this.core.guidance().state === 'idle') this.markEntry();
    }
  };
  /**
   * motion.js convenience: tracker.pose.yaw is degrees CLOCKWISE (right = +). Call it for every
   * tracked camera frame. trackerState 'lost' (the picture still comes but does not match: a blank
   * wall, a fast turn) keeps the last good heading instead: the heading source is still there, so
   * the device must not treat it as a motion-sensor outage (that grows its position error at a
   * brisk walk's pace for good). Only when no frames arrive at all (the camera stopped) does the
   * held heading run out after externalTimeoutMs and the device say LOST.
   */
  Navigator.prototype.feedCameraTrackerYaw = function (yawDegClockwise, trackerState) {
    if (trackerState === 'lost') { if (this._ext) this.feedHeading(this._ext.yaw); return; }
    this.feedHeading(-yawDegClockwise * D2R);
  };

  /* ------------------------------------------------------------ actions */
  Navigator.prototype.markEntry = function () {
    const t = this.input.mode === 'demo' ? Math.round(this.coreT) : this.coreNow();
    this.core.markEntry(t);
    this.trail = [{ x: 0, y: 0 }];
    this.markPose = this.walker ? { x: this.walker.x, y: this.walker.y, yaw: this.walker.yaw } : null;
    if (this.walker) { this.walker.trail = [{ x: this.walker.x, y: this.walker.y }]; this.walker.setEntry(this.walker.x, this.walker.y); }
    this._resetStats();
    this._emit('entry', { t });
    this._collect();
  };

  /** "Where is out?" button: the device answers with a spoken direction (or "follow hose"). */
  Navigator.prototype.whereOut = function () {
    this.core.whereOut(this.input.mode === 'demo' ? Math.round(this.coreT) : this.coreNow());
    this._collect();
  };

  /** Forget the entry (back to idle). */
  Navigator.prototype.reset = function () { this.core.reset(); this.trail = []; this.markPose = null; this._collect(); };

  /** Walk: demo -> the walker walks; other modes -> manual steps at stepRateHz. */
  Navigator.prototype.walk = function (on) {
    if (this.walker) { if (on) this.walker.stopAuto(); this.walker.setWalk(on); return; }
    if (on && !this._walkHeld) this._walkPhase = 0.999; // first step at once
    this._walkHeld = !!on;
  };
  /** Turn the demo walker by deg (+ = left). */
  Navigator.prototype.turn = function (deg) { if (this.walker) this.walker.turnBy(deg); };
  /**
   * Hold-to-turn (buttons / keys): dir +1 = left, -1 = right, 0 = release.
   * A tap turns 15 degrees; holding longer than 300 ms keeps turning at 90 deg/s.
   */
  Navigator.prototype.turnHold = function (dir) {
    if (this._turnTimer) { clearTimeout(this._turnTimer); this._turnTimer = null; }
    if (!this.walker) return;
    if (!dir) { this.walker.setTurn(0); return; }
    this.walker.turnBy(15 * dir);
    this._turnTimer = setTimeout(() => { this._turnTimer = null; if (this.walker) this.walker.setTurn(dir); }, 300);
  };
  /** Keys: another turn key is still held after one was released: keep turning that way (no 15 degree tap). */
  Navigator.prototype._turnResume = function (dir) {
    if (this._turnTimer) { clearTimeout(this._turnTimer); this._turnTimer = null; }
    if (this.walker) this.walker.setTurn(dir);
  };
  /** Auto demo: new seed, walker back at the door, entry marked, scripted walk in, then guided out. */
  Navigator.prototype.autoDemo = function (seed) {
    if (seed !== undefined) this.seed = seed; else if (this.walker && this.walker.phase !== 'manual') this.seed++;
    if (!this.walker || this.input.mode !== 'demo') this.useDemo();
    this.walker.reset(this.seed);
    this.walker.onPhase = (p) => this._emit('demo', { phase: p, result: this.walker.result });
    this._feedWalker(0, false);
    this.markEntry();
    this.walker.startAuto();
    this._emit('demo', { phase: 'inbound', result: null });
  };
  /** Demo: the walker follows the arrow out by itself. */
  Navigator.prototype.guideMeOut = function () {
    if (!this.walker) return;
    this.walker.guideOut();
    this.core.whereOut(Math.round(this.coreT));
    this._collect();
  };
  Navigator.prototype.setSpeed = function (x) { this.speed = Math.max(0.25, Math.min(16, +x || 1)); };

  /* ------------------------------------------------------------ loop */
  Navigator.prototype._feedWalker = function (dtMs, tick) {
    const s = this.walker.step(dtMs, this.g);
    const t = Math.round(this.coreT);
    this.core.onYaw(s.yaw, t);
    this.core.onAccel(s.accel, t);
    if (s.step) { this.core.onStep(t); this._emit('step', { t, source: 'demo' }); }
    if (s.whereOut) this.core.whereOut(t);
    if (tick) { this.core.tick(t); this.g = this.core.guidance(); this._demoStats(); }
  };

  /** Advance everything to now and return the guidance. Call every frame (or use run()). */
  Navigator.prototype.update = function () {
    const t = this.coreNow();
    if (this.input.mode === 'demo' && this.walker) {
      let n = 0;
      while (this._simAcc >= 10 && n < 2000) {
        this._simAcc -= 10; n++;
        this.coreT += 10;
        this._feedWalker(10, true);
      }
    } else {
      if (this.input.mode === 'camera' && this._ext && t - this._ext.t < this.o.externalTimeoutMs) {
        this.core.onYaw(this._ext.yaw, t); // hold the last tracker pose: the tracker updates slower than the IMU
        // No accelerometer in this mode: report the body's motion like the demo walker does
        // (~2.2 m/s^2 while Walk is held, ~0.2 standing). Without a resting value the device's
        // low-passed acceleration stays where the last steps left it, and after ~8 steps its
        // crawl path ("moving without steps") moved the position on while the user stood still.
        this.core.onAccel(this._walkHeld ? 2.2 : 0.2, t);
      }
      if (this._walkHeld) {
        this._walkPhase += this.o.stepRateHz * (this._lastStepT !== undefined ? (t - this._lastStepT) / 1000 : 0);
        if (this._walkPhase >= 1) {
          this._walkPhase -= Math.floor(this._walkPhase);
          this.core.onStep(t);
          this._emit('step', { t, source: 'button' });
        }
      }
      this._lastStepT = t;
      this.core.tick(t);
      this.g = this.core.guidance();
    }
    this._collect();
    return this.g;
  };

  Navigator.prototype._collect = function () {
    const g = this.g = this.core.guidance();
    g.t = Math.round(this.coreT);
    g.input = this.input.mode;
    if (g.valid) {
      const last = this.trail[this.trail.length - 1];
      if (!last || Math.hypot(g.pos.x - last.x, g.pos.y - last.y) >= 0.2) {
        this.trail.push({ x: g.pos.x, y: g.pos.y });
        if (this.trail.length > 4000) this.trail = this.trail.filter((p, i) => i % 2 === 0 || i === this.trail.length - 1);
      }
    }
    const alerts = this.core.popAlerts();
    for (const a of alerts) {
      this.alertLog.push(a);
      if (this.alertLog.length > 50) this.alertLog.shift();
      if (a.names.indexOf('FOLLOW_HOSE') >= 0) this.stats.followHoseAlerts++;
      this._emit('alert', a);
    }
    this._emit('guidance', g);
  };

  Navigator.prototype._resetStats = function () {
    this.stats = { maxEstErrM: 0, minConfidence: 1, unreliableEntries: 0, followHoseAlerts: 0, lastLevel: 0 };
  };
  Navigator.prototype._demoStats = function () {
    const g = this.g, st = this.stats;
    if (!g.valid) return;
    const tp = this.truePose();
    if (tp) st.maxEstErrM = Math.max(st.maxEstErrM, Math.hypot(tp.x - g.pos.x, tp.y - g.pos.y));
    st.minConfidence = Math.min(st.minConfidence, g.confidence);
    if (g.levelId === 2 && st.lastLevel !== 2) st.unreliableEntries++;
    st.lastLevel = g.levelId;
  };

  /** World (demo floor plan) -> entry frame of the current mark. */
  Navigator.prototype.toEntry = function (x, y) {
    const m = this.markPose || { x: 0, y: 0, yaw: 0 };
    const dx = x - m.x, dy = y - m.y, c = Math.cos(m.yaw), s = Math.sin(m.yaw);
    return { x: c * dx + s * dy, y: -s * dx + c * dy };
  };
  /** Demo only: the walker's true pose in the entry frame. */
  Navigator.prototype.truePose = function () {
    if (!this.walker) return null;
    const p = this.toEntry(this.walker.x, this.walker.y);
    p.yaw = this.walker.yaw - (this.markPose ? this.markPose.yaw : 0);
    return p;
  };

  /** Everything a map needs, in the entry frame. demo = null outside demo mode. */
  Navigator.prototype.snapshot = function () {
    const g = this.g;
    const snap = { g, crumbs: this.core.crumbs(), trail: this.trail, input: this.input, demo: null };
    if (this.walker && this.markPose) {
      const w = this.walker, plan = w.plan;
      snap.demo = {
        truePos: this.truePose(),
        truePath: w.trail.map((p) => this.toEntry(p.x, p.y)),
        walls: plan.walls.map((s) => { const a = this.toEntry(s[0], s[1]), b = this.toEntry(s[2], s[3]); return [a.x, a.y, b.x, b.y]; }),
        labels: (plan.labels || []).map((l) => Object.assign(this.toEntry(l.x, l.y), { text: l.text })),
        phase: w.phase, result: w.result, hose: w.hoseUsed,
      };
    }
    return snap;
  };

  /** Own update loop (setInterval, keeps running when rAF is throttled). */
  Navigator.prototype.run = function () {
    if (this._timer) return;
    this._timer = setInterval(() => this.update(), this.o.updateMs);
  };
  /** Stop the update loop and the sensor listeners (the track is kept; start()/use*() to go on). */
  Navigator.prototype.stop = function () {
    if (this._timer) clearInterval(this._timer);
    this._timer = null;
    if (this.sensors) this.sensors.stop();
    this._walkHeld = false;
    this.turnHold(0);
  };

  /**
   * Keyboard: ArrowUp / W hold to walk, ArrowLeft / A and ArrowRight / D turn
   * (tap 15 deg, hold = continuous). Ignored while typing in a form field.
   * Returns an unbind function.
   */
  Navigator.prototype.bindKeys = function (target) {
    target = target || this.o.win;
    const held = {};
    const map = { ArrowUp: 'walk', w: 'walk', W: 'walk', ArrowLeft: 'left', a: 'left', A: 'left', ArrowRight: 'right', d: 'right', D: 'right' };
    const typing = (e) => { const el = e.target; return el && (el.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(el.tagName || '')); };
    const down = (e) => {
      const k = map[e.key];
      if (!k || e.ctrlKey || e.metaKey || e.altKey || typing(e)) return;
      e.preventDefault();
      if (held[k]) return; // key repeat
      held[k] = true;
      if (k === 'walk') this.walk(true); else this.turnHold(k === 'left' ? 1 : -1);
    };
    const up = (e) => {
      const k = map[e.key];
      if (!k || !held[k]) return;
      held[k] = false;
      if (k === 'walk') this.walk(false);
      else if (held.left || held.right) this._turnResume(held.left ? 1 : -1);   // the other turn key is still down
      else this.turnHold(0);
    };
    const blur = () => { for (const k in held) if (held[k]) { held[k] = false; if (k === 'walk') this.walk(false); else this.turnHold(0); } };
    target.addEventListener('keydown', down);
    target.addEventListener('keyup', up);
    if (this.o.win.addEventListener) this.o.win.addEventListener('blur', blur);
    return () => { target.removeEventListener('keydown', down); target.removeEventListener('keyup', up); if (this.o.win.removeEventListener) this.o.win.removeEventListener('blur', blur); };
  };

  /**
   * Make a button act while pressed (pointer / touch / mouse). Returns an unbind function.
   * Mouse and pen act at once. A finger acts once it has rested on the button for touchDelayMs
   * (a quick tap = press + release) and lets the page scroll: give the button CSS
   * touch-action: pan-y; a swipe that starts on it ends in pointercancel (the browser scrolls)
   * or moves more than slopPx first, and then does nothing. Several fingers (or a finger and the
   * mouse) on the same button: it is released when the last one lifts.
   */
  function holdButton(el, onDown, onUp, opts) {
    const o = Object.assign({ touchDelayMs: 130, slopPx: 10 }, opts || {});
    const ptrs = new Map();   // pointerId -> where it went down
    let active = false, timer = null;
    const clear = () => { if (timer) { clearTimeout(timer); timer = null; } };
    const begin = () => { clear(); if (!active && ptrs.size) { active = true; onDown(); } };
    const start = (e) => {
      if (e.button > 0) return;
      const touch = e.pointerType === 'touch';
      if (!touch) e.preventDefault();
      ptrs.set(e.pointerId, { x: e.clientX, y: e.clientY, touch });
      if (!touch) { try { el.setPointerCapture(e.pointerId); } catch (x) { /* ignore */ } begin(); return; }
      if (!active && !timer) timer = setTimeout(begin, o.touchDelayMs);
    };
    const move = (e) => {
      const p = ptrs.get(e.pointerId);
      if (!p || active || !p.touch) return;
      if (Math.hypot(e.clientX - p.x, e.clientY - p.y) > o.slopPx) {   // a swipe, not a press
        ptrs.delete(e.pointerId);
        if (!ptrs.size) clear();
      }
    };
    const end = (e) => {
      if (!ptrs.has(e.pointerId)) return;
      ptrs.delete(e.pointerId);
      if (ptrs.size) return;
      if (active) { active = false; onUp(); }
      else if (timer) { clear(); onDown(); onUp(); }       // a quick tap: one press
    };
    const cancel = (e) => {
      if (!ptrs.has(e.pointerId)) return;
      ptrs.delete(e.pointerId);
      if (ptrs.size) return;
      clear();                                              // the browser scrolls instead
      if (active) { active = false; onUp(); }
    };
    el.addEventListener('pointerdown', start);
    el.addEventListener('pointermove', move);
    el.addEventListener('pointerup', end);
    el.addEventListener('pointercancel', cancel);
    el.addEventListener('lostpointercapture', end);
    const noMenu = (e) => e.preventDefault();
    el.addEventListener('contextmenu', noMenu);
    return () => {
      clear();
      el.removeEventListener('pointerdown', start); el.removeEventListener('pointermove', move); el.removeEventListener('pointerup', end);
      el.removeEventListener('pointercancel', cancel); el.removeEventListener('lostpointercapture', end); el.removeEventListener('contextmenu', noMenu);
    };
  }

  /** Speak with the browser's own voice (no network). Returns false if unavailable. */
  function speak(text) {
    try {
      const ss = root.speechSynthesis;
      if (!ss || !root.SpeechSynthesisUtterance || !text) return false;
      ss.speak(new root.SpeechSynthesisUtterance(text));
      return true;
    } catch (e) { return false; }
  }

  Navigator.LABELS = LABELS;
  PSNav.Navigator = Navigator;
  PSNav.holdButton = holdButton;
  PSNav.speak = speak;
})(typeof globalThis !== 'undefined' ? globalThis : this);
