/*
 * PyroSight demo navigation: the phone's own motion sensors as the IMU.
 *
 *   heading  'deviceorientation' alpha/beta/gamma -> quaternion (W3C
 *            DeviceOrientation spec, Z-X'-Y'') -> corrected for the screen
 *            orientation -> ps_quat_to_yaw() of the device code (heading of the
 *            screen's right-hand axis, counter-clockwise = +). That axis stays
 *            horizontal whether the phone is held upright (camera forward) or
 *            flat. Chrome's 'deviceorientation' is relative (gyro + accel, no
 *            magnetometer), like the BNO085 game rotation vector the device uses.
 *            Fallback when only 'devicemotion' exists: integrate rotationRate
 *            about the gravity axis.
 *   steps    'devicemotion' accelerationIncludingGravity -> PSNav.StepDetector.
 *   motion   |acceleration| (without gravity) when the browser gives it, else
 *            the detector's estimate -> ps_nav_on_linear_accel.
 *
 * start() must be called from a tap/click handler: iOS only grants motion
 * access (DeviceMotionEvent.requestPermission) during a user gesture.
 * Status: 'off' | 'asking' | 'waiting' | 'ok' | 'denied' | 'blocked' |
 * 'absent' | 'lost'; each change calls onStatus(status, message).
 */
(function (root) {
  'use strict';
  const PSNav = root.PSNav = root.PSNav || {};
  const D2R = Math.PI / 180;

  /** W3C DeviceOrientation (alpha, beta, gamma in degrees) -> quaternion [w, x, y, z]. */
  function eulerToQuat(alpha, beta, gamma) {
    const x = (beta || 0) * D2R / 2, y = (gamma || 0) * D2R / 2, z = (alpha || 0) * D2R / 2;
    const cX = Math.cos(x), cY = Math.cos(y), cZ = Math.cos(z), sX = Math.sin(x), sY = Math.sin(y), sZ = Math.sin(z);
    return [
      cX * cY * cZ - sX * sY * sZ,
      sX * cY * cZ - cX * sY * sZ,
      cX * sY * cZ + sX * cY * sZ,
      cX * cY * sZ + sX * sY * cZ,
    ];
  }

  function quatMul(a, b) {
    return [
      a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
      a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
      a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
      a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    ];
  }

  /** Same formula as ps_quat_to_yaw() in core/src/ps_nav.c. */
  function quatToYaw(q) {
    const w = q[0], x = q[1], y = q[2], z = q[3];
    return Math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z));
  }

  /**
   * Heading (rad, counter-clockwise +) of the screen's right-hand axis, and how
   * horizontal that axis is (1 = level; near 0 the heading is meaningless).
   */
  function orientationYaw(alpha, beta, gamma, screenAngleDeg) {
    return yawFromQuat(eulerToQuat(alpha, beta, gamma), screenAngleDeg);
  }
  function yawFromQuat(q, screenAngleDeg) {
    const th = -(screenAngleDeg || 0) * D2R / 2;
    q = quatMul(q, [Math.cos(th), 0, 0, Math.sin(th)]);
    const w = q[0], x = q[1], y = q[2], z = q[3];
    const r00 = 1 - 2 * (y * y + z * z), r10 = 2 * (x * y + w * z);
    return { yaw: quatToYaw(q), level: Math.hypot(r00, r10) };
  }

  function MotionSensors(opts) {
    this.o = Object.assign({
      win: root,                // object that receives the device events
      timeoutMs: 1500,          // no usable data within this -> absent / blocked; silence -> lost
      now: () => (root.performance ? root.performance.now() : Date.now()),
      onYaw: null, onStep: null, onAccel: null, onStatus: null,
      stepOptions: null,
    }, opts || {});
    this.status = 'off';
    this.message = '';
    this.steps = new PSNav.StepDetector(this.o.stepOptions);
    this._h = null;
    this.heading = 'none';      // 'orientation' | 'gyro' | 'none'
    this.stats = { orientation: 0, orientationNull: 0, motion: 0, motionNull: 0, steps: 0 };
    this.lastYaw = null;
    this._gyroYaw = 0;
    this._lastData = -1e9;
  }

  MotionSensors.prototype._set = function (status, message) {
    if (status === this.status && message === this.message) return;
    this.status = status; this.message = message;
    if (this.o.onStatus) this.o.onStatus(status, message);
  };

  /** Ask for permission (iOS) and start listening. Call from a tap. Resolves to the status. */
  MotionSensors.prototype.start = function () {
    const w = this.o.win;
    this.stop();
    const DOE = w.DeviceOrientationEvent, DME = w.DeviceMotionEvent;
    if (!DOE && !DME) {
      this._set('absent', 'This browser has no motion sensor support.');
      return Promise.resolve(this.status);
    }
    if (w.isSecureContext === false) {
      this._set('blocked', 'Motion sensors need a secure (https) page.');
      return Promise.resolve(this.status);
    }
    // A page inside another page (an iframe, such as a chat preview) gets no motion events unless
    // the page around it allows them: answer at once instead of waiting for data that never comes.
    try {
      const doc = w.document, pol = doc && (doc.permissionsPolicy || doc.featurePolicy);
      if (pol && typeof pol.allowsFeature === 'function' && (!pol.allowsFeature('accelerometer') || !pol.allowsFeature('gyroscope'))) {
        this._set('blocked', 'This page is not allowed to read the motion sensors here (the page around it does not allow it).');
        return Promise.resolve(this.status);
      }
    } catch (e) { /* no policy API: find out from the data */ }
    const asks = [];
    // Both requests must start synchronously inside the tap handler (iOS).
    try {
      if (DME && typeof DME.requestPermission === 'function') asks.push(DME.requestPermission());
      if (DOE && typeof DOE.requestPermission === 'function') asks.push(DOE.requestPermission());
    } catch (e) {
      this._set('blocked', 'The browser refused to ask for motion access (' + (e && e.name || e) + ').');
      return Promise.resolve(this.status);
    }
    if (asks.length) this._set('asking', 'Asking for permission to use motion sensors...');
    return Promise.all(asks).then((res) => {
      if (res.some((r) => r !== 'granted')) {
        this._set('denied', 'Motion sensor permission was refused. Allow motion access in the browser settings, or use the demo walk.');
        return this.status;
      }
      this._listen();
      return this.status;
    }, (e) => {
      this._set('blocked', 'Motion sensors are not allowed on this page (' + (e && (e.name || e.message) || e) + '). Inside claude.ai they are usually blocked; use the demo walk.');
      return this.status;
    });
  };

  MotionSensors.prototype._listen = function () {
    const w = this.o.win;
    this.steps.reset();
    this.heading = 'none';
    this.stats = { orientation: 0, orientationNull: 0, motion: 0, motionNull: 0, steps: 0 };
    this.lastYaw = null;
    this._lastGyroT = null;
    this._lastData = -1e9;
    this._t0 = this.o.now();
    this._lastWatch = undefined;
    const onOri = (e) => this._onOrientation(e);
    const onMot = (e) => this._onMotion(e);
    w.addEventListener('deviceorientation', onOri);
    w.addEventListener('devicemotion', onMot);
    const timer = setInterval(() => this._watch(), 250);
    this._h = { onOri, onMot, timer };
    this._set('waiting', 'Waiting for motion sensor data...');
  };

  MotionSensors.prototype.stop = function () {
    if (!this._h) return;
    const w = this.o.win;
    w.removeEventListener('deviceorientation', this._h.onOri);
    w.removeEventListener('devicemotion', this._h.onMot);
    clearInterval(this._h.timer);
    this._h = null;
    this._set('off', '');
  };

  MotionSensors.prototype._watch = function () {
    const now = this.o.now(), st = this.stats;
    // A busy page (e.g. a long model run) delays both events and this timer:
    // the stall itself is not sensor silence, so move the reference times on.
    if (this._lastWatch !== undefined && now - this._lastWatch > 750) {
      const stall = now - this._lastWatch - 250;
      this._t0 += stall; this._lastData += stall;
    }
    this._lastWatch = now;
    if (this.status === 'waiting' && now - this._t0 > this.o.timeoutMs) {
      if (st.motion > 0 && this.heading === 'none') {
        this._set('absent', 'This device reports movement but no rotation (no gyroscope), so it cannot tell which way you turn. Use the demo walk.');
      } else if (st.orientation + st.motion > 0) {
        this._set('absent', 'This device reports no motion sensor readings (a laptop or desktop usually has none). Use the demo walk.');
      } else {
        this._set('blocked', 'No motion sensor data arrived. The device has no motion sensors, or this page is not allowed to read them (inside claude.ai they are usually blocked). Use the demo walk.');
      }
    } else if (this.status === 'ok' && now - this._lastData > this.o.timeoutMs) {
      this._set('lost', 'Motion sensor data stopped.');
    }
  };

  MotionSensors.prototype._alive = function (kind) {
    this._lastData = this.o.now();
    // usable only with a heading: steps alone cannot say which way the way out is
    if ((this.status === 'waiting' || this.status === 'lost') && this.heading !== 'none') {
      const what = this.heading === 'orientation' ? 'Phone motion sensors: heading and steps.'
        : this.heading === 'gyro' ? 'Phone motion sensors: gyro heading and steps.' : 'Phone motion sensors.';
      this._set('ok', what);
    }
    return kind;
  };

  MotionSensors.prototype._screenAngle = function () {
    const w = this.o.win;
    try {
      if (w.screen && w.screen.orientation && typeof w.screen.orientation.angle === 'number') return w.screen.orientation.angle;
    } catch (e) { /* ignore */ }
    return typeof w.orientation === 'number' ? w.orientation : 0;
  };

  MotionSensors.prototype._onOrientation = function (e) {
    const st = this.stats;
    if (e.alpha === null || e.alpha === undefined || e.beta === null || e.beta === undefined) { st.orientationNull++; return; }
    st.orientation++;
    this.heading = 'orientation';
    this._alive('orientation');
    const r = orientationYaw(e.alpha, e.beta, e.gamma, this._screenAngle());
    if (r.level < 0.35) return; // screen-right axis nearly vertical: hold the last heading
    this.lastYaw = r.yaw;
    if (this.o.onYaw) this.o.onYaw(r.yaw, 'orientation');
  };

  MotionSensors.prototype._onMotion = function (e) {
    const st = this.stats;
    const g = e.accelerationIncludingGravity, rr = e.rotationRate;
    const hasG = g && g.x !== null && g.x !== undefined && g.y !== null && g.z !== null;
    if (!hasG) { st.motionNull++; return; }
    st.motion++;
    const t = this.o.now();
    // Gyro heading only when there is no orientation stream.
    if (this.heading !== 'orientation' && rr && rr.alpha !== null && rr.alpha !== undefined) {
      if (st.orientation === 0) {
        this.heading = 'gyro';
        const n = Math.hypot(g.x, g.y, g.z) || 1;
        // rotationRate: alpha about z, beta about x, gamma about y (deg/s); gravity-up unit vector.
        const rate = (rr.beta * g.x + rr.gamma * g.y + rr.alpha * g.z) / n * D2R;
        if (this._lastGyroT !== null) {
          const dt = Math.min(0.2, (t - this._lastGyroT) / 1000);
          this._gyroYaw = PSNav.wrapPi(this._gyroYaw + rate * dt);
          this.lastYaw = this._gyroYaw;
          if (this.o.onYaw) this.o.onYaw(this._gyroYaw, 'gyro');
        }
        this._lastGyroT = t;
      }
    }
    this._alive('motion');
    const found = this.steps.push(g.x, g.y, g.z, t);
    const a = e.acceleration;
    let lin = a && a.x !== null && a.x !== undefined ? Math.hypot(a.x, a.y || 0, a.z || 0) : this.steps.linear;
    // shaking or knocking the phone (the step detector's shake hold) is not the body moving:
    // without this the device's crawl path moved the position while the user stood still
    if (t < this.steps.quietUntil) lin = 0;
    if (this.o.onAccel) this.o.onAccel(lin);
    for (let i = 0; i < found.length; i++) { st.steps++; if (this.o.onStep) this.o.onStep(found[i]); }
  };

  PSNav.MotionSensors = MotionSensors;
  PSNav.eulerToQuat = eulerToQuat;
  PSNav.quatToYaw = quatToYaw;
  PSNav.orientationYaw = orientationYaw;
  PSNav.yawFromQuat = yawFromQuat;
  PSNav.quatMul = quatMul;
})(typeof globalThis !== 'undefined' ? globalThis : this);
