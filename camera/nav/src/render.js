/*
 * PyroSight demo navigation: drawing (map, arrow) and the camera-picture
 * EXIT box helper. Plain 2D canvas, no dependencies.
 *
 * Colours come from CSS custom properties on the canvas (or an ancestor), so
 * the page's light/dark tokens apply; a `theme` object passed in overrides them:
 *   --nav-bg --nav-fg --nav-muted --nav-wall --nav-trail --nav-true
 *   --nav-route --nav-door --nav-me --nav-good --nav-warn --nav-bad --nav-ring
 */
(function (root) {
  'use strict';
  const PSNav = root.PSNav = root.PSNav || {};
  const D2R = Math.PI / 180;

  const DEFAULT_THEME = {
    bg: '#101418', fg: '#f2f4f5', muted: '#8a949c', wall: '#4b5560', trail: '#6cb4ff', true: '#b9a27a',
    route: '#28ff50', door: '#28ff50', me: '#ffffff', good: '#28ff50', warn: '#ffc400', bad: '#ff4a4a', ring: '#000000',
    font: 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif',
  };
  const KEYS = ['bg', 'fg', 'muted', 'wall', 'trail', 'true', 'route', 'door', 'me', 'good', 'warn', 'bad', 'ring'];

  /** Theme from the CSS tokens visible at element el (missing ones keep defaults). */
  function themeFromCSS(el, base) {
    const t = Object.assign({}, DEFAULT_THEME, base || {});
    try {
      const cs = root.getComputedStyle(el);
      for (const k of KEYS) {
        const v = cs.getPropertyValue('--nav-' + k).trim();
        if (v) t[k] = v;
      }
      const f = cs.getPropertyValue('font-family');
      if (f) t.font = f;
    } catch (e) { /* not in a browser */ }
    return t;
  }

  function fitCanvas(canvas) {
    const dpr = Math.min(3, root.devicePixelRatio || 1);
    const w = Math.max(1, Math.round((canvas.clientWidth || canvas.width) * dpr));
    const h = Math.max(1, Math.round((canvas.clientHeight || canvas.height) * dpr));
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
    return { w, h, dpr };
  }

  /* ================================================================ map */
  /**
   * new PSNav.MapRenderer(canvas, {up: 'entry' | 'heading', theme})
   * draw(snapshot) with snapshot = navigator.snapshot().
   * 'entry' up: the direction faced when the entry was marked points up (the
   * device has no compass: magnetic north is useless in a steel building).
   */
  function MapRenderer(canvas, opts) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.o = Object.assign({ up: 'entry', theme: null, pad: 22, minScale: 4, maxScale: 60 }, opts || {});
    this.view = null; // smoothed {cx, cy, scale}
    this._themeAge = 1e9;
  }
  MapRenderer.prototype.setUp = function (up) { this.o.up = up === 'heading' ? 'heading' : 'entry'; this.view = null; };
  MapRenderer.prototype.toggleUp = function () { this.setUp(this.o.up === 'heading' ? 'entry' : 'heading'); return this.o.up; };
  MapRenderer.prototype.refreshTheme = function () { this._themeAge = 1e9; };

  MapRenderer.prototype.draw = function (snap) {
    const { w: W, h: H, dpr } = fitCanvas(this.canvas);
    if (this._themeAge++ > 30 || !this.theme) { this.theme = this.o.theme ? Object.assign({}, DEFAULT_THEME, this.o.theme) : themeFromCSS(this.canvas); this._themeAge = 0; }
    const T = this.theme, c = this.ctx, g = snap.g || {};
    const demo = snap.demo;
    c.setTransform(1, 0, 0, 1, 0, 0);
    c.fillStyle = T.bg;
    c.fillRect(0, 0, W, H);
    const px = (v) => v * dpr;
    c.font = px(12) + 'px ' + T.font;
    c.textBaseline = 'middle';

    if (!g.valid) {
      c.fillStyle = T.muted; c.textAlign = 'center';
      c.fillText(g.state === 'idle' || !g.state ? 'Mark the entry to start the map' : 'No position yet', W / 2, H / 2);
      return;
    }
    const heading = this.o.up === 'heading';
    const rot = heading ? g.yaw : 0;
    const cr = Math.cos(rot), sr = Math.sin(rot);
    const pos = g.pos, crumbs = snap.crumbs || [];

    // ---- auto zoom: everything that matters in view
    const pts = [pos, ...crumbs];
    for (let i = 0; i < snap.trail.length; i += 4) pts.push(snap.trail[i]);
    if (demo) {
      if (demo.truePos) pts.push(demo.truePos);
      for (let i = 0; i < demo.truePath.length; i += 4) pts.push(demo.truePath[i]);
      if (!heading) for (const s of demo.walls) { pts.push({ x: s[0], y: s[1] }, { x: s[2], y: s[3] }); }
    }
    // map coords: u = right on screen, v = up on screen
    const toUV = (p, cx, cy) => {
      const dx = p.x - cx, dy = p.y - cy;
      const fwd = dx * cr + dy * sr, left = -dx * sr + dy * cr;
      return { u: -left, v: fwd };
    };
    let target;
    const pad = px(this.o.pad), avW = W - 2 * pad, avH = H - 2 * pad - px(18);
    if (heading) {
      let r = 3;
      for (const p of pts) r = Math.max(r, Math.hypot(p.x - pos.x, p.y - pos.y));
      target = { cx: pos.x, cy: pos.y, scale: Math.min(avW, avH) / 2 / (r + 1) };
    } else {
      let u0 = 1e9, u1 = -1e9, v0 = 1e9, v1 = -1e9;
      for (const p of pts) { const q = toUV(p, 0, 0); u0 = Math.min(u0, q.u); u1 = Math.max(u1, q.u); v0 = Math.min(v0, q.v); v1 = Math.max(v1, q.v); }
      const spanU = Math.max(6, u1 - u0 + 2), spanV = Math.max(6, v1 - v0 + 2);
      // centre (u, v) back to entry-frame x, y (rot = 0: u = -y, v = x)
      target = { cx: (v0 + v1) / 2, cy: -(u0 + u1) / 2, scale: Math.min(avW / spanU, avH / spanV) };
    }
    target.scale = Math.max(px(this.o.minScale), Math.min(px(this.o.maxScale), target.scale));
    if (!this.view || this._dprWas !== dpr) this.view = Object.assign({}, target);
    else {
      const k = 0.2;
      this.view.cx += (target.cx - this.view.cx) * (heading ? 1 : k);
      this.view.cy += (target.cy - this.view.cy) * (heading ? 1 : k);
      this.view.scale += (target.scale - this.view.scale) * k;
    }
    this._dprWas = dpr;
    const V = this.view, sc = V.scale, ox = W / 2, oy = H / 2 - px(6);
    const S = (p) => { const q = toUV(p, V.cx, V.cy); return [ox + q.u * sc, oy - q.v * sc]; };
    const line = (arr, color, width, dash) => {
      if (arr.length < 2) return;
      c.beginPath();
      arr.forEach((p, i) => { const s = S(p); if (i) c.lineTo(s[0], s[1]); else c.moveTo(s[0], s[1]); });
      c.strokeStyle = color; c.lineWidth = px(width); c.setLineDash(dash ? dash.map(px) : []); c.lineCap = 'round'; c.lineJoin = 'round';
      c.stroke(); c.setLineDash([]);
    };

    // ---- demo: floor plan and the true path
    if (demo) {
      c.strokeStyle = T.wall; c.lineWidth = px(3); c.lineCap = 'square';
      c.beginPath();
      for (const s of demo.walls) { const a = S({ x: s[0], y: s[1] }), b = S({ x: s[2], y: s[3] }); c.moveTo(a[0], a[1]); c.lineTo(b[0], b[1]); }
      c.stroke();
      c.fillStyle = T.muted; c.textAlign = 'center';
      if (!heading && sc > px(9)) for (const l of demo.labels) { const s = S(l); c.fillText(l.text, s[0], s[1]); }
      line(demo.truePath, T.true, 2, [5, 5]);
    }

    // ---- estimated trail and breadcrumbs
    line(snap.trail, T.trail, 2.5);
    c.fillStyle = T.trail;
    for (let i = 1; i < crumbs.length; i++) { const s = S(crumbs[i]); c.beginPath(); c.arc(s[0], s[1], px(3), 0, 2 * Math.PI); c.fill(); }

    // ---- return route: position -> target crumb -> ... -> door
    const unreliable = g.levelId === 2;
    const route = [pos];
    for (let i = Math.min(g.returnTarget, crumbs.length - 1); i >= 0; i--) route.push(crumbs[i]);
    line(route, unreliable ? T.bad : T.route, 3, [8, 6]);

    // ---- entry door: 0.9 m green bar across the entry heading, at crumb 0
    const door = crumbs[0] || { x: 0, y: 0 };
    const d0 = S({ x: door.x, y: door.y - 0.45 }), d1 = S({ x: door.x, y: door.y + 0.45 }), dc = S(door);
    c.strokeStyle = T.door; c.lineWidth = px(6); c.lineCap = 'butt';
    c.beginPath(); c.moveTo(d0[0], d0[1]); c.lineTo(d1[0], d1[1]); c.stroke();
    c.fillStyle = T.door; c.font = 'bold ' + px(11) + 'px ' + T.font; c.textAlign = 'center';
    c.fillText('ENTRY', dc[0], dc[1] + px(13));

    // ---- uncertainty circle
    const sp = S(pos);
    if (g.posSigmaM > 0.05) {
      c.beginPath(); c.arc(sp[0], sp[1], Math.min(W, g.posSigmaM * sc), 0, 2 * Math.PI);
      c.fillStyle = unreliable ? 'rgba(255,74,74,0.12)' : 'rgba(108,180,255,0.12)'; c.fill();
    }

    // ---- demo: true position (hollow ring + heading tick)
    if (demo && demo.truePos) {
      const tp = S(demo.truePos), a = demo.truePos.yaw - rot;
      c.strokeStyle = T.true; c.lineWidth = px(2);
      c.beginPath(); c.arc(tp[0], tp[1], px(6), 0, 2 * Math.PI); c.stroke();
      c.beginPath(); c.moveTo(tp[0], tp[1]); c.lineTo(tp[0] - Math.sin(a) * px(14), tp[1] - Math.cos(a) * px(14)); c.stroke();
    }

    // ---- estimated position + heading wedge
    const a = g.yaw - rot; // 0 = up on screen
    const wr = px(30);
    c.beginPath(); c.moveTo(sp[0], sp[1]);
    c.arc(sp[0], sp[1], wr, -Math.PI / 2 - a - 28 * D2R, -Math.PI / 2 - a + 28 * D2R);
    c.closePath();
    c.fillStyle = 'rgba(255,255,255,0.18)'; c.fill();
    c.strokeStyle = T.me; c.lineWidth = px(1); c.stroke();
    c.beginPath(); c.arc(sp[0], sp[1], px(6.5), 0, 2 * Math.PI);
    c.fillStyle = T.me; c.fill(); c.strokeStyle = T.ring; c.lineWidth = px(2); c.stroke();

    // ---- scale bar (bottom left)
    const nice = [0.5, 1, 2, 5, 10, 20, 50, 100];
    let m = nice[0];
    for (const n of nice) if (n * sc <= W * 0.3) m = n;
    const bx = px(12), by = H - px(12);
    c.strokeStyle = T.fg; c.lineWidth = px(2); c.lineCap = 'butt';
    c.beginPath(); c.moveTo(bx, by - px(4)); c.lineTo(bx, by); c.lineTo(bx + m * sc, by); c.lineTo(bx + m * sc, by - px(4)); c.stroke();
    c.fillStyle = T.fg; c.font = px(11) + 'px ' + T.font; c.textAlign = 'left';
    c.fillText(m + ' m', bx + m * sc + px(6), by - px(2));

    // ---- orientation label (top left) and, heading-up, where the entry direction is
    c.textAlign = 'left'; c.fillStyle = T.muted;
    c.fillText(heading ? 'Heading up' : 'Entry direction up', px(10), px(12));
    if (heading) {
      const ax = W - px(22), ay = px(22), ang = -rot;
      c.save(); c.translate(ax, ay); c.rotate(-ang);
      c.fillStyle = T.muted; c.beginPath(); c.moveTo(0, -px(11)); c.lineTo(px(6), px(7)); c.lineTo(0, px(3)); c.lineTo(-px(6), px(7)); c.closePath(); c.fill();
      c.restore();
    }
    if (demo) {
      c.textAlign = 'right'; c.fillStyle = T.muted;
      c.fillText('dashed: true path', W - px(10), H - px(12));
    }
  };

  /* ================================================================ arrow */
  /**
   * new PSNav.ArrowWidget(canvas, {theme}); draw(g, {blink}) - the device's
   * eyepiece ring: arrow toward the next return point (relative to facing),
   * white dot = straight line to the door, distance and confidence level.
   * UNRELIABLE: no arrow, "FOLLOW HOSE" (as the device does).
   */
  function ArrowWidget(canvas, opts) {
    this.canvas = canvas; this.ctx = canvas.getContext('2d');
    this.o = Object.assign({ theme: null }, opts || {});
    this._themeAge = 1e9;
  }
  ArrowWidget.prototype.refreshTheme = function () { this._themeAge = 1e9; };
  ArrowWidget.prototype.draw = function (g, opt) {
    opt = opt || {};
    const { w: W, h: H } = fitCanvas(this.canvas);
    if (this._themeAge++ > 30 || !this.theme) { this.theme = this.o.theme ? Object.assign({}, DEFAULT_THEME, this.o.theme) : themeFromCSS(this.canvas); this._themeAge = 0; }
    const T = this.theme, c = this.ctx;
    c.setTransform(1, 0, 0, 1, 0, 0);
    c.clearRect(0, 0, W, H);
    const s = Math.min(W, H), cx = W / 2, cy = H / 2 - s * 0.06, r = s * 0.36;
    const col = !g || !g.valid ? T.muted : g.levelId === 0 ? T.good : g.levelId === 1 ? T.warn : T.bad;
    c.textAlign = 'center'; c.textBaseline = 'middle';
    const font = (k, bold) => (bold ? 'bold ' : '') + Math.round(s * k) + 'px ' + T.font;
    // ring
    c.lineWidth = s * 0.045; c.strokeStyle = col;
    c.beginPath(); c.arc(cx, cy, r, 0, 2 * Math.PI); c.stroke();
    if (!g || !g.valid) {
      c.fillStyle = T.fg; c.font = font(0.11, true);
      c.fillText('MARK', cx, cy - s * 0.07); c.fillText('ENTRY', cx, cy + s * 0.07);
      return;
    }
    const bottom = (txt) => { c.fillStyle = col; c.font = font(0.1, true); c.fillText(txt, cx, cy + r + s * 0.09); };
    if (g.levelId === 2) {
      const blink = opt.blink === undefined ? (Date.now() / 500 | 0) % 2 === 0 : opt.blink;
      c.fillStyle = T.bad; c.font = font(0.105, true);
      if (blink) { c.fillText('FOLLOW', cx, cy - s * 0.065); c.fillText('HOSE', cx, cy + s * 0.065); }
      bottom('NAV ±' + Math.round(g.posSigmaM) + ' M');
      return;
    }
    if (g.routeDistM < 1.0) {
      c.fillStyle = col; c.font = font(0.13, true); c.fillText('EXIT', cx, cy);
    } else {
      // arrow pointing up = ahead; + bearing (left) turns it counter-clockwise
      c.save(); c.translate(cx, cy); c.rotate(-g.routeBearingDeg * D2R);
      const L = r * 0.8;
      c.beginPath();
      c.moveTo(0, -L); c.lineTo(L * 0.55, -L * 0.1); c.lineTo(L * 0.2, -L * 0.1); c.lineTo(L * 0.2, L * 0.75);
      c.lineTo(-L * 0.2, L * 0.75); c.lineTo(-L * 0.2, -L * 0.1); c.lineTo(-L * 0.55, -L * 0.1); c.closePath();
      c.fillStyle = col; c.fill(); c.lineWidth = s * 0.012; c.strokeStyle = T.ring; c.stroke();
      c.restore();
    }
    // straight line to the door: white dot on the ring
    const a = g.homeBearingDeg * D2R;
    const dx = cx - Math.sin(a) * r, dy = cy - Math.cos(a) * r;
    c.beginPath(); c.arc(dx, dy, s * 0.032, 0, 2 * Math.PI);
    c.fillStyle = T.me; c.fill(); c.lineWidth = s * 0.012; c.strokeStyle = T.ring; c.stroke();
    bottom(Math.round(g.routeDistM) + ' M');
    if (g.state === 'lost') { c.fillStyle = T.bad; c.font = font(0.075, true); c.fillText('NO MOTION DATA', cx, cy - r - s * 0.07); }
  };

  /* ================================================================ EXIT box */
  /**
   * Where the green EXIT box goes in the camera picture (device draw_exit()).
   *
   * opts: hfovDeg (65; across the picture WIDTH, or across the longer side with
 *       fovOnLongSide: true as motion.js assumes), mirrored (false),
   *       cameraLooksBack (default = mirrored), width, height (px of the picture,
   *       optional), cameraHeightM 1.5, doorWidthM 0.9, doorHeightM 2.0, marginDeg 5.
   *
   * Sign conventions (bearing b, + = to the walker's LEFT):
   *   - back camera / eyepiece (looks where the walker faces), unmirrored:
   *       b > 0 -> left half: x = W/2 - f tan(b), f = (W/2) / tan(hfov/2)
   *       (identical to ps_display.c draw_exit()).
   *   - front (selfie) camera: it looks BACK at the walker, so its axis is the
   *     facing + 180 deg (cameraLooksBack) and the preview is mirrored. The
   *     bearing relative to the camera is bc = wrap(b - 180); raw image
   *     x = W/2 - f tan(bc); mirrored display x' = W - x. Net effect: a door
   *     behind you on your left shows on the LEFT of the mirrored preview, as
   *     in a mirror, and the edge arrow for a door ahead-left is on the left.
   *   - mirrored but the camera looks forward (cameraLooksBack: false): x' = W/2 + f tan(b).
   * Returns
   *   {kind: 'box', x, y, w, h (0..1 of the picture), px: {x, y, w, h} | null, label, distM, bearingDeg}
   *   {kind: 'route', x (0..1), label 'OUT 12M', ...}  - the door is not the next point yet but the
 *       next leg of the way out is in the picture at x: draw a green chevron/arrow there
 *   {kind: 'edge', side: 'left' | 'right', behind (bool), label, distM, bearingDeg}
 *       - the way out is outside the picture: point an arrow at that edge. behind is in the
 *         WALKER's frame (|b| > 90 deg, "turn round"), also for a camera that looks back at them.
   *   {kind: 'none', reason}  - nothing to draw (no entry, unreliable, at the door)
   * The box needs the doorway to be the next point on the way out (exitIsNext),
   * like the device; otherwise 'edge' points along the route's next leg.
   */
  function exitBox(g, opts) {
    const o = Object.assign({ hfovDeg: 65, fovOnLongSide: false, mirrored: false, cameraLooksBack: undefined, width: 0, height: 0,
      cameraHeightM: 1.5, doorWidthM: 0.9, doorHeightM: 2.0, marginDeg: 5 }, opts || {});
    const back = o.cameraLooksBack === undefined ? !!o.mirrored : !!o.cameraLooksBack;
    if (!g || !g.valid || g.state === 'idle') return { kind: 'none', reason: 'no entry marked' };
    if (g.levelId === 2) return { kind: 'none', reason: 'unreliable: follow the hose' };
    const useHome = g.exitIsNext;
    const b = useHome ? g.homeBearingDeg : g.routeBearingDeg;
    const dist = useHome ? g.homeDistM : g.routeDistM;
    if (useHome && g.homeDistM < 1.0) return { kind: 'none', reason: 'at the door' };
    const label = 'EXIT ' + Math.round(dist) + 'M';
    const bc = PSNav.wrap180(b - (back ? 180 : 0));            // bearing from the camera axis, + = camera's left
    const aspect = o.width && o.height ? o.height / o.width : 0.75;
    // focal length in picture widths; fovOnLongSide: hfovDeg spans the longer side (motion.js convention)
    const f = (o.fovOnLongSide ? Math.max(1, aspect) : 1) * 0.5 / Math.tan(o.hfovDeg / 2 * D2R);
    const half = Math.atan(0.5 / f) / D2R;                    // horizontal half field of view
    const flip = (side) => (o.mirrored ? (side === 'left' ? 'right' : 'left') : side);
    if (!useHome && Math.abs(bc) <= half) {
      // the next leg of the way out (not the door yet) is in the picture at x (0..1)
      let x = 0.5 - f * Math.tan(bc * D2R);
      if (o.mirrored) x = 1 - x;
      return { kind: 'route', x, label: 'OUT ' + Math.round(dist) + 'M', distM: dist, bearingDeg: b, cameraBearingDeg: bc };
    }
    if (Math.abs(bc) > half + o.marginDeg || !useHome) {
      // behind: relative to the walker, not the camera (a selfie camera looks back at them, so a
      // door straight ahead is "behind" the camera but must not be labelled BEHIND)
      return { kind: 'edge', side: flip(bc >= 0 ? 'left' : 'right'), behind: Math.abs(PSNav.wrap180(b)) > 90,
        label: useHome ? label : 'OUT ' + Math.round(dist) + 'M', distM: dist, bearingDeg: b, cameraBearingDeg: bc };
    }
    // device draw_exit() geometry, in units of the picture width
    const a = bc * D2R;
    let depth = g.homeDistM * Math.cos(a);
    if (depth < 0.5) depth = 0.5;
    let cx = 0.5 - f * Math.tan(a);
    let w = f * o.doorWidthM / depth;
    let top = aspect * 0.5 - f * (o.doorHeightM - o.cameraHeightM) / depth;
    let bot = aspect * 0.5 + f * o.cameraHeightM / depth;
    w = Math.max(w, 0.025);
    if (bot - top < 0.05) { const m = (top + bot) / 2; top = m - 0.025; bot = m + 0.025; }
    if (o.mirrored) cx = 1 - cx;
    const box = { kind: 'box', x: cx - w / 2, y: top / aspect, w, h: (bot - top) / aspect, label, distM: g.homeDistM, bearingDeg: b, cameraBearingDeg: bc };
    box.px = o.width && o.height ? { x: box.x * o.width, y: box.y * o.height, w: box.w * o.width, h: box.h * o.height } : null;
    return box;
  }

  PSNav.MapRenderer = MapRenderer;
  PSNav.ArrowWidget = ArrowWidget;
  PSNav.exitBox = exitBox;
  PSNav.themeFromCSS = themeFromCSS;
  PSNav.DEFAULT_THEME = DEFAULT_THEME;
})(typeof globalThis !== 'undefined' ? globalThis : this);
