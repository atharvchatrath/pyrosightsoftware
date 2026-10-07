/*
 * FireDoorNet decoder (plain JS, no dependencies). Mirrors firedoor/decode.py exactly.
 *
 * Model outputs (ONNX names heat, wh, off), Float32Arrays of length C * 32 * 40 (heat) and 2 * 32 * 40:
 *   heat : sigmoid probability, channel 0 = fire, 1 = door, 2 = window (C = 3; the older 2-class
 *          model has no window channel: C is read from the length of heat)
 *   wh   : channel 0 = box width, 1 = box height, in model-input pixels (input is 320 x 256)
 *   off  : channel 0 = x offset, 1 = y offset of the box centre inside its grid cell, [0, 1)
 * Default layout is NCHW (what ONNX / onnxruntime produce):  index = c*32*40 + y*40 + x.
 * Pass {layout: 'NHWC'} if your runtime hands back channels-last data: index = (y*40 + x)*C + c.
 *
 * The window model also outputs wh_w and off_w (same layout as wh, off): the size and offset of WINDOW
 * boxes. Pass them as {windowWh: wh_w, windowOff: off_w}; fire and door boxes always use wh, off.
 *
 * Returns [{cls: 'fire'|'door'|'window', score, x, y, w, h}] sorted by score (descending), where x, y is the
 * top-left corner and w, h the size, all normalised to [0, 1] of the frame that was fed to the model
 * (the WHOLE video frame stretched to 320 x 256) -> multiply by video width / height to draw.
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.FireDoorDecode = factory();
}(typeof self !== 'undefined' ? self : this, function () {
  'use strict';
  var CLASS_NAMES = ['fire', 'door', 'window'];
  var DEFAULTS = {
    inW: 320, inH: 256, stride: 8, gridW: 40, gridH: 32, numClasses: 3,
    thresholds: { fire: 0.50, door: 0.35, window: 0.36 },   // operating points, see MODEL.md (the camera page passes its own)
    nmsIou: 0.45, maxDet: 50, layout: 'NCHW', minScore: null, windowWh: null, windowOff: null
  };

  function iou(a, b) {
    var x0 = Math.max(a.x, b.x), y0 = Math.max(a.y, b.y);
    var x1 = Math.min(a.x + a.w, b.x + b.w), y1 = Math.min(a.y + a.h, b.y + b.h);
    var iw = x1 - x0, ih = y1 - y0;
    if (iw <= 0 || ih <= 0) return 0;
    var inter = iw * ih;
    return inter / (a.w * a.h + b.w * b.h - inter);
  }

  function decode(heat, wh, off, opts) {
    var o = {}, k;
    for (k in DEFAULTS) o[k] = DEFAULTS[k];
    if (opts) for (k in opts) if (opts[k] !== undefined) o[k] = opts[k];
    var thr = {};
    for (k in DEFAULTS.thresholds) thr[k] = DEFAULTS.thresholds[k];
    if (opts && opts.thresholds) for (k in opts.thresholds) thr[k] = opts.thresholds[k];
    var GW = o.gridW, GH = o.gridH, plane = GW * GH, nhwc = o.layout === 'NHWC';
    var C = heat.length % plane === 0 && heat.length >= plane ? Math.min(heat.length / plane, CLASS_NAMES.length) : o.numClasses;
    var CH = heat.length / plane;                // channels in the heat array itself (its NHWC stride)
    function at(arr, c, y, x) { return nhwc ? arr[(y * GW + x) * CH + c] : arr[c * plane + y * GW + x]; }
    function at2(arr, c, y, x) { return nhwc ? arr[(y * GW + x) * 2 + c] : arr[c * plane + y * GW + x]; }

    var dets = [];
    for (var c = 0; c < C; c++) {
      var t = (o.minScore !== null && o.minScore !== undefined) ? o.minScore : thr[CLASS_NAMES[c]];
      var own = CLASS_NAMES[c] === 'window' && o.windowWh && o.windowOff;
      var cwh = own ? o.windowWh : wh, coff = own ? o.windowOff : off;
      for (var gy = 0; gy < GH; gy++) {
        for (var gx = 0; gx < GW; gx++) {
          var v = at(heat, c, gy, gx);
          if (!(v >= t)) continue;
          var peak = true;                       // 3x3 local maximum (ties allowed), edges ignored
          for (var dy = -1; dy <= 1 && peak; dy++) {
            var yy = gy + dy;
            if (yy < 0 || yy >= GH) continue;
            for (var dx = -1; dx <= 1; dx++) {
              var xx = gx + dx;
              if (xx < 0 || xx >= GW) continue;
              if (at(heat, c, yy, xx) > v) { peak = false; break; }
            }
          }
          if (!peak) continue;
          var w = at2(cwh, 0, gy, gx), h = at2(cwh, 1, gy, gx);
          if (w < 1 || h < 1) continue;
          var cx = (gx + at2(coff, 0, gy, gx)) * o.stride;
          var cy = (gy + at2(coff, 1, gy, gx)) * o.stride;
          var x0 = Math.max(0, cx - w / 2), y0 = Math.max(0, cy - h / 2);
          var x1 = Math.min(o.inW, cx + w / 2), y1 = Math.min(o.inH, cy + h / 2);
          dets.push({ cls: CLASS_NAMES[c], score: v, x: x0 / o.inW, y: y0 / o.inH,
                      w: (x1 - x0) / o.inW, h: (y1 - y0) / o.inH });
        }
      }
    }
    dets.sort(function (a, b) { return b.score - a.score; });   // stable (ES2019+)
    var keep = [];
    for (var i = 0; i < dets.length && keep.length < o.maxDet; i++) {
      var d = dets[i], ok = true;
      for (var j = 0; j < keep.length; j++) {
        if (keep[j].cls === d.cls && iou(keep[j], d) > o.nmsIou) { ok = false; break; }
      }
      if (ok) keep.push(d);
    }
    return keep;
  }

  /* Fill a Float32Array [1,3,256,320] (NCHW, RGB, x = v/127.5 - 1) from RGBA pixels of a canvas
   * that already holds the whole video frame drawn (stretched) at 320 x 256. */
  function preprocessRGBA(rgba, inW, inH, out) {
    inW = inW || 320; inH = inH || 256;
    var n = inW * inH;
    out = out || new Float32Array(3 * n);
    for (var i = 0; i < n; i++) {
      out[i] = rgba[4 * i] / 127.5 - 1;
      out[n + i] = rgba[4 * i + 1] / 127.5 - 1;
      out[2 * n + i] = rgba[4 * i + 2] / 127.5 - 1;
    }
    return out;
  }

  return { decode: decode, iou: iou, preprocessRGBA: preprocessRGBA, CLASS_NAMES: CLASS_NAMES, DEFAULTS: DEFAULTS };
}));
