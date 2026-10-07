/*
 * PyroSight op-list runtime (oplist.js)
 *
 * Executes a model exported by export_oplist.py (ONNX or TFLite source) with
 * TF.js. Plain script, no bundler: defines globalThis.PSOpList. Uses only ops
 * that TF.js implements on both the WebGL and the CPU backend. See FORMAT.md.
 *
 *   const model = PSOpList.load(json, weightsArrayBuffer, {tf});
 *   const out = model.run(inputNHWC);          // {outputName: tf.Tensor}
 *   ...; tf.dispose(Object.values(out));
 *   model.dispose();                            // frees the weight tensors
 *
 * Everything run() creates is released inside tf.tidy except the returned
 * outputs; intermediates are also disposed right after their last use to
 * keep peak memory low.
 */
(function (root) {
  'use strict';

  // ---------------------------------------------------------------- utils
  let F16_LUT = null;
  function f16Table() {
    if (F16_LUT) return F16_LUT;
    F16_LUT = new Float32Array(65536);
    for (let h = 0; h < 65536; h++) {
      const s = (h & 0x8000) ? -1 : 1;
      const e = (h >> 10) & 0x1f;
      const f = h & 0x3ff;
      let v;
      if (e === 0) v = f * Math.pow(2, -24);
      else if (e === 31) v = f ? NaN : Infinity;
      else v = (1 + f / 1024) * Math.pow(2, e - 15);
      F16_LUT[h] = s * v;
    }
    return F16_LUT;
  }

  function toU8(buf) {
    let u8;
    if (buf instanceof ArrayBuffer) u8 = new Uint8Array(buf);
    else if (ArrayBuffer.isView(buf)) u8 = new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength);
    else throw new Error('PSOpList: weights must be an ArrayBuffer or typed array');
    if (u8.byteOffset % 4) u8 = u8.slice();   // keep 2/4-byte views aligned
    return u8;
  }

  function base64ToArrayBuffer(b64) {
    if (typeof atob === 'function') {
      const bin = atob(b64);
      const n = bin.length;
      const out = new Uint8Array(n);
      for (let i = 0; i < n; i++) out[i] = bin.charCodeAt(i);
      return out.buffer;
    }
    const b = Buffer.from(b64, 'base64');   // node
    return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
  }

  /** base64ToArrayBuffer in slices, awaiting yieldFn() between them (cooperative main-thread loading). */
  async function base64ToArrayBufferAsync(b64, yieldFn) {
    if (typeof atob !== 'function' || !yieldFn) return base64ToArrayBuffer(b64);
    const clean = b64.replace(/[^A-Za-z0-9+/=]/g, '');
    const pad = clean.endsWith('==') ? 2 : clean.endsWith('=') ? 1 : 0;
    const out = new Uint8Array(clean.length / 4 * 3 - pad);
    const STEP = 1 << 18;      // 256 K base64 chars -> 192 KB per slice
    let o = 0;
    for (let i = 0; i < clean.length; i += STEP) {
      const bin = atob(clean.slice(i, i + STEP));
      for (let j = 0; j < bin.length; j++) out[o++] = bin.charCodeAt(j);
      await yieldFn();
    }
    return out.buffer;
  }

  /** Decode one weight spec into a Float32Array / Int32Array. */
  function decodeWeight(spec, u8) {
    const n = spec.shape.reduce((a, b) => a * b, 1);
    const off = u8.byteOffset + spec.offset;
    const ab = u8.buffer;
    switch (spec.dtype) {
      case 'float32':
        return new Float32Array(ab.slice(off, off + 4 * n));
      case 'int32':
        return new Int32Array(ab.slice(off, off + 4 * n));
      case 'float16': {
        const h = new Uint16Array(ab, off, n);
        const lut = f16Table();
        const out = new Float32Array(n);
        for (let i = 0; i < n; i++) out[i] = lut[h[i]];
        return out;
      }
      case 'uint8': {
        const q = new Uint8Array(ab, off, n);
        const qp = spec.quant;
        const ch = qp.channels;
        const mn = new Float32Array(ab, u8.byteOffset + qp.min_offset, ch);
        const sc = new Float32Array(ab, u8.byteOffset + qp.scale_offset, ch);
        const out = new Float32Array(n);
        for (let i = 0, c = 0; i < n; i++) {
          out[i] = mn[c] + q[i] * sc[c];
          if (++c === ch) c = 0;
        }
        return out;
      }
      default:
        throw new Error('PSOpList: unknown weight dtype ' + spec.dtype);
    }
  }

  function decodeWeights(specs, weights) {
    const u8 = toU8(weights);
    const out = {};
    for (const s of specs) out[s.name] = decodeWeight(s, u8);
    return out;
  }

  function samePads(inSize, k, s, d) {
    const ek = (k - 1) * d + 1;
    const out = Math.ceil(inSize / s);
    const total = Math.max((out - 1) * s + ek - inSize, 0);
    return [Math.floor(total / 2), total - Math.floor(total / 2)];
  }

  /**
   * Resolve a pad attribute against the actual input size.
   * Returns {mode:'same'|'valid'} or {mode:'explicit', p:[t,b,l,r]}.
   */
  function resolvePad(pad, x, k, strides, dil) {
    if (pad === 'same' || pad === 'valid') return { mode: pad };
    const [t, b, l, r] = pad;
    if (t === 0 && b === 0 && l === 0 && r === 0) return { mode: 'valid' };
    const H = x.shape[1], W = x.shape[2];
    const ph = samePads(H, k[0], strides[0], dil[0]);
    const pw = samePads(W, k[1], strides[1], dil[1]);
    if (ph[0] === t && ph[1] === b && pw[0] === l && pw[1] === r) return { mode: 'same' };
    return { mode: 'explicit', p: [t, b, l, r] };
  }

  // ---------------------------------------------------------------- ops
  function makeOps(tf) {
    const fusedAct = (a) => (a && a !== 'linear' ? a : 'linear');
    const postAct = (y, a) => {
      if (!a || a === 'linear') return y;
      if (a === 'relu') return tf.relu(y);
      if (a === 'relu6') return tf.relu6(y);
      if (a === 'sigmoid') return tf.sigmoid(y);
      throw new Error('PSOpList: activation ' + a);
    };

    return {
      conv2d(op, [x, w, b], m) {
        const k = [w.shape[0], w.shape[1]];
        const dil = op.dilations || [1, 1];
        const pr = resolvePad(op.pad, x, k, op.strides, dil);
        let pad = pr.mode;
        if (pr.mode === 'explicit') {
          const p = pr.p;
          x = tf.pad(x, [[0, 0], [p[0], p[1]], [p[2], p[3]], [0, 0]]);
          pad = 'valid';
        }
        const g = op.groups || 1;
        if (g === 1) {
          return tf.fused.conv2d({ x, filter: w, strides: op.strides, pad, dilations: dil,
            bias: b, activation: fusedAct(op.act) });
        }
        // grouped convolution: per-group filters are pre-split at load time
        const ws = m.groupFilters(op, w, g);
        const bs = b ? tf.split(b, g, 0) : new Array(g).fill(undefined);
        const xs = tf.split(x, g, 3);
        const ys = xs.map((xi, i) => tf.fused.conv2d({ x: xi, filter: ws[i], strides: op.strides, pad,
          dilations: dil, bias: bs[i], activation: fusedAct(op.act) }));
        return tf.concat(ys, 3);
      },
      depthwise_conv2d(op, [x, w, b]) {
        const k = [w.shape[0], w.shape[1]];
        const dil = op.dilations || [1, 1];
        const pr = resolvePad(op.pad, x, k, op.strides, dil);
        let pad = pr.mode;
        if (pr.mode === 'explicit') {
          const p = pr.p;
          x = tf.pad(x, [[0, 0], [p[0], p[1]], [p[2], p[3]], [0, 0]]);
          pad = 'valid';
        }
        return tf.fused.depthwiseConv2d({ x, filter: w, strides: op.strides, pad, dilations: dil,
          bias: b, activation: fusedAct(op.act) });
      },
      dense(op, [x, w, b]) {
        return tf.fused.matMul({ a: x, b: w, bias: b, activation: fusedAct(op.act) });
      },
      matmul(op, [a, b]) { return tf.matMul(a, b, false, !!op.transpose_b); },
      add(op, [a, b]) { return tf.add(a, b); },
      sub(op, [a, b]) { return tf.sub(a, b); },
      mul(op, [a, b]) { return tf.mul(a, b); },
      div(op, [a, b]) { return tf.div(a, b); },
      pow(op, [a, b]) { return tf.pow(a, b); },
      maximum(op, [a, b]) { return tf.maximum(a, b); },
      minimum(op, [a, b]) { return tf.minimum(a, b); },
      relu(op, [x]) { return tf.relu(x); },
      relu6(op, [x]) { return tf.relu6(x); },
      sigmoid(op, [x]) { return tf.sigmoid(x); },
      tanh(op, [x]) { return tf.tanh(x); },
      elu(op, [x]) { return tf.elu(x); },
      exp(op, [x]) { return tf.exp(x); },
      neg(op, [x]) { return tf.neg(x); },
      abs(op, [x]) { return tf.abs(x); },
      sqrt(op, [x]) { return tf.sqrt(x); },
      leaky_relu(op, [x]) { return tf.leakyRelu(x, op.alpha); },
      clip(op, [x]) { return tf.clipByValue(x, op.min, op.max); },
      hard_sigmoid(op, [x]) { return tf.clipByValue(tf.add(tf.mul(x, op.alpha), op.beta), 0, 1); },
      hard_swish(op, [x]) { return tf.mul(x, tf.div(tf.relu6(tf.add(x, 3)), 6)); },
      cast_float(op, [x]) { return tf.cast(x, 'float32'); },
      identity(op, [x]) { return x.clone(); },
      concat(op, xs) { return tf.concat(xs, op.axis); },
      reshape(op, [x]) { return tf.reshape(x, op.shape); },
      tile(op, [x]) { return tf.tile(x, op.reps); },
      transpose(op, [x]) { return tf.transpose(x, op.perm); },
      slice(op, [x]) { return tf.slice(x, op.begin, op.size); },
      pad(op, [x]) {
        if (op.mode === 'reflect') return tf.mirrorPad(x, op.pads, 'reflect');
        return tf.pad(x, op.pads, op.value || 0);
      },
      mean(op, [x]) { return tf.mean(x, op.axes, !!op.keepdims); },
      max(op, [x]) { return tf.max(x, op.axes, !!op.keepdims); },
      sum(op, [x]) { return tf.sum(x, op.axes, !!op.keepdims); },
      softmax(op, [x]) {
        const r = x.shape.length;
        const ax = ((op.axis % r) + r) % r;
        let y = op.beta && op.beta !== 1 ? tf.mul(x, op.beta) : x;
        if (ax === r - 1) return tf.softmax(y);
        const perm = [...Array(r).keys()];
        perm[ax] = r - 1; perm[r - 1] = ax;
        return tf.transpose(tf.softmax(tf.transpose(y, perm)), perm);
      },
      resize(op, [x]) {
        if (op.mode === 'nearest') {
          return tf.image.resizeNearestNeighbor(x, op.size, !!op.align_corners, !!op.half_pixel);
        }
        return tf.image.resizeBilinear(x, op.size, !!op.align_corners, !!op.half_pixel);
      },
      max_pool(op, [x]) { return pool(tf, 'max', op, x); },
      avg_pool(op, [x]) { return pool(tf, 'avg', op, x); },
    };
  }

  function pool(tf, kind, op, x) {
    const pr = resolvePad(op.pad, x, op.k, op.strides, [1, 1]);
    let pad = pr.mode;
    if (pr.mode === 'explicit') {
      const p = pr.p;
      if (kind === 'avg' && op.count_include_pad) {
        x = tf.pad(x, [[0, 0], [p[0], p[1]], [p[2], p[3]], [0, 0]]);
        pad = 'valid';
      } else {
        // TF.js pooling skips padded cells: max ignores them, avg excludes them
        pad = [[0, 0], [p[0], p[1]], [p[2], p[3]], [0, 0]];
      }
    } else if (pr.mode === 'same' && kind === 'avg' && op.count_include_pad) {
      const ph = samePads(x.shape[1], op.k[0], op.strides[0], 1);
      const pw = samePads(x.shape[2], op.k[1], op.strides[1], 1);
      x = tf.pad(x, [[0, 0], ph, pw, [0, 0]]);
      pad = 'valid';
    }
    return kind === 'max' ? tf.maxPool(x, op.k, op.strides, pad) : tf.avgPool(x, op.k, op.strides, pad);
  }

  // ---------------------------------------------------------------- model
  function load(json, weights, opts) {
    const p = prepare(json, weights, opts);
    for (const spec of p.json.weights) p.W[spec.name] = p.tf.tensor(decodeWeight(spec, p.u8), spec.shape, spec.dtype === 'int32' ? 'int32' : 'float32');
    return build(p.json, p.W, p.tf);
  }

  /** load() with `await opts.yieldFn()` after each weight tensor (cooperative main-thread loading). */
  async function loadAsync(json, weights, opts) {
    const p = prepare(json, weights, opts);
    const yieldFn = opts && opts.yieldFn;
    try {
      for (const spec of p.json.weights) {
        p.W[spec.name] = p.tf.tensor(decodeWeight(spec, p.u8), spec.shape, spec.dtype === 'int32' ? 'int32' : 'float32');
        if (yieldFn) await yieldFn();
      }
    } catch (e) {
      for (const k in p.W) p.W[k].dispose();
      throw e;
    }
    return build(p.json, p.W, p.tf);
  }

  function prepare(json, weights, opts) {
    opts = opts || {};
    const tf = opts.tf || root.tf;
    if (!tf) throw new Error('PSOpList.load: TF.js not found (pass {tf})');
    if (typeof json === 'string') json = JSON.parse(json);
    if (json.format !== 'pyrosight-oplist') throw new Error('PSOpList.load: not a pyrosight op list');
    if (json.version !== 1) throw new Error('PSOpList.load: unsupported version ' + json.version);
    const u8 = toU8(weights);
    if (u8.byteLength < json.weights_bytes) throw new Error('PSOpList.load: weights blob too short');
    return { json, u8, tf, W: {} };
  }

  function build(json, W, tf) {
    const OPS = makeOps(tf);
    for (const op of json.ops) {
      if (!OPS[op.op]) throw new Error('PSOpList.load: unsupported op ' + op.op);
    }
    const inputNames = json.inputs.map((i) => i.name);
    const outputNames = json.outputs.map((o) => o.name);

    // last use of every intermediate, for early disposal
    const lastUse = {};
    json.ops.forEach((op, i) => op.inputs.forEach((t) => { lastUse[t] = i; }));
    const groupCache = new Map();

    const model = {
      json, meta: json.meta || {}, inputs: json.inputs, outputs: json.outputs, weights: W, tf,
      groupFilters(op, w, g) {
        let s = groupCache.get(w.id);
        if (!s) {
          s = tf.split(w, g, 3).map((t) => tf.keep(t));   // survive run()'s tidy
          groupCache.set(w.id, s);
        }
        return s;
      },
      /**
       * run(input | {name: tensor}, {outputs?: [names]}) -> {name: tf.Tensor}
       * The input tensor is not disposed. Returned tensors belong to the caller.
       */
      run(inputs, ropts) {
        ropts = ropts || {};
        const want = ropts.outputs || outputNames;
        if (inputs instanceof tf.Tensor) {
          if (inputNames.length !== 1) throw new Error('PSOpList.run: model has several inputs');
          inputs = { [inputNames[0]]: inputs };
        }
        for (const n of inputNames) {
          if (!inputs[n]) throw new Error('PSOpList.run: missing input ' + n);
        }
        const keepSet = new Set(want);
        return tf.tidy(() => {
          const env = Object.assign({}, W);
          for (const n of inputNames) {
            let t = inputs[n];
            if (t.dtype !== 'float32') t = tf.cast(t, 'float32');
            env[n] = t;
          }
          const ops = json.ops;
          for (let i = 0; i < ops.length; i++) {
            const op = ops[i];
            const args = op.inputs.map((t) => {
              const v = env[t];
              if (v === undefined) throw new Error('PSOpList: tensor ' + t + ' not computed (op ' + i + ')');
              return v;
            });
            const y = OPS[op.op](op, args, model);
            env[op.outputs[0]] = y;
            for (const t of op.inputs) {
              if (lastUse[t] === i && !(t in W) && inputNames.indexOf(t) < 0 && !keepSet.has(t)) {
                const v = env[t];
                if (v && !v.isDisposed && v !== y) v.dispose();
                delete env[t];
              }
            }
          }
          const out = {};
          for (const n of want) {
            if (!env[n]) throw new Error('PSOpList.run: unknown output ' + n);
            out[n] = env[n];
          }
          return out;
        });
      },
      /**
       * runAsync(input | {name: tensor}, {outputs?, yieldFn}) -> Promise<{name: tf.Tensor}>
       * Same result as run(), op by op with `await yieldFn()` in between (yieldFn decides whether
       * to give the event loop a turn), so a page that runs the model on its main thread stays
       * responsive. Each op runs in its own tf.tidy; nothing leaks if an op throws.
       */
      async runAsync(inputs, ropts) {
        ropts = ropts || {};
        const yieldFn = ropts.yieldFn;
        if (!yieldFn) return this.run(inputs, ropts);
        const want = ropts.outputs || outputNames;
        if (inputs instanceof tf.Tensor) {
          if (inputNames.length !== 1) throw new Error('PSOpList.run: model has several inputs');
          inputs = { [inputNames[0]]: inputs };
        }
        for (const n of inputNames) {
          if (!inputs[n]) throw new Error('PSOpList.run: missing input ' + n);
        }
        const keepSet = new Set(want);
        const env = Object.assign({}, W);
        const owned = new Set();      // tensors made here (casts, op outputs): disposed unless returned
        try {
          for (const n of inputNames) {
            let t = inputs[n];
            if (t.dtype !== 'float32') { t = tf.cast(t, 'float32'); owned.add(t); }
            env[n] = t;
          }
          const ops = json.ops;
          for (let i = 0; i < ops.length; i++) {
            const op = ops[i];
            const args = op.inputs.map((t) => {
              const v = env[t];
              if (v === undefined) throw new Error('PSOpList: tensor ' + t + ' not computed (op ' + i + ')');
              return v;
            });
            const y = tf.tidy(() => OPS[op.op](op, args, model));
            env[op.outputs[0]] = y;
            if (!args.includes(y) && !(op.outputs[0] in W)) owned.add(y);
            for (const t of op.inputs) {
              if (lastUse[t] === i && !(t in W) && inputNames.indexOf(t) < 0 && !keepSet.has(t)) {
                const v = env[t];
                if (v && !v.isDisposed && v !== y && owned.has(v)) { v.dispose(); owned.delete(v); }
                delete env[t];
              }
            }
            await yieldFn();
          }
          const out = {};
          for (const n of want) {
            if (!env[n]) throw new Error('PSOpList.run: unknown output ' + n);
            out[n] = env[n];
            owned.delete(env[n]);
          }
          return out;
        } finally {
          for (const t of owned) { if (!t.isDisposed) t.dispose(); }
        }
      },
      dispose() {
        for (const k in W) W[k].dispose();
        for (const s of groupCache.values()) s.forEach((t) => t.dispose());
        groupCache.clear();
      },
    };
    return model;
  }

  /** Convenience for embedded assets: {json, weightsB64} (or {json, weights: ArrayBuffer}). */
  function loadEmbedded(asset, opts) {
    const w = asset.weights ? asset.weights : base64ToArrayBuffer(asset.weightsB64);
    return load(asset.json, w, opts);
  }

  /** loadEmbedded() that yields while decoding and uploading (opts.yieldFn). */
  async function loadEmbeddedAsync(asset, opts) {
    const y = opts && opts.yieldFn;
    const w = asset.weights ? asset.weights : await base64ToArrayBufferAsync(asset.weightsB64, y);
    return loadAsync(asset.json, w, opts);
  }

  root.PSOpList = {
    version: 1,
    load,
    loadAsync,
    loadEmbedded,
    loadEmbeddedAsync,
    decodeWeights,
    decodeWeight: (spec, buf) => decodeWeight(spec, toU8(buf)),
    base64ToArrayBuffer,
    base64ToArrayBufferAsync,
    samePads,
  };
})(typeof globalThis !== 'undefined' ? globalThis : this);
