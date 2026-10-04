# PyroSight Camera: browser model runtime + people/face detector

Everything here runs in the browser on TF.js (WebGL backend, CPU fallback),
with no network access at run time: the models are embedded as base64 in
`dist/people_assets.js`. Video frames never leave the device.

```
runtime/
  oplist.js            JS op-list runtime (global PSOpList)          <- inline in the page
  people.js            person + face detector (global PSPeople)      <- inline in the page
  dist/people_assets.js  embedded models (global PS_PEOPLE_ASSETS, 5.16 MB) <- inline in the page
  export_oplist.py     ONNX / TFLite -> op list (JSON + weights blob, optional .js with base64)
  repack_cocossd.py    COCO-SSD TF.js graph -> person-only, per-channel uint8, no preprocessor
  build_assets.py      rebuilds both models and writes dist/people_assets.js (+ --demo test page)
  FORMAT.md            op-list / packed-weights format
  LICENSES.md          licences of models, data and tools
  models/              build outputs (op lists, packed COCO-SSD)
  tests/               parity tests, corpus evaluation, browser test (see "Tests")
  tools/make_people_testset.py   builds ../testdata/people
```

## Page integration (exact interface)

Inline, in this order, as plain `<script>` blocks (none contains `</script`):

1. `camera/node_modules/@tensorflow/tfjs/dist/tf.min.js` (TF.js 4.22.0, global `tf`; contains the WebGL and CPU backends only, no wasm)
2. `camera/runtime/oplist.js` (global `PSOpList`)
3. `camera/runtime/people.js` (global `PSPeople`)
4. `camera/runtime/dist/people_assets.js` (global `PS_PEOPLE_ASSETS = {person, face, info}`)
5. the fire/door model, exported with `export_oplist.py --js` (global `PS_OPLIST_ASSETS[name]`)

```js
// once
const backend = await PSPeople.setupBackend(tf, { allowSoftwareWebGL: false }); // 'webgl' | 'cpu'
PSPeople.backendInfo;            // {backend, renderer, software}
const info = await PSPeople.init(tf, PS_PEOPLE_ASSETS, { hfovDeg: 65 });
// info = {backend, loadMs, person: true, face: true, personClasses: ['person']}

// every frame (await it; do not start a second detect() before the first resolves)
const r = await PSPeople.detect(videoOrCanvas, { mirror: false });
// r.display    -> draw these, WHITE (#FFFFFF), one box per person
// r.detections -> raw model boxes (persons and faces), for debugging / overlays
// r.width, r.height -> source size in px;  r.ms = {person, face, total}
```

`detect(source, opts)` accepts an `HTMLVideoElement` (returns an empty result
while `videoWidth === 0`), `HTMLCanvasElement`, `HTMLImageElement`,
`ImageData`, `ImageBitmap`, or a `tf.Tensor3D [H, W, 3]` RGB 0..255 (not
disposed). All coordinates are **normalised [0, 1] in the source image**,
`x, y` = top-left corner, `w, h` = size. With `mirror: true` x is flipped for
a mirrored selfie preview.

`r.display[i]`:

| field | meaning |
|-------|---------|
| `cls` | always `'person'` |
| `src` | `'person'` (COCO-SSD box), `'person+face'` (COCO-SSD box that contains a face; distance from the face), `'face'` (a face with no person box around it, shown as a person: face box padded 15 %) |
| `score` | 0..1 |
| `x, y, w, h` | normalised box |
| `dist` | metres, **estimate** (`distEstimate: true`); `distMin`, `distMax` a plausible range; `distSrc` `'face'` or `'height'` |
| `truncated` | box touches the frame edge (height-based distance is then an upper bound) |
| `label` | device-style text: `"1.2M"`, `"12M"`, `"<1.2M"` when truncated (same rules as `ps_display.c`) |
| `face` | the attached face detection, if any |

`r.detections[i]` = `{cls: 'person'|'face', score, x, y, w, h, dist, distMin,
distMax, distSrc, truncated, label}`; faces also carry `keypoints`: six
normalised `[x, y]` (MediaPipe order: right eye, left eye, nose tip, mouth
centre, right ear tragion, left ear tragion - "right" = the subject's).

Options (defaults in `PSPeople.DEFAULTS`, pass to `init` to change them for
all calls or to `detect` per call):

| option | default | |
|--------|---------|--|
| `personMinScore` | 0.4 | COCO-SSD person threshold (coco-ssd's own default is 0.5) |
| `faceMinScore` | 0.75 | a face shown on its own |
| `faceAttachMinScore` | 0.75 | a face that gives a person box its distance |
| `faceDecodeMinScore` | 0.5 | MediaPipe's anchor threshold before the weighted NMS |
| `personIoU`, `faceIoU` | 0.5, 0.3 | NMS IoU (coco-ssd, MediaPipe values) |
| `maxPersons`, `maxFaces` | 20, 10 | |
| `faceInPerson` | 0.6 | fraction of a face box inside a person box to merge them |
| `hfovDeg` | 65 | camera horizontal field of view used for distance |
| `faceWidthM`, `personHeightM` | 0.16, 1.7 | pinhole model sizes |
| `person`, `face` | true | switch a model off (e.g. person every Nth frame on CPU) |
| `mirror` | false | flip x |

Other calls: `PSPeople.merge(detections, W, H, opts)` (display list from raw
detections), `PSPeople.dispose()`.

Distance (pinhole, like the eyepiece firmware `ps_estimate_distances`):
`f_px = (W/2) / tan(hfov/2)`; face `d = 0.16 m * f_px / face_width_px`
(range x0.8..x1.2); person `d = 1.7 m * f_px / max(box_w_px, box_h_px)`
(range x0.55..x1.15, lower bound 0.3 m when truncated). A person box that
holds a face uses the face distance, which is far more reliable at webcam
range (a close person's box is cut by the frame). The hfov of webcams varies
(55-80 deg), so show it as approximate.

### Running another op-list model (fire/door)

```sh
python3 runtime/export_oplist.py firedoor.onnx -o out/firedoor --weights float16 --js
# -> out/firedoor.oplist.json, out/firedoor.weights.bin, out/firedoor.oplist.js
#    (PS_OPLIST_ASSETS['firedoor'] = {json, weightsB64})
node runtime/tests/parity.js out/firedoor REF   # REF from tests/ref_outputs.py, see Tests
```

```js
const m = PSOpList.loadEmbedded(PS_OPLIST_ASSETS.firedoor, { tf });
m.inputs   // [{name, shape: [1, H, W, C] (NHWC), layout: 'nhwc', onnx_shape}]
m.outputs  // [{name, shape, layout: 'native' (ONNX NCHW) | 'nhwc'}]
const out = m.run(xNHWC);      // {outputName: tf.Tensor}; input not disposed
const data = await out.heat.data();
tf.dispose(Object.values(out));
m.dispose();                   // frees weights
```

`run` executes inside `tf.tidy` (intermediates are freed as soon as they
are dead); pass `{outputs: [names]}` to fetch intermediate tensors. Use
`--nhwc-outputs` at export to skip the final NHWC->NCHW transposes.
`PSOpList.decodeWeights(specs, buffer)` decodes a weight table to typed
arrays; `PSOpList.base64ToArrayBuffer(b64)`.

## Rebuild

```sh
cd camera/runtime
python3 build_assets.py            # repack COCO-SSD + export BlazeFace + write dist/people_assets.js
python3 build_assets.py --demo     # also dist/people_demo.html: minimal self-contained camera test page
```

Inputs: `../assets/cocossd` (TF.js COCO-SSD lite_mobilenet_v2, float32),
`../assets/blazeface/blaze_face_short_range.tflite`. Both checked by sha256
against their public URLs (LICENSES.md).

## Results (all measured here; numbers from `tests/out/*.json`)

Machine: 4 shared CPU cores, no GPU, another agent training at the same
time (load average 4-7), so every timing below is pessimistic.

### Sizes

| | stored weights | in `people_assets.js` (base64 + JSON) |
|--|--|--|
| person: COCO-SSD, person head only, per-channel uint8 | 3.44 MB (from 18.0 MB float32) | 4.82 MB |
| face: BlazeFace short range, float16 | 0.23 MB | 0.34 MB |
| total `dist/people_assets.js` | | **5.16 MB** |

`tf.min.js` adds 1.5 MB. Per-channel uint8 instead of TF.js' own per-tensor
uint8: relative RMS weight error median 0.56 % (max 1.9 %) vs 1.6 % (max
8.1 %) over COCO-SSD's 88 conv filters (depthwise: 0.30 % vs 1.2 %).

### Op-list runtime parity (node, TF.js CPU backend, vs onnxruntime / TFLite)

Max absolute error over all output elements (relative to the output's
range in brackets); 3-4 random inputs + 3-8 real photos per model. Time =
median of one `run()` on the node CPU backend.

| model | float32 | float16 | uint8 (per-channel) | ms/run |
|-------|---------|---------|---------------------|--------|
| BlazeFace (TFLite, 128x128) | 1.2e-3 (2.7e-6) | 1.2e-3 (2.7e-6) | 14 px (3.3 %) / logit 28 (9 %) | 145 |
| allops (ONNX opset 11/13/17; every supported op, BN folding, grouped/depthwise/dilated conv, ceil-mode, count_include_pad, reflect pad, nearest+bilinear resize, slice, tile, reshape/transpose, Gemm, softmax) | 2.4e-7 (5e-7) | 7e-5 (7e-5) | 1.6e-3 (1.8e-3) | 270 |
| MobileNetV2-0.5 + FPN CenterNet head (ONNX, 192x256) | 2.0e-7 (5e-7) | 7.7e-5 (2e-4) | 2.4e-3 (6.2e-3) | 380 |
| PyroNet thermal detector (ml/model.py, ONNX, 1x120x160) | 1.9e-6 (6e-6) | 6.3e-5 (2e-4) | 1.4e-3 (4.3e-3) | 160 |
| hand-built pads model (asymmetric conv/depthwise/pool pads, SAME_UPPER/LOWER, dilated asymmetric) | 1.5e-4 (1.8e-7) | 0.15 (2e-4) | 4.9 (4.4e-3) | 25 |

BlazeFace's TFLite weights are float16 already, so float16 storage is
lossless; its trained filters have outliers that make uint8 visibly worse
(on the 114-image face set: 82/84 faces matched, mean IoU 0.964), which is
why the face model ships float16 (0.23 MB anyway). The test ONNX models
have random weights.

### Face detector (BlazeFace short range)

114 images (50 Open Images close-ups, 50 'general', 14 MediaPipe portraits),
`tests/face_parity.js`:

* **op list + JS decoding on identical 128x128 inputs vs Python TFLite + an
  independent numpy implementation** of MediaPipe's anchors (896 = 16x16x2 +
  8x8x6), box/keypoint decoding and weighted NMS: 84/84 faces, max
  coordinate difference < 1e-6, scores identical (float16 storage).
* **full pipeline (`PSPeople.detect`) vs the official MediaPipe Tasks
  FaceDetector** (pip mediapipe 1.0.1, same .tflite, thresholds 0.5/0.3):
  82 of 84 faces matched, mean IoU 0.972, max score difference 0.027; the 3
  unmatched faces (2 MediaPipe-only, 1 ours-only) all score 0.51-0.54 at a
  0.5 threshold. The
  lowest IoUs are faces cut by the top edge, where MediaPipe clamps the box
  origin but keeps its height. people.js letterboxes exactly like
  MediaPipe's ImageToTensorCalculator (centred square, zero border, bilinear
  without half-pixel offset); a half-pixel-centred letterbox was 45x further
  from MediaPipe's scores.

### Person detector (re-packed COCO-SSD)

Agreement with the unmodified COCO-SSD (18 MB float32 graph + the npm
package's own `detect()`), 214 images, person boxes with score >= 0.5:

* class-head pruning alone (float32): all 207 original boxes kept (mean IoU
  0.9985, mean score change 0.0015); 12 extra person boxes appear whose anchor
  was won by another class before. On the test set they are mostly real
  people (recall +0.02, precision -0.006).
* per-channel uint8 alone: 214 of 219 matched, mean IoU 0.988, mean score
  change 0.008 (max 0.07); unmatched boxes all score 0.51-0.66.

### Test corpus and thresholds (`tests/eval_people.js`)

`../testdata/people`: 200 Open Images V5 validation images (CC BY 2.0) in
four groups of 50 (close-up single face/selfie-like; 3+ people; 1-2 people;
no people, half indoor), ground-truth boxes in `manifest.json`, plus 14
MediaPipe test portraits with hand labels (`mediapipe.json`).

Person boxes (IoU >= 0.5 with a ground-truth person; group/depiction boxes ignored):

| person thr | recall | recall, people >= 20 % of frame height | precision | no-people images with a box |
|-----|-----|-----|-----|-----|
| 0.3 | 0.62 | 0.79 | 0.82 | 8 % |
| **0.4** | **0.58** | **0.76** | **0.87** | **4 %** |
| 0.5 | 0.52 | 0.70 | 0.90 | 2 % |
| 0.6 | 0.45 | 0.62 | 0.92 | 2 % |
| (original COCO-SSD @0.5) | 0.50 | 0.68 | 0.90 | 2 % |

Recall by group at 0.4: close-up 0.94, 1-2 people 0.90, crowds 0.43 (small
people: the 300x300 SSDLite input misses them; not fixable without a bigger
model).

Faces (BlazeFace box matches the ground-truth face):

| face thr | close-up face recall | no-people images with a face |
|-----|-----|-----|
| 0.5 | 0.98 | 14 % |
| 0.6 | 0.90 | 4 % |
| **0.75** | **0.88** | **0 %** |

On non-negative images, faces scoring 0.5-0.75 were mostly not real faces
(9 matched a ground-truth face, 16 did not), faces >= 0.75 were (54 vs 4),
hence `faceMinScore = faceAttachMinScore = 0.75`.

Merged display list at the chosen thresholds (person 0.4, face 0.75):

* **single close-up face gets a white person box: 48 of 50** Open Images
  close-ups (all 48 where a face is visible; the two misses are the back of a
  head under a cap and an eye macro shot). 48 boxes come from the person
  model, the face model confirms and gives the distance.
* MediaPipe portraits: every image with a person gets exactly the right number
  of white boxes (1, 1, 1, 1, 2 people, ...), except a tiny person seen from
  behind in a collage; the empty room gets none; the four hand-only photos
  get a box on the hand (COCO-SSD calls hands 'person').
* display precision 0.87 (201 correct, 29 wrong), 0 duplicate boxes for one
  person; 2 of 50 no-people images get a box, and both actually show people
  (tiny unlabelled people on a lawn, people on a banner).
* 57 of 231 person boxes carry a face-based distance.

### Leaks, browser, speed

Headless Chromium 141 (Playwright 1.56) opening `dist/people_demo.html`
from `file://` with a fake webcam (Y4M clips of the MediaPipe portraits and
an empty living room), `tests/browser_test.js`; results in
`tests/out/browser_test_webgl.json`, `browser_test_webgl_labels.json`
and `browser_test_cpu.json`.
The browser was pinned to 2 cores; no GPU here, so WebGL is Chromium's
SwiftShader software renderer.

| backend in the page | frames | ms/frame median (min-max) | person / face ms | `numTensors` every frame | parity vs node CPU (4 images) |
|---|---|---|---|---|---|
| `webgl` (SwiftShader, `?swgl=1` = `allowSoftwareWebGL`) | 2 x (100 face + 30 empty) | 1506-1555 (1290-2001) | 1327-1386 / 150-163 | 330 | same boxes; max coordinate diff 6e-7, max score diff 1.4e-6 |
| `cpu` (software WebGL refused, default) | 101 face + 31 empty | 3137 (2501-3842) | 2810 / 312 | 330 | identical (0 difference) |

* Model load + warm-up: 12 s on SwiftShader WebGL (shader compilation),
  2.7-3.0 s on CPU.
* No request other than `file:`/`data:`/`blob:` in any session.
* Boxes on the fake camera, WebGL, 100 frames of the 3-photo clip
  (`tests/out/browser_test_webgl_labels.json`, `labelCounts`): every frame
  has at least one white box; full-frame face: `person+face 0.2M` (37
  frames); portrait: one `person+face 0.7-0.8M` (31); man and woman: two
  `person+face 0.8-0.9M` (16) or, when the second face scores below 0.75 in
  that frame, one `person+face 0.8M` plus one `person 2.1-2.2M` (16). The
  empty room: no box in 30 of 30 frames.
* `numTensors` is also constant in node: 100 `detect()` calls 331 -> 331,
  the 214-image corpus 330 -> 330; `init()` twice, then `dispose()`: back to
  the count before `init()`.
* TF.js refuses a software WebGL context by default; with
  `allowSoftwareWebGL: false` (the default) such a machine gets the CPU
  backend, which here was 2x slower than SwiftShader. The flag has to be set
  before the first WebGL attempt, which `setupBackend` does.
* On the CPU fallback (about 3 s/frame here, the person model is 90 % of it)
  the page should run the person model every few frames (`{person: false}`
  in between, the face model is 0.3 s) or show a warning. With a hardware
  GPU, COCO-SSD lite is normally real-time in TF.js; that could not be
  measured on this machine.

## Tests

```sh
cd camera/runtime
python3 tests/make_onnx_models.py                 # ONNX test models -> tests/out/onnx
sh tests/run_parity_all.sh                        # export x {float32,float16,uint8} + parity table
python3 tests/face_list.py                        # tests/out/face_list.txt (114 images)
python3 tests/face_ref_tflite.py tests/out/face_list.txt tests/out/face_ref_tflite
# official MediaPipe reference (separate venv: pip install mediapipe; needs libEGL.so.1/libGLESv2.so.2,
# Chromium's copies in /opt/pw-browsers/chromium-*/chrome-linux work via LD_LIBRARY_PATH):
python tests/face_ref_mediapipe.py tests/out/face_list.txt tests/out/face_ref_mediapipe.json
python3 tests/decode_images.py ../testdata/people/manifest.json /tmp/people_rgb --with-mediapipe
node tests/face_parity.js models/blazeface_float16 /tmp/people_rgb
node tests/run_people_corpus.js /tmp/people_rgb orig tests/out/corpus/orig.json
node tests/run_people_corpus.js /tmp/people_rgb full:../models/cocossd_person:../models/blazeface_float16 tests/out/corpus/full_u8_f16.json
node tests/eval_people.js tests/out/corpus/full_u8_f16.json tests/out/corpus/orig.json
node tests/agree.js tests/out/corpus/orig.json tests/out/corpus/full_u8_f16.json
python3 tests/make_y4m.py tests/out/face.y4m ../testdata/people/mediapipe/{face_stylizer_raw_face_demo.png,portrait.jpg,man-woman-okay.jpg}
python3 tests/make_y4m.py tests/out/empty.y4m ../testdata/people/mediapipe/living_room.jpg
python3 build_assets.py --no-build --demo
# needs Playwright + Chromium already installed (PLAYWRIGHT_BROWSERS_PATH); env: ONLY=face,empty,cpu,parity,
# SWGL=1 (allow software WebGL) or 0, FACE_FRAMES, EMPTY_FRAMES, CAM_TIMEOUT_MS
SWGL=1 ONLY=face,empty,parity FACE_FRAMES=100 RGB_PREFIX=/tmp/people_rgb node tests/browser_test.js --json tests/out/browser_test_webgl.json
SWGL=0 ONLY=face,empty,parity FACE_FRAMES=100 RGB_PREFIX=/tmp/people_rgb node tests/browser_test.js --json tests/out/browser_test_cpu.json
```

## Known limits

* COCO-SSD lite at 300x300 misses small/distant people (crowd recall 0.43)
  and calls a lone hand a person. BlazeFace short range (score >= 0.75)
  finds faces out to about 1.2 m in practice (Open Images check: half of the
  faces 10-15 % of the frame wide, 1 of 93 at 5-10 %, none smaller) and misses
  about half of the faces filling the view; beyond that the white box and the
  distance come from the person model. MediaPipe's full-range model would need
  its own anchors.
* A person box whose edge is within 1 % of the frame edge counts as cut off
  (distance label "<X.XM"): COCO-SSD and BlazeFace boxes stop about 0.5 %
  short of the edge (with the earlier 0.2 %, 17 of 19 close faces read a plain
  1.2 m or more with no '<'; 1 % flags them).
* Webcam RGB is not thermal: in smoke or darkness neither model sees people.
* Distances are pinhole estimates with an assumed field of view. A person
  box without an attached face uses the box height, which is too far for a
  half-visible person whose box does not touch the frame edge (seated
  behind a table: 2.1 m shown where the face gives 0.8-0.9 m in the
  fake-camera test).
* The fake-camera browser test used still photos in a moving frame, not a
  live person.
