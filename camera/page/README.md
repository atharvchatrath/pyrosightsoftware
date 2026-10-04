# PyroSight Camera page

One self-contained web page that runs the PyroSight box overlay on the viewer's own
camera (ordinary RGB webcam or phone camera), live, in the browser:

* **people**: WHITE box `#FFFFFF` with a device-style distance estimate (`1.2M`, `<1.2M`);
* **fire**: PURPLE box `#C850FF` labelled `FIRE` (needs the fire/door model, see below);
* **way out**: GREEN `#28FF50`, a `DOOR` box for each detected door, and an `EXIT` box or
  edge arrow for the direction the viewer marked ("Mark way out", or tap the picture).

**Integrated build:** `python3 camera/build_camera.py` builds the final page with the trained
fire/door model into `camera/dist/` (see `camera/README.md` for measured results). Integration
changes here: FIRE hysteresis and detector `reset()` in the fire/door adapter, test hooks
`window.__psLastDetections` / `__psDetections` / `__psCaptureHook`, live video shown by the
`<video>` element under a transparent box canvas (keeps playing on the CPU backend), and
`tracker.reanchor()` when a mark is made while the picture no longer matches the key frame.
Fix round (2026-10-04): door threshold 0.50 and a centre zoom pass for small flames in the
fire/door adapter (it runs on every second update when one model run takes over 200 ms), plus
the life-cycle, speech, tracking and label fixes listed at the end of `camera/README.md`.

The colours are `PS_COLOR_PERSON / PS_COLOR_FIRE / PS_COLOR_EXIT` from
`core/include/pyrosight/ps_display.h`. Nothing is fetched at run time; video never leaves
the device.

```
page/
  page.html        template: title, style, markup (no scripts; the build appends them)
  app.js           page logic (camera, files, detection loop, drawing, way out, voice, save)
  motion.js        camera-turn tracker: phase correlation + DeviceOrientation fusion (PSMotion)
  build_page.py    inlines everything -> dist/
  dist/pyrosight_camera.html            standalone page (doctype/html/meta viewport), 6.80 MB (stub fire/door)
  dist/pyrosight_camera.fragment.html   same without the wrapper (for claude.ai artifacts)
  tests/           make_clips.py, motion_test.js (node), browser_test.js (Playwright), ...
  shots/           screenshots from the browser test
  LICENSES.md
```

## Build

```sh
cd camera/page
python3 build_page.py                     # default: fire/door = clearly marked stub (finds nothing)
```

Inlined in this order (each a `<script data-ps-part>`): `#ps-shell` (the page's
non-script part as JSON, used by "Save this page"), `tf.min.js` (TF.js 4.22.0, WebGL + CPU
backends; the build puts `var regeneratorRuntime;` in front so it loads without `eval`),
`runtime/oplist.js`, `runtime/people.js`, `runtime/dist/people_assets.js`, the fire/door
slot, `motion.js`, `app.js`. The build stops if the page would exceed 15 MB, if a part
contains `<!--`, or if `app.js`/`motion.js` mention an http(s) URL; it also checks that the
page's self-copy rebuild is byte-identical to `dist/pyrosight_camera.html`.

### Wiring in the fire/door model (integrator)

```sh
python3 camera/runtime/export_oplist.py camera/firedoor/export/<final>.onnx \
        -o camera/page/build/firedoor/firedoor --weights float16 --nhwc-outputs --js
python3 camera/page/build_page.py --firedoor camera/page/build/firedoor/firedoor \
        [--firedoor-meta meta.json]   # {"name", "credits", "decode": {"thresholds": {"fire": .., "door": ..}}}
```

The output prefix must end in `firedoor` (the page reads `PS_OPLIST_ASSETS.firedoor`).
`camera/firedoor/decode.js` (global `FireDoorDecode`) is inlined with it. FireDoorNet adds
about 1.43 MB (8.23 MB page). The page's preprocessing (whole frame stretched to the model
input, bilinear to 2x then 2x2 average, `x/127.5 - 1`) matches the training
`cv2.INTER_AREA` path: on 60 held-out fire/door images, mean pixel difference 0.24/255,
max heat-map difference 0.011, 48 of 48 boxes identical (`tests/preproc_check.py`, onnxruntime).

### Injectable detector interface

`app.js` takes the first of: `globalThis.PS_FIREDOOR_DETECTOR` (set by any script before
it), an adapter around `PS_OPLIST_ASSETS.firedoor` + `FireDoorDecode`, or the built-in
**stub** (page says "Not in this build yet (stub finds nothing)"). A detector is

```js
{ name, stub?, credits?,
  init: async (tf) => {},
  detect: async (pixels /* tf.Tensor3D [H,W,3] RGB 0..255, do NOT dispose */) =>
          [{ cls: 'fire' | 'door', score, x, y, w, h }],   // normalised 0..1, x,y top-left
  dispose() {} }
```

`PSCamera.setFireDoorDetector(det)` swaps it at run time.

## What the page does

* **Start camera**: `getUserMedia` with `facingMode: {ideal: 'environment'}` (back camera on
  phones), 1280x720 ideal. A front or laptop camera is shown mirrored; labels are drawn
  unmirrored. **Switch camera** appears when more than one camera exists. `?mirror=0|1`
  overrides the mirroring.
* Live video is shown by the `<video>` element itself (`object-fit: contain`, mirrored with CSS for a
  front camera) under one transparent canvas with the boxes; photos and the eyepiece view are drawn
  into the canvas. Boxes stay aligned when the window is resized. Boxes are also shifted by how far the
  camera has turned since their frame was analysed (detections can take a second).
* **Detection loop**: `requestAnimationFrame`, at most one inference in flight. Each update
  copies the frame (max 640 px) into one tensor shared by `PSPeople.detect` and the fire/door
  model, then disposes it; `tf.memory().numTensors` is shown and stays flat. Backend:
  WebGL (software WebGL allowed, since it measured faster than the CPU backend here), else CPU.
  On CPU the person model runs every second update (faces every update), with a 250 ms pause
  between updates so the page stays usable. Paused while the tab is hidden. `?backend=cpu`
  forces CPU, `?swgl=0` refuses software WebGL.
* **Camera blocked or missing** (typical inside an embedded page): a message saying why, plus
  "Analyse a photo or video" (`<input type=file accept="image/*,video/*">`; images through
  `createImageBitmap`, videos through a `blob:` URL with a `data:` URL fallback when the host
  blocks `blob:` media) and, only when `window.claude.use('downloads')` gives a namespace,
  "Save this page and open it in your browser" (`downloads.save({filename:
  'pyrosight_camera.html', data: <full standalone HTML string>})` from the click; `declined`
  does nothing, `rate_limited`/`too_large`/others show a short line, unavailable codes hide
  the button). Without `window.claude` the page behaves the same minus that button.
* **Mark way out** marks the picture centre (or snaps to a detected door near it); tapping
  the picture marks that point (snaps to the door under the tap). The direction is tracked as
  the camera turns: `motion.js` phase-correlates a 128x128 grey copy of each frame against a
  key frame (Hann window, Gaussian-weighted cross-power spectrum, sub-pixel peak, PSR
  confidence), converts the shift to yaw/pitch with an assumed 65 degree field of view across
  the long side, re-keys every ~8 degrees, and goes "lost" after 1.5 s without a match (it
  recovers when the view returns). Every match is checked against the whole picture: 4x4 tiles
  each vote for the candidate turn (the best correlation peaks, or "no turn since the last good
  pose") whose band-passed detail they match, with the rotation's perspective modelled per tile.
  A turn needs most of the voting tiles; "no turn" needs fewer and then holds the pose exactly,
  so someone walking close past a still camera does not drag the mark around. Frames more than
  300 ms apart (CPU fallback) need a clearer majority on coarser detail. When nothing is clearly
  backed the EXIT box turns dashed ("EXIT?", the last good position), then "WAY OUT LOST". DeviceOrientation, when the browser reports it (iOS
  permission asked on the Start/Mark tap), is fused in with a complementary filter: picture
  motion short-term, sensor long-term, sensor alone while the picture does not match. A green
  `EXIT` box is drawn where the mark is when in view, else an arrow at the screen edge with
  the turn angle. The page says plainly that this is camera-turn tracking, not navigation.
* **Voice** (off by default, `speechSynthesis`) and the alert log use the device phrases
  (`core/src/ps_alerts.c`): "Person ahead." / "Person ahead, to your left." (person score
  >= 0.5, side from the box centre as seen on screen, said when a person appears after a 6 s
  gap and at most every 6 s), "Fire ahead[, to your left/right].", "Way out marked.",
  "Way out is behind you. / to your left. / ahead, to your right. ..." (8 sectors; said
  0.8 s after the mark leaves the view and when the direction settles in a new sector, at
  most every 4 s; "Where's the way out?" asks on demand), "Way out tracking lost/restored."
  ("lost" once tracking has been lost for 2 s, at most every 20 s; "restored" after 2 s of
  tracking again). The same sentence is never repeated within 3-5 s. Phrases go through a
  queue: nothing being spoken is cut off, the most important waiting phrase goes next (fire,
  then people, then the rest), and phrases older than 8 s are dropped.
* **Camera trouble**: a second Start tap while the camera is opening is ignored; a camera that
  delivers no picture within 8 s is stopped with a message; a track that ends or stops sending
  frames says so and is no longer analysed; a failed "Switch camera" stays on the working one;
  a late permission answer after the viewer opened a photo is discarded. A rotation (frame
  size change) clears the way-out mark and says so.
* **Graphics reset**: if the WebGL context is lost (or a detector run hangs for 20 s), the page
  drops the stale boxes, says "Detection stopped: the graphics chip was reset. Restarting.",
  rebuilds the WebGL backend and reloads the models (CPU backend after a failure or 3 resets).
* **Eyepiece view**: grey, contrast-stretched, 160 px wide (the thermal sensor's
  resolution), nearest-neighbour upscaled, with the same boxes. Labelled on the picture and
  on the page as a look-alike: an ordinary camera sees light, not heat.

## Tests

```sh
cd camera/page
python3 tests/make_clips.py               # fake-camera clips (.y4m), WebM, test photos -> tests/out
node tests/motion_test.js --json tests/out/motion_test.json
python3 tests/preproc_check.py ../firedoor/export/main_best.onnx
python3 build_page.py
python3 ../runtime/export_oplist.py ../firedoor/export/main_best.onnx -o build/firedoor_main_best/firedoor --weights float16 --nhwc-outputs --js
python3 build_page.py --out build/dist_firedoor_test --firedoor build/firedoor_main_best/firedoor --firedoor-name "FireDoorNet (early test checkpoint)"
taskset -c 2,3 node tests/browser_test.js        # or ONLY=desktop,pan,...; screenshots -> shots/
```

Chromium is launched with `--use-fake-device-for-media-stream
--use-file-for-fake-video-capture=<clip>.y4m`; no GPU here, so WebGL runs on SwiftShader.
Results are in `tests/out/browser_test.json` and `tests/out/motion_test.json`. Last run
(2026-10-04, after the fix round): motion tests 14 of 14 passed, browser test 76 of 76 checks
passed (fire/door test build = `camera/dist/pyrosight_camera.html`, made with
`python3 camera/build_camera.py --out camera/page/build/dist_firedoor_test`).

## Limits (honest list)

* Speed on a real phone or a real GPU is not measured: this machine has no GPU. On
  SwiftShader WebGL (2 cores) one update takes about 1.5-2.5 s (people 1.0-2.0 s, faces
  0.13-0.4 s, fire/door 0.4-0.9 s, the longer ones with the centre zoom pass); the video
  itself is drawn at the camera rate.
* Tested in headless Chromium only (fake camera from a file, synthetic DeviceOrientation
  events, stubbed `speechSynthesis` and `window.claude`). Not tested on Safari/iOS, Firefox,
  Android, a real webcam, a real motion sensor, real speech output, or inside claude.ai
  itself (its exact CSP and camera permissions are unknown; the page was tested inside a
  sandboxed iframe and under a CSP without `unsafe-eval`/WebAssembly/`blob:` media).
* Turn tracking was measured on synthetic pure rotations and synthetic people walking past
  a still camera (fake camera clips). Walking, sideways steps, moving crowds, dark or
  featureless walls and fast whip-pans all degrade it; on the CPU fallback (an update every
  0.5-3 s) a turn faster than about 10 degrees/s mostly shows "EXIT?" / "WAY OUT LOST"
  rather than a position; the field of
  view is assumed (65 degrees), so the angle of a big turn can be off by 10-20 % on a
  wider or narrower camera (the mark still lines up again when you point back at it).
* The person/face models' limits are in `camera/runtime/README.md` (small/distant people,
  hands taken for people, distances are pinhole estimates). The fire/door model is not in the
  default build; the early checkpoint used for testing found 0 doors in some door photos
  (also in onnxruntime), so a door is not always boxed.
* An ordinary camera cannot see people in smoke or darkness.
