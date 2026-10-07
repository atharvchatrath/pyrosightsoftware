# PyroSight Camera page

One self-contained web page that runs the PyroSight box overlay on the viewer's own
camera (ordinary RGB webcam or phone camera), live, in the browser:

* **people**: WHITE box `#FFFFFF` with a device-style distance estimate (`1.2M`, `<1.2M`);
* **fire**: PURPLE box `#C850FF` labelled `FIRE` (needs the fire/door model, see below);
* **way out**: GREEN `#28FF50`, a `DOOR` box for each detected door, and an `EXIT` box or
  edge arrow for the direction the viewer marked ("Mark way out", or tap the picture), or,
  while **Navigation (demo)** runs, for where the device's navigation estimates the way out.

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
  build_page.py    inlines everything -> dist/ (also ../nav/dist/ps_nav.js, the navigation module)
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
non-script part as JSON, used by "Save this page"), `runtime/oplist.js`, `runtime/people.js`,
the fire/door decoder slot, `runtime/engine.js`, `motion.js`, `app.js`, `#ps-nav`
(`../nav/dist/ps_nav.js`, global `PSNav`, 195 kB: the device's navigation C code as plain
JavaScript; rebuild it with `python3 camera/nav/build_nav.py`), and then, as
`type="text/plain"` blocks the browser neither parses nor runs: `#ps-tf` (`tf.min.js`, TF.js
4.22.0, WebGL + CPU backends; the build puts `var regeneratorRuntime;` in front so it loads
without `eval`), `#ps-people-assets` (`runtime/dist/people_assets.js`) and, when wired in,
`#ps-firedoor-model`. `app.js` starts the detectors from those texts (see "Detection engine"). The build stops if the page would exceed 15 MB, if a part
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
about 1.43 MB (8.27 MB page). The page's preprocessing (whole frame stretched to the model
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

`PSCamera.setFireDoorDetector(det)` swaps it at run time. Either way the detector object
lives on the page, so the detection engine is restarted on the page's own thread (cooperative
fallback above); the built-in adapter (`PSEngine.fireDoorAdapter`) runs in the worker.

## What the page does

* **Start camera**: `getUserMedia` with `facingMode: {ideal: 'environment'}` (back camera on
  phones), 1280x720 ideal. A front or laptop camera is shown mirrored; labels are drawn
  unmirrored. **Switch camera** appears on the picture (top right) when more than one camera
  exists. `?mirror=0|1` overrides the mirroring. A photo or video being analysed stays until the
  camera really starts (where the camera is refused it is not thrown away). Nothing above a button
  changes height when a label or mode changes: "Start camera" keeps one width ("Starting…" while
  the permission prompt is open), the camera-blocked notice comes below the buttons (with "Try the
  navigation demo"), and the hint under the buttons keeps the height of its longest text.
* Live video is shown by the `<video>` element itself (`object-fit: contain`, mirrored with CSS for a
  front camera) under one transparent canvas with the boxes; photos and the eyepiece view are drawn
  into the canvas. Boxes stay aligned when the window is resized. Boxes are also shifted by how far the
  camera has turned since their frame was analysed (detections can take a second).
* **Detection engine** (`runtime/engine.js`, global `PSEngine`): the three detectors run in a
  dedicated **Web Worker** built from a `Blob` of the page's own script texts (TF.js, oplist,
  people, the models, the decoder, engine.js; assembled in 512 k-character slices with a pause
  between slices) and started with `new Worker(URL.createObjectURL(blob))`. In the worker TF.js
  uses WebGL on an `OffscreenCanvas`, else its CPU backend. Each update sends one RGBA frame
  (max 640 px, `getImageData`, the buffer is transferred, not copied) and gets the boxes back;
  one request is in flight at a time. Messages: page -> worker `init`, `detect`, `reset`;
  worker -> page `boot`, `backend`, `ready` | `failed`, `result` | `error`, `lost` (protocol in
  the header of `engine.js`). **Fallback**: when the worker cannot be created or does not
  start (no `Worker`, a CSP with `worker-src 'none'`, an error, 20 s without a `boot`; a worker
  without WebGL on `OffscreenCanvas` uses the CPU backend instead), the same engine runs on the
  page itself, after the first paint, cooperatively: TF.js and the models are added as `blob:`
  scripts (parsed off the main thread), or inline scripts where the CSP refuses those (nonce
  copied), every model load, warm-up and detector run yields to the page at least every 30 ms
  (the person model runs in 330 stages, the face and fire/door models op by op) and gives the
  browser a frame at least every 50 ms, and after each update the loop idles 120 ms or 25 % of
  the update's time. The status line then says "on the page itself (no background worker here:
  taps may lag)". `?engine=main`, `#engine=main` or `window.PS_ENGINE = 'main'` before the page
  scripts force the fallback (tests); `engine=worker` the worker. A detector set with
  `PS_FIREDOOR_DETECTOR` / `setFireDoorDetector` is an object on the page, so the engine then
  runs on the page. `tf.memory().numTensors` of the engine is shown and stays flat. Backend:
  WebGL (software WebGL allowed, since it measured faster than the CPU backend here), else CPU.
  On CPU the person model runs every second update (faces every update); in the fallback on the
  page's own thread there is also a 250 ms pause between CPU updates. Paused while the tab is hidden.
  The turn tracker runs on each new video frame, at most every 25 ms, and takes at most about 30 %
  of the main thread (on a slow CPU it runs less often, down to every 500 ms; its cost estimate
  starts at 40 ms and rises at once with a slow run); it skips the first 0.4 s of a new source.
  `?backend=cpu`
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
* **Navigation (demo)** (card under the camera): the eyepiece's own way-back-out code
  (`core/src/ps_nav.c`, `ps_config.c`, `ps_alerts.c` compiled to JavaScript in `camera/nav`, see
  `camera/nav/README.md`): dead reckoning from steps and turns since the entry, breadcrumbs, the
  device's confidence and levels (GOOD, DEGRADED below 0.6, UNRELIABLE below 0.3, with hysteresis)
  and its phrases. Nothing starts or asks for anything until a tap.
  * **Start here (mark entry)** tries, in order: the phone's **motion sensors** (the iOS
    `DeviceMotionEvent/DeviceOrientationEvent.requestPermission` calls are made inside that tap;
    no data within 1.5 s, refused or blocked -> next), else the **camera turn** (heading from the
    camera-turn tracker of `motion.js`, kept continuous across tracker resets; steps from the
    "Hold to walk" button or W / up arrow) when a camera or video is running, else the **demo
    walk** (a simulated firefighter with a simulated motion sensor, `camera/nav` DemoWalker; works
    with no camera at all). The input in use is shown with a coloured dot and a plain sentence
    (e.g. "Motion sensors not used: No motion sensor data arrived ... inside claude.ai they are
    usually blocked"). Pressed again while running, it re-marks the entry where you are.
  * **Demo walk** (hold to walk, Left / Right: tap 15 degrees, hold to keep turning; keys W/A/D or
    the arrows), **Auto demo** (walks ~38 m in through two corridors and a room, looks around, asks
    "where is out?", then follows the arrow back), **Speed** (1x-8x, auto demo / walk-out only),
    **Guide me out** (demo: the walker follows the arrow out by itself; otherwise the device says
    the way: "Way out is ahead, to your left."), **Stop** (navigation off, the map is kept).
  * The card shows the device ring (`PSNav.ArrowWidget`: arrow to the next breadcrumb, white dot =
    straight line to the door, distance, green GOOD / amber DEGRADED, FOLLOW HOSE when
    UNRELIABLE), the phrase, route and straight-line distances, confidence with its level, and the
    map (`PSNav.MapRenderer`: entry door, estimated path and breadcrumbs, the way back, position
    with heading and uncertainty circle; in the demo also the floor plan and the true path;
    "Heading up" toggle). They are drawn only while the card is on screen: the ring at most 10
    times a second (the eyepiece's rate) and only when what it shows changed, the map 5 times a
    second (at least once a second while unchanged). At 20 / 10 redraws a second they had cost
    about 3 % more of the main thread under Chromium's 4x CPU throttling.
  * **The picture**: while navigation runs, its marker replaces the "Mark way out" mark (the mark
    is kept and comes back after Stop; taps on the picture do not mark and the hint under the
    buttons says why). Under the picture, the row "Mark way out / Where's the way out?" is replaced,
    in the same space, by the navigation's "Left / Hold to walk / Right / Guide me out", so on a
    phone the picture with its EXIT marker and these buttons are on screen together (the same
    buttons are also in the navigation card, which is below the camera). Buttons that do not apply
    to the current input are disabled, not hidden, so nothing moves. Hold buttons let the page
    scroll: a finger acts after resting 130 ms, a swipe that starts on them scrolls. The marker is
    `PSNav.exitBox()` (the device's `draw_exit()` geometry: door 0.9 x 2.0 m, camera 1.5 m up,
    sized by distance): a green **EXIT box** ("EXIT 5M") when the way out is in the field of view,
    "**EXIT? 5M**" dashed when the confidence is DEGRADED, a chevron "OUT 12M" when the next leg
    (not yet the door) is in view, a green **edge arrow** otherwise ("EXIT 7M BEHIND"), and **no
    box but FOLLOW HOSE** (with "NAV ±4M") when UNRELIABLE. In the demo walk the picture stands in
    for the eyepiece (it looks where the walker faces, never mirrored); with real sensors or the
    camera turn, a front (mirrored) camera looks back at you, so a way out behind you shows in its
    picture. FireDoorNet DOOR boxes are drawn as before. **Eyepiece view** adds the device ring at
    the top centre (`draw_nav()` geometry scaled from 320 x 240: ring, arrow, home dot, distance,
    MARK ENTRY, blinking FOLLOW HOSE, NO IMU). With **no camera** (blocked, as expected inside
    claude.ai) the picture area shows a plain simulated eyepiece view with the same marker and
    ring while navigation runs ("SIMULATED VIEW: NO CAMERA").
  * **Voice**: the device's phrases from its own alert queue ("Entry point marked.", "Way out is
    ..." when the confidence drops to DEGRADED and every 15 s while it stays there, "Navigation
    estimate unreliable. Follow the hose line out." every 20 s, "Navigation restored.", "You are
    at the entry point.") go through the page's speech queue and voice toggle, below fire and
    people (the hose warning at the level of "Way out tracking lost"), into the Spoken alerts log;
    the same phrase is not repeated within 3 s unless asked for with a button.
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
* **Graphics reset and stalls**: if the WebGL context is lost, the page drops the stale boxes,
  says "Detection stopped: the graphics chip was reset. Restarting.", and restarts the detectors
  (CPU backend after a failure, or after 3 resets / stalls with no completed WebGL update in
  between). A run that takes over 3x the usual time (at least 5 s) drops its out-of-date boxes and
  shows "Detection paused…". The detectors count as stalled only when they go silent: the worker
  posts a beat every second (silent for 8 s on WebGL, 30 s on the CPU, or 3x the longest silence
  seen while it worked); on the page itself, the cooperative engine stops reaching its yield
  points for 20 s (or 3x its longest step). A worker whose run never finishes while it still beats
  is restarted after 60 s on WebGL / 180 s on the CPU for the first update after a start, else
  after max(20 s, 8x the usual update); and never twice in a row without one completed update
  (a slow device is told "Detection is very slow on this device. Still trying." and kept running,
  instead of reloading the models forever).
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
taskset -c 2,3 node tests/browser_test.js        # or ONLY=desktop,pan,nav,navcam,...; screenshots -> shots/ (nav: ../shots/)
```

Navigation scenarios (`ONLY=nav,navcam`, 43 checks; screenshots `camera/shots/nav_*.png`):
`nav` runs at 390 px, light and dark, with no camera at all (as inside claude.ai): no motion
permission request on load; "Start here" calls `requestPermission` inside the tap (stubbed, iOS
style), finds no sensor data and falls back to the demo walk, which the input indicator says;
holding "Hold to walk" for 2.5 s gives 5 steps and moves the position 3.1 m; the simulated view
shows the edge arrow, then a green EXIT box (3.6-4.8 k green pixels) once the walker faces the way
out; Guide me out draws the arrow and puts "Way out is ..." in the log and the (stubbed) voice; the
walker reaches the door (0.12 / 0.18 m off); Auto demo at 8x finishes in 9.6 s, 1.04 m from the
door, with the EXIT box shown on the way out; Stop restores the mark hint; no horizontal scroll and
every visible button at least 44 px tall. `navcam` (camera clips): Mark way out works with
navigation off, is disabled (mark kept) while it runs and works again after Stop; demo walk over
the camera picture shows the EXIT box when facing the way out; eyepiece view shows the device ring;
with the confidence lowered (smaller `navConfScaleM`) the marker turns to dashed "EXIT? 3M"
(DEGRADED), then to no box + FOLLOW HOSE with the hose phrase spoken (UNRELIABLE); with a front
camera and no sensors "Start here" uses the camera turn, and after walking forward the way out
behind you is boxed in the front camera's picture; on the 120 degree panning clip the navigation
heading follows the camera-turn tracker (span 122 degrees, median difference 0, max 4.1-8.1 degrees over three runs: the test samples the two values a moment apart while the camera turns).

Last full run (2026-10-05, final build with navigation, run alone): 120 of 120 checks passed
(`pan`: mark error median 8.0 px, max 18.5 px, yaw error max 2.43 degrees, 0 skipped frames; the
navigation heading followed the pan over 122 degrees, median difference 0, max 4.2 degrees);
motion tests 14 of 14. An earlier full run had failed only the `pan` mark check (max 38 px, 85 of
138 tracker frames skipped) while the e2e test ran at the same time on the other two cores.

Chromium is launched with `--use-fake-device-for-media-stream
--use-file-for-fake-video-capture=<clip>.y4m`; no GPU here, so WebGL runs on SwiftShader.
Results are in `tests/out/browser_test.json` and `tests/out/motion_test.json`. Last run
(2026-10-05, after the worker change): motion tests 14 of 14 passed, browser test 77 of 77 checks
passed (fire/door test build = `camera/dist/pyrosight_camera.html`, made with
`python3 camera/build_camera.py --out camera/page/build/dist_firedoor_test`). Two changes to the
browser test, both intended: the strict-CSP host sets `worker-src 'none'`, so the page's `blob:`
worker is refused and it runs the detectors itself (as designed); the CSP-report check now accepts
that `worker-src` report, and one new check confirms the fallback (`engineMode === 'main'`, with
the worker error recorded).

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
