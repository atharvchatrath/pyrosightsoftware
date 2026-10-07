# PyroSight Camera

One web page that puts the PyroSight eyepiece boxes on **your own camera** (laptop webcam or phone
camera), live, inside the browser:

| What it sees | Box | Label |
|---|---|---|
| a person, including just a face close to the camera | **WHITE** `#FFFFFF` | rough distance, device style: `0.6M`, `2.9M`, `<1.2M` (cut off by the frame edge) |
| fire / flames | **PURPLE** `#C850FF` | `FIRE` |
| a door | **GREEN** `#28FF50` | `DOOR` |
| a window (added 2026-10-07; at most 4 shown, none on a door) | **GREEN** `#28FF50` | `WINDOW` |
| the way out you marked ("Mark way out", or tap the picture; a tap on a DOOR or WINDOW box marks the way out there) | **GREEN** `#28FF50` | `EXIT`, or an edge arrow `EXIT 120°` when you have turned away |
| the way out as the navigation estimates it (while **Navigation (demo)** runs; it replaces the mark) | **GREEN** `#28FF50` | `EXIT 5M` sized by distance, `EXIT? 5M` dashed when unsure, an edge arrow when out of view, no box but `FOLLOW HOSE` when unreliable |

The colours are `PS_COLOR_PERSON / PS_COLOR_FIRE / PS_COLOR_EXIT` from
`core/include/pyrosight/ps_display.h`. **Video never leaves the device**: the three detectors run in
the page with TensorFlow.js (in a background worker made from the page's own text, or on the page
itself where workers are not allowed), nothing is fetched or uploaded at run time, and the page says so.

It also has spoken alerts (off by default, the device's phrases from `core/src/ps_alerts.c`), an
"Eyepiece view" (grey, 160 px wide, labelled *look-alike, not thermal*), "Where's the way out?",
a photo/video fallback when the camera is blocked, "Save this page" when it runs as a claude.ai
artifact with the downloads capability, and a **demo navigation** (below).

## Navigation (demo)

The card under the camera runs the eyepiece's own way-back-out navigation: `core/src/ps_nav.c`,
`ps_config.c` and `ps_alerts.c` compiled to plain JavaScript (`nav/`, global `PSNav`, see
`nav/README.md`; parity with the native C build: 0 integer mismatches over 250 k commands). It
counts steps and turns from where you marked the entry, keeps breadcrumbs, and rates its own
confidence (GOOD, DEGRADED below 0.6, UNRELIABLE below 0.3).

* **Start here (mark entry)** uses the phone's motion sensors (asking for permission on iOS, from
  that tap only), else the camera's turns as the heading with "Hold to walk" for steps (when the
  camera or a video runs), else a **demo walk**: a simulated firefighter with a simulated motion
  sensor whose estimate drifts like a real one. The card says which input it uses and why.
  **Demo walk** / **Auto demo** start the simulation directly (works with no camera at all,
  as inside claude.ai); **Guide me out** makes the demo walker follow the arrow back, or asks the
  device for the way ("Way out is ahead, to your left."); **Stop** turns navigation off.
* The card shows the device's ring (arrow to the next breadcrumb, distance, colour by confidence),
  the phrase, distances, confidence and a map (estimated path, the way back; in the demo also the
  floor plan and the true path).
* While it runs, the green way-out marker in the picture comes from the navigation instead of
  "Mark way out" (the mark returns after Stop): an EXIT box sized by distance when the way out is
  in view (device `draw_exit()` geometry), dashed `EXIT?` when DEGRADED, an edge arrow when it is
  not in view, and no box but `FOLLOW HOSE` when UNRELIABLE. The eyepiece view adds the device's
  ring at the top; with no camera the picture area shows a plain simulated eyepiece view.
* Its phrases go through the page's voice toggle, speech queue (below fire and people) and the
  Spoken alerts log.
* It is a demo: a phone in your hand is not the body-worn sensor, real-sensor walking was only
  tested with emulated sensor events, and inside claude.ai the motion sensors are expected to be
  blocked (then it runs the demo walk). Measured results are under "Navigation results" below.

## Open it

* **As a file:** double-click `dist/pyrosight_camera.html` (9.5 MB, works offline from `file://`),
  press **Start camera**, allow the camera. Chrome, Edge and Firefox allow the camera on local files.
  A laptop/front camera is shown mirrored; the back camera of a phone is not.
* **As a claude.ai artifact:** publish `dist/pyrosight_camera.fragment.html` (the same page without
  the doctype/html wrapper) with `capabilities: {downloads: true}`. The artifact capability list for
  this account (checked 2026-10-03) has no camera capability, so inside the artifact frame the camera
  is expected to be blocked (not tested inside claude.ai). The page then says "Camera access is
  blocked", offers "Analyse a photo or video", and "Save this page and open it in your browser",
  which saves the full standalone page (byte-identical to `dist/pyrosight_camera.html`); opened from
  the downloads folder it gets the camera.
* Point it at your face: a white box with a distance appears within one or two detector updates. Point
  it at a fire video or a candle (best near the middle of the picture): a purple FIRE box. Point it at a
  closed door, face-on: a green DOOR box, which you can tap to mark the way out there. The door
  threshold is strict, so often no box appears; then tap the door in the picture or press "Mark way out".
  Point it at a window seen face-on with its whole frame in the picture: a green WINDOW box, which you can
  tap too. Windows are found in fewer than half of the rooms tried (see Limits).

Detector updates come about every 2 s on this test machine (no GPU, software WebGL; every 3.2-3.6 s
with WebGL switched off); on a laptop or phone GPU they should be several times faster (not measured,
see Limits). The video itself plays at the camera's frame rate in every case.

## Rebuild

```sh
python3 camera/build_camera.py              # -> camera/dist/pyrosight_camera.html + .fragment.html + .build.json
python3 camera/build_camera.py --people     # also rebuild the person/face models (runtime/build_assets.py)
python3 camera/build_camera.py --no-firedoor  # fire/door stub build (finds nothing; the page says so)
```

`build_camera.py` exports `firedoor/export/firedoor.onnx` with `runtime/export_oplist.py` (float16
weights, NHWC outputs, base64) to `build/firedoor/firedoor.oplist.js`, writes the fire/door settings to
`build/firedoor/meta.json` (thresholds fire 0.50 / door 0.50 / window 0.36, the FIRE hysteresis: on at 0.50,
kept while an overlapping box stays at 0.35 or more, and the centre zoom pass: middle 50 % of the
frame, FIRE from it at 0.60 or more; see `firedoor/MODEL.md` and "Fire/door settings" below), then runs
`page/build_page.py`, which inlines, in order: `runtime/oplist.js`, `runtime/people.js`,
`firedoor/decode.js` + settings, `runtime/engine.js`, `page/motion.js`, `page/app.js` (all small, run
by the browser as normal scripts), then three **`<script type="text/plain">` blocks that the browser
does not parse or run**: TF.js 4.22.0 (`tf.min.js`), `runtime/dist/people_assets.js` (person + face
weights) and the fire/door model, each split over elements of at most 128 k characters
(`id="ps-tf"`, then `data-ps-of="ps-tf"`, ...) so the browser's HTML parser can pause between them.
`app.js` hands those texts, with the small scripts, to a background worker (see "Responsiveness"
below). The page is written as **pure ASCII** (`\uXXXX` escapes in scripts, `&#x...;` in the
markup): Chromium decodes an all-ASCII page on a fast 8-bit path, and the few non-ASCII characters
had made it decode the 8 MB page at about a third of that speed, in tasks of 250+ ms at 4x CPU
throttling. It then checks both outputs and stops on any failure:

* size under 15 MB (base64 counted);
* `<title>PyroSight Camera</title>`;
* no external resource in the markup (no `<script src>`, `<link>`, `<img src>` other than `data:`,
  `<iframe>`, `@import`, `url(...)`), and no `fetch`, `XMLHttpRequest`, `import()`, `WebSocket`,
  `sendBeacon`, `importScripts`, `SharedWorker`, `serviceWorker` or `http(s)://` in the page's own
  code (app, engine, motion, oplist, people, decoder, the navigation module `nav/dist/ps_nav.js`);
* exactly one `new Worker(url)`, in `app.js`, where `url` is `URL.createObjectURL(blob)` of a `Blob`
  built only from the page's own script texts (no network: a `blob:` URL of the page's own text);
* TF.js and the two model blocks (and every continuation element) are `type="text/plain"`, and
  the scripts the browser runs on load stay under 450 kB (`main_thread_script_bytes` in the build
  report; 406 kB now: page code 211 kB + the navigation module 195 kB, which has to run on the page
  because it reads the motion sensors and draws the map; it takes 5-6 ms to run there, 20-50 ms at
  4-6x CPU throttling);
* the page is pure ASCII;
* the fragment has no doctype/html/head/body wrapper; the standalone page has them.

The 16 `http(s)` strings left are inside `tf.min.js` (licence and docs links, core-js URL feature
tests, TF.js's HTTP model loader that the page never calls); they are listed in
`dist/pyrosight_camera.build.json`. The browser tests confirm the page makes **no** request other than
`file:`, `data:` and `blob:`.

Page parts (MB): TF.js 1.47, person + face models 5.16, fire/door/window model + decoder 2.42 (1.43 before
the window class), page code and markup and engine 0.24, navigation module 0.21. Total 9.52 MB.

## Tests

```sh
cd camera
python3 tests/firedoor_ref.py                         # onnxruntime references (32 held-out inputs)
node tests/firedoor_parity.js build/firedoor/firedoor # op-list runtime (TF.js CPU) vs onnxruntime
python3 tests/make_e2e_clips.py --clips-dir DIR       # barcoded fake-camera clips (~0.8 GB, regenerable)
taskset -c 2,3 node tests/e2e_test.js --secs 60       # Playwright + fake camera, WebGL and CPU fallback
taskset -c 2,3 node tests/e2e_layout.js               # 390 px dark / desktop light screenshots
taskset -c 2,3 node tests/responsiveness_test.js --configs webgl,cpu,webgl-4x,webgl-main,csp-iframe-noworker
                                                      # click latency + long tasks while loading/detecting
python3 tests/face_scale_check.py DIR && node tests/face_scale_check.js DIR   # distance-label sanity check
node tests/e2e_shots.js                               # key screenshots from single-image clips
cd page && node tests/motion_test.js && node tests/browser_test.js   # the page's own suites (still pass)
cd page && ONLY=nav,navcam taskset -c 2,3 node tests/browser_test.js # navigation in the page (43 checks)
cd nav && node tests/geom_test.js && node tests/walker_test.js && node tests/stepdetect_test.js \
       && node tests/parity_test.js && node tests/pw_demo_test.js   # the navigation module itself
taskset -c 2,3 node tests/responsiveness_test.js --nav --configs webgl,cpu,webgl-4x   # with the auto demo running
```

The page agent's own suites were re-run after the fix round (2026-10-04): `page/tests/motion_test.js`
14 of 14 passed, including the re-anchor test and two new ones (a person walking close past a still
camera: at most 0.02 degrees of yaw; frames far apart during a fast turn: 0 confident wrong poses),
and `page/tests/browser_test.js` 76 of 76 checks passed. For that run its fire/door test build was
made with `python3 build_camera.py --out page/build/dist_firedoor_test`, byte-identical to
`dist/pyrosight_camera.html`.

The e2e clips are slideshows of held-out test images (3 s each at 10 fps, 640x480), with a frame
number barcode in the bottom 8 rows. A capture hook set by the test reads it from the exact frame
handed to the detectors, so every detector update is compared with the ground-truth boxes of the
image on screen. The page exposes `window.__psLastDetections` and `window.__psDetections` (the last
1000 updates: boxes normalised to the camera frame, per-model ms, backend, the engine's
`tf.memory().numTensors` as `tensors`, and `engine`: `'worker'` or `'main'`). With the worker there is
no `tf` global on the page; `PSCamera.state.numTensors` has the count.

## Responsiveness (2026-10-05: the detectors moved off the page's main thread)

Before this change the page parsed TF.js, decoded the 6.6 MB of models and ran every detector on the
page's main thread: a tap could wait 5-6 s while the models loaded (25-42 s with Chromium's 4x/6x CPU
throttling), and on the CPU backend 80-99 % of the time during detection. Now:

* **The detectors run in a Web Worker** (`runtime/engine.js`, started by `page/app.js` from a `Blob`
  of the page's own script texts, so nothing is fetched). The page's main thread never parses TF.js
  or the models (they are `type="text/plain"`), never waits for a detector, and sends one RGBA frame
  per update (transferred, not copied). TF.js uses WebGL on an `OffscreenCanvas` in the worker, else
  its CPU backend. A lost WebGL context restarts the worker (tested with `PSCamera.debugEngine('lose-context')`:
  boxes back after 21 s, tensors flat at 456; in the fallback the engine restarts on the page, boxes
  back after 23 s, and 3 tensors of the lost context stay counted: 456 -> 459, then flat).
* **Stalls**: the worker posts a beat every second; a detector that goes silent (8 s on WebGL, 30 s
  on the CPU, or 3x the longest silence seen while it worked; 20 s for the fallback's yield points)
  is restarted; boxes from a run 3x longer than usual are dropped ("Detection paused…"). A first
  update after a start is not judged by a fixed 20 s, and a stall restart is never repeated
  without a completed update in between (a slow phone keeps trying instead of reloading forever).
  Three graphics resets or stalls switch to the CPU only when no WebGL update completed in between.
* **Fallback on the main thread** when the worker cannot start (a CSP with `worker-src 'none'`, no
  `Worker`, an error or 20 s without a sign of life, a custom fire/door detector; a worker without
  WebGL on `OffscreenCanvas` uses TF.js's CPU backend instead): after the first paint, TF.js and the
  models run as `blob:` scripts (compiled off the main thread) or else inline scripts, each started
  only after 1 s without a tap or key (at most 8 s of waiting: running one blocks the page for
  0.3-0.5 s on a slow phone), and every model
  load, warm-up and detector run gives the browser a turn at least every 30 ms and a rendered frame
  at least every 50 ms (the person model as 330 stages, the face and fire/door models op by op,
  bit-exact with the one-shot runs), with an idle gap after each update. The status line then says
  "on the page itself (no background worker here: taps may lag)". `?engine=main`, `#engine=main` or
  `window.PS_ENGINE = 'main'` forces it (tests).
* **The page loads lighter:** pure ASCII (see Rebuild), the text blocks split so the HTML parser can
  pause, and the turn tracker limited to about 30 % of the main thread (on a slow CPU it runs less
  often, down to 5 times a second, instead of starving taps). The buttons work from the first paint;
  the picture shows "LOADING DETECTORS…" and the status "Loading detectors…" until the first boxes.

Measured with `tests/responsiveness_test.js` (extends the lead's freeze check): headless Chromium,
390x844 phone viewport, fake camera (`e2e_still_face.y4m`), all Chromium processes on 2 of this
machine's 4 cores (`taskset -c 2,3`), no GPU (WebGL = SwiftShader). Clicks at 0.5, 0.6, 1.0 ("Start
camera"), 1.1, 2, 3, 5, 5.2 and 8 s after navigation and 16 more while boxes come in; **click ->
visible** = the click event's timestamp to the first rendered frame after the page's handler ran.
"Worst" counts handled clicks on the intended button; the old page also lost or misdirected 1-6 taps
per run (handled seconds later, after the layout had moved). Long tasks are main-thread tasks over
50 ms after the first paint. Results: `tests/out/responsiveness.json` (old page = the build before
this change, same harness, same day).

| configuration | old page: "Start camera" / worst click | longest task (>200 ms) | blocked in detection | new page: "Start camera" / worst click | longest task (>200 ms) | blocked in detection |
|---|---|---|---|---|---|---|
| WebGL | 5347 / 5746 ms | 5756 ms (3) | 0.3 % | 12 / 29 ms | 64 ms (0) | 0 % |
| CPU only (no WebGL) | 5271 / 5773 ms | 5774 ms (13) | 79.6 % | 20 / 47 ms | 61 ms (0) | 0 % |
| WebGL, 4x CPU throttling | 340 / 8279 ms | 8744 ms (7) | 19.2 % | 31 / 146 ms | 175 ms (0) | 7.9 % |
| WebGL, 6x | 858 / 7633 ms | 9162 ms (11) | 57.6 % | 118 / 197 ms | 216 ms (1) | 20.8 % |
| CPU only, 4x | 210 / 25621 ms | 25477 ms (9) | 93.2 % | 45 / 181 ms | 180 ms (0) | 13.2 % |
| CPU only, 6x | 363 / 41868 ms | 42097 ms (10) | 99.4 % | 59 / 275 ms | 207 ms (1) | 34.6 % |
| sandboxed iframe, strict CSP (`script-src 'unsafe-inline' blob:; worker-src blob:` ...) | 5384 / 5788 ms | 5779 ms (4) | 3.6 % | 13 / 26 ms | 80 ms (0) | 0 % |
| same, 4x | 286 / 7979 ms | 8629 ms (6) | 16.5 % | 42 / 118 ms | 192 ms (0) | 9.6 % |
| same CSP with `worker-src 'none'` (fallback) | 5229 / 5636 ms | 5610 ms (3) | 0.4 % | 79 / 114 ms | 310 ms (1) | 0.4 % |
| same, 4x (fallback) | (as above) | | | 43 / 132 ms | 592 ms (3) | 8.4 % |
| fallback forced, WebGL (`?engine=main`) | (= WebGL row) | | | 140 / 186 ms | 299 ms (1) | 0 % |
| fallback forced, CPU only | (= CPU row) | | | 68 / 223 ms | 255 ms (2) | 33.5 % |
| fallback forced, WebGL, 4x / 6x | | | | 568 / 568 ms; 32 / 229 ms | 712 ms (2); 894 ms (4) | 5.4 %; 27.3 % |
| fallback forced, CPU only, 4x | | | | 507 / 1760 ms | 2536 ms (64) | 57.2 % |
| whole browser on 1 core, WebGL, 4x | 262 / 20742 ms | 21125 ms (17) | 51.3 % | 155 / 354 ms | 269 ms (5) | 24.2 % |
| whole browser on 1 core, CPU only | 5884 / 6272 ms | 6262 ms (14) | 94 % | 18 / 54 ms | 99 ms (0) | 0 % |

Detector speed with the worker, from `tests/e2e_test.js` (60 s clips, same day, same machine; old
page run with `--page`): WebGL faces clip, median update 1617 ms (people 930, faces 127, fire/door
455 ms), 39 updates in 60 s, vs 1674 ms (923 / 124 / 635 ms), 38 updates on the old page. CPU only:
median 3857 ms (people 2882, faces 321, fire/door 1903 ms), 16 updates in 60 s, vs 3095 ms
(2161 / 267 / 1696 ms), 20 updates: the worker now shares the 2 cores with a main thread that keeps
drawing, tracking and answering taps, where the old page simply froze. Boxes are unchanged: faces
clip, a white box in 39/39 updates (every ground-truth person in 38), none off a person (CPU 16/16);
fire clip, FIRE in 37/37 updates (35 on the fire); doors clip, DOOR in 10/37 (all on a door); no
FIRE in the 92 updates on no-fire images; tensors flat at 456; no request to any host
(`tests/out/e2e_results_worker.json`). The worker takes 10-14 s to load the person/face models on
SwiftShader (first boxes 17-18 s after opening, as before).

Remaining slow spots (all measured above): the main-thread fallback cannot split TF.js's own start-up
(one 300 ms task here, 700-900 ms at 4-6x throttling) or a single CPU-backend operation (up to 255 ms
here, 2.5 s at 4x throttling on the CPU backend); at 6x throttling the page's own drawing, video and
turn tracking keep the main thread 64-80 % busy, so a tap now and then waits 200-275 ms; with the
whole browser on one core the worker competes with the page (taps up to 354 ms). Chromium's CPU
throttling slows only the main thread, not the worker, so the throttled rows show the page's own cost,
not a slow phone's detector speed. Not tested on a real phone, a real GPU, Safari or Firefox (Safari
has WebGL in workers only from version 17; older ones would run the worker on TF.js's CPU backend).

## Navigation results (2026-10-05, after adding Navigation (demo) to the page)

Same machine and harnesses as above (headless Chromium, no GPU, Chromium pinned to 2 cores).

* **The navigation module** (`nav/`, unchanged by the integration) still passes its own tests:
  `geom_test.js` (36 checks: direction words and the EXIT box), `walker_test.js` (closed loop),
  `stepdetect_test.js`, `parity_test.js` (C vs JS, stress run of 181,195 commands: 0 integer, crumb
  or alert mismatches, max float difference 8e-8) and `pw_demo_test.js` (its own demo page in
  Chromium, 35 of 35, including emulated motion sensors and iOS-style permission granted/denied).
* **The page** (`page/tests/browser_test.js` on the final build, run alone): 120 of 120 checks,
  43 of them the navigation scenarios `nav` and `navcam` (listed in `page/README.md`); the `pan` mark
  test as before (median 8.0 px, max 18.5 px, 0 skipped frames). Motion tests 14 of 14. (An earlier
  full run failed only that `pan` check, max 38 px with 85 of 138 tracker frames skipped, while the
  e2e test ran on the other two cores.) Screenshots: `shots/nav_*.png` (390 px light and dark, demo walk,
  simulated view with the EXIT box, auto demo, desktop camera with the EXIT box, eyepiece ring,
  DEGRADED `EXIT? 3M`, UNRELIABLE `FOLLOW HOSE`, front camera).
* **Detectors unchanged** (`tests/e2e_test.js --secs 60`, run alone, on the build before the last
  change to the card's redraw rates, which touches only navigation drawing): WebGL faces clip, median
  update 1640 ms (people 970, faces 127, fire/door 618 ms), 38 updates (before navigation: 1617 ms,
  39); a white box in 38/38 updates, 37 with every ground-truth person, none off a person. Fire clip
  1578 ms, FIRE in 36/36 updates (34 on the fire). Doors clip 1600 ms, DOOR in 9/37 updates (all on a
  door). CPU only: 3440 / 3012 / 3903 ms, 18 / 17 / 16 updates (faces 18/18; FIRE 17/17, 16 on the
  fire; DOOR 5/16, all on a door). No FIRE in any of the 109 updates on no-fire images, tensors flat at
  456, no request to any host, no page errors (`tests/out/e2e_results_nav.json`).
* **Responsiveness** (`tests/responsiveness_test.js`, same 17 configurations as above; results in
  `tests/out/responsiveness_nav.json`, A/B runs in `tests/out/responsiveness_nav_ab.json`). Cells:
  "Start camera" / worst click -> visible; longest main-thread task after first paint (number over
  200 ms); share of the detection phase blocked by long tasks. "Off" = navigation in the page but not
  started (2-3 runs: ranges); "auto demo" = `--nav`: the auto demo at its fastest speed (8x),
  restarted whenever it ends, with its map and ring drawn even when scrolled off screen (the worst
  case), for the whole detection phase (1-2 runs).

| configuration | before navigation (1 run) | navigation in the page, off | auto demo 8x running | main thread busy in detection: off / demo |
|---|---|---|---|---|
| WebGL | 12 / 29 ms; 64 ms (0); 0 % | 12-18 / 33 ms; 60-62 ms (0); 0 % | 14 / 31 ms; 68 ms (0); 0 % | 18.6-19.4 % / 22.4 % |
| CPU only | 20 / 47 ms; 61 ms (0); 0 % | 17-19 / 25-47 ms; 64-72 ms (0); 0 % | 11 / 29 ms; 72 ms (0); 0.4 % | 22.4-22.9 % / 30.6 % |
| WebGL, 4x CPU throttling | 31 / 146 ms; 175 ms (0); 7.9 % | 29-43 / 122-154 ms; 142-143 ms (0); 6.7-8.9 % | 22-27 / 138-262 ms; 165-218 ms (0-1); 10.3-13.6 % | 55.2-56.8 % / 73.2-77.2 % |
| WebGL, 6x | 118 / 197 ms; 216 ms (1); 20.8 % | 41-55 / 209-234 ms; 172-212 ms (0-1); 22.5-28.8 % | 71-98 / 180-375 ms; 164-192 ms (0); 24.9-34 % | 72.7-77.2 % / 86.5-90 % |
| CPU only, 4x | 45 / 181 ms; 180 ms (0); 13.2 % | 33-34 / 151-228 ms; 140-182 ms (0); 10.8-20.9 % | 27 / 293 ms; 168 ms (0); 17.8 % | 66.8-71.4 % / 89.7 % |
| CPU only, 6x | 59 / 275 ms; 207 ms (1); 34.6 % | 34-165 / 272-555 ms; 216-275 ms (1); 29.9-38.9 % | 33 / 407 ms; 270 ms (2); 41.4 % | 83.6-89.9 % / 95.3 % |
| fallback forced, WebGL | 140 / 186 ms; 299 ms (1); 0 % | 84-126 / 156-179 ms; 234-322 ms (1); 0 % |  | 24.4-25.7 % / - |
| fallback forced, CPU only | 68 / 223 ms; 255 ms (2); 33.5 % | 42-157 / 202-216 ms; 259-306 ms (2-3); 32.9-41.1 % |  | 81.3-83.1 % / - |
| fallback forced, WebGL, 4x | 568 / 568 ms; 712 ms (2); 5.4 % | 27-44 / 569-669 ms; 564-623 ms (2-3); 2.5-11 % |  | 74.7-76.4 % / - |
| fallback forced, WebGL, 6x | 32 / 229 ms; 894 ms (4); 27.3 % | 47-68 / 395-858 ms; 832-936 ms (4-6); 23.2-27 % |  | 84.6-86.6 % / - |
| fallback forced, CPU only, 4x | 507 / 1760 ms; 2536 ms (64); 57.2 % | 30-330 / 564-1063 ms; 887-961 ms (67-97); 54.3-54.8 % |  | 87.1-90.5 % / - |
| sandboxed iframe, strict CSP | 13 / 26 ms; 80 ms (0); 0 % | 18-19 / 21-25 ms; 52-136 ms (0); 0-0.8 % |  | 32.6-34.9 % / - |
| same, 4x | 42 / 118 ms; 192 ms (0); 9.6 % | 25-26 / 144-201 ms; 129-174 ms (0); 6.7-11.6 % |  | 74.4-74.9 % / - |
| same CSP, `worker-src 'none'` (fallback) | 79 / 114 ms; 310 ms (1); 0.4 % | 114-144 / 144-162 ms; 205-274 ms (1); 0 % | 114 / 146 ms; 296 ms (1); 0 % | 40.8-41.7 % / 48.1 % |
| same, 4x (fallback) | 43 / 132 ms; 592 ms (3); 8.4 % | 25-27 / 195-197 ms; 539 ms (4-5); 7.2-9 % |  | 82.9-84.9 % / - |
| whole browser on 1 core, WebGL, 4x | 155 / 354 ms; 269 ms (5); 24.2 % | 46-87 / 565-793 ms; 332-341 ms (6-9); 14-20.3 % | 152-174 / 328-547 ms; 315-322 ms (4-9); 21.5-30.3 % | 74.3-79.6 % / 94-94.7 % |
| whole browser on 1 core, CPU only | 18 / 54 ms; 99 ms (0); 0 % | 12-13 / 30-85 ms; 74-113 ms (0); 0 % |  | 25.2-26.2 % / - |

  * **Navigation off:** taps as before within run-to-run noise. The worst click is one sample out of
    25 and moves a lot between runs of the same build (1 core: 565-793 ms with navigation, 332-533 ms
    in 3 runs of the same page with the navigation script removed; the 354 ms before was one run);
    those worst clicks fall in single long tasks while the worker loads on the same core. Script
    time on the main thread is unchanged, but in the A/B runs the main thread was busier with the
    navigation script present, all of it outside script (webgl-4x +6, cpu-4x +6, cpu-6x +7,
    1 core +12 percentage points in the harness, 2-3 runs each; +2 to +5 points in 12-window direct
    measurements at 6x). Its cause was not found: it is not the card's canvases (blanking them
    changed nothing), the IntersectionObserver (disabling it changed nothing) or timers (none run
    while navigation is off). The script itself runs once at load: 5-6 ms here, 20-50 ms at 4-6x.
  * **Auto demo running:** without CPU throttling no tap is slower (worst 29-31 ms) and the main
    thread is 3-8 points busier; under 4-6x throttling it is 5-23 points busier, and in the
    2-core rows the worst tap rose from at most 154 / 234 / 228 ms (WebGL 4x / 6x, CPU 4x, off) to
    262 / 375 / 293 ms (the 1-core and CPU 6x rows stay inside their off range). Measured directly
    at 4x with the camera and detection running: the 8x simulation itself takes about 5 % of the
    main thread, drawing the card about 2-3 %, its text and the picture marker the rest. After that
    measurement the card was made cheaper (the ring is drawn at most 10 times a second and only
    when it changes, the map 5 times a second; at the earlier 20 / 10 per second the demo cost 3
    points more at 4x). The page's default demo speed is 2x, a quarter of the simulation work.
* **Not tested:** a real phone (its motion sensors, iOS's permission prompt, a body-worn sensor),
  walking with real steps, real speech output, and the page inside claude.ai. Motion sensors were
  only emulated (`pw_demo_test.js`) or stubbed (`requestPermission` in the browser test).

## Measured results

### Fire/door model in the page's runtime vs onnxruntime

`tests/firedoor_parity.js`: the op-list runtime (TF.js CPU backend in node) against onnxruntime on
32 held-out inputs (12 FireNET fire, 10 Open Images door, 8 hard negatives, the 2 `sample_io` tensors),
preprocessed exactly like `firedoor/data.py`.

| Weights | max abs error heat / wh (px) / off | Decoded boxes at fire 0.50, door 0.35 | at the 0.35 hysteresis floor |
|---|---|---|---|
| float32 (2.0 MB) | 9.5e-7 / 5.5e-4 / 2.7e-6 | 15 of 15 identical (IoU 1.000, score diff 4e-7) | 19 of 19 |
| **float16 (shipped, 1.04 MB)** | 1.1e-3 / 0.94 / 2.8e-3 | 14 of 15 (min IoU 0.999, score diff 9e-4); the missing box scores 0.5002 in onnxruntime, right on the 0.50 threshold | 19 of 19 (min IoU 0.998) |

Op list: 77 ops (conv2d 43, depthwise 20, add 12, nearest resize 2; ReLU6/ReLU/sigmoid fused), all 3x3
convolutions with the explicit symmetric padding the model needs. Node TF.js CPU: 730-770 ms per
frame; all weight tensors are freed on `dispose()` (126 -> 0).

### Fire/door settings in the page (changed in the fix round, 2026-10-04)

* **Door threshold 0.50** (the model's own default is 0.35). On the quality verifier's Open Images
  val/test photos, 0.35 put a DOOR box on 38 % of real doors and on 31 % of door look-alikes
  (wardrobes, windows, fridges); 0.50 gives 19 % and 7 %. Fewer doors are found, but a green DOOR
  box is more often a door. Retraining with the look-alikes as hard negatives is the real fix.
* **Centre zoom pass for small flames.** After the whole-frame pass, the model also looks at the
  middle 50 % x 50 % of the frame (2x zoom) and keeps fire boxes from it at 0.60 or more (stricter
  than the whole frame's 0.50, because zooming also magnifies lamps and LEDs). Measured in node on
  the verifier's 30 candle/lighter flames from FireNET validation pasted at each size (all inside
  the centre area), flame found:

  | flame width / frame width | 2 % | 3 % | 5 % | 8 % | 12 % | 20 % |
  |---|---|---|---|---|---|---|
  | whole frame only | 8/30 | 16/30 | 22/30 | 21/30 | 21/30 | 10/30 |
  | + centre zoom (shipped) | **18/30** | **24/30** | 23/30 | 21/30 | 21/30 | 10/30 |

  In the page (WebGL, a fake-camera clip of 10 of those flames per size, 6 s each, drifting a few
  pixels like a hand-held camera), updates with FIRE on the flame:

  | | 2 % wide | 3 % wide |
  |---|---|---|
  | before (whole frame only) | 14/35 updates; 5 of 10 flames ever boxed | 19/31; 6 of 10 |
  | after (centre zoom) | **18/30**; 6 of 10 | **26/29**; 10 of 10 |

  Cost: photos with a false FIRE box among 871 Open Images non-fire photos went from 17 to 24. On
  the verifier's deliberately fire-like clip (24 photos of sunsets, LEDs, lamps, car lights, red
  clothes; 2 min in the page, run back to back with the old build) FIRE was shown in 26 of 54
  updates with 11 onsets, against 28 of 64 with 8 onsets before. It also costs
  a second model run. Run on every update it made a software-WebGL update here 2.3 s instead of
  1.8 s (faces clip, same day), so when one model run takes over 200 ms the centre pass runs on
  every second update and its boxes are reused once in between (on the CPU fallback: on the updates
  that skip the person model); on a fast GPU it runs on every update. A reused box is dropped when
  the middle of the picture has changed since (a 16 x 12 grey thumbnail differs by 25 grey levels or
  more on average; measured offline on the test clips over 2 s: 8-17 for a still photo with
  hand-held drift or a burning car on CCTV, 37-44 across a cut, mostly over 25 during a pan), so
  it does not stay on after a cut or a large turn. Small flames away from the middle of the
  picture get no help.

### End-to-end in Chromium with a fake camera (`tests/e2e_test.js`, re-run 2026-10-04 after the fix round)

Headless Chromium 141 (Playwright 1.56), pinned to 2 cores, nothing else running, no GPU: WebGL
runs on SwiftShader (software). Each clip ran for 60 s after the first detection; every detector
update was read from `window.__psDetections` and matched to the image on screen through the frame
barcode (all updates matched). "WebGL off" = Chromium started with `--disable-gpu
--disable-software-rasterizer --disable-webgl --disable-webgl2`, so the page falls back to the
TF.js CPU backend by itself. Matching rule: a box counts as on its object at IoU >= 0.3 (fire and
doors: or its centre inside the ground-truth box). "Before" is the 2026-10-03 run of the
integration build (door threshold 0.35, no centre zoom pass).

| Backend | Clip (held-out images, 3 s each) | Updates | WHITE (people) | PURPLE (FIRE) | GREEN (DOOR) |
|---|---|---|---|---|---|
| WebGL | faces: 8 close-up people | 31 | person boxed in 31/31 updates; 0 of 31 boxes off a person | none (0/31) | none |
| WebGL | group: 8 photos with 3-6 people | 29 | a person boxed in 29/29; 89 of 99 people found (90 %); 0 of 96 boxes off a person | **false FIRE in 4/29** updates, all on one motion-blurred concert photo in orange stage light (before: 6/37, same photo) | none |
| WebGL | nopeople: 8 photos, no people | 30 | **0/30** updates with a white box | none (0/30) | none (before: 9 boxes on 2 photos that do show doors, below 0.50) |
| WebGL | fire: 12 FireNET test fire photos | 29 | (not scored) | FIRE in **29/29** updates, on the fire in 26/29; 23/29 without the hysteresis (before: 37/38, 35/38, 30/38) | none |
| WebGL | firevideo: CCTV car fire, fire in every frame | 29 | (not scored) | FIRE in **27/29**; 22/29 without the hysteresis (before: 34/37, 29/37) | none |
| WebGL | doors: 10 door photos | 29 | (not scored) | none (0/29) | DOOR in **4/29** updates, all on a door; 3 of 10 doors boxed at least once (before, at 0.35: 19/36, 7 of 10) |
| WebGL | lights: lamp, bulb, sunset, dusk, sunlight, street light, neon, Christmas tree, orange, pumpkin, hand, monitor | 28 | (not scored) | **0/28** false FIRE | none |
| CPU (WebGL off) | faces | 18 | person boxed in 18/18 (before: 17/18) | none | none |
| CPU (WebGL off) | fire | 16 | (not scored) | FIRE in 16/16, on the fire 15/16 (before: 22/22, 21/22) | none |
| CPU (WebGL off) | doors | 18 | (not scored) | none | DOOR in 5/18, all on a door; 3 of 10 doors boxed at least once (before: 13/20, 10/20 on a door) |

The hysteresis numbers on the slideshow clips are slightly optimistic: a box kept at 0.35-0.50 can
carry over a cut to the next photo when the two fire boxes overlap (in real video, consecutive
frames show the same scene). In the first run of this round, without the thumbnail check, reused
centre-pass boxes from the concert photo also crossed the cut onto the next photo (false FIRE in
7/29 group updates); with the check, none did.

Re-run 2026-10-07 with the window class (WebGL, same machine and rules, `--only
windows,doors,fire,faces,nopeople,lights`): **windows** clip (10 held-out photos of rooms with
windows, `make_e2e_clips.py`): WINDOW in 9 of 17 updates, all 9 on a window; 6 of the 9 photos shown
got a WINDOW box at least once (the 10th was never analysed in the 60 s). Fire
28/29 updates on the fire, faces 32/32, DOOR 5/26 (all on a door), no white or purple box on the
no-people or no-fire photos. WINDOW boxes on photos without labelled windows: a TV screen (nopeople,
5 updates), a bright bulb (lights, 3), and real but unlabelled windows (a shop front; a building front
on the doors clip), plus one glazed door while its DOOR score was under 0.50. Median update 1.97-2.35 s (fire,
door and window model 0.62-1.00 s; it is about 30 % more compute than fire/door alone), tensors flat
at 506.

Speed and memory (same machine, all 10 runs). This machine ran about 12 % slower on 2026-10-04 than
on 2026-10-03: the unchanged integration build, re-run back to back with this one on the faces and
fire clips, took a median 1.77-1.80 s per WebGL update (1.57-1.64 s the day before). This build took
2.11-2.19 s in that back-to-back run: the centre zoom pass adds about 0.3-0.4 s per update here.

| | WebGL (SwiftShader, software) | CPU backend (WebGL off) |
|---|---|---|
| detectors loaded, first boxes | load 16.7-18.1 s; first boxes 19.9-21.6 s after Start camera (old build, same day: 19.9-20.6 s) | load 4.9-6.5 s; first boxes 9.6-11.6 s after Start (old build, same day: 8.7-10.2 s) |
| one detector update | median 1.94-2.20 s (p90 2.31-2.59 s): people 1.16-1.27 s, faces 0.16-0.17 s, fire/door 0.43-0.47 s, or 0.88-0.93 s on the updates that also run the centre pass (every second one) | one update every 3.2-3.6 s on average (16-18 updates a minute; old build, same day: 17-20): people 2.4-2.9 s (every 2nd update), faces 0.28-0.32 s, fire/door 1.0-1.2 s, or 2.2-2.7 s with the centre pass (on the updates without the person model) |
| video shown (frames presented, `getVideoPlaybackQuality`) | 10 fps (the clip's rate; 15 fps for the 15 fps clip) | 9.4-9.8 fps: the `<video>` element keeps playing while a detector blocks the page (the box layer itself only redraws between detector runs) |
| `tf.memory().numTensors` per update | 457 in every update of every run (flat, no leak) | 457, flat |
| network requests other than file:/data:/blob: | 0 | 0 |
| page errors | 0 | 0 (2 console warnings: TF.js reports that WebGL is unavailable) |

Box colours: in the verifier's page runs on this build (small-flame and fire-like clips), every
drawn box had its exact colour in the edge band (person 255,255,255: 53 of 53 boxes; FIRE
200,80,255: 95 of 95); the key screenshots contain the exact DOOR/EXIT green 40,255,80.

Interactions checked: "Mark way out" puts a green EXIT box at the picture centre (in view,
tracked); tapping a DOOR box marks the way out at that door (`mark.door = true`, in view, tracked;
checked in `tests/e2e_shots.js` on the single door photo; in this round's slideshow run no DOOR box
was on screen at a moment the test could tap it); "Eyepiece view" renders grey (max channel
difference 0 in a sample patch); 390 px wide dark page and 1280 px light page have no horizontal
scroll (`tests/e2e_layout.js`).

Screenshots in `shots/`: `key_face_white.png`, `key_group_white.png`, `key_fire_purple.png`,
`key_door_green.png`, `key_door_exit_marked.png`, `key_exit_marked.png`, `key_eyepiece.png`
(single-image clips, `tests/e2e_shots.js`), plus `e2e_*.png` from the slideshow runs
(`e2e_false_fire_stage_light.png` shows the false FIRE on the concert photo) and `e2e_mobile_dark.png` /
`e2e_desktop_light.png` full pages. `e2e_cpu_faces_white.png`, `e2e_cpu_doors_green.png` and
`e2e_false_fire_stage_light.png` are from the 2026-10-03 build (this round's CPU runs never had a
box on screen while the same photo was shown, so the test did not retake them); the DOOR box in
`e2e_cpu_doors_green.png` scored 0.36, below today's 0.50 threshold. `e2e_mobile_dark.png` (a
layout check) was taken just after a slideshow cut: its white 3.0M and FIRE boxes belong to the
previous photo (a cook beside a pan flare) and are drawn over the burning car until the next
detector update about 2 s later. They show Open Images (CC BY 2.0) and FireNET (MIT) photos;
credits are in `LICENSES.md`.

## Limits (read before trusting a box)

* **Not a safety device.** It is a demo of the eyepiece overlay on an ordinary camera.
* **RGB, not thermal.** An ordinary camera sees light, not heat: it cannot see people through smoke or
  in the dark, and "Eyepiece view" only imitates the thermal picture (the page says so).
* **Fire model trained on small public datasets.** FireDoorNet saw 357 box-labelled fire photos
  (FireNET) plus about 3,000 door photos and 3,700 hard negatives from Open Images, on CPU for 107
  minutes. Held-out: fire AP@0.5 0.727 (90 FireNET test images, about +/-0.05), box recall 0.54 at
  the default threshold; it misses small or distant flames (41 % of the CCTV clip's fire frames at
  0.50, 61 % with the hysteresis the page uses). A candle or lighter flame 2-3 % of the frame wide is
  found in 60-80 % of frames when it is near the middle of the picture (centre zoom pass) and in
  27-53 % elsewhere. False FIRE boxes: 1.5 % of 976 hard-negative photos at 0.50, mostly low sun,
  glowing bulbs, LEDs and warm lamps (the zoom pass raises this a little: 17 -> 24 of 871 Open Images
  photos); candle and lighter flames count as fire. The hysteresis keeps a false FIRE box on longer
  once it appears (+41 % on a deliberately fire-like clip, same number of onsets). Fireplaces,
  fire on a TV or phone screen and fire filling the whole view are also unreliable. In the e2e
  run above, a motion-blurred concert photo in orange stage light was boxed as FIRE.
* **Doors are the weak class** (test AP 0.32, recall 0.26): clear, face-on doors are found; open,
  side-on, glass and small doors are missed. Wardrobes, windows and fridges were boxed as doors
  often enough (31 % at the model's 0.35) that the page now uses 0.50: 7 % of look-alikes, but only
  19 % of real doors in the verifier's photos.
* **Windows are weak too** (added 2026-10-07, details in `firedoor/MODEL.md`, "Window class"). On 40
  held-out photos of rooms with windows, a WINDOW box lands on a window in 17; on Open Images test
  photos with a big window (at least 5 % of the picture) in 32 %. Windows seen at an angle, cut by the
  picture's edge or showing only bright sky are often missed. False WINDOW boxes: about 1 in 10 photos
  of mirrors, pictures, TVs or wardrobes and of window-free scenes (lampshades, bright bulbs, framed pictures,
  TV screens, signs).
  A WINDOW box on a DOOR box is dropped (a glazed door is a door), and only the 4 strongest are drawn.
* **People:** COCO-SSD lite at 300x300 misses small or distant people (overall recall 0.58, 0.76 for
  people at least 20 % of the frame height, crowds 0.43) and can take a lone hand for a person.
  BlazeFace short range finds faces out to about 1.2 m in practice (and misses about half of the faces
  that fill the view); farther away the white box and its distance come from the person model.
* **Distances are rough pinhole estimates** with an assumed 65 degree field of view (webcams are
  55-80). Face-based distances use a 0.16 m face width; on 49 close-up photos the BlazeFace box
  width measured against the eye distance (0.063 m average) came out at a median 0.156 m
  (`tests/face_scale_check.js`), so the size assumption holds and the field of view dominates the
  error. A person box with no face uses body height, which reads too far for a half-visible person.
* **Exit tracking is camera motion only.** "Mark way out" remembers a direction and follows it from
  how the picture moves (plus the phone's motion sensor when it reports one). It does not know where
  you are or how far you walked; fast whip-pans, dark or featureless walls and cuts lose it ("WAY
  OUT LOST"), and turn angles can be 10-20 % off on a lens wider or narrower than 65 degrees.
  Each picture match is checked tile by tile, so a person walking across a still camera no longer
  drags the mark away (fake-camera walker clip: 0.02 degrees of drift, was 212). Two weak spots
  remain: on the CPU fallback (an update every 3-4 s) a turn faster than about 10 degrees/s shows
  "EXIT?" or "WAY OUT LOST" instead of a position, and a mark made while a person fills most of the
  picture can still slip by up to about 5 degrees (4 of 100 marks in the simulation).
* **Speed on real devices is unmeasured.** This machine has no GPU: WebGL ran on SwiftShader
  (Chromium's software renderer) on 2 cores. Real laptop/phone GPUs should be much faster, but that
  was not tested, and neither were Safari/iOS, Firefox, Android, a real webcam, a real motion
  sensor, real speech output or the page inside claude.ai itself.
* **CPU fallback is slow** (one update every 3-4 s here). The video keeps playing at the camera
  rate because the page shows the `<video>` element itself under a transparent box layer, and with
  the detectors in a worker the turn tracking, alerts and taps keep going between updates; only the
  boxes wait for the next update. Where the host refuses the worker the detectors share the page's
  thread and taps can take 0.2-0.3 s (see "Responsiveness").

## Licences

All models, datasets and test images are listed in `LICENSES.md` (TF.js, COCO-SSD and BlazeFace
Apache-2.0; FireDoorNet from MobileNetV2 ImageNet weights Apache-2.0, FireNET MIT, Open Images
CC BY 4.0 annotations / CC BY 2.0 images). The CGTN/CCTV news footage in the FireNET repo was used
for local testing only and is not in the page or any screenshot.

## What changed in integration (camera/page, camera/runtime untouched except where noted)

* `build_camera.py` (new): one command for the final page, with the checks above.
* `page/app.js`: real fire/door detector wired in through the existing adapter, FIRE hysteresis
  (`PS_FIREDOOR_META.hysteresis`), detector `reset()` on source change, test hooks
  (`window.__psLastDetections`, `window.__psDetections`, optional `window.__psCaptureHook`), live
  video shown by the `<video>` element (smooth on the CPU fallback; frame rate read from
  `getVideoPlaybackQuality`), and "Mark way out" re-anchors the turn tracker when the picture had
  stopped matching (before, a mark made right after a fast turn or cut went "lost" at once).
* `page/motion.js`: `MotionTracker.reanchor()` (+ a unit test in `page/tests/motion_test.js`).
* `page/page.html`: one About paragraph on the fire/door model's limits.
* `firedoor/decode.js`, `runtime/*`: unchanged.

## What changed in the fix round (2026-10-04, after the functional and quality verification)

Camera and page life cycle (`page/app.js`):
* Pressing Start twice no longer leaves a second camera running after Stop.
* A camera that sends no picture within 8 s is stopped with "The camera started but sent no picture";
  one that stops sending shows "No picture from the camera"; a track that ends, even without an
  `ended` event, gives "Camera stopped: the camera was switched off or disconnected". A camera that cannot open no longer drops the one that was
  working, and a late camera permission no longer replaces a photo opened in the meantime.
* A video file that does not autoplay starts on a tap, and blank frames before the first real frame
  are not analysed.
* A lost WebGL context or a stalled detector run is detected; detection restarts by itself, the
  stale boxes are removed, and the page says so. After 3 failures it stays on the CPU backend.
* Spoken alerts go through a queue with priorities: FIRE is said before a person, and a third phrase
  no longer cuts off the first two. "Way out tracking lost" is said only after 2 s of lost
  tracking and at most every 20 s; "restored" only after a "lost" and 2 s of steady tracking.
* Rotating the phone (the frame size changes) still clears the way-out mark, because the tracker
  cannot follow a rotation of the picture, but the page now says so.
* Labels are drawn after all boxes (FIRE on top) and move when another label takes their spot,
  so a person box no longer hides the FIRE label.
* On the CPU fallback the last person boxes are kept (shifted by the camera turn) for the updates
  that skip the person model, so white boxes no longer disappear on every second update: on the
  verifier's people clip with WebGL off, a white box in 34 of 35 updates (32 on a person), was 28 of
  45 (25 on a person).

Detection (`page/app.js`, `build_camera.py`, `runtime/people.js`):
* Door threshold 0.50 instead of 0.35, and the centre zoom pass for small flames (see "Fire/door
  settings in the page").
* The `<` (cut off by the frame edge) flag on a distance label is set when a face or person box
  ends within 1 % of the frame edge (was 0.2 %: cut-off boxes stop 0.2-1 % short of the edge), so a
  person right at the camera no longer reads a plain 1.3-2.2 m. On the verifier's stored
  detections, plain labels of 1.2 m or more on cut-off close faces went from 13 to 0 (Open Images
  close-ups) and from 5 to 0 (synthetic faces at the edge).

Way-out tracking (`page/motion.js`):
* Each picture match is checked tile by tile (a 4x4 grid, normalised cross-correlation on
  band-passed detail). A shift that only a moving person supports is rejected in favour of "no
  turn". After a pause or a dropped stretch of frames, a coarser detail band is used. Before the
  tracker re-keys on a match against the previous frame, it checks that the match also agrees with
  the key frame.
* With WebGL off, a turn the tracker cannot follow now shows "EXIT?" or "WAY OUT LOST" instead of
  a confident wrong position.

Documentation: the hysteresis wording (it keeps false FIRE on longer once one starts), the BlazeFace
range (about 1.2 m, not 2 m), the door threshold, the zoom pass and the tracking limits are now
stated in this file, `firedoor/MODEL.md`, `runtime/README.md`, `page/README.md`, the page's About
text and a comment in `firedoor/decode.js` (its code is unchanged).
