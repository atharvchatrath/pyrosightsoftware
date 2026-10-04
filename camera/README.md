# PyroSight Camera

One web page that puts the PyroSight eyepiece boxes on **your own camera** (laptop webcam or phone
camera), live, inside the browser:

| What it sees | Box | Label |
|---|---|---|
| a person, including just a face close to the camera | **WHITE** `#FFFFFF` | rough distance, device style: `0.6M`, `2.9M`, `<1.2M` (cut off by the frame edge) |
| fire / flames | **PURPLE** `#C850FF` | `FIRE` |
| a door | **GREEN** `#28FF50` | `DOOR` |
| the way out you marked ("Mark way out", or tap the picture; a tap on a DOOR box marks that door) | **GREEN** `#28FF50` | `EXIT`, or an edge arrow `EXIT 120°` when you have turned away |

The colours are `PS_COLOR_PERSON / PS_COLOR_FIRE / PS_COLOR_EXIT` from
`core/include/pyrosight/ps_display.h`. **Video never leaves the device**: the three detectors run in
the page with TensorFlow.js, nothing is fetched or uploaded at run time, and the page says so.

It also has spoken alerts (off by default, the device's phrases from `core/src/ps_alerts.c`), an
"Eyepiece view" (grey, 160 px wide, labelled *look-alike, not thermal*), "Where's the way out?",
a photo/video fallback when the camera is blocked, and "Save this page" when it runs as a claude.ai
artifact with the downloads capability.

## Open it

* **As a file:** double-click `dist/pyrosight_camera.html` (8.2 MB, works offline from `file://`),
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
`build/firedoor/meta.json` (thresholds fire 0.50 / door 0.50, the FIRE hysteresis: on at 0.50,
kept while an overlapping box stays at 0.35 or more, and the centre zoom pass: middle 50 % of the
frame, FIRE from it at 0.60 or more; see `firedoor/MODEL.md` and "Fire/door settings" below), then runs
`page/build_page.py`, which inlines, in order: TF.js 4.22.0 (`tf.min.js`), `runtime/oplist.js`,
`runtime/people.js`, `runtime/dist/people_assets.js` (person + face weights), the fire/door model +
`firedoor/decode.js`, `page/motion.js`, `page/app.js`, into `page/page.html`. It then checks both
outputs and stops on any failure:

* size under 15 MB (base64 counted);
* `<title>PyroSight Camera</title>`;
* no external resource in the markup (no `<script src>`, `<link>`, `<img src>` other than `data:`,
  `<iframe>`, `@import`, `url(...)`), and no `fetch`, `XMLHttpRequest`, `import()`, `WebSocket`,
  `sendBeacon`, `Worker` or `http(s)://` in the page's own code (app, motion, oplist, people, decoder);
* the fragment has no doctype/html/head/body wrapper; the standalone page has them.

The 16 `http(s)` strings left are inside `tf.min.js` (licence and docs links, core-js URL feature
tests, TF.js's HTTP model loader that the page never calls); they are listed in
`dist/pyrosight_camera.build.json`. The browser tests confirm the page makes **no** request other than
`file:`, `data:` and `blob:`.

Page parts (MB): TF.js 1.47, person + face models 5.16, fire/door model + decoder 1.43, page code and
markup 0.16. Total 8.23 MB.

## Tests

```sh
cd camera
python3 tests/firedoor_ref.py                         # onnxruntime references (32 held-out inputs)
node tests/firedoor_parity.js build/firedoor/firedoor # op-list runtime (TF.js CPU) vs onnxruntime
python3 tests/make_e2e_clips.py --clips-dir DIR       # barcoded fake-camera clips (~0.8 GB, regenerable)
taskset -c 2,3 node tests/e2e_test.js --secs 60       # Playwright + fake camera, WebGL and CPU fallback
taskset -c 2,3 node tests/e2e_layout.js               # 390 px dark / desktop light screenshots
python3 tests/face_scale_check.py DIR && node tests/face_scale_check.js DIR   # distance-label sanity check
node tests/e2e_shots.js                               # key screenshots from single-image clips
cd page && node tests/motion_test.js && node tests/browser_test.js   # the page's own suites (still pass)
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
1000 updates: boxes normalised to the camera frame, per-model ms, backend, `tf.memory().numTensors`).

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
* **CPU fallback is slow** (one update every 3.2-3.6 s here). The video keeps playing at the camera
  rate because the page shows the `<video>` element itself under a transparent box layer, but the
  boxes, turn tracking and alerts only refresh between detector runs.

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
