# PyroSight demo navigation (`camera/nav`)

The PyroSight way-back-out navigation, running in a web page: the **device's own C code**
(`core/src/ps_nav.c`, `ps_config.c`, `ps_alerts.c`) compiled to plain JavaScript, fed either by
the phone's motion sensors (real walking) or by a simulated walker (demo walk), with a map,
the eyepiece-style arrow and a helper that says where the green EXIT box goes in a camera picture.

One classic script, `dist/ps_nav.js` (190 KB), defines the global `PSNav`. No WebAssembly at run
time, no eval, no Worker, no network, no `alert()/confirm()/prompt()`; nothing touches the DOM
at load (it is safe to include in a Worker blob too). `demo.html` is a self-contained test page.

```
nav/
  build_nav.py          C -> wasm (zig) -> JS (wasm2js) + src/*.js -> dist/ps_nav.js, demo.html, dist/nav_demo.html
  src/nav_api.c         C API shim over the device code (same file for the native parity build)
  src/core.js           PSNav.Core: one instance of the C code; guidance(), crumbs(), popAlerts()
  src/stepdetect.js     PSNav.StepDetector: steps from accelerationIncludingGravity
  src/sensors.js        PSNav.MotionSensors: deviceorientation/devicemotion, iOS permission, absent/blocked detection
  src/walker.js         PSNav.DemoWalker: simulated firefighter + simulated BNO085 (from sim/sim_world.c)
  src/navigator.js      PSNav.Navigator: the one object a page uses (inputs, clock, events, keyboard)
  src/render.js         PSNav.MapRenderer, PSNav.ArrowWidget, PSNav.exitBox, theme tokens
  src/demo.src.html     template of the test page
  demo.html             built test page (fragment: no doctype/html/head/body)
  dist/ps_nav.js        built bundle (inline this)
  dist/nav_demo.html    demo.html with a doctype + viewport wrapper, to open directly
  tests/                parity_test.js (+ parity_native.c), stepdetect_test.js, geom_test.js,
                        walker_test.js, pw_demo_test.js (Playwright)
```

## Build and test

```sh
python3 camera/nav/build_nav.py           # needs: pip ziglang; binaryen wasm2js (WASM2JS=... or found automatically)
node camera/nav/tests/parity_test.js       # native gcc build vs JS build, same inputs
node camera/nav/tests/stepdetect_test.js   # synthetic accelerometer traces + orientation -> heading
node camera/nav/tests/geom_test.js         # direction words, EXIT box sides (mirrored / unmirrored), box size
node camera/nav/tests/walker_test.js 40    # auto demo closed loop over 40 seeds
node camera/nav/tests/pw_demo_test.js [--shots DIR]   # Playwright, Chromium at 390 px
```

Measured here (2-core Linux container, Node 22, Chromium headless; no phone, no real sensors, no GPU):

* **Parity C vs JS**: 3 auto-demo walks (~22 k commands each) + a 600 s stress walk (181 k commands:
  turns, trail at the 128-crumb limit and thinned, 2 s IMU outage -> LOST, crawling, "where is
  out?", re-marked entry, reset). Every integer (state, level, steps, crumbs, return target, alert
  phrases) identical; largest float difference 8e-8 relative, 5e-7 m in position.
* **Step detector**: walking 1.4-2.2 steps/s, hand-held, soft, in a pocket, at 60 Hz and 30 Hz:
  within 0-4 steps of 56-88 (<= 4.5%; the misses are the first steps of a walk); standing still, hand tremor, pocket jostle, shaking at
  4/5/6/8 Hz, knocks every 2 s: 0 steps; knocks every 0.6 s (a walking rhythm): 0-3 of 66;
  walk/stand/shake/walk: 66 of 68.
* **Auto demo, 40 seeds** (each with its own stride bias, gyro drift, scale error, missed steps):
  34/40 (85%) end within 1.5 m of the door; mean 0.85 m, median 0.76 m, p90 1.73 m, worst 2.14 m.
  6/40 runs dropped to UNRELIABLE ("Navigation estimate unreliable. Follow the hose line out.";
  the walker then follows the hose and ends 0.1-0.3 m from the door); every run went DEGRADED at
  some point (lowest confidence mean 0.32). Estimated vs true position differs by up to 1.8 m on
  average (max 3.5 m), so the two paths are visibly different on the map. ~40 m in, 70-110 s per run.
* **Playwright** (35 checks): auto fallback to the demo walk with a plain "no motion sensor data"
  message 1.5 s after load; 390 px with no horizontal scroll; every button/select >= 44 px; Walk
  hold, Left tap (15 deg), Right hold (continuous), keyboard arrows/WASD; auto demo to the end
  (EXIT box seen in the camera picture); heading-up toggle; emulated `deviceorientation` +
  `devicemotion` (walk 5 s, turn left 90, walk 5 s) -> 18 steps, heading +90, "behind" /
  "behind-left"; a 5 s busy main thread keeps the track (confidence 0.949 -> 0.945); sensor
  silence -> "lost"; iOS-style `requestPermission` denied / granted; dark theme; no errors or requests.
  `update()` + map + arrow drawing: ~0.1 ms per frame of JavaScript (canvas raster time not included).

## Interface

Everything hangs off the global `PSNav` (also `module.exports` in node).

### `new PSNav.Navigator(options)`

| option | default | meaning |
|---|---|---|
| `mode` | `'auto'` | `'auto'` (sensors, else demo), `'sensors'`, `'demo'`, `'camera'` |
| `win` | `globalThis` | object that receives `deviceorientation` / `devicemotion` and key events |
| `now` | `performance.now` | clock (ms) |
| `maxGapMs` | `250` | one update never advances the C clock more than this (see Clock) |
| `sensorTimeoutMs` | `1500` | no usable sensor data within this -> absent/blocked; silence -> lost |
| `externalTimeoutMs` | `2000` | camera-heading mode: hold the last fed yaw this long |
| `seed` | `1` | first demo seed (`autoDemo()` without a seed uses the next one) |
| `demoSpeed` | `1` | time factor for the auto demo / guide me out (manual demo is always real time) |
| `stepRateHz` | `1.8` | Walk button / key cadence |
| `stepLengthM` | `null` | override the device stride (0.62 m, heavy gear) |
| `headingSnap` | `null` | override the device's heading-snap heuristic (on) |
| `autoMarkEntry` | `true` | mark the entry where an input starts delivering heading |
| `updateMs` | `33` | `run()` interval |

**Starting and inputs**

* `nav.autoStart()` - call on page load (no tap needed). Where the browser needs a tap before it
  shares motion data (iOS `DeviceMotionEvent.requestPermission`), it starts the demo walk and
  `input.detail` says to tap "Use phone sensors". Elsewhere it is `start()`. Returns a Promise of `input`.
* `nav.start()` - from a tap: try the sensors; in `'auto'` mode fall back to the demo walk when
  no usable data arrives within 1.5 s, permission is refused, or values are all null.
* `nav.useSensors()` - from a tap (iOS permission). Promise of the sensor status. No fallback.
* `nav.useDemo({seed?, reason?})` - demo walk: walker at the door, entry marked, manual control.
* `nav.useCameraHeading(detail?)` - heading only from `feedHeading()`; steps from `walk(true)`.
* Switching input resets the track (another clock and heading reference).

**Heading hook (camera-turn tracker)**

* `nav.feedHeading(yawRad)` - counter-clockwise radians (+ = turned LEFT), any fixed reference.
  Used in `'camera'` mode (ignored otherwise). The last value is re-fed to the C code on every
  update for `externalTimeoutMs`, because the tracker updates slower than the device IMU
  timeout (300 ms); after that the device goes LOST and its confidence decays.
* `nav.feedCameraTrackerYaw(yawDegClockwise, trackerState)` - motion.js convenience:
  `tracker.pose.yaw` is degrees CLOCKWISE (right = +). When `trackerState === 'lost'` (the picture
  still comes but does not match: a blank wall, a fast turn) the last good heading is fed again
  instead, so the device does not treat it as a motion-sensor outage; only when no frames come at
  all does the held heading run out after `externalTimeoutMs`. In this mode the module also feeds
  the device a body acceleration (2.2 m/s^2 while Walk is held, 0.2 standing), as the demo walker
  does, so its crawl path ("moving without counted steps") stays off while the user stands still.
  E.g. after `const r = tracker.update(grey, W, H, t)`: `nav.feedCameraTrackerYaw(r.pose.yaw, r.state)`.

**Actions**: `markEntry()`, `whereOut()` (the device answers with an alert), `reset()` (back to
"entry not marked"), `walk(on)` (demo: the walker; other modes: steps at `stepRateHz`),
`turn(deg)` (demo, + = left), `turnHold(dir)` (+1 left, -1 right, 0 release; tap = 15 deg,
hold > 300 ms = 90 deg/s), `autoDemo(seed?)`, `guideMeOut()` (demo walker follows the arrow
out by itself), `setSpeed(x)`.

**Loop**: `nav.update()` advances everything and returns the guidance (call per frame), or
`nav.run()` (own `setInterval`, keeps going when rAF is throttled) / `nav.stop()` (stops the loop
and sensor listeners, keeps the track).

**Keyboard / buttons**: `nav.bindKeys(target?)` -> unbind. ArrowUp/W hold = walk, ArrowLeft/A
and ArrowRight/D = turn (tap 15 deg, hold continuous; releasing one turn key while the other is
still held keeps turning that way); ignored while typing in a form field.
`PSNav.holdButton(el, onDown, onUp, {touchDelayMs: 130, slopPx: 10})` -> unbind, for
press-and-hold buttons (no context menu). Mouse and pen act at once; a finger acts once it has
rested on the button for 130 ms (a quicker tap = one press), and a swipe that starts on the button
(more than 10 px, or the browser's own scroll: `pointercancel`) does nothing, so the page still
scrolls. Several fingers on one button: released when the last lifts. Give hold buttons
`touch-action: pan-y; user-select: none`.

**State**: `nav.g` (last guidance), `nav.input`, `nav.stats` (demo: `maxEstErrM`, `minConfidence`,
`unreliableEntries`, `followHoseAlerts`), `nav.alertLog` (last 50), `nav.walker`
(`phase`: `manual|inbound|scan|outbound|done`, `result`: `{how: 'exit'|'hose'|'outside'|'timeout',
doorErrorM, tOutS, hose, ref: 'door'|'mark'}`; measured from where the entry was marked, which is
the floor plan's door unless Start here was pressed again inside: `ref` says which), `nav.stalls`
(updates clamped by `maxGapMs`). While following the hose, a walker pinned at a corner for 3 s
heads for the furthest hose point in plain sight (within 8 m), or back to the previous one; the
hose is the path walked in, not the path walked since "guide me out".

**Events**: `nav.on(type, fn)` returns an unsubscribe function.

| type | payload |
|---|---|
| `'guidance'` | guidance object, after every update |
| `'alert'` | `{text, parts: [phrase ids], names: ['WAY_OUT_IS', 'DIR_LEFT'], prio, t}` from the device's alert queue: "Entry point marked.", "Way out is to your left." (on dropping to DEGRADED, then every 15 s, and on `whereOut()`), "Navigation estimate unreliable. Follow the hose line out." (UNRELIABLE, every 20 s), "Navigation restored.", "You are at the entry point." The module does not speak; `PSNav.speak(text)` uses the browser voice if the page wants it. |
| `'input'` | `input` (below), on every change |
| `'step'` | `{t, source: 'sensors'|'demo'|'button'}` |
| `'entry'` | `{t}` |
| `'demo'` | `{phase, result}` |

**Guidance object** (`nav.g`, also `PSNav.Core#guidance()`):

```
state 'idle'|'tracking'|'lost'   valid (entry marked)
routeBearingDeg, homeBearingDeg  relative to where the user faces, + = LEFT (route = next breadcrumb, home = straight line)
routeDistM, homeDistM, posSigmaM, confidence (0..1)
level 'GOOD'|'DEGRADED'|'UNRELIABLE', levelId 0|1|2   (device thresholds 0.6 / 0.3, hysteresis 0.05, ps_alerts.c)
exitIsNext                       the door is the next point on the way out
word, homeWord                   'ahead'|'ahead-left'|'left'|'behind-left'|'behind'|'behind-right'|'right'|'ahead-right'
phraseIds, phrase                "Way out is ahead, to your left." or "Navigation estimate unreliable. Follow the hose line out."
followHose (= UNRELIABLE), atExit (route < 1 m: the device ring shows EXIT)
pos {x, y} (entry frame), yaw (rad), distWalkedM, steps, turns, nCrumbs, returnTarget, headingSigmaDeg, untrackedS
t (core clock ms), input (mode)
```

**Input object** (`nav.input`): `{mode: 'none'|'sensors'|'demo'|'camera', status: 'off'|'asking'|
'waiting'|'ok'|'denied'|'blocked'|'absent'|'lost', label, detail, heading, steps}`. Show
`label` + `detail` as the "which input is active" indicator; `detail` is a plain sentence
(e.g. "Motion sensors not used: No motion sensor data arrived. The device has no motion
sensors, or this page is not allowed to read them (inside claude.ai they are usually blocked).
Use the demo walk."). The sensors count as `'ok'` only once a heading (orientation or gyro) has
arrived: a phone that reports acceleration but no rotation ends as `'absent'` ("no gyroscope").
A frame whose permissions policy does not allow `accelerometer` and `gyroscope`
(`document.featurePolicy.allowsFeature`, e.g. an iframe with `allow=""`) is `'blocked'` at
once, inside the tap, instead of after 1.5 s without data.

### Rendering

```js
const map = new PSNav.MapRenderer(mapCanvas, { up: 'entry' });   // or 'heading'; map.toggleUp()
const arrow = new PSNav.ArrowWidget(arrowCanvas);
function frame() { map.draw(nav.snapshot()); arrow.draw(nav.g); requestAnimationFrame(frame); }
```

* Canvases are sized from their CSS size x devicePixelRatio on every draw; give them a CSS
  width/height (the demo uses `width: 100%; height: min(78vw, 460px)` for the map, 120-150 px
  square for the arrow).
* Map: entry door (green bar, "ENTRY"), estimated trail + breadcrumbs (blue), return route to
  the door (green dashed; red when UNRELIABLE), position dot + heading wedge, 1-sigma circle,
  scale bar, "Entry direction up" / "Heading up" (the device has no compass, so "north" is the
  direction faced when the entry was marked; heading-up shows a small arrow for that
  direction), auto-zoom. Demo only: floor plan, true path (dashed, muted) and true position ring.
* Arrow: device ring. Green GOOD / amber DEGRADED; arrow to the next breadcrumb (relative to
  facing), white dot = straight line to the door, distance; "EXIT" under 1 m; UNRELIABLE: no
  arrow, blinking "FOLLOW HOSE" and "NAV +-N M" (as the eyepiece); "MARK ENTRY" when idle.
* Colours from CSS custom properties on the canvas or an ancestor (re-read every 30 draws or on
  `refreshTheme()`), or a `theme` object: `--nav-bg --nav-fg --nav-muted --nav-wall --nav-trail
  --nav-true --nav-route --nav-door --nav-me --nav-good --nav-warn --nav-bad --nav-ring`.
  The EXIT/door green is `#28ff50` (`PS_COLOR_EXIT`) by default.

### EXIT box in the camera picture: `PSNav.exitBox(g, opts)`

Same rules and geometry as the device's `draw_exit()` (`core/src/ps_display.c`): only when the
door is the next point on the way out, not UNRELIABLE, at least 1 m away, within half the field
of view + 5 deg; door 0.9 m wide, 2.0 m tall, camera 1.5 m above the floor; depth = d cos(bearing).

`opts`: `hfovDeg` (65; across the picture width; `fovOnLongSide: true` for motion.js's long-side
convention), `mirrored`, `cameraLooksBack` (defaults to `mirrored`), `width`, `height` (picture px,
optional), `cameraHeightM`, `doorWidthM`, `doorHeightM`, `marginDeg`.

Returns, in **display** coordinates (already mirrored when `mirrored` is true):

* `{kind: 'box', x, y, w, h, px, label: 'EXIT 5M', distM, bearingDeg, cameraBearingDeg}` - x, y
  top-left, 0..1 of the picture (`px` in pixels when width/height were given);
* `{kind: 'route', x, label: 'OUT 12M', ...}` - the door is not next yet, but the next leg of the
  way out is in the picture at x: draw a green chevron there;
* `{kind: 'edge', side: 'left'|'right', behind, label, ...}` - outside the picture: arrow at that edge;
* `{kind: 'none', reason}` - no entry, unreliable ("follow the hose"), or at the door.

Left/right (bearing b, + = to the user's left; f = (W/2)/tan(hfov/2)):

* back camera / eyepiece (looks where the user faces), not mirrored: x = W/2 - f tan(b), so a
  door on the user's left is on the left of the picture (identical to `draw_exit()`);
* front (selfie) camera, mirrored preview: it looks BACK at the user, so its axis is facing + 180;
  bc = wrap(b - 180), raw x = W/2 - f tan(bc), displayed x = W - raw x. A door behind the user on
  their left appears on the LEFT of the preview (as in a mirror); a door ahead-left (not in view)
  gets a LEFT edge arrow. Screen side = the user's side in both cases.
* `mirrored: true, cameraLooksBack: false` (a mirrored view of a forward camera): x = W/2 + f tan(b).

In `camera/page/app.js` terms: `PSNav.exitBox(nav.g, { hfovDeg: effectiveHfov(src.w, src.h),
mirrored: src.mirror, width: src.w, height: src.h })` and draw the box with
`toCanvas(box, fit, false)` (it is already mirrored; do not flip twice), `COL.exit`, label `box.label`.

### Lower-level pieces

`PSNav.Core` (`markEntry(t)`, `onYaw(rad, t)`, `onStep(t)`, `onAccel(mag, t)`, `tick(t)`,
`whereOut(t)`, `reset()`, `guidance()`, `crumbs()`, `popAlerts()`, `raw()`, `setCfg(name, v)`,
`getCfg(name)`), `PSNav.StepDetector` (`push(ax, ay, az, tMs)` -> step times; `.linear`),
`PSNav.MotionSensors`, `PSNav.DemoWalker`, `PSNav.directionWord(b)`, `PSNav.DIR_WORDS`,
`PSNav.PHRASE`, `PSNav.LEVELS`, `PSNav.orientationYaw(alpha, beta, gamma, screenAngle)`,
`PSNav.speak(text)`.

## Conventions and design notes

* Frames: entry frame = origin at the marked doorway, +x = heading when the entry was marked,
  +y = left; yaw counter-clockwise (turning left increases it); bearings + = left. Same as the device.
* **Clock**: the C code has its own millisecond clock. Outside demo mode it follows
  `performance.now()`, but one update advances it at most `maxGapMs` (250 ms). A page that freezes
  for seconds (the camera page's model runs: ~5 s at start, ~2 s per update on this machine) is
  not an IMU outage; without the clamp the device code would declare the IMU lost after 300 ms
  and collapse its confidence. Steps a phone takes during such a freeze are still lost (the
  browser does not deliver motion events while the page is busy). The sensor watchdog also
  discounts stalls. In demo mode the clock is the walker's simulated time in 10 ms steps.
* **Motion sensor lost**: like the device's health supervisor (`ps_system_tick`), the C shim
  (`nav_api.c`) says "Motion sensor lost." once when the heading / acceleration stream stops for
  longer than the IMU timeout (300 ms of the clamped clock) while navigating.
* Steps: peaks 0.25-1.4 s apart count (slow, careful walking down to ~0.7 steps a second). While
  the step detector mutes a shake or a knock, the device is fed no linear acceleration, so shaking
  the phone while standing does not move the position through the device's crawl path.
* EXIT edge arrow: `behind` is in the user's frame (|bearing| > 90 deg), also for a front camera
  that looks back at the user (a door straight ahead is not labelled BEHIND there).
* Phone heading: `deviceorientation` -> quaternion (W3C Z-X'-Y'') -> screen-orientation
  correction -> `ps_quat_to_yaw()` (heading of the screen's right-hand axis). That axis stays
  level whether the phone is upright or flat, portrait or landscape; when it is nearly vertical
  the last heading is held. Chrome's `deviceorientation` is relative (no magnetometer), like the
  BNO085 game rotation vector. Fallback: integrate `rotationRate` about gravity (untested on a device).
* Demo walker (from `sim/sim_world.c`): same floor plan (corridor A, corridor B, room), the
  device sim's walk-out policy (arrow only; hose when UNRELIABLE) and IMU model: stride bias
  N(-6%, 4%) per seed and 5% per-step noise, gyro drift 0.02-0.05 deg/s per seed + random walk,
  0.5-1.5% scale-factor error on turns, 0.003 rad noise, 1-3% missed steps. Changes from the
  simulator, so the walker behaves like a person: it walks along a wall or corridor instead of
  bumping into it or crossing it at a slant (bumping made the IMU heading differ from the motion
  and dominated the error); the room route uses axis-aligned or clearly diagonal legs (walking
  straight 5-12 deg off an axis is where the device's heading snap misleads itself: with a
  perfect IMU the old route ended 2.1 m off); the room scan is a left-right sweep instead of a
  360 deg spin. The auto demo route is ~38 m in.

## Not tested / limits

* No real phone, real motion sensors or real walking here: the sensor path is tested with
  synthetic accelerometer traces and emulated `DeviceOrientationEvent`/`DeviceMotionEvent`
  in Chromium. Real step-detection accuracy, iOS behaviour and heading quality are unmeasured.
* A phone held in the hand is not the device's body-worn IMU; expect larger errors than the demo.
* Inside the claude.ai artifact sandbox motion events are most likely blocked; the module then
  says so and runs the demo walk.
* The iOS permission flow is tested only with a stubbed `requestPermission`.
* Canvas raster cost on a phone is unmeasured (JS cost ~0.1 ms per frame here).
