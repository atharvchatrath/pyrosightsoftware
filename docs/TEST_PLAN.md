# PyroSight test and validation plan

Covers design step 8 (testing and validation) and the validation part of step 2.
Each test says what is already automated in this repository and what needs the
real hardware, a smoke chamber or a training building.

## 0. Automated, runs on any Linux machine

| What | Command | Checks |
|---|---|---|
| Core unit tests | `cmake -S . -B build && cmake --build build && (cd build && ctest)` | thermal denoise and mapping, detector, NMS, CenterNet decode, distance, dead reckoning, way-back arrow, crawl and IMU-loss handling, alert levels and hysteresis, power modes, full-system integration including camera/IMU loss |
| Firmware protocol tests | `make -C firmware/test_host test` | Lepton VoSPI packet/segment assembly, CRC, resync; BNO085 SHTP/SH-2 parsing; audio partition index |
| ML tests | `python3 -m pytest ml/tests -q` | model-input code and decoder bit-parity with the C core, dataset round trip, model shapes |
| Scenario simulator | `./build/sim/ps_sim --scenario corridor --out run/` | one closed-loop search-and-exit with ground truth |
| Navigation bench | `python3 sim/monte_carlo.py --seeds 20` | 4 scenarios x 20 seeds: exit success, nav error, detector P/R |

Simulator results at the time of writing (20 seeds each, threshold detector, heading snap on):

| scenario | out by arrow | out by hose (after "unreliable" warning) | failed | median max nav error | worst |
|---|---|---|---|---|---|
| corridor (36 m route, 6% stride error) | 19/20 | 1/20 | 0/20 | 1.44 m | 3.56 m |
| long (3 search laps, 12% stride error, high drift) | 0/20 | 20/20 | 0/20 | 4.02 m | 7.29 m |
| crawl (15 m crawled, no steps) | 0/20 | 20/20 | 0/20 | 3.27 m | 4.25 m |
| dropout (3 s camera + 4 s IMU outage) | 0/20 | 20/20 | 0/20 | 5.13 m | 7.32 m |

"Out by hose" is the designed behaviour, not a failure: the device judged its
own estimate unreliable, said so, and the simulated firefighter followed the hose.
The thing to watch is an arrow that is still shown while wrong. Across these
80 runs, the largest position error while the arrow was still on screen during
the walk-out was 3.6 m (corridor) and 4.7 m (long search). An earlier version
showed an 8 m error after an IMU outage; uncertainty growth during outages was
made deliberately pessimistic as a result, so short outages now end in
"follow hose".

The threshold (non-AI) detector finds fire reliably (precision ~0.95, recall 1.0)
but people poorly against a hot gas layer (precision 0.1-0.3, recall 0.05-0.25).
That is why spoken "Person" call-outs come only from the neural detector.

## 1. Thermal pipeline (design step 1)

| Test | Method | Pass criteria |
|---|---|---|
| Frame rate | Firmware stats log over 10 min | >= 8.5 frames/s delivered, 0 resync storms; `frames_dropped` < 1% |
| Radiometry | Blackbody or calibrated hot plate at 35, 100, 300 C, low-gain TLinear | within the Lepton spec (+-5 C or 5%, low gain) |
| Noise | Uniform target, 100 frames, compare raw vs denoised | temporal SD reduced >= 2x (unit test shows 5.0 -> 0.9 dC on synthetic noise) without lag on a moving hand |
| Display contrast with fire in view | 300 C source in corner of a room | door frames and people still distinguishable (unit test `test_display_mapping`) |

## 2. Detection accuracy in smoke (design steps 2 and 8)

Smoke chamber or fire-training building, recorded with the actual Lepton in
low-gain radiometric mode. Use `ml/eval.py`, which slices by smoke level.

| Condition | Targets | Pass criteria (proposal) |
|---|---|---|
| Clear, 1-12 m | standing, crouched, crawling, lying people in turnout gear and in plain clothes | person recall >= 0.9 at <= 0.1 false alarms per minute |
| Light smoke (visibility 3-10 m) | same | recall >= 0.85 |
| Dense smoke (visibility < 1 m) | same | recall >= 0.8; this is where the eyepiece earns its keep |
| Hot gas layer > 100 C | people below the layer | recall >= 0.8; check people that appear cooler than background |
| Hard negatives | radiators, hot pipes, lamps, sunlit windows, dogs, mannequins at room temp | false "person" call-outs <= 1 per 10 minutes |
| Fire | flames, glowing embers, hot surfaces | recall >= 0.95; report hot-surface confusion separately |
| Distance | people at measured 2, 4, 6, 8, 10 m | median error <= 25% standing; report crouched/lying separately |

## 3. Navigation (design step 4)

1. **Calibrate** each wearer: walk a measured 20 m three times upright and three
   times crouched, spin in place 3 turns; run `tools/calibrate_steps.py`.
2. **Course tests** in a training building, wearer in full gear, eyes covered
   or smoke on, with a spotter: mark entry, follow a set route of 20/40/80 m
   with 2-8 turns, press "where is out", walk out following only the arrow.
   Record final distance from the door and every audio prompt.
3. **Pass criteria (proposal):** at 40 m and 4 turns, 90% of runs end within
   2 m of the door; when the device says "unreliable, follow hose", it was
   actually > 2 m off at least 50% of the time (the warning is not crying wolf);
   a confident arrow (no warning) pointing more than 45 deg wrong: never.
4. Repeat crawling for part of the route, and with a 5 s IMU unplug.

## 4. Latency and responsiveness

| Path | Budget | How measured |
|---|---|---|
| Photon to display | <= 150 ms | LED blink in view, high-speed phone camera through the eyepiece |
| Frame to detections | <= 62 ms (the ~16 fps inference budget) | firmware `last_infer_us`, logged every 5 s |
| Thermal pipeline (denoise + maps) | <= 10 ms on ESP32-P4 | firmware stats; host reference ~1 ms |
| Head turn to arrow update | <= 50 ms | IMU at 100 Hz, display at 30 fps |
| Button to first audio | <= 300 ms | scope on button GPIO and headphone output |

## 5. Edge cases

| Case | Expected behaviour | Covered by |
|---|---|---|
| Lepton stops (cable, VoSPI desync) | "Thermal camera lost" spoken, THERMAL LOST on screen, boxes cleared, nav continues; auto-resync and "restored" | `test_system`, sim `dropout`, firmware resync tests |
| IMU silent | "Motion sensor lost", NO IMU on screen, nav state LOST, uncertainty grows at walking pace, recovers on data | `test_nav`, `test_system`, sim `dropout` |
| Crawling | motion without steps advanced at 0.3 m/s, uncertainty grows, usually ends in "follow hose" | `test_nav`, sim `crawl` |
| Long search, drift | confidence falls, "Way out is ..." every 15 s, then "unreliable, follow hose" | sim `long` |
| Model missing or too slow | falls back to the threshold detector, status shows TH | firmware detector component |
| Battery low / critical | "Battery low", then "Battery critical, exit now"; inference rate and brightness reduced, nothing switched off | `test_alerts_power` |
| Entry never marked | screen says MARK ENTRY, no arrow, no nav alerts | display and system code |
