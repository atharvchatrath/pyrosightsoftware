# PyroSight software architecture

## One core, two hosts

```
                ┌──────────────────────── core/ (portable C99, no heap) ────────────────────────┐
                │ ps_thermal   denoise (motion-adaptive IIR + 3x3 median), display window,      │
                │              detail enhancement, fixed model-input curve                      │
                │ ps_detect    threshold detector, NMS, CenterNet decode, distance from size    │
                │ ps_nav       step + heading dead reckoning, turns, breadcrumbs, loop pruning, │
                │              heading snap, uncertainty relative to the way out                │
                │ ps_alerts    phrase ids, priority queue, confidence levels with hysteresis    │
                │ ps_display   RGB565 compositor: thermal, boxes, arrow, status, night palette  │
                │ ps_power     battery %, low/critical modes, rate and brightness policy        │
                │ ps_system    owns all of the above; the API firmware and simulator drive      │
                └───────────────────────────────────────────────────────────────────────────────┘
                         ▲                                               ▲
         firmware/ (ESP-IDF, ESP32-P4)                        sim/ (Linux host)
         Lepton VoSPI+CCI, BNO085 SH-2,                       ray-cast thermal scenes,
         ESP-DL detector, MIPI-DSI OLED + PPA,                BNO085 model, closed-loop
         ES8311 audio, buttons, battery ADC                   walk-out, metrics, PPM/GIF
```

Every algorithm lives once, in `core/`. The firmware adds drivers and tasks;
the simulator adds a synthetic world. A behaviour seen in simulation is the
behaviour the device will have.

## Data flow per frame

| Step | Where | Rate | Notes |
|---|---|---|---|
| VoSPI capture, 4 segments x 60 packets | firmware/components/lepton | 8.7 Hz | low gain, TLinear 0.1 K, converted to int16 deci-C |
| Denoise, display map, model input | `ps_system_on_frame` | 8.7 Hz | ~1 ms on a desktop core; budget 10 ms on P4 |
| Detection | ESP-DL model, or `ps_detect_classical` | each new frame, cap 16/s | 62 ms budget; falls back to threshold detector |
| Distances, peaks, person call-outs | `ps_system_on_detections` | per detection run | spoken call-outs only from the neural detector |
| IMU yaw, steps, linear accel | firmware/components/bno085 | 100 Hz | game rotation vector: no magnetometer indoors |
| Health, nav confidence, nav alerts | `ps_system_tick` | 100 Hz | |
| Compose eyepiece frame | `ps_system_render` | 30 Hz | arrow tracks head turns between thermal frames |
| Scale and send to micro-OLED | PPA + MIPI-DSI | 30 Hz | 320x240 -> 960x720 |
| Audio | `ps_system_next_alert` -> PCM clips -> ES8311 | on demand | |

The camera delivers ~8.7 frames/s, so the "~16 fps inference" target is a
latency budget (62 ms per inference), not a throughput the camera can feed.
The display runs faster than the camera so the arrow responds to head turns
immediately.

## Navigation design choices

* **Breadcrumb retrace, not straight line.** The arrow points to the next
  breadcrumb on the way in, reversed. A small white dot on the ring shows the
  straight-line direction to the door for context. Straight lines go through
  walls.
* **Loop pruning.** Coming back within 1.5 m of an older crumb deletes the
  detour after it, so wandering around a room collapses to the shortest known
  path.
* **Confidence relative to the way out.** Stride and heading errors are shared
  by the trail and the position estimate, so retracing cancels most of them.
  The model counts stride bias over the route still ahead, heading drift since
  each remaining crumb was dropped, and displacement that was never measured
  (crawling, IMU outage). Confidence = 1 / (1 + (sigma / 4 m)^2).
* **Levels.** Above 0.6: green arrow, silent. 0.3 to 0.6: yellow arrow and
  "Way out is ..." every 15 s. Below 0.3: no arrow, blinking FOLLOW HOSE and the
  spoken warning every 20 s. Hysteresis 0.05.
* **Heading snap.** Walking straight within 12 deg of the doorway wall's axes
  nudges the heading toward the axis. In the unit test this cut cross-track
  error over 40 steps from 1.26 m to 0.52 m. It is a heuristic and can be
  turned off (`heading_snap`).
* **Crawling.** Motion with no counted steps advances the estimate at 0.3 m/s
  and grows uncertainty quickly; in simulation that cut crawl error from
  ~15 m to ~3 m, and those runs still end in "follow hose", which is correct.

## Display

320x240 RGB565 composed by the core, scaled 3x by the ESP32-P4 PPA. White-hot,
iron and night (amber) palettes. Every HUD element has a black halo so it reads
against both a white-hot ceiling and black floor. Pixels above the fire
threshold are drawn solid orange. Box colours have one meaning each:

* **white box**: a person, with a distance label (`<` prefix when the person
  is cut off by the frame edge, meaning the real distance is smaller);
* **purple box**: fire, labelled FIRE;
* **green box**: the exit. Drawn where the device believes the doorway is,
  sized for a 0.9 x 2.0 m door at the dead-reckoned distance, and only when
  the doorway is the next point on the way out (so it never points through a
  wall the route goes around) and navigation is not UNRELIABLE.

Colours are `PS_COLOR_PERSON`, `PS_COLOR_FIRE`, `PS_COLOR_EXIT` in
`ps_display.h`.

## Memory (ESP32-P4)

`ps_system_t` is ~300 KB and lives in PSRAM. Two RGB565 frame buffers at
320x240 are 150 KB each; the scaled DSI buffers at panel resolution are the
largest allocation and also live in PSRAM.

## No radio

Nothing in the firmware enables WiFi or Bluetooth. All logs stay on the device
(serial during development).
