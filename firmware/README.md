# PyroSight firmware (ESP32-P4)

ESP-IDF firmware for the PyroSight thermal eyepiece. All algorithms live in
the portable core library (`../core`), compiled unchanged into the firmware by
`components/pyrosight_core` (sources referenced in place, never copied). This
directory adds only drivers, tasks and glue code around `ps_system_t`
(see `core/include/pyrosight/ps_system.h`).

```
firmware/
  CMakeLists.txt  sdkconfig.defaults  partitions.csv
  main/                  app_main, tasks, board_pins.h, Kconfig.projbuild, idf_component.yml
  components/
    pyrosight_core/      core/src/*.c built as an IDF component
    lepton/              FLIR Lepton 3.5: VoSPI assembler, CCI, SPI/VSYNC driver
    bno085/              BNO085: SHTP/SH-2 parser, INT-driven I2C driver
    microoled/           MIPI-DSI micro-OLED, PPA 3x upscale, DCS brightness
    audio/               ES8311 + I2S, clip partition, fallback beeps
    detector/            ESP-DL v3 model wrapper (C++ with a C API)
    power/               battery ADC, DFS ceiling from the power policy
    buttons/             debounced short/long press
  test_host/             host unit tests for the hardware-independent code
```

## Build and flash

Requires ESP-IDF v5.3 or later (v5.4 recommended) with ESP32-P4 support.

```sh
cd firmware
idf.py set-target esp32p4
idf.py menuconfig           # PyroSight -> pins, panel timing, volume, ...
idf.py build flash monitor  # app + partition table
```

Managed components (`espressif/esp-dl`, `espressif/esp_codec_dev`) are fetched
by the component manager on the first build.

Data partitions are flashed separately, so a new model or new voice clips
need no firmware rebuild:

```sh
# Neural detector (ESP-DL .espdl exported from ml/ with esp-ppq), <= 4 MB
../tools/flash_partitions.sh -p /dev/ttyUSB0 --model path/to/pyrosight.espdl
# Voice clips: builds build/audio/audio.bin from core/src/ps_alerts.c, then flashes it
../tools/flash_partitions.sh -p /dev/ttyUSB0 --audio
# Disable the neural detector (classical detector only)
../tools/flash_partitions.sh -p /dev/ttyUSB0 --erase-model
```

`tools/make_audio_clips.py` uses `espeak-ng` when installed and writes
placeholder chirps otherwise; recorded clips can be dropped in with
`--override-dir`. An empty or missing `audio` partition is not fatal: every
alert then plays a priority-coded beep pattern. An empty `model` partition
selects the classical detector.

Partition layout (16 MB flash):

| name    | type/subtype | offset   | size   | content                         |
|---------|--------------|----------|--------|---------------------------------|
| nvs     | data/nvs     | 0x9000   | 24 KB  |                                 |
| factory | app          | 0x10000  | 3 MB   | firmware                        |
| model   | data/spiffs  | 0x310000 | 4 MB   | ESP-DL model (.espdl)           |
| audio   | data/0x40    | 0x710000 | 2 MB   | clip pack, see `audio_pack.h`   |

## Host tests

No ESP-IDF needed: the protocol code is kept in files without IDF includes.

```sh
make -C firmware/test_host        # builds with ASan/UBSan and runs everything
```

| test                  | covers                                                                         |
|-----------------------|--------------------------------------------------------------------------------|
| `test_lepton`         | VoSPI CRC, discard packets, segment ordering, missing/out-of-order segments, invalid (TTT=0) segments, resync, CRC auto-disable, TLinear -> deci-C, CCI sequencing on a fake camera |
| `test_shtp`           | SHTP header/build, Set Feature encoding, base timestamp/rebase, Game RV -> yaw, linear accel, step detector, unknown/truncated reports, reset complete, product ID |
| `test_audio_pack`     | partition index parser incl. corrupt images; beep patterns; parses the image produced by `tools/make_audio_clips.py` |
| `test_buttons`        | debounce, short/long classification, bounce, held-at-boot, timer wrap         |
| `test_detector_quant` | input quantisation, NHWC/NCHW dequantisation, end to end into `ps_centernet_decode` |
| `test_power_adc`      | divider maths, median filter, CPU frequency steps                              |

## Pin map (defaults, ESP32-P4-Function-EV-Board)

All pins are Kconfig options (`main/Kconfig.projbuild`), collected in
`main/board_pins.h`. The sensor pins are on the board's 2x20 header;
**check them against your board revision** before wiring.

| function           | signal                 | GPIO          | peripheral                     |
|--------------------|------------------------|---------------|--------------------------------|
| Lepton VoSPI       | SCLK / MISO / CS       | 20 / 21 / 22  | SPI2, mode 3, 16 MHz (<= 20)   |
| Lepton VSYNC       | Lepton GPIO3           | 23            | rising edge, one per segment   |
| Lepton RESET_L / PWR_DWN_L | optional       | -1 / -1       |                                |
| Lepton CCI         | I2C addr 0x2A          | sensor bus    | I2C1                           |
| BNO085             | I2C addr 0x4A, INT, RST| bus, 47, 48   | I2C1                           |
| Sensor I2C         | SDA / SCL              | 32 / 33       | I2C1, 400 kHz                  |
| ES8311 codec       | I2C addr 0x18          | SDA 7 / SCL 8 | I2C0 (on board)                |
| I2S                | MCLK/BCLK/WS/DOUT      | 13/12/10/9    | I2S0 (on board)                |
| Speaker PA enable  |                        | 53            | -1 for headphone jack only     |
| Micro-OLED         | MIPI-DSI 2 lanes       | dedicated     | DSI PHY on LDO channel 3 (2.5 V) |
| Panel reset        |                        | 27            |                                |
| Button A / B       | to GND, pull-up        | 45 / 46       |                                |
| Battery            | divider 100k/100k      | 16            | ADC1, oneshot + calibration    |

The sensors use their own I2C bus because the BNO085 stretches the clock;
`PS_SENSORS_ON_CODEC_BUS` puts them on the codec bus instead. Everything uses
the new `i2c_master` driver, which is why the codec is driven through
`esp_codec_dev` (the standalone `es8311` component uses the legacy driver,
which cannot coexist with it).

## Tasks, cores and priorities

One mutex serialises every `ps_system_*` call. Slow work (VoSPI transfer,
inference, PPA scaling, DSI, audio playback) happens outside it on private
buffers.

| task      | core | prio | trigger                  | under the lock                         | outside the lock                  |
|-----------|------|------|--------------------------|----------------------------------------|-----------------------------------|
| thermal   | 0    | 20   | Lepton VSYNC             | `ps_system_on_frame`, copy model input | VoSPI capture, assembly, TLinear  |
| imu       | 0    | 18   | BNO085 INT (100 Hz GRV)  | `on_yaw` / `on_step` / `on_linear_accel` | SHTP read and parse             |
| tick      | 0    | 16   | 10 ms                    | `ps_system_tick`                        |                                   |
| inference | 1    | 15   | new frame (task notify)  | `on_detections` or `run_classical`      | ESP-DL on a private copy          |
| display   | 1    | 10   | 1/`display_fps` (30)     | `ps_system_render` into a 320x240 fb    | PPA 3x scale, DSI flip, DCS 0x51  |
| audio     | 0    | 8    | alert queue, 50 ms poll  | `ps_system_next_alert`                  | clip / beep playback              |
| service   | 0    | 3    | 10 ms                    | buttons; battery + policy (1 s)         | ADC, esp_pm; stats log (5 s)      |

Every task is subscribed to the task watchdog (3 s, panic + reboot); long
driver waits (Lepton resync/reboot, audio playback) feed it explicitly.

Statistics are logged every 5 s: frame rate and drops, pipeline and
inference latency (last/max), neural vs classical runs and fallbacks, VoSPI
counters (packets, discards, CRC, sequence, segment order, lost frames,
resyncs), SHTP counters, display and audio counters, heap, battery, CPU clock.

## Timing budget

Lepton 3.5 delivers ~8.7 unique frames/s (115 ms), so inference runs once per
new frame; the 16 fps policy cap (62 ms) is never the limit in NORMAL mode.
Figures marked *est.* are estimates to be confirmed with the stats log on
hardware.

| stage (per frame)                         | time            | where              |
|-------------------------------------------|-----------------|--------------------|
| VoSPI: 4 valid + up to ~8 repeated (TTT=0) segments read | <= ~59 ms @ 16 MHz *est.* (SPI DMA, CPU mostly idle) | core 0 |
| `ps_system_on_frame` (denoise, maps)      | 2-4 ms *est.*   | core 0, locked     |
| inference (ESP-DL, int8)                  | <= 62 ms deadline, 30-50 ms *est.* | core 1 |
| `ps_centernet_decode` + distances         | < 1 ms          | core 1             |
| classical fallback                        | 1-3 ms *est.*   | core 1, locked     |
| render 320x240                            | 3-5 ms *est.*   | core 1, locked     |
| PPA 320x240 -> 960x720                    | 3-6 ms *est.*   | PPA, core 1 waits  |
| capture -> detections on screen           | ~115 ms frame + <= 62 ms + <= 33 ms display | |

If inference exceeds the deadline (`PS_INFER_DEADLINE_MS`, 62 ms) that frame
uses the classical detector; three overruns in a row bench the network for
30 s. In ECO / CRITICAL power modes the policy halves / quarters the inference
rate, lowers the display rate and IMU rate, caps brightness and lowers the
DFS CPU ceiling (`power_busy_begin/end` hold the ceiling during work; the CPU
idles at 40 MHz otherwise). Automatic light sleep stays off because the DSI
video stream needs its clocks (`PS_LIGHT_SLEEP` for bench builds only).

## Detector model contract

* input `[1,120,160,1]` int8: `ps_system_t.model_input` codes `c` (uint8),
  trained on `x = c/255`, quantised as `q = clamp(round_half_even(c/255 * 2^-e), -128, 127)`
  with the input tensor exponent `e` (`detector_quant.h`). With the usual
  `e = -7`, `q = c*128/255` and code 255 saturates at 127 (< 0.8 % error at
  full scale). An exponent of -8 would clip codes above 127, so the export
  should keep -7 for this input.
* outputs, 30x40 grid, stride 4, 2 channels (fire, person), int8 or int16,
  NHWC or NCHW (detected from the shape), names containing `heat`, `wh`, `off`:
  dequantised with their own exponents into channel-major float planes and
  passed to `ps_centernet_decode()` (which applies the sigmoid and NMS).

## Filling in the micro-OLED

The 0.39" micro-OLED driver is generic MIPI-DSI video mode. For a specific
panel:

1. `components/microoled/include/panel_init_seq.h`: replace the marked
   VENDOR SEQUENCE block with the datasheet's register table (power, gamma,
   MIPI interface setup, scan direction). Use `panel_init_post_video` for
   commands the panel wants after video starts.
2. menuconfig -> PyroSight -> Micro-OLED: resolution (default 1024x768),
   HSYNC/VSYNC pulse widths and porches, pixel clock, lane count and lane bit
   rate (>= pixel clock x bpp / lanes plus ~20 % overhead), RGB888 vs RGB565.
3. Brightness: DCS 0x51 with one byte by default; enable
   `PS_OLED_BRIGHTNESS_2BYTES` if the panel takes a 10/11-bit value, and
   check whether it needs DCS 0x53 (BCTRL) as sent in the init table.
4. If 960x720 does not fit, lower `PS_OLED_SCALE`; the image is always centred
   with a black border.

## Known gaps

* **Never run on hardware.** The firmware now builds for the ESP32-P4 (ESP-IDF
  v5.4.1, `.github/workflows/ci.yml` builds it on every push, and it was first
  compiled by hand on 2026-10-05: 2,131,536 byte image, DIRAM 26% used), but a
  binary that links is not a binary that works. Nothing here has driven a
  Lepton, a BNO085, a panel or a codec.
* The first compile found two faults that no host test could have caught, both
  now fixed: `microoled` used `esp_timer.h` without declaring the dependency,
  and `detector.cpp` formatted esp-dl 3.x's `ExponentInfo` with `%d`. The
  second mattered beyond the warning — `ExponentInfo` converts to `int`
  implicitly and yields the *per-tensor* exponent, so a per-channel quantised
  model would have been dequantised with a single scale and produced silent
  nonsense. Per-channel tensors are now rejected at load, which leaves the
  classical detector running.
* Lepton CCI command IDs marked `[check]` in `lepton_cci.h` (gain mode, FFC
  status, OEM reboot, TLinear auto-resolution) should be verified against the
  Lepton Software IDD for the camera's firmware; setup results are logged.
* `pwr_cpu_mhz_supported()` uses assumed DFS steps for the P4.
* The panel init sequence is a placeholder by design.
