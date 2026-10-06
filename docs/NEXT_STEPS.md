# Next steps

What is left to do on PyroSight, in rough priority order. Everything in this
repo has only been run in simulation and in a browser: nothing has run on the
real eyepiece hardware yet.

## Getting started

```sh
cmake -S . -B build && cmake --build build
(cd build && ctest)                     # core unit tests
make -C firmware/test_host              # firmware protocol tests
python3 -m pytest ml/tests -q           # ML pipeline tests (needs torch)
```

Read `README.md`, then `docs/ARCHITECTURE.md`. Datasets, training images and
test videos are not in the repo (too big); the scripts that download or
regenerate them are, see `ml/README.md` and `camera/README.md`.

## 1. Bring-up on the hardware (needs the parts)

ESP32-P4 board, FLIR Lepton 3.5, Bosch BNO085, the 0.39" micro-OLED and the
ES8311 audio codec. Details in `firmware/README.md`, "Known gaps".

- [ ] Compile `firmware/` with ESP-IDF 5.3 or 5.4. It has never been compiled
      against ESP-IDF, so expect build errors to fix first.
- [ ] Check the four Lepton command IDs marked `[check]` in
      `firmware/components/lepton/include/lepton_cci.h` against the Lepton
      Software IDD for the camera's firmware.
- [ ] Fill in the micro-OLED panel init table from the panel's datasheet. The
      current one is a placeholder (`firmware/README.md`, micro-OLED section).
- [ ] Confirm the ESP-DL model loader and tensor layout on the P4.
- [ ] Check the CPU frequency steps assumed in `pwr_cpu_mhz_supported()`.
- [ ] Get the full loop running on the device and measure frame rate (target
      about 16 fps) and battery life.

## 2. Real training data (the biggest gap)

The thermal fire and person detector was trained only on synthetic images.

- [ ] Record the real Lepton 3.5 (low-gain radiometric mode) at live-fire
      training burns and in a smoke chamber, with people in turnout gear at
      measured distances. Follow the training ground's safety officer and get
      written consent from volunteers.
- [ ] Label it, fine-tune, quantise to int8 and flash. Step-by-step commands are
      in `ml/README.md`, "Getting real weights".
- [ ] Optional improvement while retraining: feed the model `(code-128)/128`
      instead of `code/255` so int8 keeps every temperature step (needs the same
      change in `ml/` and the firmware detector).

## 3. Navigation testing

- [ ] Calibrate step length per wearer with `tools/calibrate_steps.py`.
- [ ] Run the navigation course tests in `docs/TEST_PLAN.md`, section 3, with
      the real BNO085 worn on a helmet: walking, crouching, crawling, long searches.

## 4. Camera page (browser demo, no hardware needed)

`camera/` is the webcam version: people in white, fire in purple, way out in
green, plus a navigation demo. See `camera/README.md`.

- [ ] Test it on real phones (iPhone and Android) and real webcams. So far it
      has only run in a headless Chromium browser with recorded clips.
- [ ] Doors are weak: only clear, face-on doors get boxed. Retrain FireDoorNet
      (`camera/firedoor/`) with more door images and with wardrobes, windows
      and fridges as hard negatives.
- [ ] Fire: small flames (a lighter at arm's length) are mostly missed, and
      sunsets and bright bulbs are sometimes boxed as fire.
- [ ] Phone navigation: test "Start here" with real walking (step counting and
      turns from the phone's motion sensors).

## Ground rules

- Run the tests above before every push.
- Never commit datasets, test videos or recordings of people; they are listed
  in the `.gitignore` files for a reason (size and licences).
- PyroSight is an aid, not a replacement for hose lines and search ropes. Keep
  the "follow the hose" behaviour when navigation is unsure.
