# PyroSight ML: fire and person detector

This directory has the full pipeline for the neural detector that runs on the ESP32-P4 (ESP-DL):
a synthetic data generator, training, evaluation, ONNX export, int8 quantisation and parity tests
against the device C code.

> **The weights in `runs/synthetic_v0/` are trained only on synthetic data. They show that the
> pipeline works end to end and are not usable in the field.** Do not flash them to a unit that
> anyone will rely on. The firmware's classical detector (`ps_detect_classical`) stays the default
> until a model trained on real Lepton 3.5 data passes the evaluation below. See
> [Getting real weights](#getting-real-weights).

## Device contract

The definitions are in `core/include/pyrosight/ps_thermal.h` and `ps_detect.h`. The Python code here mirrors
them bit-exactly, and `tests/` checks the Python against the C library through ctypes.

| | |
|---|---|
| Frame | Lepton 3.5, 160x120, radiometric, int16 deci-C (365 = 36.5 C) |
| Pre-processing (device) | temporal IIR + 3x3 median (`ps_thermal_denoise`), then `ps_thermal_model_code()` |
| Code curve | -20..60 C maps to 0..191 (about 0.42 C per code); 60..600 C maps to 192..255; clamped outside. Integer maths, `thermal.model_code()` |
| Input | `input` [1,1,120,160] float32 = code / 255 |
| Output `heat` | [1,2,30,40] raw logits, channel 0 fire, 1 person |
| Output `wh` | [1,2,30,40] box width, height in input pixels, >= 0 (ReLU in-model) |
| Output `off` | [1,2,30,40] sub-cell centre offset in [0,1) (Sigmoid in-model) |
| Decode | stride 4; 3x3 local max of logits >= logit(0.35); cx = (gx+off_x)*4; at most 16 detections; per-class NMS at IoU 0.5 (`ps_centernet_decode`, mirrored in `decode.py`) |
| Distance | person: f_px * 1.7 m / max(w,h), f_px = 80/tan(28.5 deg), about 147.4 (`ps_estimate_distances`) |
| Ops | Conv2d (BN folded), ReLU, nearest Resize, Add, Sigmoid. No Mul, Slice, Pad or Concat in the graph |

`model.py` (PyroNet) has 287 k parameters and needs about 54.8 M conv MACs per frame. This is an
estimate that counts only conv multiply-accumulates. It is not a measured ESP32-P4 latency.

## Files

| File | Purpose |
|---|---|
| `thermal.py` | bit-exact `ps_thermal_model_code`, the 3x3 median, model input |
| `decode.py` | Python mirror of `ps_centernet_decode`, `ps_nms`, `ps_iou`, `ps_estimate_distances` |
| `cbridge.py` | builds `core/src/*.c` into a shared library (gcc) and calls it through ctypes for the parity tests |
| `synth.py` | synthetic thermal scene generator (rooms, gas layer, fires, people, hard negatives, smoke, sensor noise) |
| `dataset.py` | on-disk format, augmentations, CenterNet target encoding, `SynthDataset` and `DiskDataset` |
| `sim_dataset.py` | runs the C simulator (`ps_sim --dump-dataset`) over scenarios x smoke and merges the output into one evaluation directory |
| `model.py` | PyroNet, BN folding, MAC counting |
| `train.py` | CPU training: focal loss on Gaussian heatmaps, L1 on wh/off; checkpoints every epoch; `--resume` |
| `eval.py` | AP@0.5 per class, P/R at 0.35, person distance error, all sliced by smoke; `--samples` writes a PNG |
| `export_onnx.py` | folds BN, fixed shape, outputs named heat/wh/off, op allow-list check, onnxruntime comparison |
| `quantize.py` | calibration `.npy`, esp-ppq to `.espdl` (or exact instructions), int8 power-of-two PTQ simulation |
| `viz.py` | PIL-only rendering |
| `tests/` | `python3 -m pytest ml/tests -q` |

### On-disk dataset format

`ps_sim --dump-dataset DIR` writes this format, so real recordings should use it too:

```
DIR/images/000123.bin   int16 little-endian deci-C, 160*120 row-major, RAW (before device denoise)
DIR/labels/000123.txt   one object per line: "cls x y w h dist_m"  (cls 0 fire, 1 person;
                        x,y top-left in pixels; dist_m 0 if unknown or for fire)
DIR/meta.csv            optional: "id,smoke" (smoke 0..1, used for the evaluation slices)
```

Loaders apply the device's 3x3 median before the code mapping. Pass `--no-denoise` to `eval.py`
for frames that were recorded after the device denoise.

## Running it

All commands run from the repo root. Requirements are Python 3, numpy, pillow and torch (CPU),
plus pytest, onnx and onnxruntime. esp-ppq is optional.

```sh
python3 -m pytest ml/tests -q                                   # parity + unit tests

# Train. Synthetic data is cached in $PS_ML_CACHE, or <out>/cache if that is unset.
PS_ML_CACHE=/tmp/ps_cache python3 -m ml.train --out ml/runs/synthetic_v0 \
    --n-train 16000 --epochs 9 --workers 2 --threads 2
# After an interruption, run the same command again with --resume (keep the same --epochs).

python3 -m ml.eval --weights ml/runs/synthetic_v0/best.pt --synthetic 500 \
    --out ml/runs/synthetic_v0/metrics_synth.json --samples ml/runs/synthetic_v0/samples.png
python3 -m ml.sim_dataset --out /tmp/sim_eval --every 6          # needs build/sim/ps_sim
python3 -m ml.eval --weights ml/runs/synthetic_v0/best.pt --data /tmp/sim_eval

python3 -m ml.export_onnx --weights ml/runs/synthetic_v0/best.pt --out ml/runs/synthetic_v0/model.onnx
python3 -m ml.quantize --weights ml/runs/synthetic_v0/best.pt --onnx ml/runs/synthetic_v0/model.onnx \
    --out-dir ml/runs/synthetic_v0
```

CPU tip: train with `channels_last` (train.py already does this), and keep `--threads` plus
`--workers` at or below the number of free cores. With too many threads, oneDNN can run more
than 10x slower.

## Results for runs/synthetic_v0

These numbers come from synthetic data. They measure how well the network learned the generator,
not how it will do on a fireground. See `runs/synthetic_v0/metrics.json` for the full breakdown.

Training: 16 000 synthetic frames with augmentation, 9 epochs, about 12 min on 4 shared CPU cores.

| Eval set | Slice | mAP@0.5 | fire AP / P / R @0.35 | person AP / P / R @0.35 | person dist. median abs rel. err |
|---|---|---|---|---|---|
| Synthetic held-out (500) | all | 0.77 | 0.84 / 0.87 / 0.84 | 0.70 / 0.86 / 0.61 | 0.19 |
| | clear | 0.77 | 0.82 / 0.84 / 0.84 | 0.72 / 0.87 / 0.65 | 0.20 |
| | light | 0.78 | 0.84 / 0.88 / 0.84 | 0.71 / 0.85 / 0.60 | 0.19 |
| | heavy | 0.77 | 0.86 / 0.90 / 0.83 | 0.67 / 0.85 / 0.58 | 0.18 |
| C simulator dump (1578) | all | 0.66 | 0.80 / 0.90 / 0.79 | 0.53 / 0.78 / 0.42 | 0.07 |
| | clear | 0.74 | 0.81 / 0.87 / 0.86 | 0.67 / 0.86 / 0.55 | 0.10 |
| | light | 0.63 | 0.83 / 0.91 / 0.85 | 0.43 / 0.67 / 0.39 | 0.07 |
| | heavy | 0.66 | 0.80 / 0.91 / 0.67 | 0.53 / 0.82 / 0.31 | 0.07 |

The simulator scenes were never used for training, so they are a small domain-shift check. Person
recall is low: the generator's people are harder and more varied than the simulator's.

**Export and quantisation.** The ONNX graph uses only Conv, Relu, Resize, Add and Sigmoid, and its
output matches onnxruntime to 3e-5. esp-ppq 1.3.11 produced `model.espdl` (345 kB, target esp32p4).
It uses per-channel power-of-two weights, per-tensor power-of-two activations, and an input
exponent of -7. The PyTorch int8 simulation (300 synthetic frames) gives:

| Scheme | mAP@0.5 | drop vs float |
|---|---|---|
| float | 0.769 | |
| per-tensor po2 weights and activations | 0.662 | -0.107 (mostly person: 0.70 to 0.50) |
| per-channel po2 weights, per-tensor activations (as esp-ppq does) | 0.767 | -0.002 |

Per-tensor weight scales are the main loss, because folded BN gives per-channel ranges. Input and
activation quantisation alone cost under 0.01 mAP.

## Getting real weights

Synthetic frames will not produce a field-usable detector. The generator lacks real flame
structure, smoke optics, reflections, turnout-gear emissivity, SCBA masks, steam, hose streams,
and the Lepton's real noise and shutter (FFC) behaviour. The plan below gets real weights.

**1. Record with the real sensor and the real pipeline.**
- Use the production Lepton 3.5 in low-gain radiometric (TLinear) mode. In high gain the sensor
  saturates at about 140 C and fires clip. Record raw int16 deci-C at full frame rate, and log
  FFC events and housing temperature.
- Record from the eyepiece itself, at helmet height and crawling height, with real head motion.
  Prefer frames before the device denoise, so training applies the same median as the device.
  If you record after the denoise, mark it and use `--no-denoise`.
- Scenes:
  - live-fire training burns (burn containers, acquired structures), at several stages from
    incipient to flashover-risk gas layers;
  - smoke chambers and smoke-house drills with cold and hot smoke, from clear to zero visibility;
  - ordinary buildings (homes, offices, corridors, stairwells, basements);
  - outdoor scenes at night and in sun.
- People: firefighters in full turnout gear with SCBA (standing, crouching, crawling, dragging a
  casualty), and casualties (manikins are not enough: heated manikins and real volunteers,
  clothed and lightly clothed) lying, slumped and partly covered.
- Measure distances with a laser rangefinder or tape marks on the floor, from 1 m to 15 m. These
  calibrate `person_height_m` and the pinhole prior, and fill `dist_m` in the labels.
- Hard negatives: radiators, hot pipes, lamps, appliances, hot walls and ceilings, sun on windows,
  reflections on glass and metal, steam, hot water from hose streams, warm hand-prints and seats.
- Follow the training ground's safety officer. Volunteers must give written consent, and the
  recordings will contain identifiable people.

**2. Label.**
- Draw a box around the visible part of each person and each open flame. Glowing embers and
  flame clusters wider than about 2 px count as fire. Hot gas layers, plumes and hot surfaces
  without flame are not fire.
- Add `dist_m` for people when it is known, and a per-frame smoke estimate (0 clear to 1 zero
  visibility) in `meta.csv`. This can come from a visible-light camera recorded next to the
  sensor, or from the chamber's obscuration meter.
- Have a second person review a sample of labels. Frames 1 s apart are highly correlated.
  **Split train, validation and test by recording session and site, never by frame.**
- Size, as a rough guide: at least about 20-50 k labelled frames from at least 10 different sites
  or burns, at least 5 k person instances in gear, and as many crawling or lying casualties as you
  can get. Hold out whole sessions for the test set. More diverse sites help more than more
  frames from one site.

**3. Public data, for pretraining only.** Check each licence before use. None of these is
radiometric Lepton data at 160x120:
- *Teledyne FLIR ADAS thermal dataset*: automotive, people outdoors.
- *KAIST Multispectral Pedestrian*: visible and thermal pairs, outdoor pedestrians.
- *LLVIP*: visible and infrared pairs, pedestrians at night.
- *FLAME* (aerial wildfire imagery that includes thermal frames).

These contain AGC/8-bit or non-Lepton imagery. Use them only for pretraining person shape,
downscaled to 160x120. Then fine-tune on the real radiometric data, because the code curve
assumes absolute temperatures.

**4. Fine-tune, check, quantise, flash.**
```sh
python3 -m ml.train --out ml/runs/real_v1 --data REAL_TRAIN --val REAL_VAL \
    --init ml/runs/synthetic_v0/best.pt --epochs 30 --lr 1e-3
python3 -m ml.eval --weights ml/runs/real_v1/best.pt --data REAL_TEST --out ml/runs/real_v1/metrics.json
python3 -m ml.export_onnx --weights ml/runs/real_v1/best.pt --out ml/runs/real_v1/model.onnx
python3 -m ml.quantize --weights ml/runs/real_v1/best.pt --onnx ml/runs/real_v1/model.onnx \
    --out-dir ml/runs/real_v1 --calib-dir REAL_CALIB --eval-dir REAL_TEST
```
- Use a few hundred real frames from varied scenes for calibration, never synthetic ones.
- Compare the float, int8-sim and on-device results with the test values from esp-ppq
  (`export_test_values=True`). The device should reproduce the int8 outputs exactly.
- Then flash the `.espdl` to the model partition (see `firmware/`). Keep the classical detector
  as the fallback.
- Set an acceptance bar before you look at test results. For example: person recall at 0.35 of
  at least 0.95 in clear and light smoke and at least 0.9 in heavy smoke, with fewer than about
  0.1 false fire alarms per minute of walking through a non-fire building.
- Re-run the C parity tests whenever `core/` changes.

## Known limitations and contract notes

- **int8 input resolution.** The input is code/255 in [0,1]. With a symmetric int8, power-of-two
  input scale, the best exponent is 2^-7. That keeps codes 0..254 at half resolution, about
  0.84 C per step in the people range instead of 0.42 C. If ESP-DL is given an int8 input tensor
  directly, consider changing the contract to (code - 128)/128, which is exact in int8. That
  change must be made in `core/` and the firmware together.
- `wh` is a regression in pixels, from 0 to about 160. With a power-of-two int8 output scale the
  step is 1-2 px. That is fine for distance at the ranges that matter, but coarse for tiny boxes.
- The person distance prior assumes the long side of the box is about 1.7 m. It overestimates
  distance for crouching and end-on people (the device's `dist_min_m` = 0.55 x estimate covers
  this) and for partly occluded people.
