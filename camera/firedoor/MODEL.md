# FireDoorNet: fire, door and window detector for ordinary RGB cameras

A small CenterNet-style detector for the PyroSight Camera web page. It finds **fire / flames**
(the page draws a PURPLE `#C850FF` box labelled FIRE), **doors** and **windows** (GREEN `#28FF50` boxes labelled DOOR and
WINDOW: ways out) in webcam or phone video.
It is **not** the thermal PyroNet: it was trained from scratch on RGB photos, starting from an ImageNet MobileNetV2.

The window class was added on 2026-10-07 as a separate branch on the frozen fire/door network, so the fire and door outputs
are bit for bit those of the earlier 2-class model and every fire and door number below still holds. See "Window class".

| | |
|---|---|
| File | `export/firedoor.onnx`, ONNX opset 13, 3,478,581 bytes (fp32; the fire/door-only model was 2,045,975) |
| Weights | 856,072 values: 1.71 MB as fp16, about 2.3 MB as fp16 base64 (fire/door alone: 501,722) |
| Compute | 203.2 M multiply-accumulates per frame at 320x256 (fire/door alone: 157.2 M) |
| Speed | fire/door alone: 5.8 ms per frame in onnxruntime, 1 CPU thread (x86 container). In the page see `camera/README.md` |
| Ops | Conv 87 (including depthwise, `group = C`), Clip 42 (ReLU6, min 0 and max 6 as initializers), Relu 20, Add 16, Resize 4, Sigmoid 3, Mul 2 (box size x 1.0), Concat 1 (the heat channels), Identity 1. BatchNorm is folded. There are no Constant nodes: every constant is an initializer |

## Input
- Name `input`, shape **[1, 3, 256, 320]** float32, **NCHW**, **RGB** channel order.
- Normalisation: **x = pixel / 127.5 - 1** (pixel 0..255, so x is in [-1, 1]). There is no mean/std per channel.
- Draw the **whole video frame stretched** to 320 wide x 256 high, with no letterbox and no crop. Training used crop aspect
  ratios from 0.75 to 1.9 (portrait to 16:9), so 4:3, 16:9 and portrait frames stretched to 5:4 all work. Box outputs
  are then normalised to the original frame directly.
- `decode.js` exports `preprocessRGBA(rgba, 320, 256, out?)`, which turns `ctx.getImageData(0,0,320,256).data` into the input
  Float32Array of length 245,760.
- The model was trained with horizontal flips, so a mirrored selfie view is fine. If the page mirrors the *display*,
  mirror the boxes too: `x' = 1 - x - w`.

## Outputs (all NCHW, batch dimension 1, stride 8 so the grid is 32 rows x 40 columns)
| name | shape | meaning |
|---|---|---|
| `heat` | [1, 3, 32, 40] | sigmoid probability that an object centre falls in this cell. Channel 0 = **fire**, 1 = **door**, 2 = **window** |
| `wh` | [1, 2, 32, 40] | box **width** (ch 0) and **height** (ch 1) in model-input pixels (of 320x256), >= 0 (ReLU), for fire and door boxes |
| `off` | [1, 2, 32, 40] | centre offset inside the cell, x (ch 0) and y (ch 1), in [0, 1) (Sigmoid), for fire and door boxes |
| `wh_w` | [1, 2, 32, 40] | the same as `wh`, for **window** boxes |
| `off_w` | [1, 2, 32, 40] | the same as `off`, for **window** boxes |

Flat index of channel c, row y, column x: `c*1280 + y*40 + x`. `heat` is a Float32Array of length 3840, the others 2560.
The fire/door-only model had no window channel and no `wh_w`/`off_w`; `decode.js` reads the number of classes from the
length of `heat` and takes window sizes and offsets from `{windowWh, windowOff}`.

## Decoding (`decode.js` = `decode.py`, tested identical)
For each class c and cell (y, x):
1. Keep the cell if `heat >= threshold[c]` and no 3x3 neighbour is strictly greater (a local maximum; ties are kept,
   neighbours outside the grid are ignored).
2. Read `w = wh[0]` and `h = wh[1]`, and drop the cell if `w < 1` or `h < 1`.
3. Centre `cx = (x + off[0]) * 8` and `cy = (y + off[1]) * 8`. The box is `[cx - w/2, cy - h/2, cx + w/2, cy + h/2]`, clipped to
   320x256, then divided by 320 and 256.
4. Sort all boxes by score, descending (a stable sort). Run greedy NMS per class with IoU > 0.45, and keep at most 50 boxes.

```js
// browser: <script> with decode.js inlined gives window.FireDoorDecode
const dets = FireDoorDecode.decode(heat, wh, off);             // default thresholds fire 0.50, door 0.35
// -> [{cls:'fire'|'door', score, x, y, w, h}], x/y = top-left, all in [0,1] of the frame
const sensitive = FireDoorDecode.decode(heat, wh, off, {thresholds: {fire: 0.40, door: 0.30}});
// {layout:'NHWC'} if a runtime returns channels-last data
```
Parity: on 30 held-out images x 2 thresholds (325 boxes), decode.js and decode.py gave 0 differences. NHWC input
gives identical results (`tools/parity_decode.py`).

## Operating thresholds (chosen on the *validation* split, then reported on *test*)
- **fire 0.50**: the lowest threshold with fire false alarms on Open Images hard-negative images at or below 2% on validation (1.7%).
- **window 0.36** (also the page's): the lowest threshold that puts a WINDOW box on at most 10 % of window look-alike photos
  (mirrors, picture frames, TVs, monitors, fridges, wardrobes, ...: 9.4 %) and at most 15 % of window-free photos (13.0 %), on the
  Open Images validation photos. See "Window class".
- **door 0.35**: door-image precision of at least 0.6 on validation (0.67). The camera page uses **door 0.50** instead:
  on Open Images val/test photos (verification run) 0.35 put a DOOR box on 38 % of real doors but also on 31 % of
  wardrobes, windows and fridges; 0.50 gives 19 % and 7 %. A green DOOR box there should mean a door, at the cost of
  finding half as many. The real fix is retraining with those look-alikes as hard negatives.
- Recommended in the page: **hysteresis for FIRE**. Switch the box on at a score of 0.50 or more, and keep it on while the
  best fire score near it stays at 0.35 or more. On video1, frame recall went from 0.41 to 0.61 with no false alarms on the
  no-fire frames. A still hard-negative scene can only turn FIRE on by crossing 0.50, so the onset rate stays the same,
  but a false FIRE box that does cross 0.50 also stays on longer: on a hard-negative clip run in the camera page
  (sunsets, LEDs, lamps, car lights) false-FIRE updates went from 27 to 38 (+41 %) with the same number of onsets.
  A "2 of last 3 frames" vote did not help here (0.41 to 0.41).
- There is a "sensitive" mode at fire 0.40. It gives more recall (table below) and about twice the false alarms on lamps, sunsets and street lights.

## Held-out test results (`runs/eval_test_final.txt`, `runs/eval_test.json`, `runs/video1_eval.json`)
None of these images or frames were used for training or for picking the threshold.

| Fire | thr 0.40 | **thr 0.50** |
|---|---|---|
| FireNET test split (official 90-image validation split, 138 boxes): **AP@0.5 = 0.727** | | |
| FireNET box recall / precision | 0.674 / 0.802 | **0.543 / 0.915** |
| FireNET images with at least one fire box (image recall) | 0.900 | **0.778** |
| Open Images fire photos (127, image-level labels) fire-present recall | 0.929 | **0.898** |
| DeepQuest Test/Fire (100) fire-present recall | 0.880 | **0.740** |
| video1.mp4 frames with visible flames (793 frames, 30 fps), recall | 0.672 | **0.414** (0.610 with hysteresis down to 0.35) |
| video1 no-fire frames (173: road before ignition, end card), false alarm | 0.000 | **0.000** |
| **False alarms** | | |
| Open Images hard negatives (976 images), fire boxes per image | 0.045 | **0.019** |
| Open Images hard negatives, images with any fire box | 3.6% | **1.5%** |
| DeepQuest Test/Neutral (100), fire boxes per image / images with a fire box | 0.05 / 4% | **0.03 / 2%** |
| DeepQuest Test/Smoke (100), images with a fire box (some contain small flames) | 8% | **2%** |

Hard-negative false-alarm rate by type at 0.50 (images with any fire box, out of n): sunset 13.6% (44), dusk 8.3% (12), light bulb 8.3% (12),
lamp 5.1% (39), street light 4.8% (42), lighting 3.8% (26), Christmas tree 3.8% (26), neon 3.7% (27), hand 3.1% (64),
sunlight 2.9% (34), monitor 2.2% (46), **human face 0% (240)**, indoor rooms 0% (170), orange 0% (57), pumpkin 0% (20),
TV 0% (47), traffic light 0% (31). The remaining failures are glowing filament bulbs, sun glare at sunset, and warm wall lights.

| Door (Open Images test split here, rotated images excluded) | thr 0.30 | **thr 0.35** | thr 0.40 |
|---|---|---|---|
| AP@0.5 on door + hard-negative images (348 door boxes) = **0.324**. AP on door images only = **0.379** | | | |
| recall | 0.345 | **0.259** | 0.204 |
| precision (all test images / door images only) | 0.41 / 0.51 | **0.55 / 0.67** | 0.68 / 0.75 |
| door boxes per image on Open Images negatives without door boxes | 0.062 | **0.031** | 0.009 |
| door boxes per image on images annotators marked "no door" (33 door-like confusers) | 0.52 | **0.24** | 0.09 |
| door boxes per image on FireNET / DeepQuest neutral | 0.07 / 0.17 | **0.01 / 0.09** | 0.01 / 0.04 |

Door is the weaker class. It finds clear, frame-filling doors (front doors, room doors seen face-on), and misses doors that
are small, open, side-on or glass. Lockers and wardrobes sometimes come out as doors. Door recall is about 26%
of all annotated doors (many are tiny background doors). The rate for the obvious door in front of the camera is higher,
but we did not measure it separately.

## Window class (added 2026-10-07; `train_window.py`, `model.py` WindowBranch, `runs/window_deep/log.jsonl`)
- **Branch, not retraining:** the fire/door network (`runs/stage2/best.pt`) is frozen and a window branch reads its backbone
  features: its own trainable copy of the backbone's last stage (blocks 13-16, stride 32, `--deep`), its own FPN-lite neck and
  head (64 channels, one depthwise-separable block), started as a copy of the door detector (window channel = door channel),
  then trained for windows. Fire and door outputs are unchanged bit for bit (validation fire AP 0.790, door AP 0.303 at every
  epoch). Retraining the whole network with a third class was tried first and lost fire AP (0.79 to 0.73-0.75 in 88 minutes).
- **Data:** Open Images "Window" boxes. Two problems found on the way: about half of the window photos are cars, buses, trains
  and planes (their windows are no way out of a building), and most building windows in it are small (on building fronts).
  Photos with a vehicle label or box are left out (`tools/oi_vehicles.py`), and 6,000 extra training photos whose largest window
  covers 8-90 % of the picture were added (`tools/oi_select_window_big.py`): 8,504 window photos in training, plus the look-alike
  photos (mirrors, picture frames, TVs, monitors, fridges, wardrobes, cupboards, closets, bookcases, whiteboards, posters,
  billboards, laptops, tablets) and verified window-free photos as negatives. Group-of window boxes are ignored.
- **Training:** 3,600 iterations x 32 images, AdamW lr 1e-3, cosine on wall-clock time, EMA, 50 minutes on 3 CPU threads; mix 50 %
  window photos, 20 % look-alikes and window-free photos, 15 % door photos, 15 % other negatives; same augmentation as fire/door.
  The last epoch had the best validation window AP (0.186).
- **Held-out test** (Open Images test split, vehicle photos left out: 357 window photos with 1,040 windows, 209 of them with a window
  covering at least 5 % of the picture; 385 look-alike and 1,685 window-free photos). A window counts as found by a WINDOW box with
  IoU >= 0.3; a WINDOW box is right when it overlaps a window that much. `runs/window_branch` is the first window branch
  (frozen features only, small windows), for comparison.

| Window, test | **thr 0.36, this model** | thr 0.36, first branch | thr 0.30 | thr 0.40 |
|---|---|---|---|---|
| photos with a window covering >= 5 %: one of those windows found | **32 %** | 14 % | 53 % | 23 % |
| windows covering 20-100 % of the picture found | **24 %** | 1 % | 41 % | 17 % |
| windows covering 5-20 % found | **35 %** | 23 % | 57 % | 23 % |
| precision of WINDOW boxes on window photos | **0.73** | 0.69 | 0.62 | 0.79 |
| look-alike photos with a WINDOW box | **9.6 %** | 14.3 % | 28 % | 4.7 % |
| window-free photos with a WINDOW box | **11.2 %** | 14.9 % | 28 % | 5.7 % |
| FireNET fire photos: WINDOW boxes per photo (some show burning houses with windows) | **0.09** | 0.07 | 0.18 | 0.07 |

  On `camera/testdata/window` (40 held-out indoor photos of rooms with windows, 55 windows, chosen without looking at any model's
  output) a window is found in **17 of 40** photos at 0.36 (27 at 0.30, 13 at 0.40); the first branch found 8 at 0.35.
- **Rebuild** (from `camera/firedoor`): `tools/oi_train_stream.py`, `tools/oi_select_window.py`, `tools/oi_select_window_big.py 6000`,
  `tools/oi_download.py` on `data/oi/selection_window.json` and on `data/oi/selection_window_big.json`, `tools/oi_vehicles.py`,
  `build_index.py`, then `train_window.py --out runs/window_deep --ch 64 --layers 1 --init-door --deep --lr 1e-3 --max-minutes 50` and
  `export_onnx.py runs/window_deep/best.pt export/firedoor.onnx`. Training stops on wall-clock time, so a rerun on another CPU gives a
  similar, not identical, model.
- **Limits:** this is the weakest class with doors. Scores are low and close together (most real windows score 0.25-0.50), so
  the threshold trades finding windows against false ones quickly. Misses: windows cut by the picture's edge, windows seen
  at an angle, windows showing only a bright sky. False WINDOW boxes: lampshades, lanterns and bright bulbs, framed pictures, TV screens,
  signs, and glazed doors (the page drops a WINDOW box that lies on a DOOR box).

## Training (see `train.py`, `data.py`, `runs/main/log.jsonl`, `runs/stage2/log.jsonl`)
- Backbone: MobileNetV2 alpha 0.5. Keras ImageNet weights were mapped into PyTorch (`model.load_keras_mnv2`) with symmetric padding.
  The mapping was checked by loading the same Keras weights *with* the classifier top. On 300 Open Images crops of ImageNet-like
  classes (orange, traffic light, TV, monitor, lamp, pumpkin), ImageNet top-5 accuracy was 0.83 with Keras' asymmetric 'same' padding
  replicated, 0.77 with the symmetric padding we export, and about chance with random weights (`runs/keras_mapping_check.txt`).
  The stem and blocks 0 to 2 (stride 2 and 4) are frozen. Everything else is fine-tuned.
- Neck and head: FPN-lite (stride 32 to 16 to 8, 64 channels, 1x1 laterals, nearest x2 upsampling, Add, depthwise-separable 3x3), then one
  depthwise-separable 3x3 head and three 1x1 heads.
- Losses: CenterNet focal loss on `heat` with per-pixel, per-class supervision masks. FireNET images do not supervise door,
  and Open Images fire photos supervise only through MIL. L1 on size relative to the box size, L1 on offset. A MIL term on
  Open Images photos with an image-level Fire/Flame/Bonfire/Campfire/Wildfire label pushes the strongest fire logit in that image above 0.
- Augmentation: random crops (area 0.3 to 1, aspect 0.75 to 1.9) and object-centred zoom crops on hard negatives (close-up faces
  filling 25-85% of the frame height, lamps, bulbs and so on), 2x2 mosaic (p 0.25), flips, brightness/contrast, saturation, hue at most
  +/-8 degrees (fire colour stays meaningful), colour cast, gamma, Gaussian and motion blur, noise, JPEG artefacts.
- Data (train split): FireNET 357 images with 494 fire boxes; Open Images 2,983 door images (4,171 door boxes), 3,668 hard-negative
  and indoor images, and 541 fire photos with image-level labels. Images whose labels mention candles, fireplaces, torches, lanterns,
  jack-o'-lanterns, fireworks, smoke, stoves or grills were left out of the negatives.
- Schedule: main run of 4,250 iterations x 32 images (86.5 min, AdamW at lr 2e-3, cosine schedule on wall-clock time, EMA of weights).
  Then a stage-2 hard-negative fine-tune: mix of 45% negatives, MIL weight 0.1, lr 5e-4, 20 min. The best validation epoch (682
  iterations) was kept. Total is about 107 min on 2 CPU threads. Selection score on validation =
  0.5 x fire AP + 0.3 x door AP + 0.2 x fire-present recall on fire photos - 0.5 x hard-negative false-alarm rate.
- Validation at the selected checkpoint: fire AP 0.790 (55 FireNET images), door AP 0.308 / 0.367 (door images only),
  hard-negative false-alarm rate 2.8% at 0.40.
- Pretrained against random initialisation (`runs/random_init_check.log` compared with `runs/main/log.jsonl`, same data and batch): after 375 iterations
  the ImageNet-initialised model had validation fire AP 0.31 (0.44 at 250 iterations), door AP 0.08 and train heat loss 1.69.
  The random-initialised model (no frozen layers) still had fire AP 0.00, door AP 0.00 and heat loss 2.41.

## Limits to be honest about
- Only 357 box-labelled fire training images (FireNET). Fire box AP numbers come from 90 test images, so they carry about +/-0.05 noise.
- Small or distant fire (under about 15 px at 320 wide) is often missed. On the CCTV clip only 41% of fire frames light up at 0.50.
- The fire class covers flames, including candles and lighters, because FireNET contains them. Glowing filament bulbs, low sun
  and some warm lamps can still trigger it. Use the hysteresis and keep 0.50.
- fp16 weights: rounding all weights to fp16 changed `heat` by at most 0.017 and `wh` by at most 8 px, and FireNET AP went from 0.727 to 0.728.
- Nothing was tested in a real browser or on a real webcam here. The page agent should run the `.y4m` clips in `camera/testdata`.

## Running it without onnxruntime (JSON op list, checked with TF.js)
- `export/firedoor_oplist_fp16.json` (1.39 MB: weights fp16 base64, about 1.0 MB of raw weights) and `export/firedoor_oplist_fp32.json` (2.7 MB)
  hold the same graph as plain JSON: `{input, outputs, nodes:[{op, inputs, outputs, attrs}], initializers:{name:{dims, dtype, b64}}}`.
  Nodes are in topological order. Conv weights are ONNX layout [O, I/group, kh, kw]. Depthwise convs have `group == O` and `I/group == 1`.
- **Padding gotcha:** every 3x3 Conv (stride 1 *and* stride 2) uses symmetric `pads [1,1,1,1]`, and 1x1 convs use 0. In TF.js, use explicit
  padding `[[0,0],[1,1],[1,1],[0,0]]` (or `tf.pad` and then 'valid'). **Do not use 'same'**: for stride 2 it pads only bottom/right, which shifts
  every feature map.
- Resize = `tf.image.resizeNearestNeighbor(x, [2H, 2W], alignCorners=false, halfPixelCenters=false)`. Clip = `tf.clipByValue(x, 0, 6)`.
- `tools/tfjs_oplist_check.js` is a roughly 60-line TF.js interpreter for this op list. Run in node against the onnxruntime reference tensors in
  `export/sample_io/`, its maximum absolute difference was heat 5e-7, wh 2e-4 px, off 1e-6 with fp32 weights, and heat 0.004, wh 2.4 px, off 0.012
  with fp16 weights. Decoded boxes were the same to 3 decimals.
- Speed: the TF.js **CPU** backend (pure JS) took about **1.5 s per frame** in node. That is too slow for every frame, so on the CPU fallback run
  fire/door every few seconds, or only when WebGL is unavailable. WebGL speed was not measured here (there is no GPU in this container).
- `export/sample_io/`: `fire.jpg`, `door.jpg` and `window.jpg` (a held-out Open Images test photo of a cabin with a window) with their
  exact model input (`*_input_1x3x256x320.f32`, little-endian float32), onnxruntime outputs (`*_heat_1x3x32x40.f32`,
  `*_{wh,off,wh_w,off_w}_1x2x32x40.f32`) and expected decoded boxes (`*_expected.json`), for checking any runtime. Rebuild them with
  `tools/make_sample_io.py`.

## Files
- `model.py`: network, Keras weight loader, BN folding, the window branch. `train.py`, `data.py`: training. `train_window.py`: the
  window branch on the frozen fire/door network. `evaluate.py`, `eval_full.py`: metrics.
- `export_onnx.py`: export, constant-to-initializer conversion, onnxruntime parity check. Maximum absolute difference against PyTorch:
  heat 1.7e-6, wh 2e-3 px, off 4.6e-6 (`export/firedoor.export.json`).
- `decode.py` (reference) and `decode.js` (browser and node, UMD: `window.FireDoorDecode` or `require`).
- `tools/`: data selection and download (`oi_select_window.py`, `oi_select_window_big.py`, `oi_vehicles.py` for the window data),
  the held-out window test photos (`make_window_testdata.py`), test-media builder, parity tests, video evaluation, visualisation.
- `LICENSES.md`: dataset and model licences.
