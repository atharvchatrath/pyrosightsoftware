# Licences: PyroSight Camera (camera/)

Everything that ends up inside `dist/pyrosight_camera.html` and everything used to build or test it.
Per-part detail: `runtime/LICENSES.md` (person and face models), `firedoor/LICENSES.md` (fire/door
model and its training data), `page/LICENSES.md` (page code and the page agent's test clips).

## Inside the page (`dist/pyrosight_camera.html`, `dist/pyrosight_camera.fragment.html`)

| Item | Licence | Notes |
|------|---------|-------|
| TensorFlow.js 4.22.0 (`tf.min.js`, inlined unchanged after one line `var regeneratorRuntime;`) | Apache-2.0 | npm `@tensorflow/tfjs` |
| Person detector: COCO-SSD `ssdlite_mobilenet_v2` (TensorFlow.js models), re-packed person-only, uint8 weights | Apache-2.0 | converted by Google from the TF Object Detection API model zoo, trained on MS-COCO; `runtime/repack_cocossd.py` |
| Face detector: MediaPipe BlazeFace short range (`blaze_face_short_range.tflite`), float16 | Apache-2.0 | `runtime/export_oplist.py` |
| FireDoorNet (fire + door detector), float16 weights | trained in this project; inherits: MobileNetV2 alpha 0.5 ImageNet weights (keras-applications, Apache-2.0), FireNET boxes and images (MIT, Copyright (c) 2019 Moses Olafenwa), Open Images annotations (CC BY 4.0, Google LLC) and images (CC BY 2.0, per-image authors in `firedoor/data` selection files) | trained by `firedoor/train.py`; the page's About card carries the credits line |
| `runtime/oplist.js`, `runtime/people.js`, `firedoor/decode.js`, `page/app.js`, `page/motion.js`, `page/page.html` | PyroSight project code | post-processing re-implemented from coco-ssd and MediaPipe calculators (Apache-2.0), no code copied verbatim |

Keep the Apache-2.0 notices (TF.js, COCO-SSD, BlazeFace) and the FireNET / Open Images attribution with any copy of the page.
The page shows the credits in its About card.

Not used anywhere in the page or its training: DeepQuest FIRE-SMOKE-DATASET (no licence stated; used only
for internal evaluation by the fire/door agent, not copied).

## End-to-end test clips (`tests/make_e2e_clips.py`, never shipped)

The fake-camera clips are built from images already in `camera/testdata`. Screenshots in `shots/e2e_*`
show some of them; credit the authors below if a screenshot is published.

| Clip | Image | Author / licence |
|------|-------|------------------|
| faces | `d96e2962d8db7d41.jpg` | mthakral, CC BY 2.0 |
| faces | `e408619cbefb92b1.jpg` | Samson Loo, CC BY 2.0 |
| faces | `56c1fd30dfa1eb42.jpg` | Charlie Kaijo, CC BY 2.0 |
| faces | `93324166a6fc25d1.jpg` | Chung Shao Tung, CC BY 2.0 |
| faces | `b3a4e6e4deadb73f.jpg` | lirneasia, CC BY 2.0 |
| faces | `9044d6abaa6196a4.jpg` | Sadasiv Swain, CC BY 2.0 |
| faces | `d0fb126ba870cfda.jpg` | The Institute for Inclusive Security, CC BY 2.0 |
| faces | `7d24c6cc1ef5b529.jpg` | Kristen Stacy, CC BY 2.0 |
| group | `f1d99418db05e6a1.jpg` | Jim Larrison, CC BY 2.0 |
| group | `c4a79ad7987f24b9.jpg` | Steve Vandenberg, CC BY 2.0 |
| group | `55047052d7c4b4d4.jpg` | K.M. Klemencic, CC BY 2.0 |
| group | `e96fe1c5acf48a98.jpg` | jbraine, CC BY 2.0 |
| group | `dd27cbcec8817a76.jpg` | UKBERRI.NET Uribe Kosta eta Erandioko agerkari digitala, CC BY 2.0 |
| group | `fc14f1cce131d551.jpg` | George Campbell, CC BY 2.0 |
| group | `3fb2521488b9b319.jpg` | Stephen Michael Barnett, CC BY 2.0 |
| group | `e39cd20c9e0a6147.jpg` | NatalieMaynor, CC BY 2.0 |
| nopeople | `05489aa3e2b24d6a.jpg` | Rocky Mountain Feline Rescue (formerly known as Animal Rescue and Adpotion Society), CC BY 2.0 |
| nopeople | `7c485a7dfe958934.jpg` | Vikalpa / Groundviews / Maatram / CPA, CC BY 2.0 |
| nopeople | `fa17861c768f5a51.jpg` | Chiu Heiyan, CC BY 2.0 |
| nopeople | `85f5cf4451545d98.jpg` | bfishadow, CC BY 2.0 |
| nopeople | `321ffb38656f7311.jpg` | Michael Gil, CC BY 2.0 |
| nopeople | `d9d4b619b1d6d406.jpg` | Kimberly Vardeman, CC BY 2.0 |
| nopeople | `678b317900696657.jpg` | Mitchell Haindfield, CC BY 2.0 |
| nopeople | `ca268d57abb702f5.jpg` | Jonathan Chen, CC BY 2.0 |
| fire | `firenet_img_1.jpg` | MIT, Copyright (c) 2019 Moses Olafenwa (FireNET validation split) |
| fire | `firenet_img_16.jpg` | MIT, Copyright (c) 2019 Moses Olafenwa (FireNET validation split) |
| fire | `firenet_img_22.jpg` | MIT, Copyright (c) 2019 Moses Olafenwa (FireNET validation split) |
| fire | `firenet_img_29.jpg` | MIT, Copyright (c) 2019 Moses Olafenwa (FireNET validation split) |
| fire | `firenet_img_8.jpg` | MIT, Copyright (c) 2019 Moses Olafenwa (FireNET validation split) |
| fire | `firenet_pic_14.jpg` | MIT, Copyright (c) 2019 Moses Olafenwa (FireNET validation split) |
| fire | `firenet_pic_20.jpg` | MIT, Copyright (c) 2019 Moses Olafenwa (FireNET validation split) |
| fire | `firenet_pic_27.jpg` | MIT, Copyright (c) 2019 Moses Olafenwa (FireNET validation split) |
| fire | `firenet_pic_6.jpg` | MIT, Copyright (c) 2019 Moses Olafenwa (FireNET validation split) |
| fire | `firenet_small_12.jpg` | MIT, Copyright (c) 2019 Moses Olafenwa (FireNET validation split) |
| fire | `firenet_small_19.jpg` | MIT, Copyright (c) 2019 Moses Olafenwa (FireNET validation split) |
| fire | `firenet_small_25.jpg` | MIT, Copyright (c) 2019 Moses Olafenwa (FireNET validation split) |
| doors | `oi_73e6ceb9800a3ece.jpg` | CAMH Foundation, CC BY 2.0 |
| doors | `oi_aa0fc9d6f82cee66.jpg` | acn1485, CC BY 2.0 |
| doors | `oi_ef06b701061b0838.jpg` | MelisaTG, CC BY 2.0 |
| doors | `oi_0224eb6948c6627e.jpg` | David Shankbone, CC BY 2.0 |
| doors | `oi_bc7b4740914995e2.jpg` | SuperChickenPerson, CC BY 2.0 |
| doors | `oi_1b3247da6dd5d578.jpg` | Dave Walker, CC BY 2.0 |
| doors | `oi_34366cc10d904e94.jpg` | thierry ehrmann, CC BY 2.0 |
| doors | `oi_716703d1bb3bbcc0.jpg` | Annie Pilon, CC BY 2.0 |
| doors | `oi_fbd2f16697562dd1.jpg` | Brian Wright, CC BY 2.0 |
| doors | `oi_3ddb20972d7ebe74.jpg` | Jim Linwood, CC BY 2.0 |
| lights | `oi_fff3ce694bc02a09.jpg` | Bruno Cordioli, CC BY 2.0 |
| lights | `oi_2cffe8b1fca33189.jpg` | theNerdPatrol, CC BY 2.0 |
| lights | `oi_353bd154107ffafd.jpg` | Eder Fortunato, CC BY 2.0 |
| lights | `oi_5ba79ea4e55a5e68.jpg` | Christian Bjarnson, CC BY 2.0 |
| lights | `oi_296b84193ba78ff4.jpg` | Alana Sise, CC BY 2.0 |
| lights | `oi_c4c5c7841f474830.jpg` | Luigi Guarino, CC BY 2.0 |
| lights | `oi_4c83af9e3cdb1741.jpg` | Ian Griffiths, CC BY 2.0 |
| lights | `oi_a03a6c26b95bb3d1.jpg` | Ian Rutherford, CC BY 2.0 |
| lights | `oi_2da4cad190c2a7b4.jpg` | Francis Vallance (Heritage Warrior), CC BY 2.0 |
| lights | `oi_f1f26b2f9caa6037.jpg` | Leszek Kozlowski, CC BY 2.0 |
| lights | `oi_a9459cbecb699cec.jpg` | matthew Hunt, CC BY 2.0 |
| lights | `oi_be907cce6918b17b.jpg` | ITU Pictures, CC BY 2.0 |
| still_face | `d96e2962d8db7d41.jpg` | mthakral, CC BY 2.0 |
| still_group | `f1d99418db05e6a1.jpg` | Jim Larrison, CC BY 2.0 |
| still_fire | `firenet_pic_14.jpg` | MIT, Copyright (c) 2019 Moses Olafenwa (FireNET validation split) |
| still_door | `oi_fbd2f16697562dd1.jpg` | Brian Wright, CC BY 2.0 |
| firevideo | `fire_road_burning_640x480.y4m` | FireNET repo `video1.mp4` = CGTN/CCTV news footage: **local testing only**, never published; no screenshot of it is taken |

Open Images box annotations used as ground truth: CC BY 4.0, Google LLC.

## Tools (not shipped)

| Tool | Licence |
|------|---------|
| Playwright 1.56 + Chromium 141 (headless, SwiftShader) | Apache-2.0 / BSD-3-Clause |
| onnxruntime (reference outputs) | MIT |
| NumPy, OpenCV, Pillow | BSD-3-Clause / Apache-2.0 / HPND |
