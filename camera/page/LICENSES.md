# Licences: camera/page (PyroSight Camera page)

## Shipped inside the page (`dist/pyrosight_camera.html`)

| Item | Licence | Notes |
|------|---------|-------|
| `page.html`, `app.js`, `motion.js`, `build_page.py` | PyroSight project code (written for this project) | The phase correlation, FFT and orientation maths in `motion.js` are written from scratch; the direction phrases follow `core/src/ps_alerts.c`. |
| TensorFlow.js 4.22.0 `tf.min.js` | Apache-2.0 | Inlined unchanged; the build adds one line before it (`var regeneratorRuntime;`) so it loads without `eval` under a strict Content-Security-Policy. |
| `camera/runtime/oplist.js`, `people.js`, `dist/people_assets.js` | see `camera/runtime/LICENSES.md` | COCO-SSD lite (TensorFlow.js models, Apache-2.0) and MediaPipe BlazeFace short range (Apache-2.0). |
| Fire/door model: **not in the default build** (stub) | — | The build with `--firedoor` adds FireDoorNet and `camera/firedoor/decode.js`; see `camera/firedoor/LICENSES.md` (MobileNetV2 ImageNet weights Apache-2.0, FireNET MIT, Open Images annotations CC BY 4.0 and images CC BY 2.0). The page then shows its credits line. |

The page shows the credits for whatever it contains (About card). Keep the Apache-2.0 notices with any copy of the page.

## Test data (never shipped in the page)

Clips are made by `tests/make_clips.py` from images already in `camera/testdata`:

| Used in | Image | Licence, author |
|---------|-------|-----------------|
| pan clip (panorama) | Open Images `000e4e7ed48c932d` | CC BY 2.0, Gabriele Cantini, flickr.com/photos/letorri/5835265287 |
| pan clip | Open Images `73e6ceb9800a3ece` | CC BY 2.0, CAMH Foundation, flickr.com/photos/camhfoundation/7704387562 |
| pan clip | Open Images `00ec4ba83d648c33` | CC BY 2.0, Zac Bowling, flickr.com/photos/zbowling/6820961330 |
| pan clip | Open Images `0224eb6948c6627e` | CC BY 2.0, David Shankbone, flickr.com/photos/shankbone/2830138700 |
| mixed / still_door clips, person_door.jpg | Open Images `4e14c94a0dda414e` (door) | CC BY 2.0, Andres EM, flickr.com/photos/jorgeandresem/1088620910 |
| people / mixed / still clips | Open Images `d96e2962d8db7d41` | CC BY 2.0, mthakral, flickr.com/photos/meechai/464919308 |
| people / mixed / still clips | Open Images `7a23330c4180f7ea` | CC BY 2.0, JamesBond0071, flickr.com/photos/55250321@N02/6810074110 |
| people clip | Open Images `56fd536256dae19b` | CC BY 2.0, Walt Stoneburner, flickr.com/photos/waltstoneburner/3373248256 |
| people clip | Open Images `a6fba3d8e01d2219` | CC BY 2.0, Bill Automata, flickr.com/photos/billautomata/8468539526 |
| people clip | Open Images `7ad997675bbf6870` | CC BY 2.0, Duru..., flickr.com/photos/sevilaydurul/3619222958 |
| people clip | Open Images `3fb2521488b9b319` | CC BY 2.0, Stephen Michael Barnett, flickr.com/photos/httpwwwflickrcomphotostopend/8584354311 |
| mixed / still_fire clips, person_fire.jpg | FireNET `firenet_img_10.jpg` | MIT, Copyright (c) 2019 Moses Olafenwa (github.com/OlafenwaMoses/FireNET) |

Open Images box annotations: CC BY 4.0, Google LLC. The screenshots in `shots/` show these
images; credit the authors above if a screenshot is published.

Not used: the FireNET `video1.mp4` clips in `camera/testdata/fire/clips` (third-party news
footage, local testing only) and the MediaPipe test portraits (no per-image licence).

## Test tools (not shipped)

| Tool | Licence |
|------|---------|
| Playwright 1.56 + Chromium 141 | Apache-2.0 / BSD-3-Clause |
| OpenCV (cv2), NumPy | Apache-2.0 / BSD-3-Clause |
| onnxruntime (preprocessing check) | MIT |
| FFmpeg with libvpx (WebM test clip) | LGPL-2.1+ / BSD |
