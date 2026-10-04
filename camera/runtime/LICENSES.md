# Licences of models, data and tools used by camera/runtime

## Shipped in the page (embedded in dist/people_assets.js)

| Item | Source | Licence | Notes |
|------|--------|---------|-------|
| COCO-SSD `ssdlite_mobilenet_v2` TF.js graph model | `storage.googleapis.com/tfjs-models/savedmodel/ssdlite_mobilenet_v2/model.json` (+ 5 shards; the URL `@tensorflow-models/coco-ssd` 2.2.3 loads; `model.json` sha256 3770b252... matches assets/cocossd) | Apache-2.0 | Converted by Google from the TensorFlow Object Detection API model zoo (Apache-2.0), trained on MS-COCO. Re-packed here by `repack_cocossd.py` (preprocessor removed, person-only class head, per-channel uint8 weights). Keep the Apache-2.0 notice with the page. |
| MediaPipe BlazeFace short range `blaze_face_short_range.tflite` | `storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite` (sha256 b4578f35... matches assets/blazeface) | Apache-2.0 | Model card: "MediaPipe BlazeFace (Short Range)". Converted to the op-list format by `export_oplist.py` (float16 weights, unchanged values). |
| TensorFlow.js 4.22.0 (`tf.min.js`, inlined by the page) | npm `@tensorflow/tfjs` | Apache-2.0 | |
| The post-processing in `people.js` | re-implemented from `@tensorflow-models/coco-ssd` (Apache-2.0) and MediaPipe's `SsdAnchorsCalculator`, `TensorsToDetectionsCalculator`, `NonMaxSuppressionCalculator`, `ImageToTensorCalculator` (Apache-2.0) | Apache-2.0 | No code copied verbatim; same algorithms and constants. |

## Test data only (camera/testdata/people, never shipped)

| Item | Source | Licence |
|------|--------|---------|
| 200 images (`images/*.jpg`, downscaled to <= 640 px) | Open Images Dataset V5, validation split, `open-images-dataset.s3.amazonaws.com/validation/<id>.jpg` | Each image CC BY 2.0 (all 200 checked); author, licence URL and Flickr page per image are in `manifest.json` (`author`, `licence`, `landing`). |
| Box annotations in `manifest.json` | Open Images V5 `validation-annotations-bbox.csv`, class names `oidv6-class-descriptions.csv`, rotation/attribution `2018_04/validation/validation-images-with-rotation.csv` | CC BY 4.0 (Google LLC) |
| 14 images in `mediapipe/` | public bucket `storage.googleapis.com/mediapipe-assets` (MediaPipe test assets) | Not stated per image in the bucket; used only for local testing, do not redistribute. Labels in `mediapipe.json` were assigned by hand. |

## Test tools (not shipped)

| Tool | Licence | Use |
|------|---------|-----|
| onnxruntime 1.30, onnx 1.17 | MIT / Apache-2.0 | ONNX reference outputs, shape probe in the exporter |
| ai_edge_litert (TFLite interpreter), `tflite` flatbuffer parser 2.18 | Apache-2.0 | TFLite reference outputs, TFLite parsing |
| PyTorch 2.14 | BSD-3-Clause | building the ONNX test models |
| mediapipe 1.0.1 (pip, in a throw-away venv) | Apache-2.0 | official FaceDetector reference in `tests/face_ref_mediapipe.py` |
| Playwright 1.56 + Chromium | Apache-2.0 / BSD | headless browser test |
