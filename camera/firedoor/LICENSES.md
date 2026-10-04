# Licences of everything used by camera/firedoor

| What | Used for | Licence / terms | Source |
|---|---|---|---|
| MobileNetV2 alpha 0.5 ImageNet weights (`assets/keras/mobilenet_v2_0.5_224_no_top.h5`) | backbone initialisation (shipped, fine-tuned, inside `export/firedoor.onnx`) | Apache-2.0 (keras-applications weights release) | storage.googleapis.com/tensorflow/keras-applications/mobilenet_v2/ |
| MobileNetV2 alpha 0.5 ImageNet weights *with classifier top* (`data/mnv2_0.5_224_top.h5`) | only to check the Keras to PyTorch weight mapping (ImageNet top-5 on Open Images crops); not shipped | Apache-2.0 | same bucket |
| ImageNet class index JSON | label names for the mapping check; not shipped | Apache-2.0 (TensorFlow/Keras) | storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json |
| FireNET dataset (502 images, Pascal VOC fire boxes) | training (412-image train split, minus 55 kept for validation), test = the official 90-image validation split | MIT, Copyright (c) 2019 Moses Olafenwa | github.com/OlafenwaMoses/FireNET (release v1.0 `fire-dataset.zip`) |
| FireNET repo `video1.mp4` | evaluation only (frame-level fire recall); short clips in `testdata/fire/clips` and `testdata/negatives/clips/road_before_fire*` | Repo is MIT, but the footage is CGTN news / CCTV video (third-party content). Keep the clips for **local testing only**; do not publish them. | github.com/OlafenwaMoses/FireNET |
| Open Images V5/V6 annotations (boxes, human-verified image labels, image metadata) | door boxes, hard-negative selection, image-level fire labels (weak MIL supervision), held-out door/negative test sets | CC BY 4.0 (Google LLC) | storage.googleapis.com/openimages/ |
| Open Images images | training and test images (resized copies in `data/oi/img`, test subsets copied to `testdata/door` and `testdata/negatives`) | Each image is listed by Open Images as CC BY 2.0 (Flickr). All 3,924 selected validation/test images are CC BY 2.0 by the metadata CSV. Author, licence and URL for every copied test image are in the `testdata/*/manifest.json` entries | open-images-dataset.s3.amazonaws.com |
| DeepQuest AI FIRE-SMOKE-DATASET | **evaluation only** (Test/Fire fire-present recall, Test/Neutral false alarms, Test/Smoke for information). Not used for training, not copied into testdata (the manifest only points to it) | **No licence stated** in the repository (checked 2026-10-03). Without a licence we treated it as all-rights-reserved, so we used it only for internal evaluation | github.com/DeepQuestAI/Fire-Smoke-Dataset |

Datasets we looked at and did not use:
- D-Fire (21k images, fire and smoke boxes) is on Google Drive, which this machine cannot reach.
- DBA-YOLO-Dataset (GitHub, ScryAbu) states no licence ("research purposes"), so we skipped it.
- FireData (LOIIIIIII) is hosted on Baidu AI Studio. Not reachable, and no licence stated.

The trained weights in `export/firedoor.onnx` come from the Apache-2.0 backbone, MIT FireNET data and CC BY Open Images data. Attribution for Open Images (CC BY 2.0 images, CC BY 4.0 annotations) should be kept in the app's credits.
