#!/usr/bin/env python3
"""Official MediaPipe Tasks FaceDetector (same blaze_face_short_range.tflite)
as an end-to-end reference. Runs in a separate venv with `pip install
mediapipe`; on a headless box without libEGL point LD_LIBRARY_PATH at a
directory with libEGL.so.1/libGLESv2.so.2 (Chromium's ANGLE copies work; the
CPU graph never touches them).

    python face_ref_mediapipe.py LIST.txt OUT.json
"""
import json
import sys

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

MODEL = sys.argv[3] if len(sys.argv) > 3 else \
    '/home/claude/pyrosight/camera/assets/blazeface/blaze_face_short_range.tflite'


def main():
    files = [l.strip() for l in open(sys.argv[1]) if l.strip()]
    det = vision.FaceDetector.create_from_options(vision.FaceDetectorOptions(
        base_options=BaseOptions(model_asset_path=MODEL), min_detection_confidence=0.5,
        min_suppression_threshold=0.3))
    out = []
    for f in files:
        img = mp.Image.create_from_file(f)
        r = det.detect(img)
        W, H = img.width, img.height
        faces = []
        for d in r.detections:
            b = d.bounding_box
            faces.append(dict(score=d.categories[0].score, x=b.origin_x / W, y=b.origin_y / H,
                              w=b.width / W, h=b.height / H, kp=[(k.x, k.y) for k in d.keypoints]))
        out.append(dict(file=f, width=W, height=H, faces=faces))
    det.close()
    json.dump({'images': out}, open(sys.argv[2], 'w'))
    print(len(out), 'images,', sum(len(o['faces']) for o in out), 'faces')


if __name__ == '__main__':
    main()
