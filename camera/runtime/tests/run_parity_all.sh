#!/bin/sh
# Export every test model with float32/float16/uint8 weights and compare the
# op-list runtime (node, TF.js CPU) with onnxruntime / TFLite.
# Usage: sh tests/run_parity_all.sh   (from camera/runtime; ONNX test models
# from tests/make_onnx_models.py, references from tests/ref_outputs.py)
set -e
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=2
O=tests/out
mkdir -p $O/parity
if [ ! -f $O/ref_blazeface.json ]; then
  python3 tests/ref_outputs.py ../assets/blazeface/blaze_face_short_range.tflite $O/ref_blazeface \
    --random 4 --images '../testdata/people/images/*.jpg' --n-images 8 --range -1 1
fi
for w in float32 float16 uint8; do
  python3 export_oplist.py ../assets/blazeface/blaze_face_short_range.tflite -o models/blazeface_$w --weights $w >/dev/null
  node tests/parity.js models/blazeface_$w $O/ref_blazeface --json $O/parity/blazeface_$w.json >/dev/null 2>&1
done
for m in allops_opset11 allops_opset13 allops_opset17 mnv2det pyronet pads; do
  if [ ! -f $O/onnx/ref_$m.json ]; then
    python3 tests/ref_outputs.py $O/onnx/$m.onnx $O/onnx/ref_$m --random 3 \
      --images '../testdata/people/images/*.jpg' --n-images 3 --range 0 1
  fi
  for w in float32 float16 uint8; do
    python3 export_oplist.py $O/onnx/$m.onnx -o $O/onnx/${m}_$w --weights $w >/dev/null
    node tests/parity.js $O/onnx/${m}_$w $O/onnx/ref_$m --json $O/parity/${m}_$w.json >/dev/null 2>&1
  done
done
python3 - <<'PY'
import glob, json, os
rows = []
for f in sorted(glob.glob('tests/out/parity/*.json')):
    d = json.load(open(f))
    rows.append(d)
    print('%-26s %-8s %8.1f ms  ' % (d['model'], d['storage'], d['medianRunMs']) +
          '  '.join('%s: abs %.3g rel %.2g' % (k, d['maxAbs'][k], d['maxRel'][k]) for k in d['maxAbs']))
json.dump(rows, open('tests/out/parity_summary.json', 'w'), indent=1)
PY
