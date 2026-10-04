#!/usr/bin/env python3
"""Write tests/out/face_list.txt: absolute paths of the 114 images used by the
face references (Open Images close-up + general groups, MediaPipe portraits)."""
import json
import os

TD = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../testdata/people'))
m = json.load(open(os.path.join(TD, 'manifest.json')))
files = [i['file'] for i in m['items'] if i['group'] in ('closeup', 'general')]
files += [i['file'] for i in json.load(open(os.path.join(TD, 'mediapipe.json')))['items']]
out = os.path.join(os.path.dirname(__file__), 'out/face_list.txt')
open(out, 'w').write('\n'.join(os.path.join(TD, f) for f in files))
print(len(files), 'files ->', out)
