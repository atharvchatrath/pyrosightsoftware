"""Held-out window test media for the page tests: camera/testdata/window/{images,manifest.json}.

    python3 tools/make_window_testdata.py

Picks Open Images photos of the *test* split with window boxes (never used for training or for
choosing thresholds), without looking at any model output: indoor scenes first (Room, Interior
design, Living room, Bedroom, Kitchen, Bathroom, ... verified), then the rest; at most 6 single
window boxes, no group-of window box, the largest window at least 5 % of the picture; not rotated.
Order is a hash of the image id. Boxes: window (and door, where present), normalised x, y, w, h.
"""
import csv
import glob
import hashlib
import json
import os
import shutil
import sys

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, R)
from data import load_index  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oi_stats import load  # noqa: E402

T = os.path.join(R, '..', 'testdata', 'window')
INDOOR = ['Room', 'Interior design', 'Living room', 'Bedroom', 'Kitchen', 'Bathroom', 'Ceiling', 'Floor',
          'Apartment', 'Home', 'Daylighting']
N = 40


def main():
    o = load()
    indoor = set().union(*[o['pos'].get(n, set()) for n in INDOOR])
    meta = {}
    for f in glob.glob(os.path.join(R, 'data/oi/*-images-with-rotation.csv')):
        for r in csv.DictReader(open(f)):
            meta[r['ImageID']] = r
    cand = []
    for it in load_index('test', ['window']):
        m = meta.get(it['oi_id'])
        if not m or m.get('Rotation') not in ('', '0.0', None):
            continue
        w = [b for b in it['boxes'] if int(b[0]) == 2]
        ng = [b for b in w if not b[5]]
        if len(ng) != len(w) or not ng or len(ng) > 6:
            continue
        if max((b[3] - b[1]) * (b[4] - b[2]) for b in ng) < 0.05:
            continue
        cand.append((0 if it['oi_id'] in indoor else 1, hashlib.md5(it['oi_id'].encode()).hexdigest(), it, m))
    cand.sort(key=lambda c: c[:2])
    pick = cand[:N]
    os.makedirs(os.path.join(T, 'images'), exist_ok=True)
    man = dict(description='Held-out Open Images (test split here) photos with Window boxes, for the window class of '
                           'FireDoorNet. Chosen by tools/make_window_testdata.py without looking at model output: indoor '
                           'scenes first, at most 6 windows, the largest at least 5 % of the picture.',
               images=[])
    for ind, _, it, m in pick:
        dst = 'oi_%s.jpg' % it['oi_id']
        shutil.copy(os.path.join(R, it['path']), os.path.join(T, 'images', dst))
        man['images'].append(dict(file='images/' + dst, expect='window', indoor=ind == 0,
                                  source='Open Images V5 ' + it['src'].split('/')[-1], license=m.get('License'),
                                  author=m.get('Author'), url=m.get('OriginalLandingURL'),
                                  boxes=[dict(cls='window' if int(b[0]) == 2 else 'door', x=b[1], y=b[2], w=b[3] - b[1], h=b[4] - b[2],
                                              group_of=bool(b[5])) for b in it['boxes'] if int(b[0]) in (1, 2)]))
    json.dump(man, open(os.path.join(T, 'manifest.json'), 'w'), indent=1)
    print('candidates', len(cand), '(indoor %d)' % sum(1 for c in cand if c[0] == 0), '-> picked', len(pick),
          '(indoor %d)' % sum(1 for c in pick if c[0] == 0))


if __name__ == '__main__':
    main()
