"""Build data/index.json: every training/eval image with normalised boxes, split and supervision flags.

kinds: fire_box (FireNET, VOC boxes; door channel unsupervised)
       fire_weak (Open Images image-level Fire/Flame/Bonfire/Campfire/Wildfire; MIL only, door unsupervised)
       door, neg (Open Images; fire channel fully negative, door channel from boxes)
"""
import glob, hashlib, json, os, re
import xml.etree.ElementTree as ET
from PIL import Image

R = os.path.dirname(os.path.abspath(__file__))
A = os.path.join(R, '..', 'assets')
items = []

def h(s, mod=100):
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16) % mod

for sp in ['train', 'validation']:
    for x in sorted(glob.glob(os.path.join(A, 'firenet/fire-dataset', sp, 'annotations', '*.xml'))):
        t = ET.parse(x).getroot()
        fn = t.find('filename').text
        p = os.path.join(A, 'firenet/fire-dataset', sp, 'images', fn)
        if not os.path.exists(p):
            p = os.path.join(A, 'firenet/fire-dataset', sp, 'images', os.path.basename(x)[:-4] + '.jpg')
        W, H = Image.open(p).size
        xw, xh = int(t.find('size/width').text), int(t.find('size/height').text)
        boxes = []
        for o in t.findall('object'):
            b = o.find('bndbox')
            x0, y0, x1, y1 = (float(b.find(k).text) for k in ['xmin', 'ymin', 'xmax', 'ymax'])
            boxes.append([0, max(0, x0 / xw), max(0, y0 / xh), min(1, x1 / xw), min(1, y1 / xh)])
        split = 'test' if sp == 'validation' else ('val' if h(fn) < 12 else 'train')
        items.append(dict(path=os.path.relpath(p, R), kind='fire_box', split=split, boxes=boxes,
                          sup=[1, 0], focus=[], src='firenet/' + sp, size=[W, H], size_mismatch=(W, H) != (xw, xh)))

sel = json.load(open(os.path.join(R, 'data/oi/selection.json')))
for v in sel:
    p = os.path.join(R, 'data/oi/img', v['id'] + '.jpg')
    if not os.path.exists(p):
        continue
    kind = 'fire_weak' if v['role'] == 'fire_weak' else ('door' if v['role'] == 'door' else 'neg')
    boxes = [[1, d[0], d[1], d[2], d[3], d[4]] for d in v['doors']]   # last = IsGroupOf
    items.append(dict(path=os.path.relpath(p, R), kind=kind, split=v['split'], boxes=boxes,
                      sup=[0, 0] if kind == 'fire_weak' else [1, 1], focus=v['focus'],
                      src='openimages/' + v['source'], role=v['role'], oi_id=v['id']))

json.dump(items, open(os.path.join(R, 'data/index.json'), 'w'))
import collections
print(collections.Counter((i['kind'], i['split']) for i in items))
print('firenet size mismatches', sum(i.get('size_mismatch', False) for i in items))
