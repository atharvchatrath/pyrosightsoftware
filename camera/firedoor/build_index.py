"""Build data/index.json: every training/eval image with normalised boxes, split and supervision flags.

Classes (box field 0): 0 fire, 1 door, 2 window. sup = [fire, door, window]: 1 = that channel is
supervised over the whole image (absent objects are negatives), 0 = ignored except inside `regions`.
kinds: fire_box (FireNET, VOC boxes; door and window channels unsupervised)
       fire_weak (Open Images image-level Fire/Flame/Bonfire/Campfire/Wildfire; MIL only, door/window unsupervised)
       door, neg (Open Images; fire channel fully negative, door channel from boxes, window channel
                  supervised when "Window" was verified for the image, see tools/oi_select_window.py)
       window, wneg (Open Images, added for the window class: window photos, verified window-free
                  photos and window look-alikes; door channel supervised when "Door" was verified)
regions: [[cls, x0, y0, x1, y1], ...] where channel cls is supervised even when sup[cls] = 0: the window
channel inside window look-alikes (mirrors, pictures, TVs, ...) and inside door boxes, the door channel
inside window boxes. Window boxes with IsGroupOf = 1 are ignored in training ("don't care" in evaluation).
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
                          sup=[1, 0, 0], focus=[], src='firenet/' + sp, size=[W, H], size_mismatch=(W, H) != (xw, xh)))

sel = json.load(open(os.path.join(R, 'data/oi/selection.json')))
wsel_path = os.path.join(R, 'data/oi/selection_window.json')
wsel = json.load(open(wsel_path)) if os.path.exists(wsel_path) else {'new': [], 'old': {}}
# more train photos whose windows are big (a window filling much of the view), see tools/oi_select_window_big.py
bsel_path = os.path.join(R, 'data/oi/selection_window_big.json')
if os.path.exists(bsel_path):
    have = {v['id'] for v in sel} | {v['id'] for v in wsel['new']}
    wsel['new'] += [v for v in json.load(open(bsel_path))['new'] if v['id'] not in have]
# window photos of cars, buses, trains, planes and boats are left out: their windows are not a way out
veh_path = os.path.join(R, 'data/oi/vehicle_ids.json')
vehicles = set(json.load(open(veh_path))) if os.path.exists(veh_path) else set()
for v in sel:
    v.update(wsel['old'].get(v['id'], {}))
for v in sel + wsel['new']:
    p = os.path.join(R, 'data/oi/img', v['id'] + '.jpg')
    if not os.path.exists(p):
        continue
    role = v['role']
    if role == 'window' and v['id'] in vehicles:
        continue
    kind = ('fire_weak' if role == 'fire_weak' else 'door' if role == 'door' else 'window' if role == 'window'
            else 'wneg' if role.startswith('wneg_') else 'neg')
    boxes = [[1, d[0], d[1], d[2], d[3], d[4]] for d in v['doors']]   # last = IsGroupOf
    wins = v.get('windows', [])
    boxes += [[2, w[0], w[1], w[2], w[3], w[4]] for w in wins]
    wsup, dsup = v.get('wsup', 0), v.get('dsup', 1)
    regions = []
    if not wsup:
        regions += [[2] + list(b[:4]) for b in v.get('wneg', [])]
        regions += [[2, d[0], d[1], d[2], d[3]] for d in v['doors'] if not d[4]]
    if not dsup:
        regions += [[1, w[0], w[1], w[2], w[3]] for w in wins if not w[4]]
    focus = v['focus']
    if kind == 'window':      # zoom crops onto a window now and then (a window filling much of the view)
        focus = [['Window'] + w[:4] for w in wins if not w[4] and (w[2] - w[0]) * (w[3] - w[1]) >= 0.01]
    elif kind == 'wneg':
        focus = [['Lookalike'] + list(b[:4]) for b in v.get('wneg', [])][:8]
    items.append(dict(path=os.path.relpath(p, R), kind=kind, split=v['split'], boxes=boxes,
                      sup=[0, 0, 0] if kind == 'fire_weak' else [1, dsup, wsup], focus=focus, regions=regions,
                      lookalikes=[list(x[:4]) for x in v.get('wneg', [])],
                      src='openimages/' + v['source'], role=role, oi_id=v['id']))

json.dump(items, open(os.path.join(R, 'data/index.json.tmp'), 'w'))
os.replace(os.path.join(R, 'data/index.json.tmp'), os.path.join(R, 'data/index.json'))
import collections
print(collections.Counter((i['kind'], i['split']) for i in items))
print('firenet size mismatches', sum(i.get('size_mismatch', False) for i in items))
