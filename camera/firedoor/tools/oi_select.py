"""Select Open Images v5/v6 images for the fire+door detector and write data/oi/selection.json.

Sources: v5 validation+test (human-verified labels, exhaustive boxes for verified classes)
and v6 train (boxes for door + hard-negative classes, image labels for fire-ish classes).
Our split for val/test-sourced images is a hash of the image id; train-sourced images are
only ever used for training.
"""
import csv, collections, hashlib, json, os, random, sys
sys.path.insert(0, os.path.dirname(__file__))
from oi_stats import load, D

random.seed(0)
o = load()
pos, boxes, srcsplit = o['pos'], o['boxes'], o['split']
cd = {}
for l in open(os.path.join(D, 'cd.csv')):
    k, v = l.rstrip('\n').split(',', 1); cd[k] = v.strip('"')

FIRE = ['Fire', 'Flame', 'Bonfire', 'Campfire', 'Wildfire']
EXCL = FIRE + ['Candle', 'Candle holder', 'Fireplace', 'Fireworks', 'Explosion', 'Smoke', 'Lava',
               "Jack-o'-lantern", 'Torch', 'Lantern', 'Oil lamp', 'Barbecue grill', 'Gas stove',
               'Stove', 'Burn', 'Wood-burning stove', 'Heat']
FOCUS_BOX = ['Lamp', 'Light bulb', 'Traffic light', 'Pumpkin', 'Orange', 'Street light', 'Television',
             'Computer monitor', 'Christmas tree', 'Flashlight', 'Human face', 'Human hand']
SKY = ['Sunset', 'Sunrise', 'Afterglow', 'Red sky at morning', 'Dusk', 'Incandescent light bulb', 'Neon',
       'Light fixture', 'Lighting', 'Sunlight']
INDOOR = ['Room', 'Interior design', 'Kitchen', 'Living room', 'Bedroom', 'Ceiling']

def hsplit(img):
    h = int(hashlib.md5(img.encode()).hexdigest()[:8], 16) % 100
    return 'train' if h < 50 else ('val' if h < 65 else 'test')

# ---------------- train-source annotations (filtered streams)
tr_pos = collections.defaultdict(set)
for img, src, lab, conf in csv.reader(open(os.path.join(D, 'train-il-filtered.csv'))):
    if conf == '1':
        tr_pos[cd.get(lab, lab)].add(img)
tr_boxes = collections.defaultdict(list)
for row in csv.reader(open(os.path.join(D, 'train-bbox-filtered.csv'))):
    img, src, lab, conf, x0, x1, y0, y1, occ, trunc, grp, dep, ins = row[:13]
    tr_boxes[img].append((cd.get(lab, lab), float(x0), float(y0), float(x1), float(y1), int(grp), int(dep)))

vt_excl = set().union(*[pos.get(n, set()) for n in EXCL])
vt_excl |= {i for i, bl in boxes.items() if any(b[0] in EXCL for b in bl)}
tr_excl = set().union(*[tr_pos.get(n, set()) for n in EXCL])
tr_excl |= {i for i, bl in tr_boxes.items() if any(b[0] in EXCL for b in bl)}

sel = {}
def add(img, source, role, split=None):
    if img in sel:
        return
    bl = (boxes if source != 'train' else tr_boxes).get(img, [])
    doors = [b[1:5] + (b[5],) for b in bl if b[0] == 'Door']
    focus = [(b[0],) + b[1:5] for b in bl if b[0] in FOCUS_BOX and not b[5]]
    sel[img] = dict(id=img, source=source, role=role,
                    split=split or (hsplit(img) if source != 'train' else 'train'),
                    doors=doors, focus=focus,
                    url=f'https://open-images-dataset.s3.amazonaws.com/{source}/{img}.jpg')

# fire (image-level, weak)
vt_fire = set().union(*[pos.get(n, set()) for n in FIRE]) - pos.get('Fireworks', set())
tr_fire = set().union(*[tr_pos.get(n, set()) for n in FIRE]) - tr_pos.get('Fireworks', set())
for img in sorted(vt_fire):
    add(img, srcsplit[img], 'fire_weak')
for img in sorted(tr_fire):
    add(img, 'train', 'fire_weak')

# doors
vt_door = sorted({i for i, bl in boxes.items() if any(b[0] == 'Door' for b in bl)} - vt_excl)
for img in vt_door:
    add(img, srcsplit[img], 'door')
tr_door = []
for img, bl in tr_boxes.items():
    if img in tr_excl:
        continue
    d = [b for b in bl if b[0] == 'Door']
    if not d or len(d) > 6 or sum(b[5] for b in d) > 1:
        continue
    if max((b[3] - b[1]) * (b[4] - b[2]) for b in d) < 0.02:
        continue
    tr_door.append(img)
random.shuffle(tr_door)
for img in sorted(tr_door[:2600]):
    add(img, 'train', 'door')

# hard negatives (val/test: verified labels)
def take(cands, n):
    c = sorted(cands); random.shuffle(c); return c[:n]
vt_boxlab = collections.defaultdict(set)
for i, bl in boxes.items():
    for b in bl:
        vt_boxlab[b[0]].add(i)
for n in FOCUS_BOX:
    k = {'Human face': 700, 'Human hand': 200}.get(n, 160)
    for img in take(vt_boxlab[n] - vt_excl, k):
        add(img, srcsplit[img], 'neg_' + n.replace(' ', '_').lower())
for n in SKY:
    for img in take(pos.get(n, set()) - vt_excl, 120):
        add(img, srcsplit[img], 'neg_' + n.replace(' ', '_').lower())
for img in take(set().union(*[pos.get(n, set()) for n in INDOOR]) - vt_excl, 500):
    add(img, srcsplit[img], 'neg_indoor')
# hard negatives (train source)
tr_boxlab = collections.defaultdict(set)
for i, bl in tr_boxes.items():
    for b in bl:
        tr_boxlab[b[0]].add(i)
for n in FOCUS_BOX:
    if n in ('Human face', 'Human hand'):
        continue
    for img in take(tr_boxlab[n] - tr_excl, 220):
        add(img, 'train', 'neg_' + n.replace(' ', '_').lower())
for n in ['Sunset', 'Sunrise', 'Afterglow', 'Incandescent light bulb', 'Neon']:
    for img in take(tr_pos.get(n, set()) - tr_excl, 300):
        add(img, 'train', 'neg_' + n.replace(' ', '_').lower())

roles = collections.Counter((v['role'].split('_')[0] if v['role'] != 'fire_weak' else 'fire_weak', v['split']) for v in sel.values())
print(sorted(roles.items()))
print('total', len(sel), 'door boxes', sum(len(v['doors']) for v in sel.values()),
      'test door boxes', sum(len(v['doors']) for v in sel.values() if v['split'] == 'test'))
json.dump(sorted(sel.values(), key=lambda v: v['id']), open(os.path.join(D, 'selection.json'), 'w'))
