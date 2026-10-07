"""Select extra Open Images TRAIN photos whose windows are big (the largest single window covers 8-90 %
of the picture), in the same entry format as the "new" entries of selection_window.json.

    python3 tools/oi_select_window_big.py 6000     # -> data/oi/selection_window_big.json ({"new": [...]})
    python3 tools/oi_download.py data/oi/selection_window_big.json
    python3 tools/oi_vehicles.py                   # vehicle photos, left out by build_index.py
    python3 build_index.py

Why: selection_window.json is dominated by small windows on building fronts, and the window detector
trained on it found few of the windows that fill much of the view (standing in a room, facing a window),
the case that matters for this page. (The indoor-first ordering below finds no indoor labels, because
cd.csv names only the boxable classes; the 6000 photos used were picked at random from all candidates.)
"""
import collections, csv, json, os, random, sys
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'oi')
random.seed(7)
WIN = 'Window'
LOOK = ['Mirror', 'Picture frame', 'Television', 'Computer monitor', 'Refrigerator', 'Wardrobe', 'Cupboard',
        'Closet', 'Bookcase', 'Whiteboard', 'Poster', 'Billboard', 'Laptop', 'Tablet computer']
INDOOR = ['Room', 'Interior design', 'Living room', 'Bedroom', 'Kitchen', 'Bathroom', 'Ceiling', 'Floor',
          'Apartment', 'Home', 'Daylighting']
FIRE = ['Fire', 'Flame', 'Bonfire', 'Campfire', 'Wildfire']
EXCL = FIRE + ['Candle', 'Candle holder', 'Fireplace', 'Fireworks', 'Explosion', 'Smoke', 'Lava',
               "Jack-o'-lantern", 'Torch', 'Lantern', 'Oil lamp', 'Barbecue grill', 'Gas stove',
               'Stove', 'Burn', 'Wood-burning stove', 'Heat']
TAKE = int(sys.argv[1]) if len(sys.argv) > 1 else 6000

cd = {}
for l in open(os.path.join(D, 'cd.csv')):
    k, v = l.rstrip('\n').split(',', 1)
    cd[k] = v.strip('"')
tr_pos, tr_neg = collections.defaultdict(set), collections.defaultdict(set)
for f in ['train-il-filtered.csv', 'train-il-window.csv']:
    for row in csv.reader(open(os.path.join(D, f))):
        if row[0] == 'ImageID':
            continue
        (tr_pos if row[3] == '1' else tr_neg)[cd.get(row[2], row[2])].add(row[0])
tr_boxes = collections.defaultdict(list)
seen = set()
for f in ['train-bbox-window.csv', 'train-bbox-filtered.csv']:
    for row in csv.reader(open(os.path.join(D, f))):
        if row[0] == 'ImageID':
            continue
        img, src, lab, conf, x0, x1, y0, y1, occ, trunc, grp, dep, ins = row[:13]
        key = (img, lab, x0, x1, y0, y1)
        if key in seen:
            continue
        seen.add(key)
        tr_boxes[img].append((cd.get(lab, lab), float(x0), float(y0), float(x1), float(y1), int(grp), int(dep)))
del seen
tr_excl = set().union(*[tr_pos.get(n, set()) for n in EXCL])
tr_excl |= {i for i, bl in tr_boxes.items() if any(b[0] in EXCL for b in bl)}
rot = {}
for f in ['train-images-selected-with-rotation.csv', 'train-rotation-window.csv']:
    for r in csv.DictReader(open(os.path.join(D, f))):
        rot.setdefault(r['ImageID'], r.get('Rotation') or '')
rotated = {i for i, v in rot.items() if v not in ('', '0.0', '0')}
have = {v['id'] for v in json.load(open(os.path.join(D, 'selection.json')))}
sw = json.load(open(os.path.join(D, 'selection_window.json')))
have |= {v['id'] for v in sw['new']} | set(sw['old'])

def area(b):
    return (b[3] - b[1]) * (b[4] - b[2])

cand = []
for img, bl in tr_boxes.items():
    if img in have or img in tr_excl or img in rotated or img not in tr_pos.get(WIN, ()) or img not in rot:
        continue
    w = [b for b in bl if b[0] == WIN]
    ng = [b for b in w if not b[5]]
    if not ng or len(ng) > 6 or sum(b[5] for b in w) > 1:
        continue
    big = max(area(b) for b in ng)
    if not 0.08 <= big <= 0.9:
        continue
    cand.append(img)
indoor = set().union(*[tr_pos.get(n, set()) for n in INDOOR])
indoor |= {i for i, bl in tr_boxes.items() if any(b[0] in ('Curtain', 'Window blind') for b in bl)}
cand.sort()
random.shuffle(cand)
ind = [i for i in cand if i in indoor]
oth = [i for i in cand if i not in indoor]
pick = ind[:int(TAKE * 0.6)]
pick += oth[:TAKE - len(pick)]
print('candidates: indoor', len(ind), 'other', len(oth), '-> picked', len(pick), '(indoor', sum(1 for i in pick if i in indoor), ')')

out = []
for img in sorted(pick):
    bl = tr_boxes[img]
    wins = [list(b[1:5]) + [b[5]] for b in bl if b[0] == WIN]
    out.append(dict(id=img, source='train', role='window', split='train',
                    doors=[b[1:5] + (b[5],) for b in bl if b[0] == 'Door'], focus=[],
                    url=f'https://open-images-dataset.s3.amazonaws.com/train/{img}.jpg',
                    windows=wins, wsup=1, wneg=[list(b[1:5]) for b in bl if b[0] in LOOK and not b[5]],
                    dsup=1 if (img in tr_pos.get('Door', ()) or img in tr_neg.get('Door', ())) else 0))
sizes = collections.Counter()
for e in out:
    a = max(area([0] + w[:4]) for w in e['windows'] if not w[4])
    sizes['8-20%' if a < 0.2 else '20-50%' if a < 0.5 else '50-90%'] += 1
print('largest-window size of picked images', dict(sizes), '; with doors', sum(1 for e in out if e['doors']))
json.dump(dict(new=out), open(os.path.join(D, 'selection_window_big.json'), 'w'))
print('wrote', os.path.join(D, 'selection_window_big.json'))
