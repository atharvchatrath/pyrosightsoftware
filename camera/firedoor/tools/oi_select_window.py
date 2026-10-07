"""Add the WINDOW class to the FireDoorNet data: select Open Images images with window boxes, window
look-alikes and verified window-free photos, and annotate every image (old and new) with its window
supervision. Writes data/oi/selection_window.json (selection.json is left as it was).

    python3 tools/oi_train_stream.py     # once: train-bbox-window.csv, train-il-window.csv
    python3 tools/oi_select_window.py    # -> data/oi/selection_window.json (+ train-rotation-window.csv)
    python3 tools/oi_download.py data/oi/selection_window.json
    python3 build_index.py

Per image (every entry of selection.json and the new ones):
  windows  [[x0, y0, x1, y1, group_of], ...]  Open Images "Window" (/m/0d4v4) boxes, normalised
  wsup     1 when "Window" was human-verified for the image (present, with its boxes, or absent): the
           window channel is supervised everywhere. 0: only inside the regions below.
  wneg     boxes of window look-alikes (mirror, picture frame, TV, monitor, fridge, wardrobe, cupboard,
           closet, bookcase, whiteboard, poster, billboard, laptop, tablet): the window channel is
           supervised (negative) inside them even when wsup = 0.
  dsup     door channel supervision for the NEW entries: 1 when "Door" was verified, else 0 (then the
           door channel is still supervised, negative, inside window boxes). Old entries keep 1.
New roles: window (val/test source: hash split; train source: train), wneg_nowindow (verified no
window), wneg_<look-alike>. Window images are kept only when the largest single window box covers at
least 1 % of the picture; rotated images are left out; fire-like labels are excluded as in oi_select.py.
"""
import collections
import csv
import hashlib
import io
import json
import os
import random
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(__file__))
from oi_stats import load, D  # noqa: E402

random.seed(1)
ROT_URL = 'https://storage.googleapis.com/openimages/2018_04/train/train-images-boxable-with-rotation.csv'
WIN = 'Window'
LOOK = ['Mirror', 'Picture frame', 'Television', 'Computer monitor', 'Refrigerator', 'Wardrobe', 'Cupboard',
        'Closet', 'Bookcase', 'Whiteboard', 'Poster', 'Billboard', 'Laptop', 'Tablet computer']
LOOK_TAKE = {'Mirror': (200, 200), 'Picture frame': (200, 200), 'Television': (160, 160), 'Computer monitor': (160, 160),
             'Refrigerator': (160, 160), 'Wardrobe': (100, 100), 'Cupboard': (120, 120), 'Closet': (100, 100),
             'Bookcase': (100, 100), 'Whiteboard': (120, 120), 'Poster': (100, 100), 'Billboard': (80, 80),
             'Laptop': (60, 60), 'Tablet computer': (40, 40)}         # (val/test source, train source)
INDOOR = ['Room', 'Interior design', 'Living room', 'Bedroom', 'Kitchen', 'Bathroom', 'Ceiling', 'Floor',
          'Apartment', 'Home', 'Daylighting']
BUILDING = ['House', 'Building', 'Facade', 'Office building', 'Tower', 'Skyscraper']
FIRE = ['Fire', 'Flame', 'Bonfire', 'Campfire', 'Wildfire']
EXCL = FIRE + ['Candle', 'Candle holder', 'Fireplace', 'Fireworks', 'Explosion', 'Smoke', 'Lava',
               "Jack-o'-lantern", 'Torch', 'Lantern', 'Oil lamp', 'Barbecue grill', 'Gas stove',
               'Stove', 'Burn', 'Wood-burning stove', 'Heat']
TRAIN_WINDOW_TAKE = (1500, 900, 600)      # train-source window images: indoor, building, other


def hsplit(img):
    h = int(hashlib.md5(img.encode()).hexdigest()[:8], 16) % 100
    return 'train' if h < 50 else ('val' if h < 65 else 'test')


def main():
    o = load()
    pos, neg, boxes, srcsplit = o['pos'], o['neg'], o['boxes'], o['split']
    cd = {}
    for l in open(os.path.join(D, 'cd.csv')):
        k, v = l.rstrip('\n').split(',', 1)
        cd[k] = v.strip('"')

    # ---------------- train-source annotations
    tr_pos, tr_neg = collections.defaultdict(set), collections.defaultdict(set)
    for f in ['train-il-filtered.csv', 'train-il-window.csv']:
        for row in csv.reader(open(os.path.join(D, f))):
            if row[0] == 'ImageID':
                continue
            img, src, lab, conf = row[:4]
            (tr_pos if conf == '1' else tr_neg)[cd.get(lab, lab)].add(img)
    tr_boxes = collections.defaultdict(list)
    seen = set()
    for f in ['train-bbox-window.csv', 'train-bbox-filtered.csv']:
        for row in csv.reader(open(os.path.join(D, f))):
            if row[0] == 'ImageID':
                continue
            img, src, lab, conf, x0, x1, y0, y1, occ, trunc, grp, dep, ins = row[:13]
            key = (img, lab, x0, x1, y0, y1)
            if key in seen:            # Door and a few classes are in both files
                continue
            seen.add(key)
            tr_boxes[img].append((cd.get(lab, lab), float(x0), float(y0), float(x1), float(y1), int(grp), int(dep)))
    del seen

    vt_excl = set().union(*[pos.get(n, set()) for n in EXCL])
    vt_excl |= {i for i, bl in boxes.items() if any(b[0] in EXCL for b in bl)}
    tr_excl = set().union(*[tr_pos.get(n, set()) for n in EXCL])
    tr_excl |= {i for i, bl in tr_boxes.items() if any(b[0] in EXCL for b in bl)}

    # ---------------- rotation (val/test files exist; train: stream the boxable list once, keep our ids)
    rot = {}
    for f in ['validation-images-with-rotation.csv', 'test-images-with-rotation.csv', 'train-images-selected-with-rotation.csv']:
        for r in csv.DictReader(open(os.path.join(D, f))):
            rot[r['ImageID']] = r.get('Rotation') or ''
    trrot_path = os.path.join(D, 'train-rotation-window.csv')
    want_ids = {i for i, bl in tr_boxes.items() if any(b[0] == WIN or b[0] in LOOK for b in bl)}
    if not os.path.exists(trrot_path):
        print('streaming train rotation list ...', flush=True)
        with urllib.request.urlopen(ROT_URL, timeout=60) as r, open(trrot_path + '.tmp', 'w', newline='') as f:
            rd = csv.DictReader(io.TextIOWrapper(r, encoding='utf-8', newline=''))
            w = csv.writer(f)
            w.writerow(['ImageID', 'Rotation'])
            for row in rd:
                if row['ImageID'] in want_ids:
                    w.writerow([row['ImageID'], row['Rotation']])
        os.replace(trrot_path + '.tmp', trrot_path)
    for r in csv.DictReader(open(trrot_path)):
        rot.setdefault(r['ImageID'], r['Rotation'])
    rotated = {i for i, v in rot.items() if v not in ('', '0.0', '0')}

    def info(img, source):
        """window/look-alike/door fields for one image."""
        vt = source != 'train'
        bl = (boxes if vt else tr_boxes).get(img, [])
        P, N = (pos, neg) if vt else (tr_pos, tr_neg)
        wins = [list(b[1:5]) + [b[5]] for b in bl if b[0] == WIN]
        wver = img in P.get(WIN, ()) or img in N.get(WIN, ())
        wsup = 1 if wver and (wins or img in N.get(WIN, ())) else 0
        wneg = [list(b[1:5]) for b in bl if b[0] in LOOK and not b[5]]
        dsup = 1 if (img in P.get('Door', ()) or img in N.get('Door', ())) else 0
        return dict(windows=wins, wsup=wsup, wneg=wneg, dsup=dsup)

    old = json.load(open(os.path.join(D, 'selection.json')))
    old_ids = {v['id'] for v in old}
    sel = {}

    def add(img, source, role, split=None):
        if img in sel or img in old_ids or img in rotated:
            return False
        bl = (boxes if source != 'train' else tr_boxes).get(img, [])
        doors = [b[1:5] + (b[5],) for b in bl if b[0] == 'Door']
        e = dict(id=img, source=source, role=role, split=split or (hsplit(img) if source != 'train' else 'train'),
                 doors=doors, focus=[], url=f'https://open-images-dataset.s3.amazonaws.com/{source}/{img}.jpg')
        e.update(info(img, source))
        sel[img] = e
        return True

    def good_window(bl):
        w = [b for b in bl if b[0] == WIN]
        ng = [b for b in w if not b[5]]
        if not ng or len(ng) > 15 or sum(b[5] for b in w) > 2:
            return False
        return max((b[3] - b[1]) * (b[4] - b[2]) for b in ng) >= 0.01

    # ---------------- window images, val/test source (exhaustive boxes; hash split)
    for img in sorted(i for i, bl in boxes.items() if i not in vt_excl and good_window(bl)):
        add(img, srcsplit[img], 'window')
    # ---------------- window images, train source (indoor scenes first, then buildings, then the rest)
    indoor = set().union(*[tr_pos.get(n, set()) for n in INDOOR])
    indoor |= {i for i, bl in tr_boxes.items() if any(b[0] in ('Curtain', 'Window blind') for b in bl)}
    bld = set().union(*[tr_pos.get(n, set()) for n in BUILDING])
    cand = sorted(i for i, bl in tr_boxes.items() if i not in tr_excl and i not in rotated and good_window(bl)
                  and i in tr_pos.get(WIN, ()))
    tiers = [[i for i in cand if i in indoor], [i for i in cand if i not in indoor and i in bld],
             [i for i in cand if i not in indoor and i not in bld]]
    for t, n in zip(tiers, TRAIN_WINDOW_TAKE):
        random.shuffle(t)
        k = 0
        for img in t:
            if k >= n:
                break
            k += add(img, 'train', 'window')
    print('train-source window candidates (indoor, building, other):', [len(t) for t in tiers])

    # ---------------- verified window-free photos (val/test source)
    for img in sorted(neg.get(WIN, set()) - vt_excl):
        if hsplit(img) == 'train' and random.random() > 0.45:
            continue
        add(img, srcsplit[img], 'wneg_nowindow')

    # ---------------- window look-alikes
    vt_lab = collections.defaultdict(set)
    for i, bl in boxes.items():
        for b in bl:
            vt_lab[b[0]].add(i)
    tr_lab = collections.defaultdict(set)
    for i, bl in tr_boxes.items():
        for b in bl:
            tr_lab[b[0]].add(i)
    for n in LOOK:
        nvt, ntr = LOOK_TAKE[n]
        role = 'wneg_' + n.replace(' ', '_').lower()
        c = sorted(vt_lab[n] - vt_excl - vt_lab[WIN]); random.shuffle(c)
        k = 0
        for img in c:
            if k >= nvt:
                break
            k += add(img, srcsplit[img], role)
        c = sorted(tr_lab[n] - tr_excl - tr_lab[WIN]); random.shuffle(c)
        k = 0
        for img in c:
            if k >= ntr:
                break
            k += add(img, 'train', role)

    # ---------------- window fields for every old entry (their door supervision stays as it was)
    old_ann = {}
    for v in old:
        e = info(v['id'], v['source'])
        e['dsup'] = 1
        old_ann[v['id']] = e

    roles = collections.Counter((v['role'] if not v['role'].startswith('wneg_') or v['role'] == 'wneg_nowindow' else 'wneg_lookalike', v['split'])
                                for v in sel.values())
    print(sorted(roles.items()))
    print('new images', len(sel), 'window boxes', sum(len(v['windows']) for v in sel.values()),
          'test window images', sum(1 for v in sel.values() if v['role'] == 'window' and v['split'] == 'test'))
    print('old entries with window supervision', sum(e['wsup'] for e in old_ann.values()), 'of', len(old_ann),
          '; with window boxes', sum(bool(e['windows']) for e in old_ann.values()),
          '; with look-alike regions', sum(bool(e['wneg']) for e in old_ann.values()))
    print('new entries with door verified', sum(v['dsup'] for v in sel.values()))
    json.dump(dict(new=sorted(sel.values(), key=lambda v: v['id']), old=old_ann),
              open(os.path.join(D, 'selection_window.json'), 'w'))


if __name__ == '__main__':
    main()
