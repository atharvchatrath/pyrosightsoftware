#!/usr/bin/env python3
"""Build the people/face test corpus under camera/testdata/people.

Picks Open Images V5 *validation* images (CC BY 2.0 images, CC BY 4.0
annotations) in four groups and stores them downscaled (max side 640 px) with
a manifest of ground-truth boxes (normalised [0,1] x/y/w/h):

  closeup  - exactly one real (non-depiction) human face, face width >= 30 %
             of the image: selfie-like, what a webcam sees.
  multi    - three or more real people boxes.
  general  - one or two people, person box height >= 30 % of the image.
  negative - no human boxes of any kind (no Person/Man/Woman/Boy/Girl/
             Human-* part, no Clothing) and Person not verified present at
             image level. Half are picked from indoor scenes.

Usage:
  python3 make_people_testset.py --csv-dir DIR --out ../../testdata/people
where DIR holds validation-annotations-bbox.csv,
validation-annotations-human-imagelabels-boxable.csv and
oidv6-class-descriptions.csv from storage.googleapis.com/openimages/.
"""
import argparse
import collections
import concurrent.futures as cf
import csv
import io
import json
import os
import random
import urllib.request

from PIL import Image

PERSON = {'/m/01g317', '/m/04yx4', '/m/03bt1vf', '/m/01bl7v', '/m/05r655'}  # Person Man Woman Boy Girl
FACE = '/m/0dzct'
HUMAN_ANY = PERSON | {FACE, '/m/02p0tk3', '/m/0dzf4', '/m/035r7c', '/m/0k65p', '/m/03q69', '/m/04hgtk',
                      '/m/014sv8', '/m/0k0pj', '/m/0283dt1', '/m/031n1', '/m/039xj_', '/m/015h_t',
                      '/m/09j2d', '/m/09j5n', '/m/01d40f', '/m/0463sg',  # clothing, footwear, dress, accessory
                      '/m/01gl_m', '/m/013_1c', '/m/06msq', '/m/0167gd'}  # mannequin statue sculpture doll
INDOOR = {'/m/0c_jw', '/m/02dgv', '/m/04bcr3', '/m/01mzpv', '/m/02crq1', '/m/03ssj5', '/m/0d4v4',
          '/m/07c52', '/m/01s105', '/m/0fqt361', '/m/0642b4', '/m/01jfm_', '/m/03fp41'}


def iou(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    iy = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = ix * iy
    u = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / u if u > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv-dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--n-closeup', type=int, default=50)
    ap.add_argument('--n-multi', type=int, default=50)
    ap.add_argument('--n-general', type=int, default=50)
    ap.add_argument('--n-negative', type=int, default=50)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--rotation-csv', default=None,
                    help='2018_04/validation/validation-images-with-rotation.csv: per-image rotation + attribution')
    a = ap.parse_args()

    boxes = collections.defaultdict(list)
    with open(os.path.join(a.csv_dir, 'validation-annotations-bbox.csv')) as f:
        for r in csv.DictReader(f):
            boxes[r['ImageID']].append(r)
    person_verified = set()
    with open(os.path.join(a.csv_dir, 'validation-annotations-human-imagelabels-boxable.csv')) as f:
        for r in csv.DictReader(f):
            if r['LabelName'] in PERSON and r['Confidence'] == '1':
                person_verified.add(r['ImageID'])

    def summarise(iid):
        rows = boxes[iid]
        persons, ignore, faces, face_ignore = [], [], [], []
        for r in rows:
            b = (float(r['XMin']), float(r['YMin']), float(r['XMax']), float(r['YMax']))
            dep = r['IsDepiction'] == '1'
            grp = r['IsGroupOf'] == '1'
            if r['LabelName'] in PERSON:
                (ignore if (dep or grp) else persons).append(b)
            elif r['LabelName'] == FACE:
                (face_ignore if (dep or grp) else faces).append(b)
        # Person/Man/Woman boxes often duplicate each other: dedupe.
        uniq = []
        for b in sorted(persons, key=lambda b: -(b[2] - b[0]) * (b[3] - b[1])):
            if all(iou(b, u) < 0.7 for u in uniq):
                uniq.append(b)
        return uniq, ignore, faces, face_ignore

    rng = random.Random(a.seed)
    ids = sorted(boxes)
    rng.shuffle(ids)
    groups = {'closeup': [], 'multi': [], 'general': [], 'negative': []}
    want = {'closeup': a.n_closeup, 'multi': a.n_multi, 'general': a.n_general, 'negative': a.n_negative}
    neg_in, neg_out = [], []
    for iid in ids:
        labels = {r['LabelName'] for r in boxes[iid]}
        persons, ignore, faces, face_ignore = summarise(iid)
        has_dep = any(r['IsDepiction'] == '1' for r in boxes[iid] if r['LabelName'] in HUMAN_ANY)
        if not (labels & HUMAN_ANY) and iid not in person_verified:
            (neg_in if labels & INDOOR else neg_out).append(iid)
            continue
        if has_dep or ignore or face_ignore:
            continue  # keep the positive groups unambiguous
        if len(faces) == 1 and (faces[0][2] - faces[0][0]) >= 0.30 and len(persons) <= 1:
            g = 'closeup'
        elif len(persons) >= 3:
            g = 'multi'
        elif 1 <= len(persons) <= 2 and min(p[3] - p[1] for p in persons) >= 0.30:
            g = 'general'
        else:
            continue
        if len(groups[g]) < want[g]:
            groups[g].append(iid)
        if all(len(groups[k]) >= want[k] for k in groups if k != 'negative') and \
                len(neg_in) + len(neg_out) > 4 * want['negative']:
            break
    n_in = min(len(neg_in), want['negative'] // 2)
    groups['negative'] = neg_in[:n_in] + neg_out[:want['negative'] - n_in]

    os.makedirs(os.path.join(a.out, 'images'), exist_ok=True)
    info = {}
    if a.rotation_csv:
        with open(a.rotation_csv) as f:
            for r in csv.DictReader(f):
                info[r['ImageID']] = r

    def fetch(iid):
        dst = os.path.join(a.out, 'images', iid + '.jpg')
        if os.path.exists(dst):
            im = Image.open(dst)
            return iid, im.size
        url = 'https://open-images-dataset.s3.amazonaws.com/validation/%s.jpg' % iid
        data = urllib.request.urlopen(url, timeout=60).read()
        im = Image.open(io.BytesIO(data)).convert('RGB')
        rot = info.get(iid, {}).get('Rotation', '')
        if rot not in ('', '0', '0.0'):   # boxes were drawn on the upright image
            im = im.rotate(float(rot), expand=True)
        s = 640.0 / max(im.size)
        if s < 1:
            im = im.resize((round(im.size[0] * s), round(im.size[1] * s)), Image.BILINEAR)
        im.save(dst, quality=90)
        return iid, im.size

    sizes = {}
    allids = [i for g in groups.values() for i in g]
    with cf.ThreadPoolExecutor(8) as ex:
        for iid, sz in ex.map(fetch, allids):
            sizes[iid] = sz

    def xywh(b):
        return [round(b[0], 5), round(b[1], 5), round(b[2] - b[0], 5), round(b[3] - b[1], 5)]

    items = []
    for g, lst in groups.items():
        for iid in lst:
            persons, ignore, faces, face_ignore = summarise(iid)
            items.append({
                'file': 'images/%s.jpg' % iid, 'source': 'openimages-v5-validation', 'id': iid,
                'group': g, 'width': sizes[iid][0], 'height': sizes[iid][1],
                'persons': [xywh(b) for b in persons],
                'faces': [xywh(b) for b in faces],
                'ignore': [xywh(b) for b in ignore + face_ignore],
                'url': 'https://open-images-dataset.s3.amazonaws.com/validation/%s.jpg' % iid,
                'author': info.get(iid, {}).get('Author', ''),
                'licence': info.get(iid, {}).get('License', ''),
                'landing': info.get(iid, {}).get('OriginalLandingURL', ''),
                'rotation': info.get(iid, {}).get('Rotation', ''),
            })
    man = {
        'description': 'PyroSight people/face test corpus. Boxes are normalised x,y,w,h in [0,1]. '
                       'persons = union of Open Images Person/Man/Woman/Boy/Girl boxes (non-depiction, '
                       'non-group, deduped at IoU 0.7); faces = Human face boxes.',
        'licence': 'Images: Open Images V5 validation, each image CC BY 2.0 by its Flickr author '
                   '(see the per-image id; author/licence lookup in validation-images-with-rotation.csv). '
                   'Annotations: CC BY 4.0 Google LLC.',
        'counts': {g: len(l) for g, l in groups.items()},
        'items': items,
    }
    with open(os.path.join(a.out, 'manifest.json'), 'w') as f:
        json.dump(man, f, indent=1)
    print(man['counts'])


if __name__ == '__main__':
    main()
