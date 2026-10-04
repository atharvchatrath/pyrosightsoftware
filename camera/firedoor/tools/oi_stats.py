"""Load Open Images v5 val+test annotations into compact python structures (cached as pickle)."""
import csv, os, pickle, collections
D = os.path.join(os.path.dirname(__file__), '..', 'data', 'oi')
CACHE = os.path.join(D, 'oi_cache.pkl')

def load():
    if os.path.exists(CACHE):
        return pickle.load(open(CACHE, 'rb'))
    cd = {}
    for l in open(os.path.join(D, 'cd.csv')):
        k, v = l.rstrip('\n').split(',', 1)
        cd[k] = v.strip('"')
    pos = collections.defaultdict(set)   # label name -> image ids (verified positive)
    neg = collections.defaultdict(set)
    split = {}
    for sp in ['validation', 'test']:
        with open(os.path.join(D, f'{sp}-annotations-human-imagelabels.csv')) as f:
            r = csv.reader(f); next(r)
            for img, src, lab, conf in r:
                (pos if conf == '1' else neg)[cd.get(lab, lab)].add(img)
                split[img] = sp
    boxes = collections.defaultdict(list)  # image id -> [(name, x0,y0,x1,y1, groupof, depiction)]
    for sp in ['validation', 'test']:
        with open(os.path.join(D, f'{sp}-annotations-bbox.csv')) as f:
            r = csv.reader(f); next(r)
            for row in r:
                img, src, lab, conf, x0, x1, y0, y1, occ, trunc, grp, dep, ins = row
                boxes[img].append((cd.get(lab, lab), float(x0), float(y0), float(x1), float(y1), int(grp), int(dep)))
                split.setdefault(img, sp)
    out = dict(pos=dict(pos), neg=dict(neg), boxes=dict(boxes), split=split)
    pickle.dump(out, open(CACHE, 'wb'))
    return out

if __name__ == '__main__':
    o = load()
    pos, boxes = o['pos'], o['boxes']
    boxlab = collections.defaultdict(set)
    for img, bl in boxes.items():
        for b in bl:
            boxlab[b[0]].add(img)
    for n in ['Fire','Flame','Bonfire','Campfire','Wildfire','Candle','Fireplace','Sunset','Sunrise','Afterglow','Lamp','Light bulb','Incandescent light bulb','Human face','Door','Room','Interior design',"Jack-o'-lantern",'Pumpkin','Traffic light','Neon','Light fixture','Orange','Street light','Television','Computer monitor','Christmas tree','Kitchen','Living room','Bedroom','Lantern','Torch','Lighting','Red','Yellow','Heat','Smoke']:
        print(f'{n:25s} label {len(pos.get(n, ())):6d}  box {len(boxlab.get(n, ())):6d}')
    nd = sum(1 for img, bl in boxes.items() for b in bl if b[0] == 'Door')
    ng = sum(1 for img, bl in boxes.items() for b in bl if b[0] == 'Door' and b[5])
    print('door boxes', nd, 'groupof', ng, 'images', len(boxlab['Door']))
    print('total images', len(o['split']))
