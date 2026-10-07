"""Stream the Open Images V6 *train* annotation CSVs once and keep only the rows the window class needs.

    python3 tools/oi_train_stream.py      -> data/oi/train-bbox-window.csv, data/oi/train-il-window.csv

The full files are 2.3 GB (boxes) and 2.5 GB (human-verified image labels); they are read from
storage.googleapis.com as a stream and never stored. Kept:
  boxes:  Window, Door, the window look-alikes used as hard negatives (Mirror, Picture frame, TV,
          monitor, fridge, wardrobe/cupboard/closet, bookcase, shelf, whiteboard, poster, billboard,
          laptop, tablet), window context (Curtain, Window blind) and the fire-like classes that
          oi_select.py excludes (Candle, Fireplace, Lantern, Torch, Wood-burning stove)
  labels: the same classes plus scene labels (Room, Interior design, Building, House, ...), positive
          and negative verifications.
"""
import csv
import io
import os
import sys
import time
import urllib.request

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'oi')
BBOX_URL = 'https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv'
IL_URL = 'https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-human-imagelabels.csv'

BOX_NAMES = ['Window', 'Door', 'Mirror', 'Picture frame', 'Television', 'Computer monitor', 'Refrigerator',
             'Cupboard', 'Wardrobe', 'Closet', 'Chest of drawers', 'Bookcase', 'Shelf', 'Whiteboard', 'Poster',
             'Billboard', 'Laptop', 'Tablet computer', 'Curtain', 'Window blind', 'Cabinetry',
             'Candle', 'Fireplace', 'Lantern', 'Torch', 'Wood-burning stove']
IL_NAMES = BOX_NAMES + ['Room', 'Interior design', 'Living room', 'Bedroom', 'Kitchen', 'Bathroom', 'Ceiling',
                        'Floor', 'Apartment', 'Home', 'House', 'Building', 'Facade', 'Skyscraper', 'Tower',
                        'Office building', 'Daylighting', 'Sash window', 'Window covering', 'Window screen',
                        'Display window', 'Glass', 'Painting', 'Car', 'Vehicle', 'Land vehicle', 'Bus', 'Train']


def mids(names):
    cd = {}
    for l in open(os.path.join(D, 'cd.csv')):
        k, v = l.rstrip('\n').split(',', 1)
        cd[v.strip('"')] = k
    missing = [n for n in names if n not in cd]
    if missing:
        sys.exit('no MID for %s' % missing)
    return {cd[n] for n in names}


def stream(url, keep, out, label_col=2):
    t = time.time()
    n = k = 0
    tmp = out + '.tmp'
    with urllib.request.urlopen(url, timeout=60) as r, open(tmp, 'w', newline='') as f:
        rd = csv.reader(io.TextIOWrapper(r, encoding='utf-8', newline=''))
        header = next(rd)
        w = csv.writer(f)
        w.writerow(header)
        for row in rd:
            n += 1
            if row[label_col] in keep:
                w.writerow(row)
                k += 1
            if n % 5_000_000 == 0:
                print('%s: %d rows read, %d kept, %.0f s' % (os.path.basename(out), n, k, time.time() - t), flush=True)
    os.replace(tmp, out)
    print('%s: done, %d rows read, %d kept, %.0f s' % (os.path.basename(out), n, k, time.time() - t), flush=True)


if __name__ == '__main__':
    stream(BBOX_URL, mids(BOX_NAMES), os.path.join(D, 'train-bbox-window.csv'))
    stream(IL_URL, mids(IL_NAMES), os.path.join(D, 'train-il-window.csv'))
