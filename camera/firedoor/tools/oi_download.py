"""Download the selected Open Images JPEGs (resized to max side 640) into data/oi/img/.

python3 tools/oi_download.py                                  # data/oi/selection.json
python3 tools/oi_download.py data/oi/selection_window.json    # the window additions (its "new" list)
python3 tools/oi_download.py data/oi/selection_window_big.json  # the big-window training photos (its "new" list)
"""
import io, json, os, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'oi')
OUT = os.path.join(D, 'img'); os.makedirs(OUT, exist_ok=True)
sel = json.load(open(sys.argv[1] if len(sys.argv) > 1 else os.path.join(D, 'selection.json')))
if isinstance(sel, dict):
    sel = sel['new']

def get(v):
    p = os.path.join(OUT, v['id'] + '.jpg')
    if os.path.exists(p):
        return 'skip'
    for attempt in range(3):
        try:
            data = urllib.request.urlopen(v['url'], timeout=30).read()
            im = Image.open(io.BytesIO(data)).convert('RGB')
            s = 640 / max(im.size)
            if s < 1:
                im = im.resize((round(im.width * s), round(im.height * s)), Image.BILINEAR)
            im.save(p + '.tmp', 'JPEG', quality=90)
            os.replace(p + '.tmp', p)
            return 'ok'
        except Exception as e:
            err = repr(e)
            time.sleep(1)
    return 'fail ' + err

t = time.time(); n = {'ok': 0, 'skip': 0, 'fail': 0}
with ThreadPoolExecutor(48) as ex:
    for i, r in enumerate(ex.map(get, sel)):
        n[r.split()[0]] += 1
        if r.startswith('fail'):
            print(r, flush=True)
        if i % 1000 == 0:
            print(i, n, f'{time.time() - t:.0f}s', flush=True)
print('done', n, f'{time.time() - t:.0f}s', flush=True)
