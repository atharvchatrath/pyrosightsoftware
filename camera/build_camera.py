#!/usr/bin/env python3
"""Build PyroSight Camera: one self-contained web page (no network at run time).

    python3 camera/build_camera.py                    # -> camera/dist/pyrosight_camera.html
                                                      #    camera/dist/pyrosight_camera.fragment.html
    python3 camera/build_camera.py --people           # also rebuild the person/face models first
    python3 camera/build_camera.py --no-firedoor      # fire/door stub (finds nothing, page says so)

Steps
  1. (--people) runtime/build_assets.py: COCO-SSD person model (uint8) + BlazeFace (float16)
     -> runtime/dist/people_assets.js
  2. runtime/export_oplist.py firedoor/export/firedoor.onnx -> build/firedoor/firedoor.oplist.js
     (float16 weights, NHWC outputs, base64 for inlining; global PS_OPLIST_ASSETS.firedoor)
  3. build/firedoor/meta.json: name, credits, decode thresholds (fire 0.50 from firedoor/MODEL.md; door 0.50,
     stricter than MODEL.md's 0.35, see FIREDOOR_META) and the FIRE hysteresis (on at 0.50, kept while >= 0.35)
  4. page/build_page.py --firedoor ... --out dist: inlines tf.min.js, oplist.js, people.js,
     people_assets.js, the fire/door model + firedoor/decode.js, motion.js, app.js and the
     navigation module nav/dist/ps_nav.js (built by nav/build_nav.py) into page.html
  5. checks on both outputs (the build fails if one does not hold):
     - size under 15 MB (base64 counted)
     - <title>PyroSight Camera</title>
     - no external resource in the markup: no <script src>, <link href>, <img src>, <iframe>,
       @import or url(...) outside data:, and no fetch / XMLHttpRequest / import() / http(s) /
       WebSocket / sendBeacon / importScripts in the page's own code (app.js, motion.js, oplist.js,
       people.js, engine.js, decode.js, nav/dist/ps_nav.js)
     - Worker: only app.js makes one, exactly once, from a blob: URL of a Blob built from the page's
       own <script data-ps-part> texts (no URL string, nothing fetched); no SharedWorker / service worker
     - TF.js and the models are <script type="text/plain"> (not run on the page's main thread);
       the scripts that do run there stay small (MAIN_THREAD_MAX_BYTES)
     - the fragment has no doctype/html/head/body wrapper; the standalone page has them
     http(s) strings that remain inside tf.min.js (licence/doc links, core-js URL feature
     tests, TF.js's unused HTTP model loader) are listed in dist/pyrosight_camera.build.json.
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RUNTIME = os.path.join(HERE, 'runtime')
PAGE = os.path.join(HERE, 'page')
FIREDOOR = os.path.join(HERE, 'firedoor')
BUILD = os.path.join(HERE, 'build')
MAX_BYTES = 15_000_000
TITLE = 'PyroSight Camera'

FIREDOOR_META = {
    'name': 'FireDoorNet (fire, door and window detector, trained on small public datasets)',
    'credits': ('Fire, doors and windows: FireDoorNet, trained here on FireNET (MIT) and Open Images '
                '(annotations CC BY 4.0 Google LLC, images CC BY 2.0) from ImageNet MobileNetV2 weights (Apache-2.0).'),
    # door 0.50, not the 0.35 of firedoor/MODEL.md: at 0.35 a DOOR box landed on wardrobes, windows and
    # fridges (31 % of look-alike photos) almost as often as on real doors (38 %); at 0.50 it is 7 % vs 19 %
    # (verify_quality, Open Images val/test). A green DOOR box should mean a door.
    # window 0.36, chosen on Open Images validation photos (vehicles left out): the lowest threshold with a WINDOW
    # box on at most 10 % of window look-alike photos (mirrors, TVs, pictures, fridges, ...: 9 %) and at most
    # 15 % of window-free photos (13 %); see firedoor/MODEL.md, "Window class"
    'decode': {'thresholds': {'fire': 0.50, 'door': 0.50, 'window': 0.36}, 'nmsIou': 0.45, 'maxDet': 50},
    'hysteresis': {'on': 0.50, 'keep': 0.35, 'iou': 0.1, 'maxGapMs': 5000},
    # centre zoom pass (page/app.js fireDoorAdapter): the middle half of the frame is also run at full
    # model input size; its FIRE boxes turn on at 0.60. Small flames (verify_quality small set, 30 each):
    # 2 % of the frame wide 8 -> 18, 3 % 16 -> 24; false FIRE on 871 Open Images non-fire photos 17 -> 24.
    'zoom': {'frac': 0.5, 'on': 0.60},
}

OWN_CODE = ['ps-oplist', 'ps-people', 'ps-engine', 'ps-motion', 'ps-app', 'ps-nav']
TEXT_PARTS = {'ps-tf', 'ps-people-assets', 'ps-firedoor-model'}   # type="text/plain": run in the worker
# executed on the main thread while the page loads: page code ~210 kB + the navigation module
# (camera/nav/dist/ps_nav.js, 195 kB, which must run on the page: it reads the motion sensors and
# draws the map); measured 5-6 ms to run there (20-50 ms at 4-6x CPU throttling)
MAIN_THREAD_MAX_BYTES = 450_000


def run(cmd, **kw):
    show = ['python3' if c == sys.executable else os.path.relpath(c, HERE) if c.startswith(HERE + os.sep) else c for c in cmd]
    print('+ ' + ' '.join(show), flush=True)
    subprocess.check_call(cmd, **kw)


def strip_scripts(html):
    return re.sub(r'(<script\b[^>]*>).*?(</script>)', r'\1\2', html, flags=re.S | re.I)


def script_bodies(html):
    """Each part's text; TF.js and the models are split over <script data-ps-part id=X> and
    <script data-ps-part data-ps-of=X> elements (page/build_page.py CHUNK), joined here."""
    bodies = {}
    for m in re.finditer(r'<script data-ps-part (?:id|data-ps-of)="([^"]+)"[^>]*>(.*?)</script>', html, flags=re.S):
        bodies[m.group(1)] = bodies.get(m.group(1), '') + m.group(2)
    return bodies


def check_page(path, standalone):
    html = open(path, encoding='utf-8').read()
    size = len(html.encode())
    problems = []
    if size >= MAX_BYTES:
        problems.append('size %d >= %d' % (size, MAX_BYTES))
    if not html.isascii():
        problems.append('page is not pure ASCII (non-ASCII text makes the browser decode it far slower)')
    if '<title>%s</title>' % TITLE not in html:
        problems.append('title is not "%s"' % TITLE)
    markup = strip_scripts(html)
    for pat, what in [(r'<script\b[^>]*\bsrc\s*=', '<script src>'), (r'<link\b', '<link>'),
                      (r'<img\b[^>]*\bsrc\s*=\s*["\']?(?!data:)', '<img src>'), (r'<iframe\b', '<iframe>'),
                      (r'@import', '@import'), (r'url\(\s*["\']?(?!data:)', 'url(...)'),
                      (r'https?://', 'http(s) URL in markup')]:
        if re.search(pat, markup, flags=re.I):
            problems.append('markup has ' + what)
    head_tags = {t: bool(re.search(r'<%s\b' % t, markup, flags=re.I)) for t in ('html', 'head', 'body')}
    doctype = markup.lstrip().lower().startswith('<!doctype html>')
    if standalone and not (doctype and head_tags['html']):
        problems.append('standalone page lacks doctype/html')
    if not standalone and (doctype or any(head_tags.values())):
        problems.append('fragment has a document wrapper: %s' % head_tags)
    bodies = script_bodies(html)
    own = {k: v for k, v in bodies.items() if k in OWN_CODE}
    # ps-firedoor is decode.js + the model's metadata; the model data itself is ps-firedoor-model
    own['ps-firedoor(decode/meta)'] = bodies.get('ps-firedoor', '')
    for k, v in own.items():
        for pat, what in [(r'\bfetch\s*\(', 'fetch('), (r'XMLHttpRequest', 'XMLHttpRequest'), (r'\bimport\s*\(', 'import('),
                          (r'https?://', 'http(s) URL'), (r'WebSocket', 'WebSocket'), (r'sendBeacon', 'sendBeacon'),
                          (r'importScripts', 'importScripts'), (r'SharedWorker', 'SharedWorker'), (r'serviceWorker', 'serviceWorker')]:
            if re.search(pat, v):
                problems.append('%s has %s' % (k, what))
    # the one worker: app.js, new Worker(url) with url = URL.createObjectURL(blob of the page's own scripts)
    for k, v in own.items():
        n = len(re.findall(r'\bnew\s+Worker\s*\(', v))
        if k != 'ps-app' and n:
            problems.append('%s makes a Worker' % k)
        if k == 'ps-app':
            if n != 1 or not re.search(r'\bnew\s+Worker\s*\(\s*url\s*\)', v) or not re.search(r'const url = URL\.createObjectURL\(blob\)', v) \
                    or not re.search(r"new Blob\(\[blob, text\.slice\(i, end\)\]", v):
                problems.append('ps-app: the Worker must be made once, from a blob: URL of the page\'s own script texts')
    attrs = {m.group(1): m.group(2) for m in re.finditer(r'<script data-ps-part id="([^"]+)"([^>]*)>', html)}
    for k in TEXT_PARTS:
        if k in attrs and 'type="text/plain"' not in attrs[k]:
            problems.append('%s should be type="text/plain" (not run on the main thread)' % k)
    for m in re.finditer(r'<script data-ps-part data-ps-of="([^"]+)"([^>]*)>', html):
        if m.group(1) not in TEXT_PARTS or m.group(1) not in attrs or 'type="text/plain"' not in m.group(2):
            problems.append('continuation element of %s must follow a text/plain part and be type="text/plain"' % m.group(1))
    executed = {k: len(v.encode()) for k, v in bodies.items() if k in attrs and 'type=' not in attrs[k]}
    if sum(executed.values()) > MAIN_THREAD_MAX_BYTES:
        problems.append('scripts run on the main thread are %d bytes (> %d): %s' % (sum(executed.values()), MAIN_THREAD_MAX_BYTES, executed))
    tf_urls = sorted(set(re.findall(r'https?://[^\s"\'\\)<>]*', bodies.get('ps-tf', ''))))
    tf_fetch = len(re.findall(r'\bfetch\s*\(', bodies.get('ps-tf', '')))
    return {'file': os.path.relpath(path, HERE) if path.startswith(HERE + os.sep) else path, 'bytes': size, 'mb': round(size / 1e6, 3),
            'sha256': hashlib.sha256(html.encode()).hexdigest(), 'title_ok': '<title>%s</title>' % TITLE in html,
            'parts': sorted(bodies), 'problems': problems, 'main_thread_script_bytes': executed,
            'tfjs_http_strings': tf_urls, 'tfjs_fetch_calls': tf_fetch}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(HERE, 'dist'))
    ap.add_argument('--people', action='store_true', help='rebuild runtime/dist/people_assets.js first')
    ap.add_argument('--no-firedoor', action='store_true', help='build with the fire/door stub')
    ap.add_argument('--onnx', default=os.path.join(FIREDOOR, 'export', 'firedoor.onnx'))
    ap.add_argument('--weights', default='float16', choices=['float16', 'float32', 'uint8'])
    a = ap.parse_args()
    env = dict(os.environ, OMP_NUM_THREADS=os.environ.get('OMP_NUM_THREADS', '2'))
    if a.people:
        run([sys.executable, os.path.join(RUNTIME, 'build_assets.py')], cwd=RUNTIME, env=env)
    cmd = [sys.executable, os.path.join(PAGE, 'build_page.py'), '--out', a.out, '--name', 'pyrosight_camera']
    if not a.no_firedoor:
        fd_dir = os.path.join(BUILD, 'firedoor')
        os.makedirs(fd_dir, exist_ok=True)
        prefix = os.path.join(fd_dir, 'firedoor')       # basename = key in PS_OPLIST_ASSETS
        run([sys.executable, os.path.join(RUNTIME, 'export_oplist.py'), a.onnx, '-o', prefix,
             '--weights', a.weights, '--nhwc-outputs', '--js',
             '--meta', 'source_onnx_sha256=%s' % json.dumps(hashlib.sha256(open(a.onnx, 'rb').read()).hexdigest())],
            env=env)
        meta_path = os.path.join(fd_dir, 'meta.json')
        with open(meta_path, 'w') as f:
            json.dump(FIREDOOR_META, f, indent=1)
        cmd += ['--firedoor', prefix, '--firedoor-meta', meta_path,
                '--firedoor-decode', os.path.join(FIREDOOR, 'decode.js')]
    run(cmd, env=env)

    report = {'firedoor': None if a.no_firedoor else {'onnx': os.path.relpath(a.onnx, HERE), 'weights': a.weights,
                                                      'meta': FIREDOOR_META},
              'pages': [check_page(os.path.join(a.out, 'pyrosight_camera.html'), True),
                        check_page(os.path.join(a.out, 'pyrosight_camera.fragment.html'), False)]}
    page_build = os.path.join(a.out, 'pyrosight_camera.build.json')
    if os.path.exists(page_build):
        report['page_build'] = json.load(open(page_build))
    with open(page_build, 'w') as f:
        json.dump(report, f, indent=1)
    bad = False
    for p in report['pages']:
        print('%s: %.2f MB, title ok: %s, parts: %s' % (p['file'], p['mb'], p['title_ok'], ', '.join(p['parts'])))
        for pr in p['problems']:
            print('  PROBLEM: ' + pr)
            bad = True
    t = report['pages'][0]
    print('http(s) strings left inside tf.min.js (library code, never requested): %d; fetch( calls in tf.min.js: %d'
          % (len(t['tfjs_http_strings']), t['tfjs_fetch_calls']))
    if bad:
        sys.exit('build checks failed')
    print('ok')


if __name__ == '__main__':
    main()
