#!/usr/bin/env python3
"""Build PyroSight Camera as one self-contained HTML page (no network at run time).

    python3 camera/page/build_page.py                       # fire/door = clearly marked stub
    python3 camera/page/build_page.py --firedoor PREFIX     # wire in the fire/door model

--firedoor PREFIX is the output prefix of
    python3 camera/runtime/export_oplist.py firedoor.onnx -o PREFIX --weights float16 --js
i.e. PREFIX.oplist.js must exist (it sets PS_OPLIST_ASSETS.firedoor). The box
decoder camera/firedoor/decode.js (global FireDoorDecode) is inlined with it
(--firedoor-decode to use another copy). --firedoor-name / --firedoor-credits
set what the page shows; --firedoor-meta FILE.json may give
{"name", "credits", "decode": {...FireDoorDecode options, e.g. thresholds}}.

Like sim/web/build_web.py: page.html is a template whose @@NAME@@ placeholders
are replaced by the inlined scripts. Outputs (in --out, default page/dist):
  pyrosight_camera.html            open directly in a browser (works offline, file://)
  pyrosight_camera.fragment.html   the same without doctype/html/head wrapper, for
                                   hosts that add their own (claude.ai artifacts)
The page can save a copy of itself ("Save this page and open it in your browser",
claude.ai downloads capability). It rebuilds that copy from #ps-shell (the
non-script part of the page, stored as JSON) plus the text of every
<script data-ps-part> element; this script checks that the rebuild is
byte-identical to pyrosight_camera.html.
"""
import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CAMERA = os.path.dirname(HERE)
ROOT = os.path.dirname(CAMERA)
TFJS = os.path.join(CAMERA, 'node_modules/@tensorflow/tfjs/dist/tf.min.js')
RUNTIME = os.path.join(CAMERA, 'runtime')
NAV_JS = os.path.join(CAMERA, 'nav', 'dist', 'ps_nav.js')   # python3 camera/nav/build_nav.py
HEAD = ('<!doctype html>\n<html lang="en">\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n')
TAIL = '</html>\n'
MAX_BYTES = 15_000_000

# Order matters: each executed script may use the globals of the ones before it. app.js comes as
# early as it can, so the buttons work while the rest of the (large) page is still being read.
# ps_nav.js (the navigation module, 195 kB, mostly the device's C code as JavaScript) comes right
# after it; app.js picks PSNav up once it exists (it is never needed before a tap).
# TF.js and the models are type="text/plain": the browser does not run them on the page's main
# thread; app.js hands their text to a Web Worker (or, as a fallback, runs them itself later,
# between taps). page/app.js ENGINE_PARTS lists the order the worker runs them in.
SCRIPTS = [  # (id, placeholder, extra attributes)
    ('ps-shell', 'SHELL', ' type="application/json"'),
    ('ps-oplist', 'OPLIST', ''),
    ('ps-people', 'PEOPLE', ''),
    ('ps-firedoor', 'FIREDOOR', ''),
    ('ps-engine', 'ENGINE', ''),
    ('ps-motion', 'MOTION', ''),
    ('ps-app', 'APP', ''),
    ('ps-nav', 'NAV', ''),             # camera/nav/dist/ps_nav.js (PSNav: the device's navigation code); after app.js
    ('ps-tf', 'TFJS', ' type="text/plain"'),
    ('ps-people-assets', 'PEOPLE_ASSETS', ' type="text/plain"'),
    ('ps-firedoor-model', 'FIREDOOR_MODEL', ' type="text/plain"'),   # only when the model is included
]

STUB_FIREDOOR = ('/* Fire/door model: NOT INCLUDED in this build. app.js uses its built-in stub, which finds\n'
                 ' * nothing, and says so on the page. Build with --firedoor PREFIX to include the model. */\n'
                 'globalThis.PS_FIREDOOR_BUILD = {included: false};\n')


def read(p):
    with open(p, encoding='utf-8') as f:
        return f.read().replace('\r\n', '\n')


def safe_inline(name, js):
    """Text placed inside <script>...</script> must not end the element early."""
    low = js.lower()
    if '</script' in low:
        js = re.sub(r'</(script)', r'<\\/\1', js, flags=re.I)
    if '<!--' in js:
        sys.exit('%s contains "<!--", which changes how browsers parse a script element' % name)
    return js


# The page is written as pure ASCII. Chromium decodes an all-ASCII chunk of the page on a fast
# 8-bit path; one non-ASCII character makes it decode the whole chunk to 16-bit, which kept the
# main thread busy for 250+ ms while the 8 MB page loaded at 4x CPU throttling (about 130 ms
# all-ASCII). Scripts get \uXXXX escapes (every non-ASCII character in app.js and tf.min.js is
# inside a string literal, where the escape means the same string), the markup gets &#x...;
# references and CSS gets \HHHHHH escapes.
def ascii_js(name, js):
    def esc(m):
        i = m.start()
        k = i - 1
        while k >= 0 and js[k] == '\\':
            k -= 1
        if (i - 1 - k) % 2 == 1:
            sys.exit('%s: a non-ASCII character follows a backslash at %d; escape it in the source' % (name, i))
        o = ord(m.group(0))
        if o > 0xFFFF:
            o -= 0x10000
            return '\\u%04x\\u%04x' % (0xD800 + (o >> 10), 0xDC00 + (o & 0x3FF))
        return '\\u%04x' % o
    return js if js.isascii() else re.sub(r'[^\x00-\x7f]', esc, js)


def ascii_markup(html):
    def css(m):
        return re.sub(r'[^\x00-\x7f]', lambda c: '\\%06x' % ord(c.group(0)), m.group(0))
    html = re.sub(r'<style\b.*?</style>', css, html, flags=re.S | re.I)
    return re.sub(r'[^\x00-\x7f]', lambda c: '&#x%x;' % ord(c.group(0)), html)


# The text/plain parts are split into elements of at most CHUNK characters: the browser's HTML
# parser reads one element's text in one go, so a 5 MB block kept it busy (no frame, no tap
# answered) for 150+ ms at a time on a slow CPU; between elements it can yield. The first element
# has the part's id, the others data-ps-of="<id>"; their texts joined give the part.
CHUNK = 1 << 17


def split_text(body, size=CHUNK):
    out, i = [], 0
    while i < len(body):
        end = min(len(body), i + size)
        while end < len(body) and body[end - 1] == '\r':   # keep a CR LF pair in one element
            end += 1
        out.append(body[i:end])
        i = end
    return out or ['']


def script_tag(sid, attrs, body):
    return '<script data-ps-part id="%s"%s>%s</script>' % (sid, attrs, body)


def part_tags(sid, attrs, body):
    if 'type="text/plain"' not in attrs:
        return [script_tag(sid, attrs, body)]
    chunks = split_text(body)
    return [script_tag(sid, attrs, chunks[0])] + \
        ['<script data-ps-part data-ps-of="%s"%s>%s</script>' % (sid, attrs, c) for c in chunks[1:]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(HERE, 'dist'))
    ap.add_argument('--firedoor', help='PREFIX of export_oplist.py --js output (PREFIX.oplist.js)')
    ap.add_argument('--firedoor-decode', default=os.path.join(CAMERA, 'firedoor', 'decode.js'))
    ap.add_argument('--firedoor-meta', help='JSON file {name, credits, decode}')
    ap.add_argument('--firedoor-name', default=None)
    ap.add_argument('--firedoor-credits', default=None)
    ap.add_argument('--name', default='pyrosight_camera')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    template = read(os.path.join(HERE, 'page.html'))
    if '<script' in template:
        sys.exit('page.html must not contain <script> elements (the build appends them)')
    shell = ascii_markup(template.rstrip('\n') + '\n\n')

    if a.firedoor:
        js_path = a.firedoor + '.oplist.js' if not a.firedoor.endswith('.js') else a.firedoor
        meta = {'name': 'FireDoorNet (fire and door detector)',
                'credits': 'Fire and doors: FireDoorNet, trained on FireNET (MIT) and Open Images '
                           '(annotations CC BY 4.0 Google LLC, images CC BY 2.0) from an ImageNet MobileNetV2 '
                           '(Apache-2.0).'}
        if a.firedoor_meta:
            meta.update(json.load(open(a.firedoor_meta)))
        if a.firedoor_name:
            meta['name'] = a.firedoor_name
        if a.firedoor_credits:
            meta['credits'] = a.firedoor_credits
        firedoor_model = read(js_path).rstrip('\n') + '\n'
        firedoor = (read(a.firedoor_decode).rstrip('\n') + '\n' +
                    'globalThis.PS_FIREDOOR_META = %s;\nglobalThis.PS_FIREDOOR_BUILD = {included: true};\n'
                    % json.dumps(meta))
        if 'PS_OPLIST_ASSETS' not in firedoor_model or 'FireDoorDecode' not in firedoor:
            sys.exit('fire/door files do not define PS_OPLIST_ASSETS / FireDoorDecode')
    else:
        firedoor_model = None
        firedoor = STUB_FIREDOOR

    shell_json = json.dumps({'head': HEAD, 'shell': shell, 'tail': TAIL}, ensure_ascii=True)
    shell_json = shell_json.replace('<', '\\u003c').replace('>', '\\u003e')
    parts = {
        'SHELL': shell_json,
        # tf.min.js bundles regenerator-runtime, whose strict-mode assignment to an undeclared
        # global falls back to Function(...) (eval). Declaring the global first keeps TF.js
        # working under a Content-Security-Policy without 'unsafe-eval' (tests/browser_test.js csp).
        'TFJS': 'var regeneratorRuntime;\n' + read(TFJS),
        'OPLIST': read(os.path.join(RUNTIME, 'oplist.js')),
        'PEOPLE': read(os.path.join(RUNTIME, 'people.js')),
        'PEOPLE_ASSETS': read(os.path.join(RUNTIME, 'dist', 'people_assets.js')),
        'FIREDOOR': firedoor,
        'ENGINE': read(os.path.join(RUNTIME, 'engine.js')),
        'MOTION': read(os.path.join(HERE, 'motion.js')),
        'APP': read(os.path.join(HERE, 'app.js')),
        'NAV': read(NAV_JS),
    }
    if 'PSNav.Navigator' not in parts['NAV']:
        sys.exit('%s does not define PSNav.Navigator (python3 camera/nav/build_nav.py)' % NAV_JS)
    if firedoor_model is not None:
        parts['FIREDOOR_MODEL'] = firedoor_model
    tags = []
    for sid, key, attrs in SCRIPTS:
        if key not in parts:
            continue
        tags.extend(part_tags(sid, attrs, safe_inline(key, ascii_js(key, parts[key]))))
    fragment = shell + '\n'.join(tags) + '\n'
    full = HEAD + fragment + TAIL

    # what the page's own "save" rebuild produces (app.js buildStandalone)
    rebuilt = HEAD + json.loads(shell_json)['shell'] + '\n'.join(tags) + '\n' + TAIL
    if rebuilt != full:
        sys.exit('self-copy rebuild would differ from the built page')
    if not full.isascii():
        sys.exit('the page is not pure ASCII')
    if len(full.encode()) > MAX_BYTES:
        sys.exit('page is %.2f MB, over the %.0f MB limit' % (len(full.encode()) / 1e6, MAX_BYTES / 1e6))
    for p in ('http://', 'https://'):
        for k in ('APP', 'MOTION', 'ENGINE', 'NAV'):
            if p in parts[k]:
                sys.exit('%s mentions %s: the page must not load anything from the network' % (k, p))

    frag_path = os.path.join(a.out, a.name + '.fragment.html')
    full_path = os.path.join(a.out, a.name + '.html')
    with open(frag_path, 'w', encoding='utf-8') as f:
        f.write(fragment)
    with open(full_path, 'w', encoding='utf-8') as f:
        f.write(full)
    sizes = {k: len(v.encode()) for k, v in parts.items()}
    info = {'files': {os.path.basename(full_path): len(full.encode()), os.path.basename(frag_path): len(fragment.encode())},
            'parts': sizes, 'firedoor_included': bool(a.firedoor), 'firedoor_source': a.firedoor}
    with open(os.path.join(a.out, a.name + '.build.json'), 'w') as f:
        json.dump(info, f, indent=1)
    for p in (full_path, frag_path):
        print('wrote %s (%.2f MB)' % (os.path.relpath(p, ROOT), os.path.getsize(p) / 1e6))
    print('parts: ' + ', '.join('%s %.2f MB' % (k, v / 1e6) for k, v in sizes.items()))
    print('fire/door: ' + ('included from ' + a.firedoor if a.firedoor else 'STUB (not included)'))


if __name__ == '__main__':
    main()
