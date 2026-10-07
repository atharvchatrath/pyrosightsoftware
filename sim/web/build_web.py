#!/usr/bin/env python3
"""Build the browser version of the simulator as one self-contained HTML file.

    python3 sim/web/build_web.py [--out sim/web/dist] [--camera FILE | --no-camera]

Needs: zig as a C-to-WebAssembly compiler (`python3 -m pip install ziglang`)
and binaryen's wasm2js (`npm install binaryen`; set WASM2JS to its path if it
is not on PATH). The page embeds three builds of the same C code and uses the
first one the browser accepts:
  1. WebAssembly with SIMD (fastest),
  2. plain WebAssembly,
  3. plain JavaScript (wasm2js), for browsers or pages where WebAssembly is
     blocked. Slower: the AI model takes ~100 ms per frame instead of ~35 ms.
plus the trained model in the .psnn format (ml/export_nnref.py).

"Try it on your own camera" (on by default): the page also carries the
PyroSight Camera page, camera/dist/pyrosight_camera.html (build that first with
`python3 camera/build_camera.py`; --camera picks another copy, --no-camera
leaves the section out). It is stored as inert text at the end of the page, in
HTML comments inside <div id="ps-cam-store" hidden> (pieces of at most 128 k
characters, so the HTML parser can pause between them), with every "%" written
as "%0" and every "-" as "%1" (so no "--" can end a comment) and a "." in front
of each piece. Comments parse about 40 % faster than the same text in
<script type="text/plain"> elements (measured in Chromium: 90 vs 146 ms for
the 8.5 MB, 396 vs 669 ms at 4x CPU throttling). On "Start camera" the page
reverses the escaping and shows the camera page in a frame (srcdoc); nothing
of it is parsed as markup or run before that tap. The section's wording follows the camera page's classes
(doors, or doors and windows once its fire/door model has a window class).

"Save this page" (claude.ai downloads capability) rebuilds the standalone page
from #ps-sim-shell, a JSON list of the page's text with the big blocks given as
element ids; this script checks that the rebuild is byte-identical.

The output is pure ASCII (\\uXXXX escapes in scripts, &#x...; in the markup):
Chromium decodes an all-ASCII page on a faster path, which matters at 11 MB.

Outputs:
  pyrosight_sim.html            open directly in a browser (works offline)
  pyrosight_sim.fragment.html   the same page without the html/head/body
                                wrapper, for hosts that add their own
"""
import argparse
import base64
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WEB = os.path.join(ROOT, "sim", "web")
CAMERA_PAGE = os.path.join(ROOT, "camera", "dist", "pyrosight_camera.html")
HEAD = ('<!doctype html>\n<html lang="en">\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n')
TAIL = "\n</html>\n"
CHUNK = 128 * 1024          # characters per inert text element
MAX_BYTES = 15 * 1000 * 1000
BIG_PARTS = ["wasm-simd", "wasm-base", "model-psnn", "ps-js-fallback", "ps-sim-shell"]


def sources():
    core = sorted(os.path.join(ROOT, "core", "src", f) for f in os.listdir(os.path.join(ROOT, "core", "src")) if f.endswith(".c"))
    sim = [os.path.join(ROOT, "sim", f) for f in ("sim_world.c", "scene.c", "nn_ref.c")]
    return core + sim + [os.path.join(WEB, "wasm_api.c")]


def zig_cc(out, extra):
    cmd = [sys.executable, "-m", "ziglang", "cc", "-target", "wasm32-wasi", "-O2", "-mexec-model=reactor",
           "-I" + os.path.join(ROOT, "core", "include"), "-I" + os.path.join(ROOT, "sim"),
           "-fvisibility=hidden", "-Wl,--export-dynamic", "-Wl,--strip-all", *extra, *sources(), "-o", out, "-lm"]
    subprocess.run(cmd, check=True)


def find_wasm2js():
    cand = os.environ.get("WASM2JS") or shutil.which("wasm2js")
    if cand:
        return cand
    for base in (ROOT, WEB):
        p = os.path.join(base, "node_modules", "binaryen", "bin", "wasm2js")
        if os.path.exists(p):
            return p
    sys.exit("wasm2js not found: `npm install binaryen` and set WASM2JS=/path/to/wasm2js")


# ---------------------------------------------------------------- ASCII output
def ascii_js(js):
    def esc(m):
        o = ord(m.group(0))
        if o > 0xFFFF:
            o -= 0x10000
            return "\\u%04x\\u%04x" % (0xD800 + (o >> 10), 0xDC00 + (o & 0x3FF))
        return "\\u%04x" % o
    # every non-ASCII character in page.html's scripts is inside a string literal or a comment,
    # where the escape means the same thing
    return re.sub(r"[^\x00-\x7f]", esc, js)


def ascii_template(html):
    """page.html as pure ASCII, showing and doing exactly the same."""
    out, pos = [], 0

    def markup(s):
        return re.sub(r"[^\x00-\x7f]", lambda c: "&#x%x;" % ord(c.group(0)), s)
    for m in re.finditer(r"(<script\b[^>]*>)(.*?)(</script>)|<style\b[^>]*>.*?</style>", html, flags=re.S):
        out.append(markup(html[pos:m.start()]))
        if m.group(1):
            out.append(m.group(1) + ascii_js(m.group(2)) + m.group(3))
        elif not m.group(0).isascii():
            sys.exit("page.html: non-ASCII character inside <style> (write it as a CSS escape)")
        else:
            out.append(m.group(0))
        pos = m.end()
    out.append(markup(html[pos:]))
    return "".join(out)


# ---------------------------------------------------------------- the camera page as inert text
def escape_embedded(text):
    """Text that can sit inside an HTML comment: no "-" at all ("%" -> "%0", "-" -> "%1")."""
    return text.replace("%", "%0").replace("-", "%1")


def unescape_embedded(text):
    return re.sub(r"%([01])", lambda m: "%" if m.group(1) == "0" else "-", text)


def camera_blocks(page):
    if "\r" in page or "\0" in page:
        sys.exit("camera page has CR or NUL characters, which the HTML parser would change")
    if not page.isascii():
        sys.exit("camera page is not pure ASCII (camera/page/build_page.py makes it so)")
    esc = escape_embedded(page)
    if unescape_embedded(esc) != page or "-" in esc:
        sys.exit("camera page: escaping failed")
    pieces = [esc[i:i + CHUNK] for i in range(0, len(esc), CHUNK)]
    # "." first: a comment may not start with ">"; with no "-" inside, nothing can end it early
    html = ('<div id="ps-cam-store" hidden data-len="%d">' % len(page) +
            "\n".join("<!--.%s-->" % p for p in pieces) + "</div>")
    return html, len(pieces)


def camera_classes(page):
    m = re.search(r"CLASS_NAMES\s*=\s*\[([^\]]*)\]", page)
    return re.findall(r"['\"](\w+)['\"]", m.group(1)) if m else []


# ---------------------------------------------------------------- save-this-page shell
STORE = re.compile(r'(<div id="ps-cam-store"[^>]*>)((?:<!--[^-]*-->\n?)*)(</div>)')


def make_shell(full, part_ids):
    """The standalone page as a list of strings, {"ref": id} for the big script blocks (the
    element's text) and {"html": "ps-cam-store"} for the stored camera page (its innerHTML)."""
    ids = "|".join(re.escape(i) for i in part_ids)
    pat = re.compile(r'(<script\b[^>]*\bid="(%s)"[^>]*>)(.*?)(</script>)' % ids, flags=re.S)
    cuts = [(m.end(1), m.start(4), {"ref": m.group(2)}) for m in pat.finditer(full)]
    cuts += [(m.end(1), m.start(3), {"html": "ps-cam-store"}) for m in STORE.finditer(full)]
    shell, pos = [], 0
    for a, b, ref in sorted(cuts, key=lambda c: c[0]):
        shell += [full[pos:a], ref]
        pos = b
    shell.append(full[pos:])
    found = [s.get("ref") or s.get("html") for s in shell if isinstance(s, dict)]
    if sorted(found) != sorted(part_ids):
        sys.exit("shell: expected parts %s, found %s" % (sorted(part_ids), sorted(found)))
    return shell


def rebuild_from_shell(full, shell):
    """What the page's pageParts() produces: the shell with each ref filled in."""
    texts = {m.group(1): m.group(2) for m in re.finditer(r'<script\b[^>]*\bid="([^"]+)"[^>]*>(.*?)</script>', full, flags=re.S)}
    for m in STORE.finditer(full):
        texts["ps-cam-store"] = m.group(2)
    return "".join(s if isinstance(s, str) else texts[s.get("ref") or s.get("html")] for s in shell)


def check_page(name, html, fragment):
    problems = []
    if len(html.encode()) >= MAX_BYTES:
        problems.append("%d bytes, not under %d" % (len(html.encode()), MAX_BYTES))
    if not html.isascii():
        problems.append("not pure ASCII")
    if "<title>PyroSight Simulator</title>" not in html:
        problems.append("title missing")
    head = html[:200].lower()
    if fragment and any(t in head for t in ("<!doctype", "<html", "<head", "<body")):
        problems.append("fragment has a document wrapper")
    if not fragment and not head.startswith("<!doctype html>"):
        problems.append("standalone page has no doctype")
    if problems:
        sys.exit("%s: %s" % (name, "; ".join(problems)))


def check_template(tpl):
    """No network in the simulator's own markup and code (the camera page's build checks its own)."""
    bad = re.findall(r"<(?:link|iframe|img|audio|video|source|embed|object)\b|\bsrc\s*=|\bhref\s*=|@import|url\(|"
                     r"\bfetch\s*\(|XMLHttpRequest|\bimport\s*\(|WebSocket|sendBeacon|importScripts|https?://", tpl)
    if bad:
        sys.exit("page.html refers to something outside the page: %s" % sorted(set(bad)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(WEB, "dist"))
    ap.add_argument("--model", default=os.path.join(ROOT, "ml", "runs", "synthetic_v0", "model.psnn"))
    ap.add_argument("--camera", default=CAMERA_PAGE, help="the PyroSight Camera standalone page to embed")
    ap.add_argument("--no-camera", action="store_true", help="leave out the 'Try it on your own camera' section")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    camera = None
    if not args.no_camera:
        if not os.path.exists(args.camera):
            sys.exit("camera page not found: %s (python3 camera/build_camera.py, or pass --no-camera)" % args.camera)
        with open(args.camera, encoding="utf-8", newline="") as f:
            camera = f.read()
        if not camera.lower().startswith("<!doctype html"):
            sys.exit("%s is not a standalone page (use pyrosight_camera.html, not the fragment)" % args.camera)

    with tempfile.TemporaryDirectory() as tmp:
        simd, base, mvp = (os.path.join(tmp, n) for n in ("simd.wasm", "base.wasm", "mvp.wasm"))
        zig_cc(simd, ["-mcpu=generic+simd128"])
        zig_cc(base, ["-mcpu=generic"])
        # wasm2js handles only the MVP feature set; double-precision activations
        # avoid Math.fround on every multiply-add in the JavaScript engine.
        zig_cc(mvp, ["-mcpu=mvp", "-DNN_REF_F64"])
        js_path = os.path.join(tmp, "pyrosight.mjs")
        subprocess.run([find_wasm2js(), mvp, "-O2", "-o", js_path], check=True)
        js = open(js_path).read()
        blobs = {k: base64.b64encode(open(p, "rb").read()).decode() for k, p in (("SIMD", simd), ("BASE", base))}

    js = re.sub(r"^export var .*$\n?", "", js, flags=re.M)
    if "import " in js.split("function asmFunc")[0]:
        sys.exit("unexpected imports in wasm2js output")
    model = base64.b64encode(open(args.model, "rb").read()).decode()

    tpl = open(os.path.join(WEB, "page.html"), encoding="utf-8").read()
    check_template(tpl)
    parts = list(BIG_PARTS)
    shell_tag = '<script type="application/json" id="ps-sim-shell">@@SHELL@@</script>'
    if camera is None:
        tpl = re.sub(r"<!--@@CAMERA_BEGIN@@-->.*?<!--@@CAMERA_END@@-->\n?", "", tpl, flags=re.S)
        tpl = tpl.rstrip("\n") + "\n" + shell_tag + "\n"
        n_pieces, classes, blocks = 0, [], ""
    else:
        classes = camera_classes(camera)
        window = "window" in classes
        tpl = (tpl.replace("<!--@@CAMERA_BEGIN@@-->", "").replace("<!--@@CAMERA_END@@-->", "")
                  .replace("@@TARGETS@@", "a person, a fire, a door or a window" if window else "a person, a fire or a door")
                  .replace("@@WAYOUT_CHIP@@", "doors and windows" if window else "doors")
                  .replace("@@CAMERA_PAGE@@", shell_tag + "\n@@CAMERA_BLOCKS@@"))
        blocks, n_pieces = camera_blocks(camera)
        parts.append("ps-cam-store")
    left = set(re.findall(r"@@[A-Z_]+@@", tpl)) - {"@@WASM_SIMD@@", "@@WASM_BASE@@", "@@MODEL@@", "@@JS_FALLBACK@@", "@@SHELL@@", "@@CAMERA_BLOCKS@@"}
    if left:
        sys.exit("page.html: unfilled placeholders %s" % sorted(left))
    page = ascii_template(tpl)
    page = (page.replace("@@WASM_SIMD@@", blobs["SIMD"]).replace("@@WASM_BASE@@", blobs["BASE"])
                .replace("@@MODEL@@", model).replace("@@JS_FALLBACK@@", js).replace("@@CAMERA_BLOCKS@@", blocks))
    page = page.rstrip("\n") + "\n"

    full = HEAD + page.rstrip("\n") + TAIL
    shell = make_shell(full, parts)
    shell_json = json.dumps(shell, ensure_ascii=True, separators=(",", ":")).replace("<", "\\u003c").replace(">", "\\u003e")
    full = full.replace("@@SHELL@@", shell_json, 1)
    page = page.replace("@@SHELL@@", shell_json, 1)
    if rebuild_from_shell(full, shell) != full:
        sys.exit("the page rebuilt from #ps-sim-shell differs from the page")
    if camera is not None:
        store = STORE.search(full)
        texts = re.findall(r"<!--\.([^-]*)-->", store.group(2)) if store else []
        if unescape_embedded("".join(texts)) != camera:
            sys.exit("the camera page read back from the text blocks differs from %s" % args.camera)
    check_page("pyrosight_sim.html", full, False)
    check_page("pyrosight_sim.fragment.html", page, True)

    frag = os.path.join(args.out, "pyrosight_sim.fragment.html")
    with open(frag, "w", encoding="ascii", newline="\n") as f:
        f.write(page)
    full_path = os.path.join(args.out, "pyrosight_sim.html")
    with open(full_path, "w", encoding="ascii", newline="\n") as f:
        f.write(full)
    for p in (full_path, frag):
        print("wrote %s (%.2f MB)" % (os.path.relpath(p, ROOT), os.path.getsize(p) / 1e6))
    if camera is None:
        print("camera section: left out (--no-camera)")
    else:
        print("camera section: %s (%.2f MB, sha256 %s...), %d text pieces, classes %s -> way out: %s" % (
            os.path.relpath(args.camera, ROOT), len(camera) / 1e6, hashlib.sha256(camera.encode()).hexdigest()[:16],
            n_pieces, classes or "(CLASS_NAMES not found)", "doors and windows" if "window" in classes else "doors"))
        print("simulator part %.2f MB, save shell %.1f kB" % ((len(full) - len(camera)) / 1e6, len(shell_json) / 1e3))


if __name__ == "__main__":
    main()
