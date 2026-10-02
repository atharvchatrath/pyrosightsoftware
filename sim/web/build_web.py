#!/usr/bin/env python3
"""Build the browser version of the simulator as one self-contained HTML file.

    python3 sim/web/build_web.py [--out sim/web/dist]

Needs: zig as a C-to-WebAssembly compiler (`python3 -m pip install ziglang`)
and binaryen's wasm2js (`npm install binaryen`; set WASM2JS to its path if it
is not on PATH). The page embeds three builds of the same C code and uses the
first one the browser accepts:
  1. WebAssembly with SIMD (fastest),
  2. plain WebAssembly,
  3. plain JavaScript (wasm2js), for browsers or pages where WebAssembly is
     blocked. Slower: the AI model takes ~100 ms per frame instead of ~35 ms.
plus the trained model in the .psnn format (ml/export_nnref.py).

Outputs:
  pyrosight_sim.html            open directly in a browser (works offline)
  pyrosight_sim.fragment.html   the same page without the html/head/body
                                wrapper, for hosts that add their own
"""
import argparse
import base64
import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WEB = os.path.join(ROOT, "sim", "web")


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(WEB, "dist"))
    ap.add_argument("--model", default=os.path.join(ROOT, "ml", "runs", "synthetic_v0", "model.psnn"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

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

    page = open(os.path.join(WEB, "page.html")).read()
    page = (page.replace("@@WASM_SIMD@@", blobs["SIMD"]).replace("@@WASM_BASE@@", blobs["BASE"])
                .replace("@@MODEL@@", model).replace("@@JS_FALLBACK@@", js))
    frag = os.path.join(args.out, "pyrosight_sim.fragment.html")
    with open(frag, "w") as f:
        f.write(page)
    full = os.path.join(args.out, "pyrosight_sim.html")
    with open(full, "w") as f:
        f.write('<!doctype html>\n<html lang="en">\n<meta charset="utf-8">\n'
                '<meta name="viewport" content="width=device-width, initial-scale=1">\n')
        f.write(page)
        f.write("\n</html>\n")
    for p in (full, frag):
        print("wrote %s (%.1f MB)" % (os.path.relpath(p, ROOT), os.path.getsize(p) / 1e6))


if __name__ == "__main__":
    main()
