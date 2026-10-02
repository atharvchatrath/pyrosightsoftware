"""ctypes bridge to the device core library (core/src/*.c), used by parity tests.

The shared library is built on demand with gcc into a temp directory:
    gcc -O2 -shared -fPIC core/src/*.c -Icore/include -o libpscore.so -lm
Set PS_CORE_LIB to use a prebuilt library instead.
"""
from __future__ import annotations

import ctypes
import glob
import os
import subprocess
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORE = os.path.join(ROOT, "core")
MAX_DET = 16


class Detection(ctypes.Structure):
    _fields_ = [
        ("cls", ctypes.c_int),
        ("score", ctypes.c_float),
        ("x", ctypes.c_float), ("y", ctypes.c_float),
        ("w", ctypes.c_float), ("h", ctypes.c_float),
        ("peak_dc", ctypes.c_int16),
        ("dist_m", ctypes.c_float),
        ("dist_min_m", ctypes.c_float), ("dist_max_m", ctypes.c_float),
        ("truncated", ctypes.c_bool),
    ]


class Detections(ctypes.Structure):
    _fields_ = [
        ("d", Detection * MAX_DET),
        ("n", ctypes.c_uint8),
        ("frame_id", ctypes.c_uint32),
        ("t_ms", ctypes.c_uint32),
        ("source", ctypes.c_int),
        ("latency_us", ctypes.c_uint32),
    ]


_lib = None


def build_lib() -> str:
    env = os.environ.get("PS_CORE_LIB")
    if env:
        return env
    srcs = sorted(glob.glob(os.path.join(CORE, "src", "*.c")))
    out_dir = os.path.join(tempfile.gettempdir(), "pyrosight_ml")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "libpscore.so")
    newest = max(os.path.getmtime(p) for p in srcs + glob.glob(os.path.join(CORE, "include", "pyrosight", "*.h")))
    if not os.path.exists(out) or os.path.getmtime(out) < newest:
        subprocess.check_call(["gcc", "-O2", "-shared", "-fPIC", *srcs,
                               "-I" + os.path.join(CORE, "include"), "-o", out, "-lm"])
    return out


def lib():
    global _lib
    if _lib is None:
        L = ctypes.CDLL(build_lib())
        L.ps_thermal_model_code.argtypes = [ctypes.c_int16]
        L.ps_thermal_model_code.restype = ctypes.c_uint8
        fp = ctypes.POINTER(ctypes.c_float)
        L.ps_centernet_decode.argtypes = [fp, fp, fp, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                          ctypes.c_float, ctypes.POINTER(Detections)]
        L.ps_centernet_decode.restype = None
        L.ps_config_default.argtypes = [ctypes.c_void_p]
        L.ps_estimate_distances.argtypes = [ctypes.c_void_p, ctypes.POINTER(Detections)]
        _lib = L
    return _lib


def c_model_code(dc: int) -> int:
    return lib().ps_thermal_model_code(int(dc))


def _to_list(d: Detections):
    return [dict(cls=d.d[i].cls, score=d.d[i].score, x=d.d[i].x, y=d.d[i].y, w=d.d[i].w, h=d.d[i].h,
                 dist_m=d.d[i].dist_m, dist_min_m=d.d[i].dist_min_m, dist_max_m=d.d[i].dist_max_m,
                 truncated=bool(d.d[i].truncated)) for i in range(d.n)]


def c_centernet_decode(heat, wh, off, stride=4, thr=0.35, with_distance=False):
    """Run the device decoder on float arrays shaped [2,gh,gw]. Returns list of dicts."""
    heat = np.ascontiguousarray(heat, dtype=np.float32)
    wh = np.ascontiguousarray(wh, dtype=np.float32)
    off = np.ascontiguousarray(off, dtype=np.float32)
    gh, gw = heat.shape[-2:]
    fp = ctypes.POINTER(ctypes.c_float)
    out = Detections()
    L = lib()
    L.ps_centernet_decode(heat.ctypes.data_as(fp), wh.ctypes.data_as(fp), off.ctypes.data_as(fp),
                          gw, gh, stride, ctypes.c_float(thr), ctypes.byref(out))
    if with_distance:
        cfg = ctypes.create_string_buffer(4096)  # ps_config_t is far smaller; layout not mirrored
        L.ps_config_default(cfg)
        L.ps_estimate_distances(cfg, ctypes.byref(out))
    return _to_list(out)
