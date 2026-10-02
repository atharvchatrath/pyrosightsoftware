"""Device-side thermal preprocessing, mirrored bit-exactly from core/src/ps_thermal.c.

Contract (see core/include/pyrosight/ps_thermal.h):
  - Temperatures are int16 deci-degrees Celsius (365 == 36.5 C).
  - The device feeds the *denoised* frame (temporal IIR + 3x3 median) to
    ps_thermal_to_model_input(), which applies ps_thermal_model_code() per pixel.
  - The network input is the uint8 code divided by 255 (float32, 1x1x120x160).
"""
from __future__ import annotations

import numpy as np

W, H = 160, 120
NUM_PIXELS = W * H


def model_code(dc) -> np.ndarray:
    """Bit-exact port of ps_thermal_model_code() (uint8 output).

        v <  -200          -> 0
        -200 <= v <  600   -> (v + 200) * 191 / 800          (C int division)
         600 <= v < 6000   -> 192 + (v - 600) * 63 / 5400
        v >= 6000          -> 255

    All numerators are non-negative in their branch, so C's truncating
    division equals Python's floor division.
    """
    v = np.asarray(dc).astype(np.int32)
    out = np.empty(v.shape, dtype=np.int32)
    out[...] = 255
    m = v < 6000
    out[m] = 192 + (v[m] - 600) * 63 // 5400
    m = v < 600
    out[m] = (v[m] + 200) * 191 // 800
    out[v < -200] = 0
    return out.astype(np.uint8)


def model_input(dc: np.ndarray) -> np.ndarray:
    """deci-C frame (H, W) -> float32 network input (1, H, W) = code / 255."""
    return (model_code(dc).astype(np.float32) / 255.0)[None]


def median3x3(dc: np.ndarray) -> np.ndarray:
    """Spatial stage of ps_thermal_denoise(): 3x3 median, border pixels untouched."""
    dc = np.asarray(dc)
    out = dc.copy()
    stack = np.stack([dc[1 + dy:H - 1 + dy, 1 + dx:W - 1 + dx]
                      for dy in (-1, 0, 1) for dx in (-1, 0, 1)])
    out[1:H - 1, 1:W - 1] = np.median(stack, axis=0).astype(dc.dtype)
    return out


def device_preprocess(dc: np.ndarray) -> np.ndarray:
    """What the device does to a raw frame before the model, minus the temporal IIR
    (which needs a sequence). Apply to raw single frames; do NOT apply twice to
    frames that were recorded after the device's denoise stage."""
    return median3x3(dc)
