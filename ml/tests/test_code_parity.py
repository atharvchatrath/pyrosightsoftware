import numpy as np

from ml import cbridge
from ml.thermal import model_code


def test_model_code_matches_c_for_every_int16():
    L = cbridge.lib()
    vals = np.arange(-32768, 32768, dtype=np.int32)
    c = np.array([L.ps_thermal_model_code(int(v)) for v in vals], dtype=np.uint8)
    py = model_code(vals.astype(np.int16))
    bad = np.nonzero(c != py)[0]
    assert len(bad) == 0, f"{len(bad)} mismatches, first at {vals[bad[:5]]}: C {c[bad[:5]]} py {py[bad[:5]]}"


def test_model_code_anchor_points():
    assert model_code(np.int16(-201)) == 0
    assert model_code(np.int16(-200)) == 0
    assert model_code(np.int16(599)) == 190
    assert model_code(np.int16(600)) == 192
    assert model_code(np.int16(5999)) == 254
    assert model_code(np.int16(6000)) == 255
    assert model_code(np.int16(365)) == (365 + 200) * 191 // 800
