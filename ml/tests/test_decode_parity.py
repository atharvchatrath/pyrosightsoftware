import numpy as np
import pytest

from ml import cbridge
from ml.decode import centernet_decode, estimate_distances


def _random_outputs(rng, n_peaks, gh=30, gw=40):
    heat = rng.normal(-6, 1.0, (2, gh, gw)).astype(np.float32)
    for _ in range(n_peaks):
        c, y, x = rng.integers(0, 2), rng.integers(0, gh), rng.integers(0, gw)
        heat[c, y, x] = rng.uniform(-2, 6)
        # neighbours a bit lower, sometimes an exact tie
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if (dy or dx) and 0 <= y + dy < gh and 0 <= x + dx < gw:
                    heat[c, y + dy, x + dx] = max(heat[c, y + dy, x + dx],
                                                  heat[c, y, x] - (0 if rng.random() < 0.1 else rng.uniform(0.1, 2)))
    wh = rng.uniform(0.0, 60, (2, gh, gw)).astype(np.float32)
    wh[:, rng.random((gh, gw)) < 0.05] = 0.5           # w/h < 1 must be skipped
    off = rng.uniform(0, 0.999, (2, gh, gw)).astype(np.float32)
    return heat, wh, off


@pytest.mark.parametrize("seed", range(25))
@pytest.mark.parametrize("n_peaks", [0, 3, 12, 40])
def test_decode_matches_c(seed, n_peaks):
    rng = np.random.default_rng(seed * 100 + n_peaks)
    heat, wh, off = _random_outputs(rng, n_peaks)
    c = cbridge.c_centernet_decode(heat, wh, off, 4, 0.35, with_distance=True)
    py = estimate_distances(centernet_decode(heat, wh, off, 4, 0.35))
    assert len(c) == len(py)
    for a, b in zip(c, py):
        assert a["cls"] == b["cls"]
        assert a["truncated"] == b["truncated"]
        for k in ("score", "x", "y", "w", "h"):
            assert abs(a[k] - b[k]) <= 1e-5 * max(1.0, abs(a[k])), (k, a, b)
        for k in ("dist_m", "dist_min_m", "dist_max_m"):
            assert abs(a[k] - b[k]) <= 1e-4 * max(1.0, abs(a[k])), (k, a, b)


def test_decode_respects_device_cap():
    rng = np.random.default_rng(1)
    heat, wh, off = _random_outputs(rng, 80)
    assert len(centernet_decode(heat, wh, off)) <= 16
    assert len(cbridge.c_centernet_decode(heat, wh, off)) <= 16
