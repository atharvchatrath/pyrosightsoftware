import numpy as np

from ml import synth
from ml.dataset import (GH, GW, DiskDataset, SynthDataset, encode_targets, read_meta, read_sample,
                        write_sample)
from ml.decode import centernet_decode


def test_disk_round_trip(tmp_path):
    root = str(tmp_path / "ds")
    samples = [synth.generate(s) for s in range(6)]
    import os
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, "meta.csv"), "w") as meta:
        meta.write("id,smoke\n")
        for i, s in enumerate(samples):
            write_sample(root, i, s.dc, s.boxes, s.cls, s.dist, s.smoke, meta)
    meta = read_meta(root)
    for i, s in enumerate(samples):
        dc, boxes, cls, dist = read_sample(root, i)
        assert dc.dtype == np.int16 and dc.shape == (120, 160)
        assert np.array_equal(dc, s.dc)
        assert np.array_equal(cls, s.cls)
        assert np.allclose(boxes, s.boxes, atol=0.051)
        assert np.allclose(dist, s.dist, atol=0.0051)
        assert abs(meta[i] - s.smoke) < 0.006
    # raw bytes are little-endian int16, 160*120
    assert os.path.getsize(os.path.join(root, "images", "000000.bin")) == 160 * 120 * 2
    ds = DiskDataset(root, augment=True)
    item = ds[0]
    assert item["x"].shape == (1, 120, 160) and float(item["x"].min()) >= 0 and float(item["x"].max()) <= 1


def test_synth_labels_sane():
    for seed in range(30):
        s = synth.generate(seed)
        assert s.dc.dtype == np.int16 and s.dc.shape == (120, 160)
        for b, c, d in zip(s.boxes, s.cls, s.dist):
            assert b[0] >= 0 and b[1] >= 0 and b[0] + b[2] <= 160 and b[1] + b[3] <= 120
            assert c in (0, 1)
            assert (1.0 <= d <= 12.0) if c == 1 else d == 0
        assert 0 <= s.smoke <= 1


def test_targets_decode_back_to_boxes():
    """Perfect network outputs (targets turned into logits) must decode to the labels."""
    boxes = np.array([[10, 20, 12, 30], [100.5, 50, 40, 15], [150, 100, 9, 19]], np.float32)
    cls = np.array([1, 0, 1])
    heat, wh, off, mask = encode_targets(boxes, cls)
    assert heat.shape == (2, GH, GW) and mask.sum() == 3
    logits = np.log(np.clip(heat, 1e-6, 1 - 1e-6) / np.clip(1 - heat, 1e-6, 1))
    dets = centernet_decode(logits.astype(np.float32), wh, off, thr=0.35)
    assert len(dets) == 3
    for d in dets:
        j = int(np.argmin([abs(d["x"] - b[0]) + abs(d["y"] - b[1]) for b in boxes]))
        assert d["cls"] == cls[j]
        assert np.allclose([d["x"], d["y"], d["w"], d["h"]], boxes[j], atol=0.01)


def test_synth_dataset_deterministic():
    a, b = SynthDataset(3, seed0=7), SynthDataset(3, seed0=7)
    assert (a[2]["x"] == b[2]["x"]).all()
