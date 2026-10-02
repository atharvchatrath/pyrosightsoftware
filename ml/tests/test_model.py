import numpy as np
import torch

from ml.model import PyroNet, count_macs, fold_model


def test_forward_shapes_and_ranges():
    m = PyroNet().eval()
    heat, wh, off = m(torch.rand(2, 1, 120, 160))
    assert heat.shape == wh.shape == off.shape == (2, 2, 30, 40)
    assert (wh >= 0).all()
    assert (off >= 0).all() and (off <= 1).all()


def test_fold_is_exact_and_budget():
    torch.manual_seed(0)
    m = PyroNet()
    for mod in m.modules():
        if isinstance(mod, torch.nn.BatchNorm2d):
            mod.running_mean.uniform_(-0.5, 0.5); mod.running_var.uniform_(0.5, 2.0)
            mod.weight.data.uniform_(0.5, 1.5); mod.bias.data.uniform_(-0.2, 0.2)
    m.eval()
    f = fold_model(m)
    assert not any(isinstance(x, torch.nn.BatchNorm2d) for x in f.modules())
    x = torch.rand(2, 1, 120, 160)
    for a, b in zip(m(x), f(x)):
        assert torch.allclose(a, b, atol=1e-4, rtol=1e-4)
    info = count_macs(f)
    assert info["conv_macs"] < 60e6


def test_onnx_export(tmp_path):
    import pytest
    pytest.importorskip("onnx")
    from ml.export_onnx import check, export
    m = PyroNet().eval()
    p = str(tmp_path / "m.onnx")
    f = export(m, p)
    rep = check(p, f, m, n=1)
    assert rep["disallowed_ops"] == []
    assert rep["outputs"] == [("heat", [1, 2, 30, 40]), ("wh", [1, 2, 30, 40]), ("off", [1, 2, 30, 40])]
    if isinstance(rep["onnxruntime_max_abs_diff"], float):
        assert rep["onnxruntime_max_abs_diff"] < 1e-3
