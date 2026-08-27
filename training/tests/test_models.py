import os
import tempfile

import numpy as np
import torch

from training.models import RallyGRU, StrokeTCN


def test_stroke_tcn_shapes():
    m = StrokeTCN()
    out = m(torch.randn(4, 30, 68))
    assert out.shape == (4, 8)


def test_rally_gru_shapes():
    m = RallyGRU()
    out = m(torch.randn(2, 100, 4))
    assert out.shape == (2, 100)


def test_stroke_tcn_determinism():
    torch.manual_seed(0)
    m = StrokeTCN()
    m.eval()
    x = torch.randn(4, 30, 68)
    with torch.no_grad():
        out1 = m(x)
        out2 = m(x)
    assert torch.equal(out1, out2)


def test_rally_gru_determinism():
    torch.manual_seed(0)
    m = RallyGRU()
    m.eval()
    x = torch.randn(2, 100, 4)
    with torch.no_grad():
        out1 = m(x)
        out2 = m(x)
    assert torch.equal(out1, out2)


def test_stroke_tcn_eval_batch_one():
    """BatchNorm1d in eval mode must not choke on batch size 1 (small-batch inference)."""
    m = StrokeTCN()
    m.eval()
    with torch.no_grad():
        out = m(torch.randn(1, 30, 68))
    assert out.shape == (1, 8)


def _onnx_export_and_check(model, x, name):
    onnxruntime = pytest_importorskip_onnxruntime()
    if onnxruntime is None:
        return
    model.eval()
    with torch.no_grad():
        expected = model(x).numpy()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, name)
        torch.onnx.export(
            model,
            x,
            path,
            input_names=["input"],
            output_names=["output"],
            opset_version=17,
            dynamo=False,
        )
        assert os.path.exists(path)
        sess = onnxruntime.InferenceSession(path, providers=["CPUExecutionProvider"])
        (actual,) = sess.run(None, {"input": x.numpy()})
        assert actual.shape == expected.shape
        np.testing.assert_allclose(actual, expected, atol=1e-4)


def pytest_importorskip_onnxruntime():
    try:
        import onnxruntime

        return onnxruntime
    except ImportError:
        return None


def test_stroke_tcn_onnx_export():
    torch.manual_seed(0)
    m = StrokeTCN()
    _onnx_export_and_check(m, torch.randn(4, 30, 68), "stroke_tcn.onnx")


def test_rally_gru_onnx_export():
    torch.manual_seed(0)
    m = RallyGRU()
    _onnx_export_and_check(m, torch.randn(2, 100, 4), "rally_gru.onnx")
