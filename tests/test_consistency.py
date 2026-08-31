"""Train/serve consistency + ONNX parity tests.

`test_train_and_serve_import_same_feature_functions` guards against the classic
train/serve skew bug: the serving pipeline must call the *exact same* feature
functions used at training time (not a re-implementation that can drift). It is
marked xfail until Task 15 creates `backend/app/pipeline.py`; Task 15 removes the
marker (CONTROLLER RULING, Task 13 carry-over 4).

`test_onnx_matches_torch` guards against the ONNX export silently changing model
behavior (e.g. wrong opset, wrong dynamic axes, wrong export mode) by exporting a
freshly-initialized `StrokeTCN` and diffing ONNX Runtime output against the PyTorch
reference on random input.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.xfail(reason="pipeline lands in Task 15")
def test_train_and_serve_import_same_feature_functions():
    import backend.app.pipeline as sp
    import shuttlesense_core.features as cf

    assert sp.stroke_window is cf.stroke_window
    assert sp.rally_frame_features is cf.rally_frame_features


def test_onnx_matches_torch(tmp_path):
    import onnxruntime as ort
    import torch

    from training.models import StrokeTCN

    m = StrokeTCN()
    m.eval()
    ck = {
        "state_dict": m.state_dict(),
        "config": {"channels": [64, 128, 128], "kernel": 5},
        "val_macro_f1": 0.0,
        "classes": ["clear", "smash", "drop", "net", "lift", "drive", "serve", "none"],
        "confusion": [],
    }
    torch.save(ck, tmp_path / "best.pt")
    r = subprocess.run(
        [
            sys.executable,
            "training/export_onnx.py",
            "--stroke-ckpt",
            str(tmp_path / "best.pt"),
            "--skip-rally",
            "--out-dir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert r.returncode == 0, r.stderr
    x = np.random.default_rng(0).normal(size=(3, 30, 68)).astype(np.float32)
    with torch.no_grad():
        ref = m(torch.from_numpy(x)).numpy()
    sess = ort.InferenceSession(str(tmp_path / "stroke_tcn.onnx"))
    out = sess.run(None, {"x": x})[0]
    np.testing.assert_allclose(out, ref, atol=1e-4)

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["stroke"]["file"] == "stroke_tcn.onnx"
    assert "rally" not in manifest
