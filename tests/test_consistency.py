"""Train/serve consistency + ONNX parity tests.

`test_train_and_serve_import_same_feature_functions` guards against the classic
train/serve skew bug: the serving pipeline must call the *exact same* feature
functions used at training time (not a re-implementation that can drift). It
was marked `xfail(strict=True)` until Task 15 created `backend/app/pipeline.py`
(`strict=True` meant the suite would go RED, not just an ignorable
expected-failure, if the test started unexpectedly passing without the marker
being removed). Task 15 has since created `backend/app/pipeline.py` and
removed the marker, so this now runs and passes unconditionally (CONTROLLER
RULING, Task 13 carry-over 4; fix round 1 item 6).

`test_onnx_matches_torch` / `test_rally_onnx_matches_torch` guard against the
ONNX export silently changing model behavior (e.g. wrong opset, wrong dynamic
axes, wrong export mode) by exporting freshly-initialized `StrokeTCN` /
`RallyGRU` checkpoints and diffing ONNX Runtime output against the PyTorch
reference on random input. Both checkpoints intentionally use a non-default
config (stroke: channels=[32,64], kernel=3; rally: hidden=16) so the
config-driven model-rebuild path in `export_onnx.py` (`StrokeTCN(channels=...,
k=...)` / `RallyGRU(hidden=...)`) is actually exercised, not just the
constructors' defaults (fix round 1 item 5).
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_train_and_serve_import_same_feature_functions():
    import backend.app.pipeline as sp
    import shuttlesense_core.features as cf

    assert sp.stroke_window is cf.stroke_window
    assert sp.rally_frame_features is cf.rally_frame_features


def test_onnx_matches_torch(tmp_path):
    import onnxruntime as ort
    import torch

    from training.models import StrokeTCN

    # Non-default config (default is channels=[64,128,128], kernel=5) so the
    # export script's config-driven rebuild is actually exercised.
    m = StrokeTCN(channels=(32, 64), k=3)
    m.eval()
    ck = {
        "state_dict": m.state_dict(),
        "config": {"channels": [32, 64], "kernel": 3},
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


def test_rally_onnx_matches_torch(tmp_path):
    import onnxruntime as ort
    import torch

    from training.models import RallyGRU

    # Non-default config (default hidden=32) so the export script's
    # config-driven rebuild (`RallyGRU(hidden=cfg["hidden"])`) is exercised.
    m = RallyGRU(hidden=16)
    m.eval()
    ck = {
        "state_dict": m.state_dict(),
        "config": {"seed": 13, "hidden": 16, "chunk": 512},
        "val_frame_f1": 0.0,
    }
    torch.save(ck, tmp_path / "best.pt")
    r = subprocess.run(
        [
            sys.executable,
            "training/export_onnx.py",
            "--rally-ckpt",
            str(tmp_path / "best.pt"),
            "--skip-stroke",
            "--out-dir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert r.returncode == 0, r.stderr

    # Shape dynamic on BOTH axes (B=3, T=137) -- neither matches the (2,512,4)
    # trace shape used inside export_onnx.py, exercising both dynamic_axes.
    x = np.random.default_rng(1).normal(size=(3, 137, 4)).astype(np.float32)
    with torch.no_grad():
        ref = m(torch.from_numpy(x)).numpy()
    sess = ort.InferenceSession(str(tmp_path / "rally_gru.onnx"))
    out = sess.run(None, {"x": x})[0]
    assert out.shape == (3, 137)
    np.testing.assert_allclose(out, ref, atol=1e-4)

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["rally"]["file"] == "rally_gru.onnx"
    assert manifest["rally"]["threshold"] == 0.6
    assert manifest["rally"]["val_frame_f1_threshold"] == 0.5
    assert "stroke" not in manifest
