import json, subprocess, sys
import numpy as np

def test_train_smoke(tmp_path):
    rng = np.random.default_rng(0)
    n = 64
    X = rng.normal(size=(n, 30, 68)).astype(np.float32)
    y = rng.integers(0, 8, size=n).astype(np.int64)
    match = np.array(["m01"] * 32 + ["m02"] * 32, dtype="<U32")
    np.savez(tmp_path / "sw.npz", X=X, y=y, match=match)
    (tmp_path / "splits.json").write_text(json.dumps(
        {"train": ["m01"], "val": ["m02"], "test": []}))
    r = subprocess.run([sys.executable, "training/train_stroke.py",
                        "--config", "training/configs/stroke_tcn.yaml",
                        "--data", str(tmp_path / "sw.npz"),
                        "--splits", str(tmp_path / "splits.json"),
                        "--out-dir", str(tmp_path / "ckpt"), "--epochs", "2"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    import torch
    ck = torch.load(tmp_path / "ckpt" / "best.pt", map_location="cpu", weights_only=False)
    assert "state_dict" in ck and "val_macro_f1" in ck and len(ck["classes"]) == 8
