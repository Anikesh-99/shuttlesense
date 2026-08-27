import json, subprocess, sys
import numpy as np


def test_train_rally_smoke(tmp_path):
    X = np.random.default_rng(0).normal(size=(2000, 4)).astype(np.float32)
    y = (np.arange(2000) // 100 % 2).astype(np.float32)
    match = np.array(["m01"] * 1000 + ["m02"] * 1000, dtype="<U32")
    np.savez(tmp_path / "rf.npz", X=X, y=y, match=match)
    (tmp_path / "splits.json").write_text(json.dumps(
        {"train": ["m01"], "val": ["m02"], "test": []}))
    r = subprocess.run([sys.executable, "training/train_rally.py",
                        "--config", "training/configs/rally_gru.yaml",
                        "--data", str(tmp_path / "rf.npz"),
                        "--splits", str(tmp_path / "splits.json"),
                        "--out-dir", str(tmp_path / "ckpt"), "--epochs", "2"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    import torch
    ck = torch.load(tmp_path / "ckpt" / "best.pt", map_location="cpu", weights_only=False)
    assert "val_frame_f1" in ck


def _make_data(tmp_path, splits):
    X = np.random.default_rng(0).normal(size=(2000, 4)).astype(np.float32)
    y = (np.arange(2000) // 100 % 2).astype(np.float32)
    match = np.array(["m01"] * 1000 + ["m02"] * 1000, dtype="<U32")
    np.savez(tmp_path / "rf.npz", X=X, y=y, match=match)
    (tmp_path / "splits.json").write_text(json.dumps(splits))


def _run_cli(tmp_path, extra_args):
    return subprocess.run([sys.executable, "training/train_rally.py",
                            "--config", "training/configs/rally_gru.yaml",
                            "--data", str(tmp_path / "rf.npz"),
                            "--splits", str(tmp_path / "splits.json"),
                            "--out-dir", str(tmp_path / "ckpt")] + extra_args,
                           capture_output=True, text=True)


def test_checkpoint_has_expected_keys(tmp_path):
    _make_data(tmp_path, {"train": ["m01"], "val": ["m02"], "test": []})
    r = _run_cli(tmp_path, ["--epochs", "2"])
    assert r.returncode == 0, r.stderr
    import torch
    ck = torch.load(tmp_path / "ckpt" / "best.pt", map_location="cpu", weights_only=False)
    assert set(ck.keys()) == {"state_dict", "config", "val_frame_f1"}
    assert ck["config"]["epochs"] == 2


def test_epochs_zero_is_rejected_not_silently_ignored(tmp_path):
    _make_data(tmp_path, {"train": ["m01"], "val": ["m02"], "test": []})
    r = _run_cli(tmp_path, ["--epochs", "0"])
    assert r.returncode != 0
    assert "epochs" in r.stderr


def test_empty_train_split_raises_clear_error(tmp_path):
    _make_data(tmp_path, {"train": [], "val": ["m01", "m02"], "test": []})
    r = _run_cli(tmp_path, ["--epochs", "1"])
    assert r.returncode != 0
    assert "train split is empty" in r.stderr


def test_empty_val_split_warns_but_still_succeeds(tmp_path):
    _make_data(tmp_path, {"train": ["m01", "m02"], "val": [], "test": []})
    r = _run_cli(tmp_path, ["--epochs", "1"])
    assert r.returncode == 0, r.stderr
    assert "val split empty" in r.stderr
    import torch
    ck = torch.load(tmp_path / "ckpt" / "best.pt", map_location="cpu", weights_only=False)
    assert ck["val_frame_f1"] == 0.0


def test_missing_split_key_raises(tmp_path):
    _make_data(tmp_path, {"train": ["m01"], "val": ["m02"]})  # no "test" key
    r = _run_cli(tmp_path, ["--epochs", "1"])
    assert r.returncode != 0
    assert "missing required key" in r.stderr
