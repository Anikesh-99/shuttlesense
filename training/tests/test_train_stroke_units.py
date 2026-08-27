import json

import numpy as np
import pytest
import torch

from training.train_stroke import compute_class_weights, load_split, run_epoch
from training.models import StrokeTCN


def _write_npz_and_splits(tmp_path, splits):
    n = 8
    X = np.zeros((n, 30, 68), dtype=np.float32)
    y = np.zeros(n, dtype=np.int64)
    match = np.array(["m01"] * 4 + ["m02"] * 4, dtype="<U32")
    data_path = tmp_path / "sw.npz"
    splits_path = tmp_path / "splits.json"
    np.savez(data_path, X=X, y=y, match=match)
    splits_path.write_text(json.dumps(splits))
    return data_path, splits_path


def test_compute_class_weights_shape_and_values():
    # 3 classes, counts [4, 1, 0] (class 2 absent from training data)
    y = np.array([0, 0, 0, 0, 1], dtype=np.int64)
    w = compute_class_weights(y, n_classes=3)
    assert w.shape == (3,)
    counts = np.array([4.0, 1.0, 0.0])
    expected = (counts.sum() / np.maximum(counts, 1)) ** 0.5
    assert torch.allclose(w, torch.tensor(expected, dtype=w.dtype))
    # rarer/absent classes get larger weight than the majority class
    assert w[1] > w[0]
    assert w[2] > w[0]


def test_compute_class_weights_all_present_uniform_counts():
    y = np.array([0, 1, 0, 1], dtype=np.int64)
    w = compute_class_weights(y, n_classes=2)
    # equal counts -> equal weights == sqrt(sum/count) == sqrt(2)
    assert torch.allclose(w, torch.tensor([np.sqrt(2.0), np.sqrt(2.0)], dtype=w.dtype))


def test_run_epoch_empty_val_guard_does_not_crash():
    model = StrokeTCN()
    model.eval()
    X = torch.empty((0, 30, 68), dtype=torch.float32)
    y = torch.empty((0,), dtype=torch.int64)
    with torch.no_grad():
        loss, f1, (t, p) = run_epoch(model, X, y, bs=8)
    assert loss == 0.0
    assert f1 == 0.0
    assert len(t) == 0 and len(p) == 0


def test_run_epoch_nonempty_smoke():
    model = StrokeTCN()
    model.eval()
    X = torch.randn(5, 30, 68)
    y = torch.randint(0, 8, (5,))
    with torch.no_grad():
        loss, f1, (t, p) = run_epoch(model, X, y, bs=2)
    assert loss == 0.0  # no optimizer passed -> no loss computed, stays default 0.0
    assert len(t) == 5 and len(p) == 5


def test_load_split_rejects_overlapping_matches(tmp_path):
    data_path, splits_path = _write_npz_and_splits(
        tmp_path, {"train": ["m01", "m02"], "val": ["m02"], "test": []}
    )
    with pytest.raises(ValueError, match="disjoint"):
        load_split(str(data_path), str(splits_path))


def test_load_split_warns_on_unknown_match_id(tmp_path, capsys):
    data_path, splits_path = _write_npz_and_splits(
        tmp_path, {"train": ["m01", "m99"], "val": ["m02"], "test": []}
    )
    out = load_split(str(data_path), str(splits_path))
    err = capsys.readouterr().err
    assert "m99" in err
    assert "zero rows" in err
    # the known match id is still loaded despite the unknown one being warned about
    assert len(out["train"][0]) == 4


def test_load_split_disjoint_ok_no_warning(tmp_path, capsys):
    data_path, splits_path = _write_npz_and_splits(
        tmp_path, {"train": ["m01"], "val": ["m02"], "test": []}
    )
    out = load_split(str(data_path), str(splits_path))
    err = capsys.readouterr().err
    assert err == ""
    assert len(out["train"][0]) == 4 and len(out["val"][0]) == 4
