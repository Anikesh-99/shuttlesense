import numpy as np
import torch

from training.train_stroke import compute_class_weights, run_epoch
from training.models import StrokeTCN


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
