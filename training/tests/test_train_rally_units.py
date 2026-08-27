import json

import numpy as np
import pytest
import torch

from training.train_rally import chunk, load_split
from training.models import RallyGRU


def test_chunk_exact_multiple_no_padding():
    X = np.arange(1024 * 4, dtype=np.float32).reshape(1024, 4)
    y = np.arange(1024, dtype=np.float32)
    Xc, yc = chunk(X, y, size=512)
    assert Xc.shape == (2, 512, 4)
    assert yc.shape == (2, 512)
    # no padding introduced -- values round-trip exactly
    assert np.array_equal(Xc.numpy(), X.reshape(2, 512, 4))
    assert np.array_equal(yc.numpy(), y.reshape(2, 512))


def test_chunk_short_remainder_is_dropped_when_a_full_chunk_exists():
    # 600 frames, size=512 -> 1 full chunk of 512, remaining 88 frames dropped
    # (per the brief's reference impl: `n = (len(X)//size)*size` then slice to n;
    # only when there isn't even one full chunk (n==0) do we pad).
    X = np.zeros((600, 4), dtype=np.float32)
    y = np.zeros(600, dtype=np.float32)
    Xc, yc = chunk(X, y, size=512)
    assert Xc.shape == (1, 512, 4)
    assert yc.shape == (1, 512)


def test_chunk_pads_short_remainder_to_one_full_chunk():
    # fewer frames than one chunk -> padded up to exactly one chunk
    X = np.ones((200, 4), dtype=np.float32)
    y = np.ones(200, dtype=np.float32)
    Xc, yc = chunk(X, y, size=512)
    assert Xc.shape == (1, 512, 4)
    assert yc.shape == (1, 512)
    # first 200 rows preserved, remainder zero-padded
    assert np.array_equal(Xc.numpy()[0, :200], X)
    assert np.array_equal(Xc.numpy()[0, 200:], np.zeros((312, 4), dtype=np.float32))
    assert np.array_equal(yc.numpy()[0, :200], y)
    assert np.array_equal(yc.numpy()[0, 200:], np.zeros(312, dtype=np.float32))


def test_chunk_returns_torch_tensors():
    X = np.zeros((512, 4), dtype=np.float32)
    y = np.zeros(512, dtype=np.float32)
    Xc, yc = chunk(X, y, size=512)
    assert isinstance(Xc, torch.Tensor) and isinstance(yc, torch.Tensor)


def _write_rally_npz_and_splits(tmp_path, splits, size_per_match=700):
    n = size_per_match * 2
    X = np.random.default_rng(0).normal(size=(n, 4)).astype(np.float32)
    y = np.zeros(n, dtype=np.float32)
    match = np.array(["m01"] * size_per_match + ["m02"] * size_per_match, dtype="<U32")
    data_path = tmp_path / "rf.npz"
    splits_path = tmp_path / "splits.json"
    np.savez(data_path, X=X, y=y, match=match)
    splits_path.write_text(json.dumps(splits))
    return data_path, splits_path


def test_load_split_chunks_never_span_two_matches(tmp_path):
    # each match has 700 frames -- with size=512 that's 1 full chunk (512 frames,
    # rows 0..511) per match plus a leftover 188 frames dropped per match; the
    # important invariant is that no chunk mixes rows from m01 and m02.
    data_path, splits_path = _write_rally_npz_and_splits(
        tmp_path, {"train": ["m01", "m02"], "val": [], "test": []}, size_per_match=700
    )
    z = np.load(data_path)
    out = load_split(str(data_path), str(splits_path), chunk_size=512)
    Xtr, ytr = out["train"]
    assert Xtr.shape[1] == 512
    # 1 chunk per match * 2 matches
    assert Xtr.shape[0] == 2
    m01_first512 = torch.from_numpy(z["X"][:512])
    m02_first512 = torch.from_numpy(z["X"][700:700 + 512])
    got = {tuple(Xtr[i].flatten().tolist()) for i in range(Xtr.shape[0])}
    assert tuple(m01_first512.flatten().tolist()) in got
    assert tuple(m02_first512.flatten().tolist()) in got


def test_load_split_pads_short_match_to_one_chunk(tmp_path):
    data_path, splits_path = _write_rally_npz_and_splits(
        tmp_path, {"train": ["m01"], "val": ["m02"], "test": []}, size_per_match=200
    )
    out = load_split(str(data_path), str(splits_path), chunk_size=512)
    Xtr, ytr = out["train"]
    assert Xtr.shape == (1, 512, 4)


def test_load_split_missing_key_raises(tmp_path):
    data_path, splits_path = _write_rally_npz_and_splits(
        tmp_path, {"train": ["m01"], "val": ["m02"]}
    )
    with pytest.raises(ValueError, match="missing required key"):
        load_split(str(data_path), str(splits_path), chunk_size=512)


def test_load_split_empty_split_returns_empty_tensor(tmp_path):
    data_path, splits_path = _write_rally_npz_and_splits(
        tmp_path, {"train": ["m01"], "val": [], "test": []}
    )
    out = load_split(str(data_path), str(splits_path), chunk_size=512)
    Xval, yval = out["val"]
    assert Xval.shape == (0, 512, 4)
    assert yval.shape == (0, 512)


def test_rally_gru_forward_shapes():
    model = RallyGRU(hidden=8)
    x = torch.randn(3, 512, 4)
    logits = model(x)
    assert logits.shape == (3, 512)
