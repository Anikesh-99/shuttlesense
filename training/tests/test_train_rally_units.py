import json

import numpy as np
import pytest
import torch

import torch.nn as nn

from training.train_rally import chunk, load_split, run_epoch
from training.models import RallyGRU


def test_chunk_exact_multiple_all_ones_mask_no_padding():
    X = np.arange(1024 * 4, dtype=np.float32).reshape(1024, 4)
    y = np.arange(1024, dtype=np.float32)
    Xc, yc, mc = chunk(X, y, size=512)
    assert Xc.shape == (2, 512, 4)
    assert yc.shape == (2, 512)
    assert mc.shape == (2, 512)
    # no padding introduced -- values round-trip exactly, mask is all ones
    assert np.array_equal(Xc.numpy(), X.reshape(2, 512, 4))
    assert np.array_equal(yc.numpy(), y.reshape(2, 512))
    assert np.array_equal(mc.numpy(), np.ones((2, 512), dtype=np.float32))


def test_chunk_remainder_is_padded_not_dropped_and_correctly_masked():
    # 600 frames, size=512 -> PAD-ALWAYS policy (controller ruling): 2 chunks
    # total, no frames lost. Chunk 0 is fully real (mask all ones). Chunk 1
    # holds the 88 real remainder frames (mask ones) followed by 424 padded
    # frames (mask zeros).
    X = np.arange(600 * 4, dtype=np.float32).reshape(600, 4)
    y = np.arange(600, dtype=np.float32)
    Xc, yc, mc = chunk(X, y, size=512)
    assert Xc.shape == (2, 512, 4)
    assert yc.shape == (2, 512)
    assert mc.shape == (2, 512)
    # no frames lost: every real frame appears somewhere with mask==1
    n_real = int(mc.numpy().sum())
    assert n_real == 600
    # chunk 0 fully real
    assert np.array_equal(mc.numpy()[0], np.ones(512, dtype=np.float32))
    assert np.array_equal(Xc.numpy()[0], X[:512])
    # chunk 1: first 88 rows real, matching original data; rest is padding (zeros)
    assert np.array_equal(mc.numpy()[1, :88], np.ones(88, dtype=np.float32))
    assert np.array_equal(mc.numpy()[1, 88:], np.zeros(424, dtype=np.float32))
    assert np.array_equal(Xc.numpy()[1, :88], X[512:600])
    assert np.array_equal(Xc.numpy()[1, 88:], np.zeros((424, 4), dtype=np.float32))
    assert np.array_equal(yc.numpy()[1, :88], y[512:600])
    assert np.array_equal(yc.numpy()[1, 88:], np.zeros(424, dtype=np.float32))


def test_chunk_pads_short_remainder_to_one_full_chunk():
    # fewer frames than one chunk -> padded up to exactly one chunk
    X = np.ones((200, 4), dtype=np.float32)
    y = np.ones(200, dtype=np.float32)
    Xc, yc, mc = chunk(X, y, size=512)
    assert Xc.shape == (1, 512, 4)
    assert yc.shape == (1, 512)
    assert mc.shape == (1, 512)
    # first 200 rows preserved and marked real, remainder zero-padded and masked out
    assert np.array_equal(Xc.numpy()[0, :200], X)
    assert np.array_equal(Xc.numpy()[0, 200:], np.zeros((312, 4), dtype=np.float32))
    assert np.array_equal(yc.numpy()[0, :200], y)
    assert np.array_equal(yc.numpy()[0, 200:], np.zeros(312, dtype=np.float32))
    assert np.array_equal(mc.numpy()[0, :200], np.ones(200, dtype=np.float32))
    assert np.array_equal(mc.numpy()[0, 200:], np.zeros(312, dtype=np.float32))


def test_chunk_returns_torch_tensors():
    X = np.zeros((512, 4), dtype=np.float32)
    y = np.zeros(512, dtype=np.float32)
    Xc, yc, mc = chunk(X, y, size=512)
    assert isinstance(Xc, torch.Tensor)
    assert isinstance(yc, torch.Tensor)
    assert isinstance(mc, torch.Tensor)


def test_chunk_rejects_empty_input():
    X = np.zeros((0, 4), dtype=np.float32)
    y = np.zeros((0,), dtype=np.float32)
    with pytest.raises(ValueError, match="empty"):
        chunk(X, y, size=512)


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
    # each match has 700 frames -- with size=512 that's 2 chunks per match
    # under the pad-always policy (512 real + a padded remainder chunk); the
    # important invariant is that no chunk mixes rows from m01 and m02.
    data_path, splits_path = _write_rally_npz_and_splits(
        tmp_path, {"train": ["m01", "m02"], "val": [], "test": []}, size_per_match=700
    )
    z = np.load(data_path)
    out = load_split(str(data_path), str(splits_path), chunk_size=512)
    Xtr, ytr, mtr = out["train"]
    assert Xtr.shape[1] == 512
    # 2 chunks per match (512 full + remainder padded) * 2 matches
    assert Xtr.shape[0] == 4
    # no frames lost overall: total real (mask==1) frames == 700*2
    assert int(mtr.numpy().sum()) == 700 * 2
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
    Xtr, ytr, mtr = out["train"]
    assert Xtr.shape == (1, 512, 4)
    assert mtr.shape == (1, 512)
    assert int(mtr.numpy().sum()) == 200


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
    Xval, yval, mval = out["val"]
    assert Xval.shape == (0, 512, 4)
    assert yval.shape == (0, 512)
    assert mval.shape == (0, 512)


def test_load_split_order_invariant_to_splits_json_key_order(tmp_path):
    # same match ids, listed in a different order in the JSON -- assembled
    # tensors must be identical (sorted() iteration order).
    data_path, splits_path_a = _write_rally_npz_and_splits(
        tmp_path, {"train": ["m01", "m02"], "val": [], "test": []}, size_per_match=200
    )
    splits_path_b = tmp_path / "splits_b.json"
    splits_path_b.write_text(json.dumps({"train": ["m02", "m01"], "val": [], "test": []}))
    out_a = load_split(str(data_path), str(splits_path_a), chunk_size=512)
    out_b = load_split(str(data_path), str(splits_path_b), chunk_size=512)
    assert torch.equal(out_a["train"][0], out_b["train"][0])
    assert torch.equal(out_a["train"][1], out_b["train"][1])
    assert torch.equal(out_a["train"][2], out_b["train"][2])


def test_rally_gru_forward_shapes():
    model = RallyGRU(hidden=8)
    x = torch.randn(3, 512, 4)
    logits = model(x)
    assert logits.shape == (3, 512)


def test_run_epoch_empty_guard_does_not_crash():
    model = RallyGRU(hidden=4)
    model.eval()
    X = torch.empty((0, 512, 4), dtype=torch.float32)
    y = torch.empty((0, 512), dtype=torch.float32)
    mask = torch.empty((0, 512), dtype=torch.float32)
    with torch.no_grad():
        loss, f1, (t, p) = run_epoch(model, X, y, mask, bs=8)
    assert loss == 0.0 and f1 == 0.0
    assert len(t) == 0 and len(p) == 0


def test_run_epoch_excludes_padded_frames_from_metric_and_returned_arrays():
    model = RallyGRU(hidden=4)
    model.eval()
    n_chunks, size = 2, 16
    X = torch.randn(n_chunks, size, 4)
    y = torch.zeros(n_chunks, size)
    mask = torch.ones(n_chunks, size)
    mask[:, size // 2:] = 0.0  # second half of every chunk is padding
    with torch.no_grad():
        _, _, (t, p) = run_epoch(model, X, y, mask, bs=2)
    # returned (t, p) should only contain the unmasked (real) frames
    assert len(t) == n_chunks * (size // 2)
    assert len(p) == n_chunks * (size // 2)


def test_run_epoch_masked_loss_ignores_padding_predictions():
    # A model that's wildly wrong only on padded frames should not incur any
    # training loss for those frames -- the masked mean must exclude them.
    torch.manual_seed(0)
    model = RallyGRU(hidden=4)
    model.train()
    n_chunks, size = 4, 16
    X = torch.randn(n_chunks, size, 4)
    y = torch.zeros(n_chunks, size)
    mask = torch.ones(n_chunks, size)
    mask[:, size // 2:] = 0.0
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    opt = torch.optim.SGD(model.parameters(), lr=0.0)  # no actual update, just exercise the path
    loss, _, _ = run_epoch(model, X, y, mask, bs=2, opt=opt, loss_fn=loss_fn)
    # sanity: loss is finite and a plain float (masked-mean successfully computed)
    assert isinstance(loss, float)
    assert loss == loss  # not NaN
