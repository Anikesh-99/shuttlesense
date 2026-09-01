import json
from pathlib import Path

import numpy as np
import pytest

from backend.app.pipeline import _chunk_frames, analyze, assign_players

MODELS_DIR = "backend/models"
_HAVE_ONNX = Path(MODELS_DIR, "stroke_tcn.onnx").exists() and Path(
    MODELS_DIR, "rally_gru.onnx"
).exists()

pytestmark = pytest.mark.skipif(
    not _HAVE_ONNX, reason="requires backend/models/*.onnx (Task 13 export)"
)


def fake_pose(video_path, target_fps):
    T = 300
    rng = np.random.default_rng(1)
    kpts = np.zeros((T, 2, 17, 2), dtype=np.float32)
    base = np.zeros((17, 2), dtype=np.float32)
    base[11], base[12], base[5], base[6] = [90, 400], [110, 400], [88, 340], [112, 340]
    kpts[:] = base
    kpts[50:200] += rng.normal(scale=15.0, size=(150, 2, 17, 2))  # movement burst
    scores = np.ones((T, 2, 17), dtype=np.float32)
    return kpts, scores, {"fps_sampled": 15.0, "width": 1280, "height": 720, "n_frames": T}


def test_analyze_produces_report_and_tracks():
    report, tracks = analyze(
        "ignored.mp4", models_dir=MODELS_DIR, pose_fn=fake_pose
    )
    assert report.n_frames == 300 and report.fps == 15.0
    assert tracks["fps"] == 15.0 and len(tracks["kpts"]) == 300
    for ev in report.strokes:
        assert 0 <= ev.frame < 300 and ev.stroke != "none"
    json.dumps(report.to_dict())  # serializable


def fake_pose_no_movement(video_path, target_fps):
    # Perfectly static pose the whole clip -> rally-frame energy features are
    # ~0 everywhere -> the rally model should not clear the manifest threshold
    # anywhere -> analyze() must raise ValueError("no rallies detected").
    T = 60
    kpts = np.zeros((T, 2, 17, 2), dtype=np.float32)
    base = np.zeros((17, 2), dtype=np.float32)
    base[11], base[12], base[5], base[6] = [90, 400], [110, 400], [88, 340], [112, 340]
    kpts[:] = base
    scores = np.zeros((T, 2, 17), dtype=np.float32)  # both players absent every frame
    return kpts, scores, {"fps_sampled": 15.0, "width": 1280, "height": 720, "n_frames": T}


def test_analyze_raises_on_no_rallies():
    with pytest.raises(ValueError, match="no rallies detected"):
        analyze("ignored.mp4", models_dir=MODELS_DIR, pose_fn=fake_pose_no_movement)


def test_analyze_reads_threshold_from_manifest_not_hardcoded():
    manifest = json.loads(Path(MODELS_DIR, "manifest.json").read_text())
    assert manifest["rally"]["threshold"] == 0.6


# --- _chunk_frames: numpy re-implementation of training.train_rally.chunk ---


def test_chunk_frames_exact_multiple_all_ones_mask_no_padding():
    X = np.arange(1024 * 4, dtype=np.float32).reshape(1024, 4)
    Xc, mask = _chunk_frames(X, size=512)
    assert Xc.shape == (2, 512, 4)
    assert mask.shape == (2, 512)
    assert np.array_equal(Xc, X.reshape(2, 512, 4))
    assert np.array_equal(mask, np.ones((2, 512), dtype=np.float32))


def test_chunk_frames_pads_remainder_and_masks_it():
    X = np.arange(600 * 4, dtype=np.float32).reshape(600, 4)
    Xc, mask = _chunk_frames(X, size=512)
    assert Xc.shape == (2, 512, 4)
    assert int(mask.sum()) == 600
    assert np.array_equal(mask[1, :88], np.ones(88, dtype=np.float32))
    assert np.array_equal(mask[1, 88:], np.zeros(424, dtype=np.float32))
    assert np.array_equal(Xc[1, :88], X[512:600])
    assert np.array_equal(Xc[1, 88:], np.zeros((424, 4), dtype=np.float32))


def test_chunk_frames_rejects_empty_input():
    with pytest.raises(ValueError, match="empty"):
        _chunk_frames(np.zeros((0, 4), dtype=np.float32), size=512)


@pytest.mark.parametrize("n_frames", [1024, 600, 200, 1])
def test_chunk_frames_matches_training_chunk(n_frames):
    # Only this test may import torch (backend/app/ code must not) -- see
    # pipeline.py's module docstring, "Rally inference contract".
    import torch

    from training.train_rally import chunk as torch_chunk

    rng = np.random.default_rng(0)
    X = rng.normal(size=(n_frames, 4)).astype(np.float32)
    y = np.zeros(n_frames, dtype=np.float32)  # unused by _chunk_frames; chunk() needs it

    Xc_np, mask_np = _chunk_frames(X, size=512)
    Xc_t, _, mask_t = torch_chunk(X, y, size=512)

    assert Xc_np.shape == tuple(Xc_t.shape)
    assert mask_np.shape == tuple(mask_t.shape)
    np.testing.assert_array_equal(Xc_np, Xc_t.numpy())
    np.testing.assert_array_equal(mask_np, mask_t.numpy())


# --- assign_players: mirrors training/extract_poses.assign_players ---


def test_assign_players_orders_by_visible_feet_depth():
    # Two detected people: person 0 has a lower (larger-y, "nearer") visible
    # ankle than person 1 -> person 0 should land in slot 0.
    kpts = np.zeros((2, 17, 2), dtype=np.float32)
    scores = np.ones((2, 17), dtype=np.float32)
    kpts[0, 15] = [100, 500]  # person 0's left ankle, low on screen (near)
    kpts[1, 15] = [100, 200]  # person 1's left ankle, high on screen (far)
    out_k, out_s = assign_players(kpts, scores)
    assert out_k[0, 15, 1] == 500
    assert out_k[1, 15, 1] == 200


def test_assign_players_pads_zero_people():
    out_k, out_s = assign_players(
        np.zeros((0, 17, 2), dtype=np.float32), np.zeros((0, 17), dtype=np.float32)
    )
    assert out_k.shape == (2, 17, 2) and out_s.shape == (2, 17)
    assert not out_k.any() and not out_s.any()
