import json
import re
from pathlib import Path

import numpy as np
import pytest

from backend.app.pipeline import _chunk_frames, analyze, assign_players

MODELS_DIR = "backend/models"
_HAVE_ONNX = Path(MODELS_DIR, "stroke_tcn.onnx").exists() and Path(
    MODELS_DIR, "rally_gru.onnx"
).exists()

# Only tests that reach analyze()'s ONNX inference path (i.e. the pose_fn
# they inject returns >0 frames, so analyze() gets past the zero-frame guard
# and constructs real ort.InferenceSessions) are guarded. analyze() reads
# manifest.json unconditionally but only builds the onnx sessions AFTER that
# guard, so the zero-frame "no rallies detected" regression test and the
# manifest-only test below run onnx-free; _chunk_frames/assign_players/
# empty-input unit tests likewise need no backend/models/*.onnx present.
requires_onnx = pytest.mark.skipif(
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


@requires_onnx
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


@requires_onnx
def test_analyze_raises_on_no_rallies():
    with pytest.raises(ValueError, match="no rallies detected"):
        analyze("ignored.mp4", models_dir=MODELS_DIR, pose_fn=fake_pose_no_movement)


def fake_pose_zero_frames(video_path, target_fps):
    # A pose extraction that produced 0 sampled frames (e.g. an unreadable /
    # zero-duration video). Must hit the SAME "no rallies detected" contract,
    # not an internal np.stack([]) crash from rally_frame_features (see
    # pipeline.py's analyze(), which hoists this check above that call).
    return (
        np.zeros((0, 2, 17, 2), dtype=np.float32),
        np.zeros((0, 2, 17), dtype=np.float32),
        {"fps_sampled": 15.0, "width": 1280, "height": 720, "n_frames": 0},
    )


def test_analyze_raises_friendly_error_on_zero_frame_pose():
    # No @requires_onnx: analyze() raises before ever constructing an
    # ort.InferenceSession for a 0-frame pose_fn (see pipeline.py), so this
    # regression test must pass even with no backend/models/*.onnx present.
    with pytest.raises(ValueError, match="no rallies detected"):
        analyze("ignored.mp4", models_dir=MODELS_DIR, pose_fn=fake_pose_zero_frames)


def fake_pose_multi_chunk(video_path, target_fps):
    # T=600 spans a full 512-frame chunk plus a padded remainder chunk under
    # the rally inference contract -- exercises the multi-chunk
    # reassembly/mask-discard path end to end (not just _chunk_frames in
    # isolation).
    T = 600
    rng = np.random.default_rng(3)
    kpts = np.zeros((T, 2, 17, 2), dtype=np.float32)
    base = np.zeros((17, 2), dtype=np.float32)
    base[11], base[12], base[5], base[6] = [90, 400], [110, 400], [88, 340], [112, 340]
    kpts[:] = base
    kpts[100:550] += rng.normal(scale=15.0, size=(450, 2, 17, 2))  # spans both chunks
    scores = np.ones((T, 2, 17), dtype=np.float32)
    return kpts, scores, {"fps_sampled": 15.0, "width": 1280, "height": 720, "n_frames": T}


@requires_onnx
def test_analyze_multi_chunk_rally_probs_cover_every_frame():
    # Regression coverage for the assert len(probs) == T check inside
    # analyze(): a 600-frame input (2 rally chunks) must produce rally
    # intervals/report bounds entirely within [0, 600), proving the
    # chunk-flatten-and-mask-discard reassembly didn't drop or duplicate any
    # frame across the chunk boundary at 512.
    report, tracks = analyze(
        "ignored.mp4", models_dir=MODELS_DIR, pose_fn=fake_pose_multi_chunk
    )
    assert report.n_frames == 600
    assert len(tracks["kpts"]) == 600
    for r in report.rallies:
        assert 0 <= r.start_frame < r.end_frame <= 600
    for ev in report.strokes:
        assert 0 <= ev.frame < 600


def fake_pose_rounding(video_path, target_fps):
    # A movement burst confined to the wrists (idx 9, 10) drives rally
    # detection while the hip/shoulder anchor keypoints (11, 12, 5, 6) stay at
    # their exact literal float values -- lets the serialization test below
    # assert on a specific, known decimal value surviving round-trip through
    # np.round + json.dumps without float32->float64 widening noise.
    T = 300
    rng = np.random.default_rng(2)
    kpts = np.zeros((T, 2, 17, 2), dtype=np.float32)
    base = np.zeros((17, 2), dtype=np.float32)
    base[11], base[12], base[5], base[6] = [90.4, 400.0], [110.0, 400.0], [88.0, 340.0], [112.0, 340.0]
    kpts[:] = base
    kpts[50:200, :, 9] += rng.normal(scale=15.0, size=(150, 2, 2))
    kpts[50:200, :, 10] += rng.normal(scale=15.0, size=(150, 2, 2))
    scores = np.ones((T, 2, 17), dtype=np.float32)
    return kpts, scores, {"fps_sampled": 15.0, "width": 1280, "height": 720, "n_frames": T}


@requires_onnx
def test_tracks_kpts_serialize_without_float64_widening():
    _, tracks = analyze("ignored.mp4", models_dir=MODELS_DIR, pose_fn=fake_pose_rounding)
    payload = json.dumps(tracks)
    # the exact literal (90.4) we planted at kpts[:, :, 11] must survive
    # round-tripping through np.round(..., 1) + json.dumps as "90.4", not
    # something like "90.4000015258789" (float32 -> python float widening).
    assert re.search(r"(?<!\d)90\.4(?!\d)", payload) is not None
    # no numeric literal anywhere in the payload has 3+ decimal digits --
    # kpts is rounded to 1 decimal, scores to 2, so a properly-rounded
    # payload never needs more than 2.
    assert re.search(r"\d\.\d{3,}", payload) is None


def test_analyze_reads_threshold_from_manifest_not_hardcoded():
    # No @requires_onnx: only reads manifest.json, never touches the .onnx
    # files themselves.
    manifest = json.loads(Path(MODELS_DIR, "manifest.json").read_text())
    assert manifest["rally"]["threshold"] == 0.6


# --- _chunk_frames: numpy re-implementation of training.train_rally.chunk ---
# (no ONNX files required for any test below this point)


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


# --- anti-drift parity: this copy must never diverge from the training original ---


def _people_case(seed: int, n_people: int, low_conf_person: int | None = None):
    """Deterministic (kpts, scores) fixture for `n_people` detected people.
    If `low_conf_person` is given, every keypoint of that person is given a
    score below the 0.3 gate, forcing assign_players's ungated max(y)
    fallback for that person's depth statistic."""
    rng = np.random.default_rng(seed)
    kpts = rng.uniform(0, 1000, size=(n_people, 17, 2)).astype(np.float32)
    scores = rng.uniform(0.3, 1.0, size=(n_people, 17)).astype(np.float32)
    if low_conf_person is not None:
        scores[low_conf_person] = rng.uniform(0.0, 0.29, size=17).astype(np.float32)
    return kpts, scores


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(seed=0, n_people=0, low_conf_person=None),
        dict(seed=1, n_people=1, low_conf_person=None),
        dict(seed=2, n_people=3, low_conf_person=None),
        # n_people=2 (not 3): with 3 people, the low-confidence person's mean
        # score is so far below the other two's that top-2-by-mean-score
        # selection excludes them entirely -- depth_stat, and therefore the
        # ungated max(y) fallback, never even runs on them (dead coverage,
        # fix round 2 item 1). With exactly 2 people, both are unconditionally
        # selected (order = top 2 of 2), so the low-confidence one is
        # guaranteed to reach depth_stat and hit its fallback branch.
        dict(seed=3, n_people=2, low_conf_person=0),  # triggers ungated fallback
    ],
    ids=["N=0", "N=1", "N=3", "N=2-low-confidence-fallback"],
)
def test_assign_players_matches_training_extract_poses(kwargs):
    # Guards against backend/app/pipeline.assign_players silently drifting
    # from training/extract_poses.assign_players (backend can't import
    # training.* at runtime, so this copy is duplicated by hand -- see both
    # modules' docstrings). Only this test needs to import training code from
    # backend/tests/, which is fine (training/ may depend on anything;
    # backend/app/ may not depend on training/).
    from training.extract_poses import assign_players as training_assign_players

    kpts, scores = _people_case(**kwargs)
    out_k_backend, out_s_backend = assign_players(kpts, scores)
    out_k_training, out_s_training = training_assign_players(kpts, scores)
    np.testing.assert_array_equal(out_k_backend, out_k_training)
    np.testing.assert_array_equal(out_s_backend, out_s_training)
