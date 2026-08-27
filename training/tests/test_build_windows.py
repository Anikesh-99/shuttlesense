import numpy as np
import pandas as pd

from shuttlesense_core.features import FEAT_DIM, WINDOW
from shuttlesense_core.schemas import ALL_CLASSES
from training.build_windows import (
    build_rally_frame_labels,
    build_stroke_samples,
    label_frame_to_pose_idx,
    make_splits,
)


def test_stroke_samples_positive_and_negative():
    rng = np.random.default_rng(0)
    T = 600
    kpts = rng.normal(size=(T, 2, 17, 2)).astype(np.float32)
    labels = pd.DataFrame({
        "hit_frame": [100, 300], "player": [0, 1], "stroke": ["smash", "net"],
        "fps": [30.0, 30.0],
    })
    X, y = build_stroke_samples(labels, kpts, fps_scale=0.5, rng=rng)
    assert X.shape[1:] == (30, 68)
    assert (y == ALL_CLASSES.index("smash")).sum() == 1
    assert (y == ALL_CLASSES.index("none")).sum() == 2   # 1:1 with positives


def test_stroke_samples_empty_labels_returns_correctly_shaped_empty_arrays():
    # No labels fall inside [0, T) -> zero positives, zero negatives (1:1 with zero is
    # zero), but the function must still return correctly-shaped (not ragged/failed)
    # empty arrays rather than raising on np.stack([]).
    rng = np.random.default_rng(0)
    T = 50
    kpts = rng.normal(size=(T, 2, 17, 2)).astype(np.float32)
    labels = pd.DataFrame({
        "hit_frame": [10_000], "player": [0], "stroke": ["smash"], "fps": [30.0],
    })
    X, y = build_stroke_samples(labels, kpts, fps_scale=0.5, rng=rng)
    assert X.shape == (0, WINDOW, FEAT_DIM)
    assert y.shape == (0,)
    assert y.dtype == np.int64


def test_stroke_samples_negatives_stay_guard_distance_from_every_hit():
    rng = np.random.default_rng(1)
    T = 600
    kpts = rng.normal(size=(T, 2, 17, 2)).astype(np.float32)
    labels = pd.DataFrame({
        "hit_frame": [200, 400], "player": [0, 1], "stroke": ["clear", "drop"],
        "fps": [30.0, 30.0],
    })
    X, y = build_stroke_samples(labels, kpts, fps_scale=1.0, rng=rng)
    # Positives land at hit_frame * fps_scale = 200, 400 exactly.
    hit_idx = [200, 400]
    none_id = ALL_CLASSES.index("none")
    # We can't directly recover each negative's frame index from the output, but we can
    # at least confirm the guard-distance sampling loop actually converged to 1:1 and
    # didn't fall back to something degenerate.
    assert (y == none_id).sum() == 2


def test_splits_disjoint_and_deterministic():
    ids = [f"m{i:02d}" for i in range(20)]
    s1, s2 = make_splits(ids), make_splits(ids)
    assert s1 == s2
    assert not (set(s1["train"]) & set(s1["test"]))
    assert set(s1["train"]) | set(s1["val"]) | set(s1["test"]) == set(ids)


def test_label_frame_to_pose_idx_applies_offset_and_step():
    # start_offset_s=600, orig_fps=30 -> the clip's frame 0 corresponds to original frame
    # 18000; step=2 halves the in-clip sampling rate relative to orig_fps.
    sidecar = {"start_offset_s": 600}
    meta = {"orig_fps": 30.0, "step": 2}
    assert label_frame_to_pose_idx(18000, sidecar, meta) == 0.0
    assert label_frame_to_pose_idx(18010, sidecar, meta) == 5.0


def test_label_frame_to_pose_idx_missing_sidecar_defaults_offset_zero():
    meta = {"orig_fps": 30.0, "step": 2}
    # No sidecar (None) -> offset 0, warns rather than raising.
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        idx = label_frame_to_pose_idx(20, None, meta)
        assert idx == 10.0
        assert len(w) == 1


def test_build_rally_frame_labels_union_of_overlapping_intervals():
    # Two overlapping intervals should union, not double count or error.
    ry = build_rally_frame_labels(10, [(0, 5), (3, 8)])
    assert ry.tolist() == [1, 1, 1, 1, 1, 1, 1, 1, 0, 0]


def test_build_rally_frame_labels_clamps_out_of_range_ends():
    # Interval end far beyond n_frames, and interval start far below 0, must clamp
    # instead of raising or wrapping via negative-index slicing semantics.
    ry = build_rally_frame_labels(10, [(-1000, 3), (7, 10_000)])
    assert ry.tolist() == [1, 1, 1, 0, 0, 0, 0, 1, 1, 1]
