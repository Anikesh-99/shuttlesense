import warnings

import numpy as np
import pandas as pd

from shuttlesense_core.features import FEAT_DIM, HIP_L, HIP_R, SH_L, SH_R, WINDOW, stroke_window
from shuttlesense_core.schemas import ALL_CLASSES
from training.build_windows import (
    build_rally_frame_labels,
    build_stroke_samples,
    check_fps_consistency,
    label_frame_to_pose_idx,
    make_splits,
)

NONE_ID = ALL_CLASSES.index("none")


def _fixed_torso_kpts(T: int, P: int = 2) -> np.ndarray:
    """(T,P,17,2) kpts with hip-center=(0,0) and shoulder-mid=(0,1) (torso scale 1) at
    every frame/slot, so `normalize_pose` is a no-op translate+scale-by-1 on every other
    keypoint. Lets tests plant an exactly-decodable signal on another keypoint (e.g. the
    nose) without normalize_pose's hip-centering/torso-scaling distorting it."""
    k = np.zeros((T, P, 17, 2), dtype=np.float32)
    k[:, :, HIP_L] = (-0.5, 0.0)
    k[:, :, HIP_R] = (0.5, 0.0)
    k[:, :, SH_L] = (-0.5, 1.0)
    k[:, :, SH_R] = (0.5, 1.0)
    return k


def test_stroke_samples_positive_and_negative():
    rng = np.random.default_rng(0)
    T = 600
    kpts = _fixed_torso_kpts(T)
    # Give both slots some (equal) motion around both hits so both clear presence and
    # both are "equally" a candidate hitter -- the label's own player id is irrelevant
    # now, only the stroke class matters for the assertions below.
    kpts[:, :, 0, 0] = np.arange(T)[:, None]
    scores = np.ones((T, 2, 17), dtype=np.float32)
    labels = pd.DataFrame({
        "hit_frame": [100, 300], "player": [0, 1], "stroke": ["smash", "net"],
        "fps": [30.0, 30.0],
    })
    X, y, n_skipped = build_stroke_samples(labels, kpts, scores, fps_scale=0.5, rng=rng)
    assert X.shape[1:] == (30, 68)
    assert (y == ALL_CLASSES.index("smash")).sum() == 1
    assert (y == ALL_CLASSES.index("none")).sum() == 2  # both hits' non-hitter slot qualifies
    assert n_skipped == 0


def test_stroke_samples_empty_labels_returns_correctly_shaped_empty_arrays():
    # No labels fall inside [0, T) -> zero positives, zero negatives (1:1 with zero is
    # zero), but the function must still return correctly-shaped (not ragged/failed)
    # empty arrays rather than raising on np.stack([]).
    rng = np.random.default_rng(0)
    T = 50
    kpts = _fixed_torso_kpts(T)
    scores = np.ones((T, 2, 17), dtype=np.float32)
    labels = pd.DataFrame({
        "hit_frame": [10_000], "player": [0], "stroke": ["smash"], "fps": [30.0],
    })
    X, y, n_skipped = build_stroke_samples(labels, kpts, scores, fps_scale=0.5, rng=rng)
    assert X.shape == (0, WINDOW, FEAT_DIM)
    assert y.shape == (0,)
    assert y.dtype == np.int64
    assert n_skipped == 0  # out-of-[0,T) hits are dropped before the skip-counted path


def test_hitter_selection_picks_the_moving_slot_not_the_static_one():
    # Slot 0 is static (energy 0) throughout; slot 1 moves right around the hit frame
    # (nonzero energy). Both are fully present (score 1.0 everywhere). The emitted
    # positive window must come from slot 1 (the mover), regardless of `labels.player`
    # (set to 0 here, i.e. the WRONG slot under the old player-indexed approach -- this
    # is the regression case for CONTROLLER RULING C1).
    rng = np.random.default_rng(0)
    T = 200
    f = 100
    kpts = _fixed_torso_kpts(T)
    kpts[:, 1, 0, 1] = 0.1 * np.sin(np.arange(T))  # slot 1's nose y oscillates -> energy > 0
    scores = np.ones((T, 2, 17), dtype=np.float32)
    labels = pd.DataFrame({
        "hit_frame": [f], "player": [0], "stroke": ["clear"], "fps": [30.0],
    })
    X, y, n_skipped = build_stroke_samples(labels, kpts, scores, fps_scale=1.0, rng=rng)
    assert n_skipped == 0
    positives = X[y == ALL_CLASSES.index("clear")]
    assert len(positives) == 1
    assert np.array_equal(positives[0], stroke_window(kpts[:, 1], f))
    # The static slot (0) still clears presence, so it must appear as a 'none' negative
    # at the same frame (mirrors inference: the classifier sees both players every frame).
    negatives = X[y == NONE_ID]
    assert any(np.array_equal(neg, stroke_window(kpts[:, 0], f)) for neg in negatives)


def test_hitter_selection_skips_hit_when_no_slot_clears_presence():
    # Both slots score below PRESENCE_THR (0.3) at the hit frame -> the hit must be
    # skipped entirely (no corrupted positive from an absent player), and counted.
    rng = np.random.default_rng(0)
    T = 100
    f = 50
    kpts = _fixed_torso_kpts(T)
    scores = np.full((T, 2, 17), 0.1, dtype=np.float32)
    labels = pd.DataFrame({
        "hit_frame": [f], "player": [0], "stroke": ["smash"], "fps": [30.0],
    })
    X, y, n_skipped = build_stroke_samples(labels, kpts, scores, fps_scale=1.0, rng=rng)
    assert n_skipped == 1
    assert X.shape == (0, WINDOW, FEAT_DIM)
    assert y.shape == (0,)


def test_stroke_samples_random_negatives_stay_guard_distance_from_every_hit():
    # Only slot 0 is ever present (slot 1 permanently absent, score 0) -> for each hit,
    # the non-hitter (slot 1) never clears presence, so ALL negatives for this labels
    # set must come from the random guard-distance fill loop, not the "other slot at
    # the hit frame" path. Each frame's nose_x is set to a unique, exactly-decodable
    # value (frame_index * 1000, undistorted by normalize_pose since torso scale is
    # pinned to 1 by _fixed_torso_kpts) so we can read back, from X itself, which real
    # frame every sampled negative actually came from and assert it honors the guard.
    rng = np.random.default_rng(0)
    T = 600
    kpts = _fixed_torso_kpts(T)
    kpts[:, :, 0, 0] = np.arange(T)[:, None] * 1000.0  # nose_x = frame_index * 1000
    scores = np.zeros((T, 2, 17), dtype=np.float32)
    scores[:, 0, :] = 1.0  # only slot 0 present, ever
    hits = [150, 400]
    labels = pd.DataFrame({
        "hit_frame": hits, "player": [0, 0], "stroke": ["smash", "clear"], "fps": [30.0, 30.0],
    })
    X, y, n_skipped = build_stroke_samples(labels, kpts, scores, fps_scale=1.0, rng=rng)
    assert n_skipped == 0
    negatives = X[y == NONE_ID]
    assert len(negatives) == 2  # 1:1 with the 2 positives, all from the random-fill path
    guard = 15
    for neg in negatives:
        # Row index WINDOW//2 (=15) of the window is exactly the (unclamped) center
        # frame `f` itself -- see core/shuttlesense_core/features.py's stroke_window
        # docstring for why the center always lands there regardless of edge clamping
        # elsewhere in the window.
        decoded_f = int(round(neg[WINDOW // 2, 0] / 1000.0))
        assert all(abs(decoded_f - h) > guard for h in hits), (decoded_f, hits)


def test_build_stroke_samples_warns_when_negative_budget_cannot_be_filled():
    # Tiny T with a guard distance that swallows nearly the whole valid range -> the
    # random-fill loop should give up after its try budget and WARN rather than loop
    # forever or silently under-report (I6 RULING).
    rng = np.random.default_rng(0)
    T = 20
    kpts = _fixed_torso_kpts(T)
    scores = np.zeros((T, 2, 17), dtype=np.float32)
    scores[:, 0, :] = 1.0  # only slot 0 present -> no "other slot" negatives possible
    labels = pd.DataFrame({
        "hit_frame": [10], "player": [0], "stroke": ["smash"], "fps": [30.0],
    })
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        X, y, n_skipped = build_stroke_samples(labels, kpts, scores, fps_scale=1.0, rng=rng)
        assert any("could not sample" in str(warning.message) for warning in w)


def test_splits_disjoint_and_deterministic():
    ids = [f"m{i:02d}" for i in range(20)]
    s1, s2 = make_splits(ids), make_splits(ids)
    assert s1 == s2
    assert not (set(s1["train"]) & set(s1["test"]))
    assert set(s1["train"]) | set(s1["val"]) | set(s1["test"]) == set(ids)


def test_label_frame_to_pose_idx_applies_offset_and_step_when_fps_match():
    # start_offset_s=600, orig_fps=30, step=2, labels_fps=orig_fps -> ratio=1, so this
    # matches the pre-I1 simpler formula's numbers exactly.
    sidecar = {"start_offset_s": 600}
    meta = {"orig_fps": 30.0, "step": 2}
    assert label_frame_to_pose_idx(18000, sidecar, meta, labels_fps=30.0) == 0.0
    assert label_frame_to_pose_idx(18010, sidecar, meta, labels_fps=30.0) == 5.0


def test_label_frame_to_pose_idx_rescales_when_labels_fps_differs_from_orig_fps():
    # CONTROLLER RULING I1: hit_frame is rescaled by orig_fps/labels_fps BEFORE the
    # offset/step are applied. orig_fps=30, labels_fps=25 -> ratio=1.2.
    sidecar = {"start_offset_s": 0}
    meta = {"orig_fps": 30.0, "step": 1}
    assert label_frame_to_pose_idx(100, sidecar, meta, labels_fps=25.0) == 120.0


def test_label_frame_to_pose_idx_missing_sidecar_defaults_offset_zero():
    meta = {"orig_fps": 30.0, "step": 2}
    # No sidecar (None) -> offset 0, warns rather than raising.
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        idx = label_frame_to_pose_idx(20, None, meta, labels_fps=30.0)
        assert idx == 10.0
        assert len(w) == 1


def test_check_fps_consistency_warns_on_large_relative_mismatch():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        check_fps_consistency({"orig_fps": 30.0}, labels_fps=25.0, match_id="m1")
        assert len(w) == 1


def test_check_fps_consistency_silent_within_relative_tolerance():
    # 30 vs 29.97 (diff=0.03) is within the ruled literal tolerance (0.002*30=0.06) --
    # this documents the ruled formula's actual behavior (it does NOT, in fact, flag
    # the real-world 29.97-vs-30 nominal-rate gap; see build_windows.py's
    # check_fps_consistency docstring and task-8-report.md "Fix round 1" for the
    # flagged discrepancy against the ruling's parenthetical claim).
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        check_fps_consistency({"orig_fps": 30.0}, labels_fps=29.97, match_id="m1")
        assert len(w) == 0


def test_build_rally_frame_labels_union_of_overlapping_intervals_inclusive_end():
    # Two overlapping intervals should union, not double count or error. Inclusive-end
    # convention (M1 RULING): (0,5) covers indices 0..5, (3,8) covers 3..8 -> union 0..8.
    ry = build_rally_frame_labels(10, [(0, 5), (3, 8)])
    assert ry.tolist() == [1, 1, 1, 1, 1, 1, 1, 1, 1, 0]


def test_build_rally_frame_labels_clamps_out_of_range_ends_inclusive():
    # Interval end far beyond n_frames, and interval start far below 0, must clamp
    # instead of raising or wrapping via negative-index slicing semantics. Inclusive-end:
    # (-1000,3) covers 0..3, (7,10000) covers 7..9.
    ry = build_rally_frame_labels(10, [(-1000, 3), (7, 10_000)])
    assert ry.tolist() == [1, 1, 1, 1, 0, 0, 0, 1, 1, 1]


def test_main_wiring_offset_and_hitter_selection_end_to_end():
    # ~20-line integration test tying label_frame_to_pose_idx (frame alignment) and
    # build_stroke_samples (hitter selection) together the way main() actually wires
    # them, without needing a real npz/argparse invocation.
    sidecar = {"start_offset_s": 600}
    meta = {"orig_fps": 30.0, "step": 2}
    labels_fps = 30.0
    pose_idx = label_frame_to_pose_idx(18100, sidecar, meta, labels_fps)
    assert pose_idx == 50.0

    T = 200
    kpts = _fixed_torso_kpts(T)
    kpts[:, 1, 0, 1] = 0.1 * np.sin(np.arange(T))  # slot 1 is the mover
    scores = np.ones((T, 2, 17), dtype=np.float32)
    labels = pd.DataFrame({
        "hit_frame": [pose_idx], "player": [1], "stroke": ["clear"], "fps": [labels_fps],
    })
    rng = np.random.default_rng(0)
    X, y, n_skipped = build_stroke_samples(labels, kpts, scores, fps_scale=1.0, rng=rng)
    assert n_skipped == 0
    positive = X[y == ALL_CLASSES.index("clear")][0]
    assert np.array_equal(positive, stroke_window(kpts[:, 1], 50))
