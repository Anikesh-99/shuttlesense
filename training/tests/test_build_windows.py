import json
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
    main,
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


def test_random_negatives_never_come_from_a_sub_threshold_confidence_slot():
    # Pin the random-fill negative-sampling loop's presence gate (build_windows.py's
    # module docstring, "Hitter selection" / Fix round 2 item 1) SPECIFICALLY --
    # not confounded with the separate all-zero-window guard also present in that
    # loop. Slot 1 has PLAUSIBLE, NON-ZERO geometry (same torso fixture as slot 0,
    # plus a large, decodable, non-zero nose_x offset) but sub-threshold confidence
    # scores (0.1, below PRESENCE_THR=0.3) everywhere -- i.e. "a real-looking pose
    # the tracker just wasn't confident about", not an absent/degenerate slot. Only
    # the presence check can reject it: `np.any(window)` would happily pass slot 1's
    # window since it's non-zero. If the presence check were deleted, slot 1 draws
    # would sometimes be accepted (rng.integers(0,2) picks either slot ~50/50 with no
    # other rejection reason), which this test's decoding would catch.
    #
    # Verified empirically (not just by inspection) that this test fails if the
    # presence check is disabled: monkeypatching `build_windows.PRESENCE_THR` to a
    # value below every score in this fixture (so the presence gate never rejects
    # anything, equivalent to deleting the check) makes this test fail on the
    # decoded-slot-1-leaked assertion below, with the same seed=0 RNG used here.
    rng = np.random.default_rng(0)
    T = 600
    kpts = _fixed_torso_kpts(T)
    kpts[:, 0, 0, 0] = 1.0  # slot 0: nose_x pinned to a small, distinguishable constant
    kpts[:, 1, 0, 0] = 100_000.0  # slot 1: plausible/non-zero geometry, large decodable marker
    scores = np.zeros((T, 2, 17), dtype=np.float32)
    scores[:, 0, :] = 1.0  # slot 0: always confidently present
    scores[:, 1, :] = 0.1  # slot 1: non-zero pose, but sub-threshold confidence everywhere
    # Several hits, all necessarily from slot 0 (slot 1 never clears presence), so the
    # random-fill loop must supply several negatives -- more draws means a check-removed
    # regression is overwhelmingly likely to be caught (each draw is an independent
    # ~50/50 shot at slot 1 once the presence gate stops filtering it out).
    hits = [100, 200, 300, 400, 500]
    labels = pd.DataFrame({
        "hit_frame": hits, "player": [0] * 5, "stroke": ["smash"] * 5, "fps": [30.0] * 5,
    })
    X, y, n_skipped = build_stroke_samples(labels, kpts, scores, fps_scale=1.0, rng=rng)
    assert n_skipped == 0
    positives = X[y != NONE_ID]
    assert len(positives) == 5  # all 5 hits resolved to slot 0 (the only eligible slot)
    negatives = X[y == NONE_ID]
    assert len(negatives) == 5  # 1:1 budget; slot 1 never clears presence for an
    # "other slot at hit" negative either, so all 5 must come from the random-fill loop
    for neg in negatives:
        decoded_nose_x = neg[WINDOW // 2, 0]  # center-row nose_x (position column 0)
        assert decoded_nose_x < 100_000.0, (
            "a random negative was emitted from the sub-threshold-confidence slot "
            "(slot 1) despite its plausible, non-zero geometry -- the presence gate "
            "did not reject it"
        )


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


def test_label_frame_to_pose_idx_rejects_non_positive_or_nan_labels_fps():
    # NIT fix (Fix round 2, item 3): a zero/negative/NaN labels_fps would otherwise
    # silently produce an infinite/NaN/sign-flipped ratio instead of failing loudly.
    sidecar = {"start_offset_s": 0}
    meta = {"orig_fps": 30.0, "step": 1}
    for bad in (0.0, -25.0, float("nan")):
        try:
            label_frame_to_pose_idx(100, sidecar, meta, labels_fps=bad)
            assert False, f"expected ValueError for labels_fps={bad!r}"
        except ValueError:
            pass


def test_check_fps_consistency_warns_on_large_absolute_mismatch():
    # A full 25-vs-30 nominal-rate mix-up (diff=5.0) must warn under the ADJUSTED I1
    # ruling's absolute tolerance (1e-3).
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        check_fps_consistency({"orig_fps": 30.0}, labels_fps=25.0, match_id="m1")
        assert len(w) == 1


def test_check_fps_consistency_warns_on_2997_vs_30_snap_mismatch():
    # ADJUSTED I1 ruling: the absolute tolerance (1e-3) is tight enough to actually
    # catch the real 29.97-vs-30 nominal-rate gap (diff=0.03), unlike Fix round 1's
    # first-pass relative tolerance (0.002*orig_fps=0.06), which was looser than this
    # gap and silently missed it -- see build_windows.py's check_fps_consistency
    # docstring and task-8-report.md "Fix round 1" / "Fix round 2" for that history.
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        check_fps_consistency({"orig_fps": 30.0}, labels_fps=29.97, match_id="m1")
        assert len(w) == 1


def test_check_fps_consistency_silent_within_float_noise():
    # A tiny sub-tolerance difference (well below 1e-3), representative of float
    # round-off rather than a genuine label/video fps disagreement, must stay silent.
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        check_fps_consistency({"orig_fps": 30.0}, labels_fps=30.0 + 1e-5, match_id="m1")
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


def test_main_writes_expected_window_via_real_npz_and_sidecar(tmp_path):
    # Fix round 2, item 2: a REAL end-to-end test -- builds a synthetic pose npz,
    # provenance sidecar, and labels.parquet on disk in a tmp dir, invokes main()
    # itself (via an explicit argv list, not sys.argv), and asserts the WRITTEN
    # stroke_windows.npz contains the expected window: hit_frame=18100, offset=600s,
    # step=2, orig_fps=30 -> pose_idx=50, and slot 1 (the mover) is the hitter.
    mid = "Some_Match_Id"
    T = 200
    kpts = _fixed_torso_kpts(T)
    kpts[:, 1, 0, 1] = 0.1 * np.sin(np.arange(T))  # slot 1 is the mover -> real hitter
    scores = np.ones((T, 2, 17), dtype=np.float32)
    meta = {
        "fps_sampled": 15.0, "orig_fps": 30.0, "width": 1280, "height": 720,
        "n_frames": T, "step": 2, "n_source_frames": T * 2,
    }

    poses_dir = tmp_path / "poses"
    poses_dir.mkdir()
    np.savez_compressed(
        poses_dir / f"{mid}.npz", kpts=kpts, scores=scores, meta=json.dumps(meta)
    )

    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    (videos_dir / f"{mid}.json").write_text(json.dumps({"start_offset_s": 600}))

    labels_path = tmp_path / "labels.parquet"
    pd.DataFrame({
        "match_id": [mid], "video_file": [f"{mid}.mp4"], "fps": [30.0],
        "rally_id": [1], "hit_frame": [18100], "player": [1], "stroke": ["clear"],
        "rally_start_frame": [18070], "rally_end_frame": [18130], "rally_winner": [1],
    }).to_parquet(labels_path, index=False)

    out_dir = tmp_path / "out"
    main([
        "--labels", str(labels_path),
        "--poses", str(poses_dir),
        "--videos-dir", str(videos_dir),
        "--out-dir", str(out_dir),
    ])

    written = np.load(out_dir / "stroke_windows.npz")
    assert written["match"][0] == mid
    clear_id = ALL_CLASSES.index("clear")
    positives = written["X"][written["y"] == clear_id]
    assert len(positives) == 1
    assert np.array_equal(positives[0], stroke_window(kpts[:, 1], 50))
