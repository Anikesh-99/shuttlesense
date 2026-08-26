import numpy as np
from shuttlesense_core.features import (
    FEAT_DIM, WINDOW, normalize_pose, rally_frame_features, stroke_window,
)

def _pose(cx=100.0, cy=200.0, torso=50.0):
    # unset keypoints default to (cx, cy), not the origin, so that whole-pose
    # translation (varying cx/cy) is a rigid shift of every keypoint.
    k = np.tile(np.array([cx, cy], dtype=np.float32), (17, 1))
    k[11] = [cx - 10, cy]; k[12] = [cx + 10, cy]          # hips
    k[5] = [cx - 12, cy - torso]; k[6] = [cx + 12, cy - torso]  # shoulders
    k[9] = [cx - 30, cy - torso - 20]                      # a wrist
    return k

def _scale_pose(kpts, cx, cy, factor):
    # Scale every keypoint's offset from (cx, cy) by `factor`, keeping (cx, cy)
    # fixed. Since (cx, cy) is exactly the hip midpoint for `_pose()` output,
    # this scales the whole pose (including torso length) uniformly around the
    # hip center, so normalize_pose's output should be unchanged.
    origin = np.array([cx, cy], dtype=np.float32)
    return (kpts - origin) * factor + origin

def test_normalize_translation_and_scale_invariant():
    a = normalize_pose(_pose(100, 200, 50))
    b = normalize_pose(_pose(500, 900, 50))
    c = normalize_pose(_scale_pose(_pose(100, 200, 50), 100, 200, 2.0))
    np.testing.assert_allclose(a, b, atol=1e-5)          # translation invariant
    np.testing.assert_allclose(a, c, atol=1e-5)          # scale invariant
    hip_mid = (a[11] + a[12]) / 2
    np.testing.assert_allclose(hip_mid, [0, 0], atol=1e-5)  # centered

def test_stroke_window_shape_and_edge_clamp():
    # Vary torso (articulation), not cx/cy, so consecutive frames are actually
    # distinguishable after normalize_pose (which is translation-invariant).
    seq = np.stack([_pose(100, 200, 50 + t) for t in range(40)])  # (40,17,2)
    w = stroke_window(seq, center=2)                          # near left edge
    assert w.shape == (WINDOW, FEAT_DIM)
    assert w.dtype == np.float32

    # Window layout: idx = clip(arange(center - w//2, center + w - w//2), 0, T-1).
    # For center=2, w=30: arange(-13, 17) -> raw values -13..16 (30 values).
    # Values -13..0 (14 of them) all clip to 0, so window rows 0..13 (14 rows)
    # all read frame 0; row 14 onward reads raw index 1, 2, 3, ... (unclamped).
    n_clamped = 14
    pos_a, vel_a = w[:, :34], w[:, 34:]
    # Clamped region: every row is exactly frame 0's normalized pose ...
    assert np.allclose(pos_a[:n_clamped], pos_a[0])
    # ... and frame-to-frame velocity is zero throughout the clamped region.
    assert np.allclose(vel_a[:n_clamped], 0.0)
    # First unclamped row (index 0 -> index 1) must show nonzero velocity and
    # a different position, since torso (and hence the normalized shape)
    # changes frame to frame once we're off the clamped edge.
    assert not np.allclose(pos_a[n_clamped], pos_a[0])
    assert not np.allclose(vel_a[n_clamped], 0.0)

def test_rally_features_motion_vs_still():
    # normalize_pose is translation-invariant (see test above), so whole-body
    # translation alone produces no motion energy by design; use a per-frame
    # changing torso (articulation) to simulate genuine pose motion instead.
    still = np.stack([_pose()] * 20)
    moving = np.stack([_pose(100, 200, 50 + 3 * t) for t in range(20)])
    kpts = np.stack([moving, still], axis=1)              # (20,2,17,2)
    scores = np.ones((20, 2, 17), dtype=np.float32)
    f = rally_frame_features(kpts, scores)
    assert f.shape == (20, 4)
    assert f[5:, 0].mean() > f[5:, 1].mean() + 1e-4        # mover has more energy
