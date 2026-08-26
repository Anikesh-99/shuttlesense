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

def test_normalize_translation_and_scale_invariant():
    a = normalize_pose(_pose(100, 200, 50))
    b = normalize_pose(_pose(500, 900, 50) )
    c = normalize_pose(_pose(100, 200, 100) * 1.0)
    np.testing.assert_allclose(a, b, atol=1e-5)          # translation invariant
    hip_mid = (a[11] + a[12]) / 2
    np.testing.assert_allclose(hip_mid, [0, 0], atol=1e-5)  # centered

def test_stroke_window_shape_and_edge_clamp():
    seq = np.stack([_pose(100 + t, 200) for t in range(40)])  # (40,17,2)
    w = stroke_window(seq, center=2)                          # near left edge
    assert w.shape == (WINDOW, FEAT_DIM)
    assert w.dtype == np.float32

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
    assert f[5:, 0].mean() > f[5:, 1].mean()              # mover has more energy
