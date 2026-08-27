"""Pose normalization and windowing features, shared by training and serving.

Contracts (load-bearing for downstream tasks 8 and 15):

- `normalize_pose`: hip-centers and torso-scales a single `(17,2)` COCO-17
  pose so it is translation- and scale-invariant. Output is `(17,2)` float32.

- `stroke_window`: builds a fixed-size `(WINDOW, FEAT_DIM)` = `(30, 68)`
  float32 feature window around a center frame. Each row is
  `[pos(34) | vel(34)]`, i.e. the first 34 columns are the frame's flattened
  normalized `(17,2)` position and the last 34 are the frame-to-frame
  velocity (difference from the previous row in the window; the first row's
  velocity is 0). The window is *past-heavy*: it spans `w // 2` frames before
  `center` through `w - w//2 - 1` frames after (for w=30: 15 back, center, 14
  forward, with `center` itself falling at window index `w // 2`).
  Out-of-range frame indices (before frame 0 or past the last frame) clamp to
  the nearest valid edge frame, so near a clip boundary multiple window rows
  read the same source frame (constant position, zero velocity) until the
  index enters the valid range.

  **`stroke_window` does NOT consult confidence scores at all** -- it will
  happily build a window from an all-zero (absent/unconfident) player slot if
  handed one; it has no way to know a slot is meaningless. Callers that pick
  *which* of a match's 2 pose slots to window (e.g.
  `training/build_windows.py`'s hitter-selection logic) MUST gate on
  `scores`/presence themselves *before* calling this, never after.

- `rally_frame_features`: for a `(T,2,17,2)` keypoints array and a
  `(T,2,17)` confidence-scores array, returns `(T,4)` float32 with column
  order `[energy_p0, energy_p1, conf_p0, conf_p1]`, where `energy_p*` is the
  per-frame mean absolute frame-to-frame displacement of that player's
  normalized keypoints (0.0 at frame 0, since there is no previous frame),
  and `conf_p*` is that player's mean keypoint confidence for the frame.
"""
from __future__ import annotations
import numpy as np

WINDOW = 30
FEAT_DIM = 68
HIP_L, HIP_R, SH_L, SH_R = 11, 12, 5, 6
COCO_EDGES = [
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12),
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6),
]

def normalize_pose(kpts: np.ndarray) -> np.ndarray:
    center = (kpts[HIP_L] + kpts[HIP_R]) / 2.0
    shoulder_mid = (kpts[SH_L] + kpts[SH_R]) / 2.0
    torso = float(np.linalg.norm(shoulder_mid - center))
    scale = torso if torso > 1e-6 else 1.0
    return ((kpts - center) / scale).astype(np.float32)

def stroke_window(kpts_seq: np.ndarray, center: int, w: int = WINDOW) -> np.ndarray:
    T = kpts_seq.shape[0]
    idx = np.clip(np.arange(center - w // 2, center + w - w // 2), 0, T - 1)
    norm = np.stack([normalize_pose(kpts_seq[i]) for i in idx])     # (w,17,2)
    pos = norm.reshape(w, -1)                                       # (w,34)
    vel = np.diff(pos, axis=0, prepend=pos[:1])                     # (w,34)
    return np.concatenate([pos, vel], axis=1).astype(np.float32)    # (w,68)

def rally_frame_features(kpts_all: np.ndarray, scores_all: np.ndarray) -> np.ndarray:
    T, P = kpts_all.shape[0], kpts_all.shape[1]
    out = np.zeros((T, 2 * P), dtype=np.float32)
    for p in range(P):
        norm = np.stack([normalize_pose(kpts_all[t, p]) for t in range(T)])
        disp = np.abs(np.diff(norm, axis=0)).mean(axis=(1, 2))      # (T-1,)
        out[1:, p] = disp
        out[:, P + p] = scores_all[:, p].mean(axis=1)
    return out
