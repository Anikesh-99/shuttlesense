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
