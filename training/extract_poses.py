"""Video -> per-frame 2-player keypoint tracks, via rtmlib's `Body` (RTMPose, COCO-17).

See `training/notes/shuttleset-format.md` ("Label-to-video alignment") for why the fps
actually reported by the downloaded video stream must be reconciled against the
label-derived `resolved_fps` before any frame-level alignment is trusted -- this module
only extracts poses at a target sampling rate; it does not attempt that reconciliation
itself (that belongs to whatever consumes `meta['fps_sampled']`/`meta['orig_fps']`, e.g.
Task 15's backend pipeline).

rtmlib return-convention notes (verified empirically against rtmlib 0.0.16, not just
read from source, before writing `extract()` -- see task-7-report.md for the full
investigation):
- `Body(mode="balanced", backend="onnxruntime", device="cpu")(frame)` returns
  `(keypoints, scores)` with `keypoints.shape == (N, 17, 2)` (dtype float64) and
  `scores.shape == (N, 17)` (dtype float32), where N is the number of detected people.
  This already matches `assign_players`'s documented input shape -- no reshaping needed,
  only a float32 cast on keypoints before returning (to satisfy the npz `float32` contract).
- **N is never 0 in practice**: rtmlib's underlying `RTMPose.__call__` defaults to a
  full-frame bounding box (`[[0, 0, w, h]]`) whenever the upstream detector (`YOLOX`,
  a person-only "humanart" detector) finds zero people, so pose estimation always runs
  on at least one (possibly bogus, low-confidence) box. `assign_players`'s zero-people
  padding branch is exercised defensively (and is unit-tested), but is not expected to
  trigger during real video decoding.
"""
from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np


def assign_players(kpts_people: np.ndarray, scores_people: np.ndarray):
    """(N,17,2),(N,17) for N detected people -> ((2,17,2),(2,17)).

    Picks the 2 highest-confidence people; player 0 = larger mean y (nearer camera,
    i.e. the bottom player in a typical broadcast badminton camera angle).
    Pads with zeros if fewer than 2 people are detected.
    """
    out_k = np.zeros((2, 17, 2), dtype=np.float32)
    out_s = np.zeros((2, 17), dtype=np.float32)
    if len(kpts_people) == 0:
        return out_k, out_s
    order = np.argsort(-scores_people.mean(axis=1))[:2]
    chosen = sorted(order, key=lambda i: -kpts_people[i, :, 1].mean())  # bottom first
    for slot, i in enumerate(chosen):
        out_k[slot], out_s[slot] = kpts_people[i], scores_people[i]
    return out_k, out_s


def extract(video_path: str, target_fps: float = 15.0):
    """Sample `video_path` at ~`target_fps` and run 2-person pose extraction on each
    sampled frame.

    Returns `(kpts, scores, meta)`:
    - `kpts`: `(T, 2, 17, 2) float32`
    - `scores`: `(T, 2, 17) float32`
    - `meta`: dict with `fps_sampled, orig_fps, width, height, n_frames`
    """
    from rtmlib import Body  # deferred: avoids paying model-download cost at import time

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"could not open video: {video_path}")
    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(int(round(orig_fps / target_fps)), 1)
    body = Body(mode="balanced", backend="onnxruntime", device="cpu")
    kpts_l, scores_l, i = [], [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            people_k, people_s = body(frame)
            k2, s2 = assign_players(
                np.asarray(people_k, dtype=np.float32), np.asarray(people_s, dtype=np.float32)
            )
            kpts_l.append(k2)
            scores_l.append(s2)
        i += 1
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    meta = dict(
        fps_sampled=orig_fps / step,
        orig_fps=orig_fps,
        width=w,
        height=h,
        n_frames=len(kpts_l),
    )
    if not kpts_l:
        return (
            np.zeros((0, 2, 17, 2), dtype=np.float32),
            np.zeros((0, 2, 17), dtype=np.float32),
            meta,
        )
    return (
        np.stack(kpts_l).astype(np.float32),
        np.stack(scores_l).astype(np.float32),
        meta,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video")
    ap.add_argument("--out-dir", default="training/data/processed/poses")
    ap.add_argument("--fps", type=float, default=15.0)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    k, s, m = extract(a.video, a.fps)
    mid = os.path.basename(a.video).rsplit(".", 1)[0]
    np.savez_compressed(f"{a.out_dir}/{mid}.npz", kpts=k, scores=s, meta=json.dumps(m))
    print(f"{mid}: kpts={k.shape} scores={s.shape} @ {m['fps_sampled']:.2f}fps (orig {m['orig_fps']:.2f}fps, {m['width']}x{m['height']})")


if __name__ == "__main__":
    main()
