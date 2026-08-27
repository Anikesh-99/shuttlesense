"""Video -> per-frame 2-player keypoint tracks, via rtmlib's `Body` (RTMPose, COCO-17).

See `training/notes/shuttleset-format.md` ("Label-to-video alignment") for why the fps
actually reported by the downloaded video stream must be reconciled against the
label-derived `resolved_fps` before any frame-level alignment is trusted -- this module
only extracts poses at a target sampling rate; it does not attempt that reconciliation
itself (that belongs to whatever consumes `meta['fps_sampled']`/`meta['orig_fps']`, e.g.
Task 15's backend pipeline). **`meta` alone cannot map an npz frame index back to the
original video's `frame_num` labels** -- if the source video is a trimmed segment (as the
one verification clip in this repo is, via `yt-dlp --download-sections`), the segment's
start offset within the *full* match video is only recorded in the sidecar
`training/data/raw/videos/<match_id>.json` (`{url, download_section, start_offset_s}`),
not in the npz. A consumer must add `start_offset_s * orig_fps` to any in-clip frame index
before comparing it to a label's `frame_num`.

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

**Reading `meta` back out of the npz**: `np.savez_compressed` stores the `json.dumps(m)`
string as a 0-d `numpy.ndarray` of dtype `<U...>` (object/unicode scalar), not a plain str
-- `json.loads(npz["meta"])` will fail. Use either `json.loads(str(npz["meta"]))` or
`json.loads(npz["meta"].item())`.

**Zero-padded slots are common, not rare** (controller-ruling follow-up, "Fix round 1"):
on the one verification clip in this repo (900 sampled frames, 60s @ 15fps), slot 1
(the second player) was entirely zero (`scores[:, 1] == 0` for all 17 keypoints) in
~18% of sampled frames -- almost entirely broadcast cutaways (close-ups, replays,
crowd shots) that contain 0 or 1 real court players, not real rally frames with a
missed detection. **A zeroed slot means "absent this frame", not "detected at pixel
(0,0)"** -- any consumer must gate on `scores` (e.g. `scores[t, slot].mean() > 0`) before
trusting `kpts[t, slot]`, never assume both slots are populated for every frame.
"""
from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np


def assign_players(kpts_people: np.ndarray, scores_people: np.ndarray, score_thr: float = 0.3):
    """(N,17,2),(N,17) for N detected people -> ((2,17,2),(2,17)).

    Person **selection**: the top-2 people by mean confidence across all 17 keypoints
    (`np.argsort(..., kind="stable")`, so ties break by original detection order
    deterministically).

    Depth **ordering** (slot 0 = nearer/bottom, slot 1 = farther/top) -- CONTROLLER
    RULING, post-review, replaces an earlier mean-y rule: per selected person, compute
    `depth = max(y over that person's keypoints with score >= score_thr)`, i.e. the
    y-coordinate of their lowest (screen-space) *visible* keypoint, used as an
    approximation of foot/ground position. If a person has **no** keypoint clearing
    `score_thr` (e.g. a heavily-occluded or low-confidence detection), fall back to the
    ungated `max(y)` over all 17 keypoints so a depth value always exists. The person
    with the larger depth statistic gets slot 0 (nearer/bottom); the other gets slot 1.

    Why not mean-y (the original rule)? Measured on a real 60s broadcast clip (see
    task-7-report.md, "Fix round 1"): a lunging/diving player's keypoint centroid
    (mean y) can rise sharply as limbs spread up and out, even though their feet --
    the actual lowest, most depth-informative visible point -- are still planted at
    the bottom of the frame. This caused mean-y-based ordering to disagree with a
    feet-position-based ordering in 125/736 (~17%) of real two-detection sampled
    frames on the verification clip, including the very first spot-check frame
    reviewed (sample 50: a diving near-court player was mean-y-misclassified as the
    far player). max-y-of-visible-keypoints does not have this failure mode because a
    raised limb never *lowers* the max; it can only be beaten by an even-lower
    (larger-y) visible keypoint on the same person.

    Pads with zeros if fewer than 2 people are detected. **A zeroed output slot means
    "no second player detected this frame", not "detected at pixel (0,0)"** -- always
    gate on the returned `scores` before trusting a slot's `kpts`.
    """
    out_k = np.zeros((2, 17, 2), dtype=np.float32)
    out_s = np.zeros((2, 17), dtype=np.float32)
    if len(kpts_people) == 0:
        return out_k, out_s
    order = np.argsort(-scores_people.mean(axis=1), kind="stable")[:2]

    def depth_stat(i: int) -> float:
        y = kpts_people[i, :, 1]
        visible = y[scores_people[i] >= score_thr]
        return float(visible.max()) if visible.size else float(y.max())

    chosen = sorted(order, key=lambda i: -depth_stat(i))  # largest depth (nearest) first
    for slot, i in enumerate(chosen):
        out_k[slot], out_s[slot] = kpts_people[i], scores_people[i]
    return out_k, out_s


def extract(video_path: str, target_fps: float = 15.0):
    """Sample `video_path` at ~`target_fps` and run 2-person pose extraction on each
    sampled frame.

    Returns `(kpts, scores, meta)`:
    - `kpts`: `(T, 2, 17, 2) float32`
    - `scores`: `(T, 2, 17) float32`
    - `meta`: dict with `fps_sampled, orig_fps, width, height, n_frames, step,
      n_source_frames` (the last two are additive fields, not in the original Task 7
      contract: `step` is the frame-decimation stride actually used, and
      `n_source_frames` is the total number of frames decoded from the source video,
      i.e. `n_frames * step` approximately, modulo the final partial stride).

    See the module docstring for: (a) why `meta` alone is insufficient to map a sampled
    frame index back to a label's `frame_num` for a trimmed clip (needs the sidecar
    provenance JSON), (b) how to read `meta` back out of the saved npz, and (c) why a
    zero-valued player slot means "absent", not "at the origin".
    """
    assert target_fps > 0, "target_fps must be > 0"
    from rtmlib import Body  # deferred: avoids paying model-download cost at import time

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"could not open video: {video_path}")
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    if orig_fps is None or not np.isfinite(orig_fps) or orig_fps <= 0:
        orig_fps = 30.0  # guard: some containers/backends report 0/NaN fps
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
    n_source_frames = i
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    meta = dict(
        fps_sampled=orig_fps / step,
        orig_fps=orig_fps,
        width=w,
        height=h,
        n_frames=len(kpts_l),
        step=step,
        n_source_frames=n_source_frames,
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
