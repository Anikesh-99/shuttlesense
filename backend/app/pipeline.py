"""Video -> `MatchReport` + tracks payload, the analysis pipeline over the ONNX
models exported by `training/export_onnx.py` (`backend/models/*.onnx` +
`backend/models/manifest.json`).

`backend/` MUST NOT import `torch` or `training.*` (serving is ONNX Runtime CPU
only, per the global constraints); every feature/smoothing/schema function used
here is imported from `shuttlesense_core` -- the same functions
`training/build_windows.py` uses to build the training tensors -- so
train/serve skew is structurally impossible (enforced by
`tests/test_consistency.py::test_train_and_serve_import_same_feature_functions`,
which asserts `pipeline.stroke_window is shuttlesense_core.features.stroke_window`
etc.).

Rally inference contract (RULED, see `training/train_rally.py`'s module
docstring ~L24-49): `RallyGRU` was only ever trained on fixed-size (512-frame)
chunks produced by `training.train_rally.chunk`'s PAD-ALWAYS-WITH-MASK policy
(the final partial chunk of a sequence is zero-padded up to exactly one more
full chunk, never a ragged remainder), with the bidirectional GRU's hidden
state reset at every chunk boundary. A full-sequence single forward pass is
NOT equivalent to training conditions and must not be used. `chunk()` itself
operates on torch tensors and lives in `training/`, which this module must not
import, so `_chunk_frames` below is a pure-numpy re-implementation of the exact
same pad+mask logic (padding computation only -- no randomness, no model code),
cross-checked against the torch original by
`backend/tests/test_pipeline.py::test_chunk_frames_matches_training_chunk`
(that test file, not this module, is allowed to import torch).

Rally threshold: read from `manifest.json`'s `"rally"."threshold"` (0.6) at
`analyze()` call time -- never hardcoded, never re-derived here (RULED).

Pose extraction: `extract_poses_onnx` below is a self-contained copy of
`training/extract_poses.py`'s rtmlib-driven `extract()`/`assign_players()`
(backend must not import `training.extract_poses`). The player-assignment rule
is the score-gated lowest-visible-keypoint ("feet") depth heuristic from that
module's CURRENT implementation (slot 0 = nearer/bottom of frame, by the
per-person max-y among keypoints clearing `score>=0.3`, falling back to
ungated max-y if none clear it) -- NOT a plain mean-y rule. A zeroed slot means
"no player detected this frame", not "detected at the origin"; callers must
gate on `scores` before trusting `kpts` for a slot (mirrored here via
`PRESENCE_THR`, matching `training/build_windows.py`'s own gate).

Per-frame slot identity is NOT persistent/tracked across frames (no player
re-identification in Phase 1) -- slot 0 vs slot 1 at frame `t` need not be the
same physical player as slot 0 vs slot 1 at frame `t + 1`. This is acceptable
for Phase 1 inference because stroke windows are evaluated per-slot per-frame
(the model only ever sees one player's own trajectory at a time, never a
cross-slot comparison), and the rally frame-features / court-side identity
question is out of scope for Phase 1 (see the spec-deviation note in the plan's
global constraints re: court-relative features being a Phase 2 feature-flagged
addition, currently off).

rtmlib install note: `backend/requirements.txt` deliberately omits `rtmlib`
(it pulls in a GUI-flavored opencv that conflicts with
`opencv-python-headless`, already required by FastAPI's static/video-duration
code); the Docker image (Task 20) installs it separately with `--no-deps`
after the headless opencv is in place. Locally, it's already present in the
dev venv (installed for `training/`'s own pose-extraction step).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

from shuttlesense_core.features import COCO_EDGES, rally_frame_features, stroke_window
from shuttlesense_core.schemas import (
    ALL_CLASSES,
    NONE_CLASS,
    MatchReport,
    RallyInterval,
    StrokeEvent,
)
from shuttlesense_core.smoothing import probs_to_intervals, suppress_events

RALLY_CHUNK_SIZE = 512  # must match training/configs/rally_gru.yaml's `chunk` (512)
PRESENCE_THR = 0.3  # matches training/extract_poses.assign_players's own score gate
                     # and training/build_windows.py's PRESENCE_THR


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _chunk_frames(X: np.ndarray, size: int = RALLY_CHUNK_SIZE):
    """Numpy re-implementation of `training.train_rally.chunk`'s
    PAD-ALWAYS-WITH-MASK policy (X-only; that function also chunks a parallel
    `y` label array, which doesn't exist at inference time). See the module
    docstring's "Rally inference contract" for why this must not simply import
    `training.train_rally.chunk` (it's torch-based, and lives in `training/`,
    which `backend/` must not import) and how its numpy-vs-torch parity is
    tested.

    Splits a single contiguous `(T, F)` frame sequence into non-overlapping
    `size`-frame chunks, zero-padding the final partial chunk up to exactly
    one more full chunk (never dropping/shortening a remainder). Returns
    `(Xc, mask)`: `Xc` is `(n_chunks, size, F)` float32, `mask` is
    `(n_chunks, size)` float32 (1.0 = real frame, 0.0 = pad). Raises
    `ValueError` on an empty input, mirroring the torch original.
    """
    if len(X) == 0:
        raise ValueError(
            "_chunk_frames received an empty sequence (0 frames) -- nothing to chunk"
        )
    n_full = len(X) // size
    remainder = len(X) - n_full * size
    if remainder > 0:
        pad = size - remainder
        Xp = np.pad(X, ((0, pad), (0, 0)))
        mask = np.ones(len(Xp), dtype=np.float32)
        mask[len(X):] = 0.0
        n_chunks = n_full + 1
    else:
        Xp = X
        mask = np.ones(len(Xp), dtype=np.float32)
        n_chunks = n_full
    Xc = Xp.reshape(n_chunks, size, X.shape[1]).astype(np.float32)
    return Xc, mask.reshape(n_chunks, size)


def assign_players(kpts_people: np.ndarray, scores_people: np.ndarray, score_thr: float = 0.3):
    """(N,17,2),(N,17) for N detected people -> ((2,17,2),(2,17)).

    Self-contained copy of `training/extract_poses.assign_players`'s CURRENT
    (score-gated feet-depth) logic -- see that module's docstring for the full
    rationale (mean-y ordering was replaced because a lunging/diving player's
    keypoint centroid can rise even while their feet stay planted at the
    bottom of the frame). Kept in lockstep here only because `backend/` must
    not import `training.*`; do not let this drift from the original without
    updating both.

    Person selection: top-2 by mean confidence (`argsort` stable, so ties
    break by original detection order). Depth ordering: per selected person,
    `depth = max(y over keypoints with score >= score_thr)` (falls back to the
    ungated `max(y)` if no keypoint clears the threshold); larger depth
    (nearer/bottom of frame) gets slot 0. Pads with zeros if fewer than 2
    people are detected -- a zeroed slot means "absent", not "at the origin".
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


def extract_poses_onnx(video_path: str, target_fps: float = 15.0):
    """Sample `video_path` at ~`target_fps` and run 2-person pose extraction on
    each sampled frame via rtmlib's `Body` (RTMPose, COCO-17), ONNX Runtime CPU
    backend. Self-contained sibling of `training/extract_poses.extract` (see
    the module docstring for why this isn't a shared import).

    Returns `(kpts, scores, meta)`: `kpts` is `(T, 2, 17, 2) float32`, `scores`
    is `(T, 2, 17) float32`, `meta` is a dict with `fps_sampled, orig_fps,
    width, height, n_frames, step, n_source_frames`.
    """
    if target_fps <= 0:
        raise ValueError(f"target_fps must be > 0, got {target_fps!r}")
    import cv2
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


def analyze(
    video_path: str,
    models_dir: str,
    target_fps: float = 15.0,
    pose_fn=None,
) -> tuple[MatchReport, dict]:
    """Run the full analysis pipeline: pose extraction -> rally segmentation
    (RallyGRU, chunked per the rally inference contract) -> per-rally,
    per-player stroke classification (StrokeTCN, sliding windows, stride 4) ->
    `MatchReport` + a JSON-serializable tracks payload
    (`{"fps", "edges", "kpts", "scores"}`, keypoints rounded to 1 decimal and
    scores to 2 for JSON size).

    `pose_fn` defaults to `extract_poses_onnx`; tests inject a fake
    `(video_path, target_fps) -> (kpts, scores, meta)` to avoid running rtmlib.

    Raises `ValueError("no rallies detected")` if the rally model finds no
    play interval clearing the manifest threshold/min-length -- callers (the
    worker) are expected to translate this into a user-facing message.
    """
    pose_fn = pose_fn or extract_poses_onnx

    # Manifest read before pose_fn so a missing/broken models_dir fails fast
    # rather than after the (potentially expensive, real-rtmlib)
    # pose-extraction pass. ONNX sessions, deliberately, are NOT constructed
    # here -- they're built further below, AFTER the zero-frame guard, so a
    # 0-frame pose extraction can raise the "no rallies detected" friendly
    # path (see below) without requiring backend/models/*.onnx to even be
    # present.
    manifest = json.loads((Path(models_dir) / "manifest.json").read_text())
    rally_threshold = manifest["rally"]["threshold"]  # RULED: never hardcode/re-derive

    kpts, scores, meta = pose_fn(video_path, target_fps)
    fps = float(meta["fps_sampled"])
    if kpts.shape[0] == 0:
        # Hoisted ABOVE rally_frame_features on purpose: that function's
        # per-player np.stack([...]) over an empty frame range raises its own
        # (unfriendly) ValueError before ever reaching a T==0 check placed
        # after it -- a 0-frame pose extraction must surface the same "no
        # rallies detected" contract callers (the worker) already handle.
        raise ValueError("no rallies detected")

    rally_sess = ort.InferenceSession(str(Path(models_dir) / "rally_gru.onnx"))
    stroke_sess = ort.InferenceSession(str(Path(models_dir) / "stroke_tcn.onnx"))

    rf = rally_frame_features(kpts, scores)  # (T,4)
    T = rf.shape[0]

    Xc, mask = _chunk_frames(rf, size=RALLY_CHUNK_SIZE)
    chunk_logits = rally_sess.run(None, {"x": Xc})[0]  # (n_chunks, size)
    chunk_probs = _sigmoid(chunk_logits)
    real = mask.reshape(-1) > 0
    probs = chunk_probs.reshape(-1)[real]  # (T,) -- pad-position outputs discarded
    if len(probs) != T:
        # Not a bare assert: a bare `assert` is stripped under `python -O`,
        # which would let a reassembly bug silently corrupt downstream
        # interval detection instead of failing loudly.
        raise RuntimeError(
            f"chunked rally probs length {len(probs)} != T={T} -- "
            "_chunk_frames reassembly bug"
        )

    intervals = probs_to_intervals(
        probs, threshold=rally_threshold, min_len=int(fps), merge_gap=int(fps / 2)
    )
    if not intervals:
        raise ValueError("no rallies detected")

    events = []
    for s, e in intervals:
        for player in (0, 1):
            wins, kept_frames = [], []
            for f in range(s, e, 4):
                # presence gate mirrors training/build_windows.py's PRESENCE_THR
                # gate: an absent/undetected slot's window is never scored.
                if scores[f, player].mean() < PRESENCE_THR:
                    continue
                wins.append(stroke_window(kpts[:, player], f))
                kept_frames.append(f)
            if not wins:
                continue
            logits = stroke_sess.run(None, {"x": np.stack(wins).astype(np.float32)})[0]
            p = np.exp(logits - logits.max(axis=1, keepdims=True))
            p /= p.sum(axis=1, keepdims=True)
            for f, pr in zip(kept_frames, p):
                c = int(pr.argmax())
                if ALL_CLASSES[c] != NONE_CLASS:
                    events.append(
                        {
                            "frame": f,
                            "player": player,
                            "stroke": ALL_CLASSES[c],
                            "confidence": float(pr[c]),
                        }
                    )
    events = suppress_events(events, min_gap=int(fps / 3))

    report = MatchReport(
        fps=fps,
        width=int(meta["width"]),
        height=int(meta["height"]),
        n_frames=int(meta["n_frames"]),
        rallies=[RallyInterval(int(s), int(e)) for s, e in intervals],
        strokes=[
            StrokeEvent(**{k: ev[k] for k in ("frame", "player", "stroke", "confidence")})
            for ev in events
        ],
    )
    tracks = {
        "fps": fps,
        "edges": COCO_EDGES,
        # Round in float64, not float32: np.round on a float32 array keeps
        # float32's nearest-representable-value error (e.g. 90.4 stored as
        # 90.400001526...), which .tolist()'s float32->python-float widening
        # then exposes verbatim (json.dumps prints "90.4000015258789" instead
        # of "90.4", ~3x the payload size for no reason). Casting to float64
        # BEFORE rounding lets np.round produce the float64 value closest to
        # the intended decimal, which reprs/serializes cleanly.
        "kpts": np.round(kpts.astype(np.float64), 1).tolist(),
        "scores": np.round(scores.astype(np.float64), 2).tolist(),
    }
    return report, tracks
