"""labels.parquet + pose npz files -> training tensors (stroke windows, rally-frame
labels) with deterministic match-level splits.

Consumes: `training/data/processed/labels.parquet` (Task 6 schema -- see
`training/notes/shuttleset-format.md` "Output contract" for the leakage warnings: this
module never touches `rally_winner`, only uses `rally_start_frame`/`rally_end_frame` as
rally-boundary metadata for the *play/no-play* rally-frame tensor, not as a per-stroke
input feature), and per-match pose npz files (Task 7 contract:
`kpts:(T,2,17,2) float32`, `scores:(T,2,17) float32`, `meta` a JSON string readable via
`json.loads(str(npz["meta"]))`).

**Frame alignment (controller carry-over from Task 7).** The pose npz's frame axis is
*clip-relative sampled frames*, not the original video's frame numbering that
`labels.parquet.hit_frame` uses. If the source video for a match is a trimmed clip (as
the one verification clip in this repo is), the clip's start offset within the full
match video is recorded only in the provenance sidecar
`training/data/raw/videos/<match_id>.json` (`{url, download_section, start_offset_s}`),
never in the npz itself. `label_frame_to_pose_idx()` implements the full mapping:

    pose_idx = (hit_frame - start_offset_s * orig_fps) / step

If the sidecar file is missing for a match, we assume `start_offset_s = 0` (i.e. the
pose npz covers the video from its true start) and emit a `UserWarning` rather than
raising -- a match without a sidecar is not necessarily an error (offset really may be
zero), but silently guessing zero without any signal is worth flagging.

`build_stroke_samples()` keeps the simpler `fps_scale`-only signature from the original
task brief (a single multiplicative scale from label-frame-units to pose-frame-units,
no offset) so it stays trivially unit-testable without needing a sidecar fixture. The
full offset-aware mapping happens in `main()`, which pre-transforms each match's
`hit_frame`/`rally_start_frame`/`rally_end_frame` columns through
`label_frame_to_pose_idx()` before calling `build_stroke_samples()` with `fps_scale=1.0`
(i.e. the column already holds pose-frame-unit values, so the multiply is a no-op).

**Rally interval overlap (controller carry-over from Task 6).** Task 6's padded rally
windows (`rally_start_frame`/`rally_end_frame`, padded +/-15/+30 raw frames around each
rally's first/last hit) can overlap between adjacent rallies, and `rally_end_frame` has
no upper clamp against the video's actual frame count. `build_rally_frame_labels()`
treats overlapping intervals as a union (a pose frame is `play=1` if it falls in *any*
rally interval) and explicitly clamps both interval ends into `[0, n_frames)` rather
than relying on Python's slice-index wraparound semantics for out-of-range bounds.

**Leakage contract**: `rally_winner` is never read by this module. Only
`rally_start_frame`/`rally_end_frame` (rally-level, but safe as *boundary* metadata --
see the notes file's leakage warning #2 for why `rally_winner`/`rally_end_frame` must
never become a per-stroke input feature; we only use `rally_end_frame` here to build a
frame-level play/no-play mask, not as a feature value itself) are consulted.

Produces (see `training/dvc.yaml` for the pipeline wiring):
- `training/data/processed/stroke_windows.npz`: `X:(N,30,68) float32`,
  `y:(N,) int64` (index into `ALL_CLASSES`), `match:(N,) <U...>`. Negatives (`none`)
  sampled 1:1 with positives, at frames >= a guard distance away from every hit.
- `training/data/processed/rally_frames.npz`: per-match concatenated `X:(M,4) float32`,
  `y:(M,) float32` (1.0 inside any rally interval, union semantics), `match:(M,) <U...>`.
- `training/data/processed/splits.json`: `{"train": [...], "val": [...], "test": [...]}`
  -- 70/15/15 by match, seed 13, sorted then shuffled deterministically.

**Deviation from the task brief**: the brief's `match` dtype was `<U32`, but real
ShuttleSet `match_id` folder names run up to 85 characters -- a fixed `<U32` silently
truncates them (verified against the real labels.parquet). `main()` sizes the `match`
column's dtype from `max(len(match_id))` over the actual labels dataframe instead of
hardcoding 32, so no match_id is ever truncated regardless of dataset naming length.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import warnings

import numpy as np
import pandas as pd

from shuttlesense_core.features import FEAT_DIM, WINDOW, rally_frame_features, stroke_window
from shuttlesense_core.schemas import ALL_CLASSES, NONE_CLASS

DEFAULT_VIDEOS_DIR = "training/data/raw/videos"
FPS_CONSISTENCY_TOL = 0.1
NEG_GUARD_FRAMES = 15  # >= ~1s at 15fps sampled; negatives must stay this far from every hit


def build_stroke_samples(labels: pd.DataFrame, kpts: np.ndarray, fps_scale: float, rng):
    """Positive stroke windows (from `labels`) + 1:1 sampled `none` negatives.

    `labels.hit_frame` is mapped into a pose-frame index via `round(hit_frame *
    fps_scale)` -- see the module docstring for why `main()` pre-transforms hit_frame
    into pose-frame units and calls this with `fps_scale=1.0` for the offset-aware path;
    this function itself stays offset-free and simple for direct unit testing.

    Returns `(X, y)`: `X:(N,30,68) float32`, `y:(N,) int64`. If zero labels fall inside
    the valid `[0, T)` pose-frame range, returns correctly-shaped empty arrays (no
    negatives are sampled either, since 1:1 with zero positives is zero) rather than
    raising on `np.stack([])`.
    """
    T = kpts.shape[0]
    Xs, ys, hit_idx = [], [], []
    for _, r in labels.iterrows():
        f = int(round(r["hit_frame"] * fps_scale))
        if not (0 <= f < T):
            continue
        Xs.append(stroke_window(kpts[:, int(r["player"])], f))
        ys.append(ALL_CLASSES.index(r["stroke"]))
        hit_idx.append(f)
    n_neg, guard = len(Xs), NEG_GUARD_FRAMES
    tries = 0
    while n_neg > 0 and tries < 10000:
        f = int(rng.integers(0, T))
        tries += 1
        if all(abs(f - h) > guard for h in hit_idx):
            p = int(rng.integers(0, 2))
            Xs.append(stroke_window(kpts[:, p], f))
            ys.append(ALL_CLASSES.index(NONE_CLASS))
            n_neg -= 1
    if not Xs:
        return np.zeros((0, WINDOW, FEAT_DIM), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return np.stack(Xs), np.asarray(ys, dtype=np.int64)


def label_frame_to_pose_idx(hit_frame: float, sidecar: dict | None, meta: dict) -> float:
    """Map an original-video-relative label frame number into a (fractional, caller
    rounds) index into the pose npz's clip-relative sampled-frame axis.

        pose_idx = (hit_frame - start_offset_s * orig_fps) / step

    `sidecar` is the parsed `training/data/raw/videos/<match_id>.json` provenance dict
    (or `None` if that file doesn't exist for this match, in which case
    `start_offset_s` is assumed to be 0 and a `UserWarning` is emitted -- a genuinely
    un-trimmed source video legitimately has offset 0, but guessing that without any
    sidecar signal is worth flagging rather than silently assuming). `meta` is the pose
    npz's `meta` dict (must have `orig_fps` and `step`).
    """
    if sidecar is None:
        warnings.warn(
            "label_frame_to_pose_idx: no provenance sidecar given -- assuming "
            "start_offset_s=0 (unverified)",
            stacklevel=2,
        )
        start_offset_s = 0.0
    else:
        start_offset_s = float(sidecar.get("start_offset_s", 0.0))
    orig_fps = float(meta["orig_fps"])
    step = float(meta["step"])
    return (float(hit_frame) - start_offset_s * orig_fps) / step


def load_sidecar(match_id: str, videos_dir: str = DEFAULT_VIDEOS_DIR) -> dict | None:
    """Load `<videos_dir>/<match_id>.json`, or return None if it doesn't exist (handled
    by `label_frame_to_pose_idx` as offset=0 + warning)."""
    path = os.path.join(videos_dir, f"{match_id}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def check_fps_consistency(meta: dict, labels_fps: float, match_id: str) -> None:
    """Warn (do not raise) if the pose npz's `orig_fps` disagrees with the label-derived
    per-match fps by more than `FPS_CONSISTENCY_TOL` -- a mismatch means the video
    reconciliation described in `training/notes/shuttleset-format.md`
    "Label-to-video alignment" hasn't held for this match, and frame-level alignment
    downstream should be treated with suspicion."""
    orig_fps = float(meta["orig_fps"])
    if abs(orig_fps - labels_fps) > FPS_CONSISTENCY_TOL:
        warnings.warn(
            f"match {match_id!r}: pose meta orig_fps={orig_fps} disagrees with labels "
            f"fps={labels_fps} by more than {FPS_CONSISTENCY_TOL} -- frame alignment "
            "for this match may be unreliable",
            stacklevel=2,
        )


def build_rally_frame_labels(n_frames: int, intervals) -> np.ndarray:
    """`(n_frames,) float32` play/no-play mask: 1.0 wherever a pose frame falls inside
    ANY `(start, end)` interval in `intervals` (union semantics -- overlapping rally
    windows are not assumed disjoint). Each interval's ends are independently clamped to
    `[0, n_frames]` (rounding to the nearest int) rather than relying on Python's
    negative-index slice wraparound, which would silently misbehave for a very negative
    start or end."""
    ry = np.zeros(n_frames, dtype=np.float32)
    for s, e in intervals:
        si = max(0, int(round(s)))
        ei = min(n_frames, int(round(e)))
        if si < ei:
            ry[si:ei] = 1.0
    return ry


def make_splits(match_ids, seed: int = 13) -> dict[str, list[str]]:
    """70/15/15 match-level split: sort ids for determinism, then shuffle with a fixed
    seed (so re-running produces the identical split, and no match ever leaks across
    train/val/test since the split key is the whole match, not a stroke/frame)."""
    ids = sorted(match_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n = len(ids)
    a, b = int(n * 0.7), int(n * 0.85)
    return {"train": ids[:a], "val": ids[a:b], "test": ids[b:]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", default="training/data/processed/labels.parquet")
    ap.add_argument("--poses", default="training/data/processed/poses")
    ap.add_argument("--videos-dir", default=DEFAULT_VIDEOS_DIR)
    ap.add_argument("--out-dir", default="training/data/processed")
    a = ap.parse_args()

    labels = pd.read_parquet(a.labels)
    rng = np.random.default_rng(13)
    SX, Sy, Sm, RX, Ry, Rm = [], [], [], [], [], []

    # Real ShuttleSet match_id folder names run up to 85 chars (e.g.
    # "Hans-Kristian_Solberg_Vittinghus_Lee_Cheuk_Yu_TOYOTA_THAILAND_OPEN_2021_QuarterFinals")
    # -- the brief's literal `<U32` dtype silently truncates any match_id longer than 32
    # chars (verified: it truncated a real 70-char id mid-name), which would corrupt any
    # downstream match-level lookup against splits.json. Size the dtype from the actual
    # data instead of hardcoding a width that's too small for this dataset.
    match_dtype = f"<U{max((len(m) for m in labels.match_id.unique()), default=1)}"

    npz_paths = sorted(glob.glob(f"{a.poses}/*.npz"))
    for npz_path in npz_paths:
        mid = os.path.basename(npz_path).removesuffix(".npz")
        ml = labels[labels.match_id == mid]
        if ml.empty:
            print(f"[build_windows] {mid}: no labels for this match_id, skipping npz")
            continue

        z = np.load(npz_path, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        labels_fps = float(ml["fps"].iloc[0])
        check_fps_consistency(meta, labels_fps, mid)
        sidecar = load_sidecar(mid, a.videos_dir)

        pose_hit_frame = ml["hit_frame"].apply(
            lambda hf: label_frame_to_pose_idx(hf, sidecar, meta)
        )
        ml_pose = ml.assign(hit_frame=pose_hit_frame)
        X, y = build_stroke_samples(ml_pose, z["kpts"], fps_scale=1.0, rng=rng)
        none_id = ALL_CLASSES.index(NONE_CLASS)
        n_pos = int((y != none_id).sum())
        n_neg = int((y == none_id).sum())
        print(
            f"[build_windows] {mid}: {len(ml)} labels in match, pose clip has "
            f"{z['kpts'].shape[0]} sampled frames -> {n_pos} positive + {n_neg} "
            f"negative stroke windows"
        )
        SX.append(X)
        Sy.append(y)
        Sm.append(np.full(len(y), mid, dtype=match_dtype))

        rf = rally_frame_features(z["kpts"], z["scores"])
        rally_bounds = ml.drop_duplicates("rally_id")
        intervals = [
            (
                label_frame_to_pose_idx(r["rally_start_frame"], sidecar, meta),
                label_frame_to_pose_idx(r["rally_end_frame"], sidecar, meta),
            )
            for _, r in rally_bounds.iterrows()
        ]
        ry = build_rally_frame_labels(len(rf), intervals)
        RX.append(rf)
        Ry.append(ry)
        Rm.append(np.full(len(ry), mid, dtype=match_dtype))

    if not SX:
        SX, Sy, Sm = [np.zeros((0, WINDOW, FEAT_DIM), dtype=np.float32)], [np.zeros((0,), dtype=np.int64)], [np.zeros((0,), dtype=match_dtype)]
    if not RX:
        RX, Ry, Rm = [np.zeros((0, 4), dtype=np.float32)], [np.zeros((0,), dtype=np.float32)], [np.zeros((0,), dtype=match_dtype)]

    os.makedirs(a.out_dir, exist_ok=True)
    np.savez_compressed(
        f"{a.out_dir}/stroke_windows.npz",
        X=np.concatenate(SX), y=np.concatenate(Sy), match=np.concatenate(Sm),
    )
    np.savez_compressed(
        f"{a.out_dir}/rally_frames.npz",
        X=np.concatenate(RX), y=np.concatenate(Ry), match=np.concatenate(Rm),
    )
    with open(f"{a.out_dir}/splits.json", "w") as f:
        json.dump(make_splits(labels.match_id.unique().tolist()), f, indent=2)

    print(
        f"[build_windows] wrote stroke_windows.npz (N={len(np.concatenate(Sy))}), "
        f"rally_frames.npz (M={len(np.concatenate(Ry))}), splits.json "
        f"({labels.match_id.nunique()} matches) -> {a.out_dir}"
    )


if __name__ == "__main__":
    main()
