"""labels.parquet + pose npz files -> training tensors (stroke windows, rally-frame
labels) with deterministic match-level splits.

Consumes: `training/data/processed/labels.parquet` (Task 6 schema -- see
`training/notes/shuttleset-format.md` "Output contract" for the leakage warnings: this
module never touches `rally_winner`, only uses `rally_start_frame`/`rally_end_frame` as
rally-boundary metadata for the *play/no-play* rally-frame tensor, not as a per-stroke
input feature), and per-match pose npz files (Task 7 contract:
`kpts:(T,2,17,2) float32`, `scores:(T,2,17) float32`, `meta` a JSON string readable via
`json.loads(str(npz["meta"]))`).

**Hitter selection (CONTROLLER RULING C1, replaces `labels.player`-indexed slot lookup).**
`labels.parquet.player` encodes MATCH identity (`0` = the player who eventually wins the
whole match, per the Task 6 notes -- fixed for the entire match) while a pose npz's slot
axis (`kpts[:, 0]` vs `kpts[:, 1]`) encodes per-frame COURT DEPTH (`assign_players`'s
slot 0 = nearer/bottom of frame *this frame*, Task 7). These are NOT the same thing and
are not even stably related: real badminton players change ends between games (and after
game-deciding intervals), so "match winner" can be the near-court player in one rally and
the far-court player in another rally of the very same match. Indexing `kpts[:,
labels.player]` (the original brief's approach, kept through this task's first pass)
silently mixes up which slot's pose actually gets windowed for a stroke whenever ends
have flipped since the match's start -- a real correctness bug, not merely a leakage
concern, because it can hand the model a window of the *non-hitting* player's pose
labeled with the hitting player's stroke class.

Fix: `build_stroke_samples` no longer reads `labels.player` for slot selection at all
(the column stays in `labels.parquet` purely as stroke-event metadata). Instead, for
each labeled hit at pose-frame index `f`, it asks the POSE DATA ITSELF which slot was
the hitter, via `_slot_signal`:

- `presence[p]` = slot `p`'s mean keypoint confidence *at frame f* (`scores[f, p].mean()`)
  -- "is this slot even a visible player at the moment of the hit". A slot must clear
  `PRESENCE_THR` (0.3, matching `assign_players`'s own score gate in `extract_poses.py`)
  to be eligible at all.
- `energy[p]` = slot `p`'s mean |frame-to-frame delta| of `normalize_pose`-normalized
  keypoints, pooled over the local window `[f - 3, f + 3]` (clamped to the valid frame
  range) -- "how much is this slot moving right around the hit", used as a cheap proxy
  for "this is the player who just swung" (the non-hitter is typically far more static
  at the instant of contact).

The **hitter slot** = `argmax(energy)` among the slots clearing `PRESENCE_THR`. If NO
slot clears presence at `f` (both players absent/undetected -- e.g. a broadcast cutaway
at the exact hit frame), or the hitter's emitted window would be literally all-zero
(defensive: catches a presence-gate false-pass on data that's still degenerate), the hit
is SKIPPED entirely -- no positive is emitted for it, and it is counted in the returned
skip count rather than silently producing a corrupted or empty-but-labeled sample.

The **non-hitter slot's** window at the same frame `f` is ALSO emitted, as a `none`
negative, whenever *that* slot independently clears `PRESENCE_THR` -- this mirrors
inference time, where the running classifier sees both players' windows every frame and
must correctly say "none" for whichever one didn't just hit. These "other-slot" negatives
count toward the 1:1 negative budget; any remaining budget (positives emitted minus
other-slot negatives already emitted) is filled by uniformly-random frames that stay more
than `NEG_GUARD_FRAMES` away from every considered hit frame (this old fallback keeps
the previous behavior for negative diversity/coverage away from stroke contexts entirely,
not just away from a specific player's stroke).

**Frame alignment (I1 RULING, generalizes the Task 7 carry-over).** The pose npz's frame
axis is *clip-relative sampled frames*, not the original video's frame numbering that
`labels.parquet.hit_frame` uses (itself expressed in the label's own per-match
`resolved_fps`, see notes §(e)). If the source video for a match is a trimmed clip, the
clip's start offset within the full match video is recorded only in the provenance
sidecar `training/data/raw/videos/<match_id>.json` (`{url, download_section,
start_offset_s}`), never in the npz itself. `label_frame_to_pose_idx()` implements the
full mapping:

    pose_idx = (hit_frame * (orig_fps / labels_fps) - start_offset_s * orig_fps) / step

The `orig_fps / labels_fps` ratio first rescales `hit_frame` (expressed in the *label's*
resolved fps) into the pose npz's own `orig_fps` domain, before the clip offset (itself
in `orig_fps` units) is subtracted and the sampling `step` divided out. When
`labels_fps == orig_fps` the ratio is exactly 1 and this reduces to the simpler
`(hit_frame - start_offset_s * orig_fps) / step` used before this generalization --
mismatches between the two are exactly what `check_fps_consistency` warns about.

If the sidecar file is missing for a match, we assume `start_offset_s = 0` (i.e. the
pose npz covers the video from its true start) and emit a `UserWarning` rather than
raising -- a match without a sidecar is not necessarily an error (offset really may be
zero), but silently guessing zero without any signal is worth flagging.

`build_stroke_samples()` keeps the simpler `fps_scale`-only signature from the original
task brief (a single multiplicative scale from label-frame-units to pose-frame-units,
no offset) so it stays trivially unit-testable without needing a sidecar fixture. The
full offset/ratio-aware mapping happens in `main()`, which pre-transforms each match's
`hit_frame`/`rally_start_frame`/`rally_end_frame` columns through
`label_frame_to_pose_idx()` before calling `build_stroke_samples()` with `fps_scale=1.0`
(i.e. the column already holds pose-frame-unit values, so the multiply is a no-op).

**Rally interval overlap + inclusive end (Task 6 carry-over + M1 RULING).** Task 6's
padded rally windows (`rally_start_frame`/`rally_end_frame`, padded +/-15/+30 raw frames
around each rally's first/last hit) can overlap between adjacent rallies, and
`rally_end_frame` has no upper clamp against the video's actual frame count.
`build_rally_frame_labels()` treats overlapping intervals as a union (a pose frame is
`play=1` if it falls in *any* rally interval), explicitly clamps both interval ends into
`[0, n_frames)` rather than relying on Python's slice-index wraparound semantics for
out-of-range bounds, and treats **both endpoints as INCLUSIVE**: the pose frame at
exactly `round(end)` is itself considered "in play" (not the first out-of-play frame
after it), since `rally_end_frame` names the last padded frame of the rally, not an
exclusive boundary. Internally this means the half-open python slice used to set `1.0`
is built as `[start, end + 1)`, not `[start, end)`.

**Leakage contract**: `rally_winner` is never read by this module. Only
`rally_start_frame`/`rally_end_frame` (rally-level, but safe as *boundary* metadata --
see the notes file's leakage warning #2 for why `rally_winner`/`rally_end_frame` must
never become a per-stroke input feature; we only use `rally_end_frame` here to build a
frame-level play/no-play mask, not as a feature value itself) are consulted.
`labels.player` is likewise never used for anything except as passthrough metadata
tagging which stroke-event row a class label came from -- it plays no role in slot
selection (see "Hitter selection" above).

Produces (see `training/dvc.yaml` for the pipeline wiring):
- `training/data/processed/stroke_windows.npz`: `X:(N,30,68) float32`,
  `y:(N,) int64` (index into `ALL_CLASSES`), `match:(N,) <U...>`. Negatives (`none`)
  are a mix of "the non-hitter's window at a real hit frame" (see "Hitter selection")
  and uniformly-random frames at least `NEG_GUARD_FRAMES` from every hit, targeting a
  1:1 ratio with positives.
- `training/data/processed/rally_frames.npz`: per-match concatenated `X:(M,4) float32`,
  `y:(M,) float32` (1.0 inside any rally interval, union + inclusive-end semantics),
  `match:(M,) <U...>`.
- `training/data/processed/splits.json`: `{"train": [...], "val": [...], "test": [...]}`
  -- 70/15/15 by match, seed 13, sorted then shuffled deterministically.

**Deviation from the task brief**: the brief's `match` dtype was `<U32`, but real
ShuttleSet `match_id` folder names run up to 85 characters -- a fixed `<U32` silently
truncates them (verified against the real labels.parquet). `main()` sizes the `match`
column's dtype from `max(len(match_id))` over the actual labels dataframe instead of
hardcoding 32, so no match_id is ever truncated regardless of dataset naming length.

**Correction (I4 RULING) to this module's Fix-round-0 report claim**: an earlier version
of `task-8-report.md` claimed zero-keypoint player slots were "already gated" by
`stroke_window`/`rally_frame_features` before this fix round. That claim was FALSE:
`stroke_window` has never consulted `scores` at all (see the corrected docstring note in
`core/shuttlesense_core/features.py`), and `rally_frame_features` only *surfaces*
`conf_p*` as an output column -- it does not exclude/gate anything using it. The actual,
real mitigation for "don't window an absent player" is the presence gate introduced in
this fix round (`PRESENCE_THR`, in `_slot_signal`/`build_stroke_samples` above); before
this round there was no presence gating anywhere in this module.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import warnings
import zlib

import numpy as np
import pandas as pd

from shuttlesense_core.features import (
    FEAT_DIM,
    WINDOW,
    normalize_pose,
    rally_frame_features,
    stroke_window,
)
from shuttlesense_core.schemas import ALL_CLASSES, NONE_CLASS

DEFAULT_VIDEOS_DIR = "training/data/raw/videos"
FPS_CONSISTENCY_ABS_TOL = 1e-3  # absolute fps tolerance, frames/sec (ADJUSTED I1 RULING)
NEG_GUARD_FRAMES = 15  # >= ~1s at 15fps sampled; random negatives must stay this far from every hit
PRESENCE_THR = 0.3  # matches extract_poses.assign_players's own score gate
ENERGY_HALF_WINDOW = 3  # +/- frames around a hit used for the hitter-selection energy signal

NONE_ID = ALL_CLASSES.index(NONE_CLASS)


def _slot_signal(kpts: np.ndarray, scores: np.ndarray, f: int, half_window: int = ENERGY_HALF_WINDOW):
    """Per-slot `(energy, presence)` signal at pose-frame `f`, used by
    `build_stroke_samples` to pick which pose slot actually hit the shuttle (see module
    docstring, "Hitter selection").

    - `presence[p]`: slot `p`'s mean keypoint confidence AT frame `f` itself (not the
      window) -- "is this slot even a real, visible player at the moment of the hit".
    - `energy[p]`: slot `p`'s mean |frame-to-frame delta| of `normalize_pose`-normalized
      keypoints, pooled over consecutive frame pairs in `[f - half_window, f +
      half_window]` (clamped to `[0, T - 1]`) -- "how much is this slot moving around the
      hit". Falls back to `0.0` if the clamped window collapses to a single frame (no
      consecutive pair to diff).

    Returns `(energy, presence)`, each a length-`P` list of floats (`P = kpts.shape[1]`,
    normally 2 per the Task 7 contract).
    """
    T, P = kpts.shape[0], kpts.shape[1]
    lo = max(0, f - half_window)
    hi = min(T - 1, f + half_window)
    energy = []
    for p in range(P):
        if hi > lo:
            norm = np.stack([normalize_pose(kpts[t, p]) for t in range(lo, hi + 1)])
            energy.append(float(np.abs(np.diff(norm, axis=0)).mean()))
        else:
            energy.append(0.0)
    presence = [float(scores[f, p].mean()) for p in range(P)]
    return energy, presence


def build_stroke_samples(
    labels: pd.DataFrame, kpts: np.ndarray, scores: np.ndarray, fps_scale: float, rng
):
    """Positive stroke windows (signal-selected hitter slot, from `labels`) + `none`
    negatives (non-hitter's window at the same hit frame, plus random guard-distance
    fill), targeting a 1:1 positive:negative ratio.

    `labels.hit_frame` is mapped into a pose-frame index via `round(hit_frame *
    fps_scale)` -- see the module docstring for why `main()` pre-transforms hit_frame
    into pose-frame units (including the fps ratio and clip offset) and calls this with
    `fps_scale=1.0`; this function itself stays offset-free and simple for direct unit
    testing. `labels.player` is NOT read (see module docstring, "Hitter selection") --
    the hitter slot is picked from `kpts`/`scores` via `_slot_signal`, independent of
    which match-identity id happened to be recorded on that stroke-event row.

    Returns `(X, y, n_skipped)`: `X:(N,30,68) float32`, `y:(N,) int64`, and `n_skipped`
    (int) counting hits that were dropped because no slot cleared `PRESENCE_THR` at that
    frame, or the selected hitter's window was degenerately all-zero. If zero positives
    end up emitted, returns correctly-shaped empty arrays (no negatives are added either)
    rather than raising on `np.stack([])`. Warns (does not raise) if the random-fill
    negative-sampling loop can't reach the 1:1 target within its try budget.
    """
    T, P = kpts.shape[0], kpts.shape[1]
    Xs, ys, hit_idx = [], [], []
    n_skipped = 0
    for _, r in labels.iterrows():
        f = int(round(r["hit_frame"] * fps_scale))
        if not (0 <= f < T):
            continue
        hit_idx.append(f)
        energy, presence = _slot_signal(kpts, scores, f)
        eligible = [p for p in range(P) if presence[p] >= PRESENCE_THR]
        if not eligible:
            n_skipped += 1
            continue
        hitter = max(eligible, key=lambda p: energy[p])
        window = stroke_window(kpts[:, hitter], f)
        if not np.any(window):
            n_skipped += 1
            continue
        Xs.append(window)
        ys.append(ALL_CLASSES.index(r["stroke"]))
        if P == 2:
            other = 1 - hitter
            if presence[other] >= PRESENCE_THR:
                Xs.append(stroke_window(kpts[:, other], f))
                ys.append(NONE_ID)

    n_pos = sum(1 for y in ys if y != NONE_ID)
    n_already_neg = sum(1 for y in ys if y == NONE_ID)
    n_neg_needed = max(0, n_pos - n_already_neg)
    guard, tries = NEG_GUARD_FRAMES, 0
    while n_neg_needed > 0 and tries < 10000:
        f = int(rng.integers(0, T))
        tries += 1
        if all(abs(f - h) > guard for h in hit_idx):
            p = int(rng.integers(0, P))
            Xs.append(stroke_window(kpts[:, p], f))
            ys.append(NONE_ID)
            n_neg_needed -= 1
    if n_neg_needed > 0:
        warnings.warn(
            f"build_stroke_samples: could not sample {n_neg_needed} more guard-distance "
            f"negative(s) after {tries} tries (T={T}, {len(hit_idx)} hit(s) to avoid, "
            f"guard={guard}) -- returning fewer negatives than the 1:1 target",
            stacklevel=2,
        )

    if not Xs:
        return (
            np.zeros((0, WINDOW, FEAT_DIM), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            n_skipped,
        )
    return np.stack(Xs), np.asarray(ys, dtype=np.int64), n_skipped


def label_frame_to_pose_idx(
    hit_frame: float, sidecar: dict | None, meta: dict, labels_fps: float
) -> float:
    """Map an original-video-relative label frame number into a (fractional, caller
    rounds) index into the pose npz's clip-relative sampled-frame axis.

        pose_idx = (hit_frame * (orig_fps / labels_fps) - start_offset_s * orig_fps) / step

    (CONTROLLER RULING I1: generalizes the previous offset-only formula with an
    `orig_fps / labels_fps` rescale, so a genuine fps mismatch between the label's
    resolved fps and the pose npz's `orig_fps` doesn't silently misalign frames. When
    the two agree exactly, the ratio is 1 and this is numerically identical to the
    simpler pre-I1 formula.)

    `sidecar` is the parsed `training/data/raw/videos/<match_id>.json` provenance dict
    (or `None` if that file doesn't exist for this match, in which case
    `start_offset_s` is assumed to be 0 and a `UserWarning` is emitted -- a genuinely
    un-trimmed source video legitimately has offset 0, but guessing that without any
    sidecar signal is worth flagging rather than silently assuming). `meta` is the pose
    npz's `meta` dict (must have `orig_fps` and `step`). `labels_fps` is the label's own
    per-match resolved fps (`labels.parquet.fps`, see notes §(e)).
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
    ratio = orig_fps / float(labels_fps)
    return (float(hit_frame) * ratio - start_offset_s * orig_fps) / step


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
    per-match fps by more than `FPS_CONSISTENCY_ABS_TOL` (ADJUSTED CONTROLLER RULING I1,
    second pass -- an ABSOLUTE tolerance, replacing both the original fixed 0.1-frame
    absolute tolerance AND the first-pass relative-tolerance attempt, whose `0.002 *
    orig_fps` constant turned out to be arithmetically looser than the real 29.97-vs-30
    gap it was meant to catch, per Fix round 1's flagged discrepancy).

    Rationale for why an absolute (not relative) tolerance is correct here, and why it
    can be this tight: `label_frame_to_pose_idx`'s `orig_fps / labels_fps` ratio term
    (CONTROLLER RULING I1) already makes the hit-frame -> pose-index MAPPING exact for
    *any* labels/video fps pair -- correctness of frame alignment does not depend on
    `orig_fps` and `labels_fps` agreeing. This check is therefore purely INFORMATIONAL:
    it flags an *unexpected* disagreement between the label-derived fps and the pose
    npz's measured fps (e.g. a wrong sidecar, a mismatched video file, or a fps-
    resolution bug upstream), not a correctness precondition for the mapping itself.
    Because it's just a diagnostic signal, a small ABSOLUTE tolerance (`1e-3` fps) is
    appropriate: it's comfortably above float round-off noise (`orig_fps`/`labels_fps`
    are typically snapped to 2-3 decimal places, see notes §(e)) while still catching
    every real nominal-rate mismatch in this dataset -- both the small 29.97-vs-30 gap
    (`|30 - 29.97| = 0.03 > 1e-3`) and a full 25-vs-30 mix-up (`5.0 > 1e-3`).
    """
    orig_fps = float(meta["orig_fps"])
    if abs(orig_fps - labels_fps) > FPS_CONSISTENCY_ABS_TOL:
        warnings.warn(
            f"match {match_id!r}: pose meta orig_fps={orig_fps} disagrees with labels "
            f"fps={labels_fps} by more than {FPS_CONSISTENCY_ABS_TOL} fps -- frame "
            "alignment for this match may be unreliable",
            stacklevel=2,
        )


def build_rally_frame_labels(n_frames: int, intervals) -> np.ndarray:
    """`(n_frames,) float32` play/no-play mask: 1.0 wherever a pose frame falls inside
    ANY `(start, end)` interval in `intervals` (union semantics -- overlapping rally
    windows are not assumed disjoint). Each interval's ends are independently clamped to
    `[0, n_frames]` (rounding to the nearest int) rather than relying on Python's
    negative-index slice wraparound, which would silently misbehave for a very negative
    start or end.

    **Interval end convention: INCLUSIVE** (CONTROLLER RULING M1) -- both `start` and
    `end` themselves are considered "in play", matching `rally_end_frame`'s semantics as
    the last padded frame of the rally, not an exclusive boundary one-past it.
    Implemented as `end_index = round(end) + 1` before clamping, to convert to Python's
    half-open slice convention.
    """
    ry = np.zeros(n_frames, dtype=np.float32)
    for s, e in intervals:
        si = max(0, int(round(s)))
        ei = min(n_frames, int(round(e)) + 1)
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


def _match_rng(seed: int, match_id: str) -> np.random.Generator:
    """Independent RNG per match, derived deterministically from `(seed, match_id)` via
    a stable string hash (`zlib.crc32`, NOT Python's built-in `hash()`, which is
    randomized per-process by default and would make this non-reproducible across runs).
    Minor hygiene fix (M3): a single `rng` threaded across all matches in a loop makes
    each match's negative sampling depend on how many random draws every *previous*
    match in the loop consumed -- i.e. adding/removing/reordering an unrelated match's
    npz file would silently change another match's sampled negatives. Per-match RNGs
    make each match's output reproducible in isolation."""
    h = zlib.crc32(match_id.encode("utf-8"))
    return np.random.default_rng([seed, h])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", default="training/data/processed/labels.parquet")
    ap.add_argument("--poses", default="training/data/processed/poses")
    ap.add_argument("--videos-dir", default=DEFAULT_VIDEOS_DIR)
    ap.add_argument("--out-dir", default="training/data/processed")
    ap.add_argument("--seed", type=int, default=13)
    a = ap.parse_args()

    labels = pd.read_parquet(a.labels)
    SX, Sy, Sm, RX, Ry, Rm = [], [], [], [], [], []
    total_skipped = 0

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
        rng = _match_rng(a.seed, mid)

        pose_hit_frame = ml["hit_frame"].apply(
            lambda hf: label_frame_to_pose_idx(hf, sidecar, meta, labels_fps)
        )
        ml_pose = ml.assign(hit_frame=pose_hit_frame)
        X, y, n_skipped = build_stroke_samples(
            ml_pose, z["kpts"], z["scores"], fps_scale=1.0, rng=rng
        )
        total_skipped += n_skipped
        n_pos = int((y != NONE_ID).sum())
        n_neg = int((y == NONE_ID).sum())
        print(
            f"[build_windows] {mid}: {len(ml)} labels in match, pose clip has "
            f"{z['kpts'].shape[0]} sampled frames -> {n_pos} positive + {n_neg} "
            f"negative stroke windows ({n_skipped} hits skipped: no eligible slot / "
            "degenerate window)"
        )
        SX.append(X)
        Sy.append(y)
        Sm.append(np.full(len(y), mid, dtype=match_dtype))

        rf = rally_frame_features(z["kpts"], z["scores"])
        rally_bounds = ml.drop_duplicates("rally_id")
        intervals = [
            (
                label_frame_to_pose_idx(r["rally_start_frame"], sidecar, meta, labels_fps),
                label_frame_to_pose_idx(r["rally_end_frame"], sidecar, meta, labels_fps),
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
        f"({labels.match_id.nunique()} matches), {total_skipped} total hits skipped "
        f"-> {a.out_dir}"
    )


if __name__ == "__main__":
    main()
