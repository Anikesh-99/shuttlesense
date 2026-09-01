"""Task 19: build the pre-analyzed sample matches that back the zero-action
demo (`GET /api/samples` + `/api/samples/{id}/{report,tracks,video}`, see
`backend/app/routes.py`'s module docstring for the read-side contract).

For each entry in `scripts/samples.yaml` this:
  1. Cuts a `clip_duration_s`-long analysis clip out of the already-downloaded
     `source_video` (itself a *trim of the full YouTube match*, per that
     file's `.json` sidecar -- see `load_clip_offset_s`), starting at
     `clip_start_s` seconds into that downloaded file. The cut is re-encoded
     at the SOURCE's own fps (never resampled) so wall-clock<->frame-index
     correspondence is exact and reproducible.
  2. Runs `backend.app.pipeline.analyze` on that clip -- the exact same
     ONNX-Runtime pipeline `worker.py` runs for a real upload -- producing a
     `MatchReport` + tracks payload.
  3. Patches `winner` onto each detected `RallyInterval` by mapping it back
     to ShuttleSet's `labels.parquet` (see `patch_winners` for the frame-math
     and the RULED end-frame-convention conversion).
  4. Writes `meta.json` (including `players`: `[winner_name, loser_name]`
     real competitor names looked up from `set/match.csv` -- Task 19 Fix
     round 1, see `lookup_player_names`; this is the SCORE-RACE identity,
     deliberately independent of the on-screen skeleton's per-frame
     court-side slot, which has no reliable correspondence to it -- see
     that function's docstring)/`report.json`/`tracks.json`, and a separate
     hosting-weight re-encode of the analysis clip as `video.mp4`
     (<=720p, ~2Mbps, +faststart), verifying via `ffprobe -count_frames`
     that the re-encode didn't drop/duplicate a single frame (a silent
     frame-count drift there would desync the report's frame-index-based
     rally/stroke markers from the served video -- RULED to fail loudly
     instead, see `reencode_for_hosting`).

Run:
    .venv/bin/python scripts/build_samples.py

Idempotent: reruns overwrite each sample's output directory outright (no
partial-skip logic -- these are small, fast, offline builds, unlike
`bulk_extract.py`'s multi-hour rtmlib passes).
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))  # so `backend.app.pipeline` / `shuttlesense_core` import cleanly

from backend.app.pipeline import analyze  # noqa: E402
from shuttlesense_core.schemas import MatchReport  # noqa: E402

DEFAULT_SAMPLES_YAML = REPO_ROOT / "scripts" / "samples.yaml"
DEFAULT_LABELS = REPO_ROOT / "training" / "data" / "processed" / "labels.parquet"
DEFAULT_MATCH_CSV = REPO_ROOT / "training" / "data" / "raw" / "shuttleset" / "set" / "match.csv"
DEFAULT_MODELS_DIR = REPO_ROOT / "backend" / "models"
DEFAULT_OUT_DIR = REPO_ROOT / "backend" / "samples"

# Must match backend.app.config.Settings.target_fps's default -- the pose
# sampling rate `worker.py` uses for a real upload. Sample reports need to be
# built at the same rate a real analysis run would use, or the demo would be
# subtly unrepresentative of what a real upload produces.
TARGET_FPS = 15.0

MIN_RALLIES = 3  # brief's Step 2 acceptance bar ("verify each report has >= 3 rallies")


def _run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"command failed ({proc.returncode}): {' '.join(cmd)}\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )


def ffprobe_fps(path: Path) -> float:
    """Exact source fps via `Fraction("30/1")`-style parsing of ffprobe's
    `r_frame_rate` -- avoids the float-rounding drift a naive `eval`/`float()`
    of e.g. `"24000/1001"` would introduce, which would compound into a
    several-frame `full_match_start_frame` error over a multi-minute offset."""
    proc = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate",
            "-of", "default=noprint_wrappers=1:nokey=1", str(path),
        ],
        capture_output=True, text=True, check=True,
    )
    return float(Fraction(proc.stdout.strip()))


def ffprobe_nb_frames(path: Path) -> int:
    """Exact decoded frame count (NOT the container's often-approximate
    `nb_frames` tag) via `-count_frames`, used both to compute the analysis
    clip's own fps (cross-check) and to verify the hosting re-encode below
    didn't drop/duplicate frames (RULED, see `reencode_for_hosting`)."""
    proc = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
            "-show_entries", "stream=nb_read_frames",
            "-of", "default=noprint_wrappers=1:nokey=1", str(path),
        ],
        capture_output=True, text=True, check=True,
    )
    return int(proc.stdout.strip())


def load_clip_offset_s(source_video: Path) -> float:
    """`source_video`'s `.json` sidecar (written by the Task 12 download
    step) records `start_offset_s`: the FULL-MATCH-video second at which
    this already-downloaded, already-trimmed clip begins. ShuttleSet's
    `labels.parquet` frame numbers are indexed against the full match, not
    this trimmed file, so every downstream frame-number comparison must
    route through this offset."""
    sidecar = source_video.with_suffix(".json")
    meta = json.loads(sidecar.read_text())
    return float(meta["start_offset_s"])


def cut_analysis_clip(source: Path, start_s: float, duration_s: float, fps: float, out_path: Path) -> None:
    """Re-encode a `duration_s`-long window of `source` starting at `start_s`
    into `out_path`, AT THE SOURCE'S OWN FPS (`-r fps`, never resampled) so
    `analysis_clip` frame `j` <-> `start_s + j/fps` seconds into `source`
    holds exactly (no VFR->CFR surprises). `-ss` before `-i` with a
    re-encoded (non-`-c copy`) output is frame-accurate in modern ffmpeg
    (it decodes-and-discards up to the timestamp rather than seeking to the
    nearest keyframe), which matters here because `start_s`/`clip_start_s`
    values were chosen to land close to (not necessarily exactly on) a
    labeled rally boundary."""
    _run([
        "ffmpeg", "-y", "-ss", str(start_s), "-i", str(source), "-t", str(duration_s),
        "-r", str(fps), "-c:v", "libx264", "-crf", "18", "-preset", "veryfast",
        "-pix_fmt", "yuv420p", "-an", str(out_path),
    ])


def reencode_for_hosting(src: Path, dst: Path) -> None:
    """<=720p, ~2Mbps, `+faststart` re-encode of `src` for hosting weight
    (RULED, see task-19 controller rulings #5). Re-encoded AT THE SAME FPS AS
    `src` (`-r <fps>`, read back via `ffprobe_fps` rather than assumed) --
    the report/tracks were computed against `src`'s own frame timeline, so
    changing fps or dropping/duplicating frames here would desync the
    report's frame-index-based rally/stroke markers from what actually plays
    at that point in the served video. Frame-count equality is verified by
    the caller via `ffprobe_nb_frames` (not here), which is what actually
    catches a silent drop/duplicate -- this function only fixes the
    encode-time knobs that would otherwise risk one."""
    fps = ffprobe_fps(src)
    _run([
        "ffmpeg", "-y", "-i", str(src), "-vf", "scale=-2:720", "-r", str(fps),
        "-b:v", "2M", "-maxrate", "2M", "-bufsize", "4M",
        "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", "-an", str(dst),
    ])


def patch_winners(
    report: MatchReport, match_id: str, full_match_start_frame: int, step: int, labels: pd.DataFrame,
) -> int:
    """Map each `report.rallies[i]` to the ShuttleSet-labeled rally it
    overlaps most (by frame count) for `match_id`, and copy that label's
    `rally_winner` onto `RallyInterval.winner` (RULED, task-19 controller
    ruling #2). Sets `winner = None` (never copies the `-1` sentinel --
    ruling #4) for any detected rally with zero overlap against every
    labeled rally -- "no confident label match".

    End-frame convention (RULED, ruling #3): `report`'s `RallyInterval.
    end_frame` is END-EXCLUSIVE (the schema's own docstring); `labels.
    parquet`'s `rally_end_frame` is END-INCLUSIVE. This function converts
    the LABEL side to exclusive (`rally_end_frame + 1`) before comparing, so
    both sides of every overlap computation are exclusive-end frame ranges.

    `full_match_start_frame` + `step` translate a detected rally's
    SAMPLED-frame-index units (what `RallyInterval.start_frame`/`end_frame`
    are actually in -- indices into the ~`TARGET_FPS`-sampled pose sequence,
    per `pipeline.analyze`, NOT raw source-video frame numbers) into
    full-match raw-video frame numbers: sampled index `j` <-> raw source
    frame `j * step` <-> full-match raw frame `full_match_start_frame +
    j * step`. `step` must be the same source-fps/TARGET_FPS ratio
    `extract_poses_onnx` used internally to build that sampled sequence in
    the first place (the caller derives it from the analysis clip's own
    measured fps and the returned `report.fps`, see `build_sample`).

    Returns the number of rallies that got a non-null winner.
    """
    sub = labels[labels["match_id"] == match_id].drop_duplicates("rally_id")
    if sub.empty:
        raise RuntimeError(f"no labels.parquet rows found for match_id={match_id!r}")
    label_intervals = [
        (int(row.rally_start_frame), int(row.rally_end_frame) + 1, int(row.rally_winner))
        for row in sub.itertuples()
    ]

    n_matched = 0
    for rally in report.rallies:
        full_start = full_match_start_frame + rally.start_frame * step
        full_end = full_match_start_frame + rally.end_frame * step  # already exclusive
        best_overlap = 0
        best_winner = None
        for lstart, lend, lwinner in label_intervals:
            overlap = min(full_end, lend) - max(full_start, lstart)
            if overlap > best_overlap:
                best_overlap = overlap
                best_winner = None if lwinner == -1 else lwinner
        rally.winner = best_winner
        if best_winner is not None:
            n_matched += 1
    return n_matched


def lookup_player_names(match_csv: pd.DataFrame, match_id: str) -> list[str]:
    """`[winner_name, loser_name]` real names for `match_id`, from
    ShuttleSet's `set/match.csv` (`video` column == our `match_id`).

    Task 19 Fix round 1: these are the display names for the SCORE-RACE
    series, deliberately NOT a claim about which on-screen skeleton is
    which -- `report.rallies[].winner` (what the score race plots) is
    ShuttleSet's `player` id, which per `training/notes/shuttleset-
    format.md` (f) is a MATCH-scoped identity where `0` is fixed to be
    "the player who eventually wins the match" (`match.csv.winner`) and `1`
    is fixed to "the player who loses" (`match.csv.loser`) -- confirmed
    100% (44/44 matches) in that note. This is entirely independent of
    `pipeline.analyze`'s per-frame court-side skeleton slot (which has no
    persistent identity at all -- see `pipeline.py`'s docstring); an
    empirical check (see task-19-report.md's "Fix round 1") found no
    reliable correspondence between the two (68.6% and 5.7% agreement on
    the two samples respectively -- i.e. not trustworthy either way), which
    is why this function's output must only ever label the score race, not
    the skeleton.

    Names are `.title()`-cased for display (ShuttleSet's raw casing is
    inconsistent, e.g. `"Anders ANTONSEN"`, `"CHOU Tien Chen"`). Raises
    loudly if `match_id` has no `match.csv` row -- every `match_id` this
    script is ever called with comes from `labels.parquet`, which IS
    ShuttleSet, so a miss here means a real data mismatch, not a
    legitimately-absent optional field.
    """
    rows = match_csv[match_csv["video"] == match_id]
    if rows.empty:
        raise RuntimeError(f"no match.csv row found for match_id={match_id!r} -- can't derive player names")
    row = rows.iloc[0]
    return [str(row["winner"]).title(), str(row["loser"]).title()]


def build_sample(entry: dict, labels: pd.DataFrame, match_csv: pd.DataFrame, models_dir: Path, out_root: Path) -> dict:
    sample_id = entry["id"]
    source_video = REPO_ROOT / entry["source_video"]
    if not source_video.is_file():
        raise RuntimeError(f"sample {sample_id!r}: source_video not found: {source_video}")

    offset_s = load_clip_offset_s(source_video)
    clip_fps = ffprobe_fps(source_video)

    sample_dir = out_root / sample_id
    if sample_dir.exists():
        shutil.rmtree(sample_dir)
    sample_dir.mkdir(parents=True)

    analysis_clip = sample_dir / "_analysis_clip.mp4"
    print(f"[build_samples] {sample_id}: cutting analysis clip from {source_video.name} "
          f"@ {entry['clip_start_s']}s +{entry['clip_duration_s']}s (source fps={clip_fps})", flush=True)
    cut_analysis_clip(source_video, entry["clip_start_s"], entry["clip_duration_s"], clip_fps, analysis_clip)

    nb_analysis_frames = ffprobe_nb_frames(analysis_clip)
    print(f"[build_samples] {sample_id}: analysis clip has {nb_analysis_frames} frames "
          f"({nb_analysis_frames / clip_fps:.1f}s)", flush=True)

    print(f"[build_samples] {sample_id}: running analyze()...", flush=True)
    report, tracks = analyze(str(analysis_clip), str(models_dir), target_fps=TARGET_FPS)
    print(f"[build_samples] {sample_id}: {len(report.rallies)} rallies, "
          f"{len(report.strokes)} strokes detected", flush=True)
    if len(report.rallies) < MIN_RALLIES:
        raise RuntimeError(
            f"sample {sample_id!r}: only {len(report.rallies)} rallies detected "
            f"(need >= {MIN_RALLIES}) -- pick a different/longer clip window"
        )

    # step: the source-fps/TARGET_FPS sampling ratio extract_poses_onnx used
    # internally (see patch_winners' docstring). Recovered from the ACTUAL
    # returned report.fps (== clip_fps / step exactly) rather than
    # recomputed from TARGET_FPS in isolation, and cross-checked against
    # that independent computation -- a mismatch would mean this script's
    # assumptions about analyze()'s internal sampling have drifted from its
    # real behavior, which must fail loudly, not silently mis-map winners.
    step_from_fps = round(clip_fps / report.fps)
    step_from_target = max(round(clip_fps / TARGET_FPS), 1)
    if step_from_fps != step_from_target:
        raise RuntimeError(
            f"sample {sample_id!r}: step mismatch -- clip_fps/report.fps={step_from_fps} "
            f"vs clip_fps/TARGET_FPS={step_from_target}; analyze()'s internal sampling "
            "assumption has drifted from this script's"
        )
    step = step_from_fps

    full_match_start_frame = round((offset_s + entry["clip_start_s"]) * clip_fps)
    n_matched = patch_winners(report, entry["match_id"], full_match_start_frame, step, labels)
    print(f"[build_samples] {sample_id}: {n_matched}/{len(report.rallies)} rallies got a patched winner", flush=True)
    if n_matched == 0:
        raise RuntimeError(f"sample {sample_id!r}: zero rallies matched a labeled winner")

    players = lookup_player_names(match_csv, entry["match_id"])
    print(f"[build_samples] {sample_id}: players (score-race identity, from ShuttleSet) = {players}",
          flush=True)

    (sample_dir / "report.json").write_text(json.dumps(report.to_dict(), indent=2))
    (sample_dir / "tracks.json").write_text(json.dumps(tracks))
    (sample_dir / "meta.json").write_text(
        json.dumps({"id": sample_id, "title": entry["title"], "players": players}, indent=2)
    )

    final_video = sample_dir / "video.mp4"
    print(f"[build_samples] {sample_id}: re-encoding for hosting (<=720p, ~2Mbps, +faststart)...", flush=True)
    reencode_for_hosting(analysis_clip, final_video)
    nb_final_frames = ffprobe_nb_frames(final_video)
    if nb_final_frames != nb_analysis_frames:
        raise RuntimeError(
            f"sample {sample_id!r}: hosting re-encode changed frame count "
            f"{nb_analysis_frames} -> {nb_final_frames}; report/tracks frame indices "
            "would no longer line up with the served video"
        )
    print(f"[build_samples] {sample_id}: frame count preserved ({nb_final_frames} frames)", flush=True)
    analysis_clip.unlink()

    return {
        "id": sample_id,
        "n_rallies": len(report.rallies),
        "n_strokes": len(report.strokes),
        "n_winners_matched": n_matched,
        "video_frames": nb_final_frames,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--samples-yaml", type=Path, default=DEFAULT_SAMPLES_YAML)
    ap.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    ap.add_argument("--match-csv", type=Path, default=DEFAULT_MATCH_CSV)
    ap.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = ap.parse_args()

    spec = yaml.safe_load(args.samples_yaml.read_text())
    entries = spec["samples"]
    labels = pd.read_parquet(args.labels)
    match_csv = pd.read_csv(args.match_csv)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for entry in entries:
        results.append(build_sample(entry, labels, match_csv, args.models_dir, args.out_dir))

    print("\n[build_samples] summary:")
    for r in results:
        print(f"  {r['id']}: {r['n_rallies']} rallies, {r['n_strokes']} strokes, "
              f"{r['n_winners_matched']} winners matched, {r['video_frames']} video frames")


if __name__ == "__main__":
    main()
