"""Task 12a: download one mid-match segment per queued ShuttleSet match via yt-dlp,
and write the Task-7-format provenance sidecar JSON alongside each clip.

Reads a manifest JSON (list of `{match_id, split, fps, url, start_s, dur_s}` dicts --
see `.superpowers/sdd/2026-08-26-shuttlesense-phase1/task-12a-report.md` for how the
12 matches/windows in this run were chosen) and, for each entry:

1. Computes `download_section = '*START-END'` (MM:SS or HH:MM:SS, matching the style
   already used by the one pre-existing sidecar from Task 7) from `start_s`/`dur_s`.
2. Runs `yt-dlp -f 'bv*[height<=720]' --download-sections <section> ...`, ONE retry on
   a non-zero exit (transient network/extractor failures only -- a genuinely
   unavailable video, e.g. geo-blocked or removed, will fail identically on the retry
   and is reported as such so the caller can substitute another match).
3. On success, writes `<videos-dir>/<match_id>.json` with the same
   `{url, download_section, start_offset_s, note}` shape as the Task 7 sidecar.

Does NOT run pose extraction (see `scripts/bulk_extract.sh`/`.py`) or any alignment
verification (ffprobe fps check + visual spot-check) -- those are separate steps in the
Task 12a procedure, run after every download in this manifest has succeeded or been
substituted.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

# Fix Round 1 (post-task-12a review): yt-dlp's default behavior SKIPS the download
# entirely (exit 0, no error) when the destination file already exists on disk. This
# previously left a stale clip (from an earlier/different segment) paired with a
# freshly-written sidecar describing the NEW segment -- a real video/sidecar mismatch
# that only got caught by manual ffprobe-checking after the fact (see
# task-12a-report.md, "Deviation" in Section 2). Two independent guards close this:
#   1. `--force-overwrite` (on by default): unlink the destination path before invoking
#      yt-dlp, so a stale pre-existing file can never be silently kept.
#   2. A post-download `ffprobe`-measured duration assertion against the manifest's
#      requested `dur_s` (defense in depth even with (1) -- also catches partial/
#      truncated downloads, or a `--download-sections` request yt-dlp fulfilled from a
#      differently-bounded source). The sidecar is NOT written on a mismatch; the run
#      is reported as a failure for that match instead of silently shipping bad
#      provenance metadata.
#
# Fix Round 2 (Task 12b carry-over): (1) above previously unlinked `out_path` BEFORE
# invoking yt-dlp -- a real destructive-pre-delete gap: if yt-dlp then failed/crashed
# before writing anything (network drop, extractor error, etc.), the match was left with
# NO video at all where a perfectly good one existed moments before, and worse, a crash
# between the unlink and a successful-looking-but-wrong write could leave a half-written
# `out_path` paired with an about-to-be-written sidecar. Replaced with TEMP-PATH STAGING:
# yt-dlp writes to `out_path + ".part.mp4"` (never touching `out_path` itself), the
# ffprobe duration assertion runs against that temp path, and only once it passes is the
# temp path atomically `os.replace()`d onto `out_path` -- immediately followed by the
# sidecar write. `out_path` is therefore either the OLD (still valid) file or the NEW
# (verified) file at every point in time, never missing/half-written; a failed attempt
# leaves the old video+sidecar pair untouched and simply removes its own leftover temp
# file.
DURATION_TOL_S = 3.0  # absolute tolerance; --download-sections snaps to keyframes, so
# a couple seconds of slack around the requested window is expected and not a bug.


def fmt_ts(seconds: int) -> str:
    """`seconds` -> `MM:SS` (or `H:MM:SS` if >= 1 hour), matching yt-dlp's accepted
    `--download-sections '*START-END'` time syntax and the style of the existing
    Task 7 sidecar (`*10:00-11:00`)."""
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def ffprobe_duration_s(path: str, ffprobe: str = "ffprobe") -> float:
    """Container-reported duration in seconds, via `ffprobe -show_entries
    format=duration`. Raises `RuntimeError` if `ffprobe` itself fails (missing binary,
    unreadable/corrupt file, etc.) -- a caller should treat that as a download failure,
    not silently trust an un-probed file."""
    proc = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True,
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        raise RuntimeError(f"ffprobe failed on {path!r}: {proc.stderr.strip()}")
    return float(proc.stdout.strip())


def download_one(
    entry: dict,
    videos_dir: str,
    yt_dlp: str = "yt-dlp",
    force_overwrite: bool = True,
    ffprobe: str = "ffprobe",
    duration_tol_s: float = DURATION_TOL_S,
) -> tuple[bool, str]:
    match_id = entry["match_id"]
    start_s, dur_s = int(entry["start_s"]), int(entry["dur_s"])
    end_s = start_s + dur_s
    section = f"*{fmt_ts(start_s)}-{fmt_ts(end_s)}"
    out_path = os.path.join(videos_dir, f"{match_id}.mp4")
    # Fix Round 2: stage the download at a TEMP path, never at `out_path` itself, so a
    # failed/crashed attempt can never leave `out_path` missing or half-written -- see
    # the module-level "Fix Round 2" comment above for the full rationale.
    tmp_path = out_path + ".part.mp4"
    cmd = [
        yt_dlp, "-f", "bv*[height<=720]",
        "--download-sections", section,
        "--extractor-args", "youtube:player_client=default",
        "-o", tmp_path,
        entry["url"],
    ]
    last_err = ""
    for attempt in (1, 2):  # ONE retry on transient failure, per controller ruling
        if force_overwrite and os.path.exists(tmp_path):
            # Unlink the TEMP path (not out_path) first -- yt-dlp silently SKIPS (exit
            # 0, no download) when its own destination already exists, which is exactly
            # the bug this guard closes; a leftover temp path from a prior failed
            # attempt must not fool the next attempt into skipping either.
            print(f"[bulk_download] {match_id}: force-overwrite, removing stale {tmp_path}", file=sys.stderr)
            os.remove(tmp_path)
        print(f"[bulk_download] {match_id}: attempt {attempt}: {' '.join(cmd)}", file=sys.stderr)
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode == 0 and os.path.exists(tmp_path):
            try:
                actual_dur = ffprobe_duration_s(tmp_path, ffprobe)
            except RuntimeError as e:
                last_err = f"{match_id}: post-download ffprobe check failed: {e}"
                continue
            if abs(actual_dur - dur_s) > duration_tol_s:
                last_err = (
                    f"{match_id}: duration mismatch -- manifest expects {dur_s}s "
                    f"({section}), ffprobe reports {actual_dur:.1f}s on {tmp_path!r} "
                    f"(tolerance {duration_tol_s}s). NOT writing sidecar, NOT touching "
                    f"the existing {out_path!r} (if any). This is the yt-dlp "
                    "silent-skip failure mode (stale pre-existing file kept on a no-op "
                    "download, exit 0) -- pass force_overwrite=True (default) or "
                    "manually delete the stale temp file and retry."
                )
                continue
            # Verified good -- atomically promote the temp file onto out_path. Any
            # pre-existing out_path (old video) is replaced only now, at the last
            # possible moment, after the new file has already passed verification.
            os.replace(tmp_path, out_path)
            sidecar = {
                "url": entry["url"],
                "download_section": section,
                "start_offset_s": start_s,
                "note": (
                    "yt-dlp --download-sections trims relative to the full match video "
                    "hosted at `url`. `start_offset_s` must be added to any in-clip "
                    "sampled-frame time (frame_index_in_clip / meta['fps_sampled']) "
                    "before it can be compared to a ShuttleSet label's frame_num (which "
                    "indexes the full match video, not this trimmed clip). This file "
                    "exists because npz `meta` alone (see extract_poses.py docstring) "
                    "has no way to represent this offset. Post-download ffprobe "
                    f"duration check passed ({actual_dur:.1f}s vs manifest {dur_s}s, "
                    f"tolerance {duration_tol_s}s)."
                ),
            }
            with open(os.path.join(videos_dir, f"{match_id}.json"), "w") as f:
                json.dump(sidecar, f, indent=2)
                f.write("\n")
            return True, ""
        last_err = proc.stderr[-2000:]
    # Both attempts failed (or verification failed both times) -- clean up any leftover
    # temp file so it can't be mistaken for a real artifact or confuse a future rerun's
    # force-overwrite unlink logic; out_path (old video, if any) is left untouched.
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    return False, last_err


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("manifest")
    ap.add_argument("--videos-dir", default="training/data/raw/videos")
    ap.add_argument("--yt-dlp", default="yt-dlp")
    ap.add_argument("--ffprobe", default="ffprobe")
    ap.add_argument(
        "--force-overwrite", dest="force_overwrite", action="store_true", default=True,
        help="Unlink an existing destination .mp4 before downloading (default: on). "
             "Closes the yt-dlp silent-skip-if-exists bug (Fix Round 1).",
    )
    ap.add_argument(
        "--no-force-overwrite", dest="force_overwrite", action="store_false",
        help="Disable the pre-download unlink (NOT recommended -- reintroduces the "
             "silent-skip-if-exists failure mode; only the post-download ffprobe "
             "duration check would still catch it, and only for the sidecar, not the "
             "stale video file itself).",
    )
    ap.add_argument("--duration-tol-s", type=float, default=DURATION_TOL_S)
    a = ap.parse_args()

    os.makedirs(a.videos_dir, exist_ok=True)
    with open(a.manifest) as f:
        entries = json.load(f)

    failures = []
    for entry in entries:
        ok, err = download_one(
            entry, a.videos_dir, a.yt_dlp,
            force_overwrite=a.force_overwrite, ffprobe=a.ffprobe,
            duration_tol_s=a.duration_tol_s,
        )
        status = "OK" if ok else "FAILED"
        print(f"[bulk_download] {entry['match_id']}: {status}")
        if not ok:
            failures.append(entry["match_id"])
            print(err, file=sys.stderr)

    if failures:
        print(f"[bulk_download] {len(failures)} FAILED after retry: {failures}", file=sys.stderr)
        sys.exit(1)
    print(f"[bulk_download] all {len(entries)} matches downloaded successfully")


if __name__ == "__main__":
    main()
