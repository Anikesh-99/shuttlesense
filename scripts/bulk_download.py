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


def fmt_ts(seconds: int) -> str:
    """`seconds` -> `MM:SS` (or `H:MM:SS` if >= 1 hour), matching yt-dlp's accepted
    `--download-sections '*START-END'` time syntax and the style of the existing
    Task 7 sidecar (`*10:00-11:00`)."""
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def download_one(entry: dict, videos_dir: str, yt_dlp: str = "yt-dlp") -> tuple[bool, str]:
    match_id = entry["match_id"]
    start_s, dur_s = int(entry["start_s"]), int(entry["dur_s"])
    end_s = start_s + dur_s
    section = f"*{fmt_ts(start_s)}-{fmt_ts(end_s)}"
    out_path = os.path.join(videos_dir, f"{match_id}.mp4")
    cmd = [
        yt_dlp, "-f", "bv*[height<=720]",
        "--download-sections", section,
        "--extractor-args", "youtube:player_client=default",
        "-o", out_path,
        entry["url"],
    ]
    last_err = ""
    for attempt in (1, 2):  # ONE retry on transient failure, per controller ruling
        print(f"[bulk_download] {match_id}: attempt {attempt}: {' '.join(cmd)}", file=sys.stderr)
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode == 0 and os.path.exists(out_path):
            sidecar = {
                "url": entry["url"],
                "download_section": section,
                "start_offset_s": start_s,
                "note": (
                    "yt-dlp --download-sections trims relative to the full match video "
                    "hosted at `url`. `start_offset_s` must be added to any in-clip "
                    "sampled-frame time (frame_index_in_clip / meta['fps_sampled']) "
                    "before it can be compared to a ShuttleSet label's frame_num (which "
                    "indexes the full match video, not this trimmed clip)."
                ),
            }
            with open(os.path.join(videos_dir, f"{match_id}.json"), "w") as f:
                json.dump(sidecar, f, indent=2)
            return True, ""
        last_err = proc.stderr[-2000:]
    return False, last_err


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("manifest")
    ap.add_argument("--videos-dir", default="training/data/raw/videos")
    ap.add_argument("--yt-dlp", default="yt-dlp")
    a = ap.parse_args()

    os.makedirs(a.videos_dir, exist_ok=True)
    with open(a.manifest) as f:
        entries = json.load(f)

    failures = []
    for entry in entries:
        ok, err = download_one(entry, a.videos_dir, a.yt_dlp)
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
