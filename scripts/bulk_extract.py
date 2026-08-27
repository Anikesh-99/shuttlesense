"""Task 12a: run `training/extract_poses.py` over every downloaded clip in
`training/data/raw/videos/*.mp4`, at most 2 in parallel (CPU-bound rtmlib workers
sharing this machine's cores -- more than 2 was judged likely to cause thrashing
rather than speedup, per the controller's "2 parallel workers max" ruling).

Meant to be run under `nohup` and polled (each clip takes minutes on CPU), e.g.:

    nohup .venv/bin/python scripts/bulk_extract.py \
        > /tmp/bulk_extract.log 2>&1 &

Skips a clip if `training/data/processed/poses/<match_id>.npz` already exists AND
`--skip-existing` is passed (default: on), so a partial/interrupted run can be
resumed by rerunning the same command.
"""
from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed


def run_one(video_path: str, out_dir: str, fps: float, python_bin: str) -> tuple[str, bool, float, str]:
    mid = os.path.basename(video_path).rsplit(".", 1)[0]
    t0 = time.time()
    proc = subprocess.run(
        [python_bin, "training/extract_poses.py", video_path, "--out-dir", out_dir, "--fps", str(fps)],
        capture_output=True, text=True,
    )
    dt = time.time() - t0
    ok = proc.returncode == 0
    log = proc.stdout + proc.stderr
    return mid, ok, dt, log


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--videos-dir", default="training/data/raw/videos")
    ap.add_argument("--out-dir", default="training/data/processed/poses")
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--python-bin", default=".venv/bin/python")
    ap.add_argument("--skip-existing", action="store_true", default=True)
    ap.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    videos = sorted(glob.glob(os.path.join(a.videos_dir, "*.mp4")))
    todo = []
    for v in videos:
        mid = os.path.basename(v).rsplit(".", 1)[0]
        npz = os.path.join(a.out_dir, f"{mid}.npz")
        if a.skip_existing and os.path.exists(npz):
            print(f"[bulk_extract] {mid}: npz already exists, skipping", flush=True)
            continue
        todo.append(v)

    print(f"[bulk_extract] {len(todo)} clip(s) to process, {a.workers} worker(s)", flush=True)

    results = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(run_one, v, a.out_dir, a.fps, a.python_bin): v for v in todo}
        for fut in as_completed(futs):
            mid, ok, dt, log = fut.result()
            status = "OK" if ok else "FAILED"
            print(f"[bulk_extract] {mid}: {status} in {dt:.1f}s", flush=True)
            print(log, flush=True)
            results.append((mid, ok, dt))

    n_ok = sum(1 for _, ok, _ in results if ok)
    print(f"[bulk_extract] done: {n_ok}/{len(results)} succeeded", flush=True)
    if n_ok < len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
