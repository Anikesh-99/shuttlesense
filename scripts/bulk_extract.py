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


def verify_npz(npz_path: str) -> str:
    """Load `npz_path` and sanity-check its contract (Fix Round 1 minor: a subprocess
    exit code of 0 is NOT sufficient evidence the npz is real/usable -- a truncated
    write, a `--out-dir` typo that silently wrote nowhere the caller expects, or a
    zero-frame extraction would all still exit 0). Returns `""` if the file passes, or
    a non-empty description of what's wrong (caller treats a non-empty return as a
    failure, matching a non-zero exit code)."""
    if not os.path.exists(npz_path):
        return f"expected output npz missing: {npz_path!r}"
    try:
        import numpy as np
        z = np.load(npz_path, allow_pickle=False)
        for key in ("kpts", "scores", "meta"):
            if key not in z:
                return f"{npz_path!r} missing expected array {key!r}"
        if z["kpts"].shape[0] == 0:
            return f"{npz_path!r} has zero sampled frames (kpts.shape={z['kpts'].shape})"
        if z["kpts"].shape[0] != z["scores"].shape[0]:
            return (
                f"{npz_path!r} kpts/scores frame-count mismatch: "
                f"{z['kpts'].shape[0]} vs {z['scores'].shape[0]}"
            )
    except Exception as e:  # noqa: BLE001 -- any load failure is a real verify failure
        return f"{npz_path!r} failed to load/parse: {e!r}"
    return ""


def run_one(
    video_path: str, out_dir: str, fps: float, python_bin: str, timeout_s: float | None = None,
) -> tuple[str, bool, float, str]:
    mid = os.path.basename(video_path).rsplit(".", 1)[0]
    npz_path = os.path.join(out_dir, f"{mid}.npz")
    t0 = time.time()
    try:
        proc = subprocess.run(
            [python_bin, "training/extract_poses.py", video_path, "--out-dir", out_dir, "--fps", str(fps)],
            capture_output=True, text=True, timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as e:
        dt = time.time() - t0
        out = (e.stdout or b"")
        err = (e.stderr or b"")
        out = out.decode() if isinstance(out, bytes) else out
        err = err.decode() if isinstance(err, bytes) else err
        log = out + err + f"\n[bulk_extract] {mid}: TIMEOUT after {timeout_s}s (per-clip limit)"
        return mid, False, dt, log
    dt = time.time() - t0
    log = proc.stdout + proc.stderr
    if proc.returncode != 0:
        return mid, False, dt, log
    # exit 0 alone is not proof of a real, usable npz -- verify it before declaring OK
    # (Fix Round 1 minor: "verify-npz-after-exit-0").
    problem = verify_npz(npz_path)
    if problem:
        log += f"\n[bulk_extract] {mid}: exit 0 but post-hoc npz verification failed: {problem}"
        return mid, False, dt, log
    return mid, True, dt, log


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--videos-dir", default="training/data/raw/videos")
    ap.add_argument("--out-dir", default="training/data/processed/poses")
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--python-bin", default=".venv/bin/python")
    ap.add_argument("--skip-existing", action="store_true", default=True)
    ap.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    ap.add_argument(
        "--timeout-s", type=float, default=None,
        help="Per-clip wall-clock timeout in seconds passed to subprocess.run "
             "(default: no timeout). Task 12a observed this machine's clip extraction "
             "time trend from ~1725s up to ~2893s under sustained 2-worker load "
             "(thermal-throttling-like); set a generous timeout (e.g. 6000-7200s) if "
             "running unattended and a hung clip should not silently stall the whole "
             "batch indefinitely.",
    )
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
        futs = {ex.submit(run_one, v, a.out_dir, a.fps, a.python_bin, a.timeout_s): v for v in todo}
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
