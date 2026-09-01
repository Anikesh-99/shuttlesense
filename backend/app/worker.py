"""Job-processing loop: claims queued rows from the sqlite job queue
(`backend.app.db`) and runs `backend.app.pipeline.analyze` against them.

Threading contract (see `backend/app/db.py`'s module docstring): sqlite3
connections must not cross threads, so every function here that touches the
database opens its OWN `db.connect()` and closes it before returning --
`run_forever` is expected to be started as a background thread (Task 16:
`threading.Thread(target=run_forever, daemon=True)`), never handed a
connection created elsewhere.

`db.claim_next` can raise `sqlite3.OperationalError` under write-lock
contention (another connection's `BEGIN IMMEDIATE` still holding the lock past
this connection's busy timeout) -- per `db.py`'s documented contract this is
transient, not fatal. `run_forever`'s poll loop catches it around the
`run_once` call and simply retries on the next iteration; `run_once` itself
does not swallow it (letting `db.connect()`/`claim_next` failures propagate to
callers that need to observe them, e.g. tests). Any OTHER exception escaping
`run_once` (a bug in `analyze`, a disk-full error writing outputs, etc.) is
also caught by `run_forever` -- logged via `traceback.print_exc()`, then the
loop sleeps and continues -- so a single bad job can never permanently kill
the background poll thread; per-job failures are meant to be recorded on the
job row (see `run_once`), not to propagate past this loop.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
import traceback
from pathlib import Path

from backend.app import db
from backend.app.pipeline import analyze

# Friendly, user-facing message for the one ValueError analyze() raises on
# purpose (no rally interval found) -- everything else records str(exc)
# verbatim, since it's an unexpected/internal failure worth the real detail.
FRIENDLY_NO_RALLY_MSG = "This video doesn't look like a badminton match we can analyze."


def db_path(settings) -> Path:
    """Where the job queue's sqlite file lives, derived from `settings.data_dir`.
    Shared here so Task 16's routes (which create jobs) and this worker (which
    claims/finishes them) agree on the same path without duplicating it."""
    return Path(settings.data_dir) / "jobs.sqlite"


def job_video_path(settings, row) -> str:
    """Where an uploaded job's source video lives, per the Task 15 brief's
    fixed convention: `<data_dir>/uploads/<job_id>/<filename>`."""
    return str(Path(settings.data_dir) / "uploads" / row["id"] / row["filename"])


def job_output_dir(settings, job_id: str) -> Path:
    return Path(settings.data_dir) / "jobs" / job_id


def _atomic_write_json(path: Path, obj) -> None:
    """Write `obj` as JSON to `path` atomically: write to a sibling `.tmp`
    file first, then `os.replace` it into place. Prevents a reader (the API,
    Task 16) from ever observing a partially-written `report.json`/
    `tracks.json`, and prevents a crash mid-write from leaving a corrupt file
    at the final path. On failure (e.g. `json.dumps` blowing up on a bad
    object, or the write itself failing), the orphaned `.tmp` file is
    unlinked before the exception propagates, so a failed job doesn't also
    litter the output directory with stale temp files."""
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp_path.write_text(json.dumps(obj))
        os.replace(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def run_once(settings) -> bool:
    """Claim and process exactly one queued job.

    Returns `True` if a job was claimed (regardless of whether analysis then
    succeeded or failed -- both outcomes still "processed" the claimed job),
    `False` if the queue was empty (nothing to do). Opens and closes its own
    db connection (see module docstring).

    Both `analyze()` and the report/tracks JSON writes happen inside the same
    try/except: a write failure (e.g. disk full) must mark the job `failed`
    exactly like an analysis failure would, not leave it silently stuck in
    `processing` forever.
    """
    conn = db.connect(db_path(settings))
    try:
        row = db.claim_next(conn)
        if row is None:
            return False
        job_id = row["id"]
        try:
            video_path = job_video_path(settings, row)
            report, tracks = analyze(
                video_path, settings.models_dir, target_fps=settings.target_fps
            )
            out_dir = job_output_dir(settings, job_id)
            out_dir.mkdir(parents=True, exist_ok=True)
            _atomic_write_json(out_dir / "report.json", report.to_dict())
            _atomic_write_json(out_dir / "tracks.json", tracks)
        except ValueError as exc:
            if str(exc) == "no rallies detected":
                db.finish(conn, job_id, "failed", error=FRIENDLY_NO_RALLY_MSG)
            else:
                db.finish(conn, job_id, "failed", error=str(exc))
            return True
        except Exception as exc:  # noqa: BLE001 -- worker boundary: never crash the loop
            traceback.print_exc()
            db.finish(conn, job_id, "failed", error=str(exc))
            return True

        db.finish(conn, job_id, "done")
        return True
    finally:
        conn.close()


def run_forever(settings, poll_interval: float = 1.0) -> None:
    """Poll the job queue forever, processing one job at a time. Intended to
    run in its own daemon thread (Task 16). Sleeps `poll_interval` seconds
    whenever there was nothing to claim, a transient
    `sqlite3.OperationalError` was hit, or (defensively) `run_once` raised
    anything else -- see module docstring for why the broad catch exists."""
    while True:
        try:
            claimed = run_once(settings)
        except sqlite3.OperationalError:
            claimed = False
        except Exception:  # noqa: BLE001 -- see module docstring
            traceback.print_exc()
            claimed = False
        if not claimed:
            time.sleep(poll_interval)
