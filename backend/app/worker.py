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
callers that need to observe them, e.g. tests).
"""
from __future__ import annotations

import json
import sqlite3
import time
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


def run_once(settings) -> bool:
    """Claim and process exactly one queued job.

    Returns `True` if a job was claimed (regardless of whether analysis then
    succeeded or failed -- both outcomes still "processed" the claimed job),
    `False` if the queue was empty (nothing to do). Opens and closes its own
    db connection (see module docstring).
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
        except ValueError as exc:
            if str(exc) == "no rallies detected":
                db.finish(conn, job_id, "failed", error=FRIENDLY_NO_RALLY_MSG)
            else:
                db.finish(conn, job_id, "failed", error=str(exc))
            return True
        except Exception as exc:  # noqa: BLE001 -- worker boundary: never crash the loop
            db.finish(conn, job_id, "failed", error=str(exc))
            return True

        out_dir = job_output_dir(settings, job_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "report.json").write_text(json.dumps(report.to_dict()))
        (out_dir / "tracks.json").write_text(json.dumps(tracks))
        db.finish(conn, job_id, "done")
        return True
    finally:
        conn.close()


def run_forever(settings, poll_interval: float = 1.0) -> None:
    """Poll the job queue forever, processing one job at a time. Intended to
    run in its own daemon thread (Task 16). Sleeps `poll_interval` seconds
    whenever there was nothing to claim, or a transient
    `sqlite3.OperationalError` was hit (see module docstring)."""
    while True:
        try:
            claimed = run_once(settings)
        except sqlite3.OperationalError:
            claimed = False
        if not claimed:
            time.sleep(poll_interval)
