"""SQLite-backed job queue for the analysis worker.

Schema (single table, ``jobs``):
    id TEXT PRIMARY KEY       -- uuid4 hex
    filename TEXT             -- original uploaded filename
    status TEXT               -- "queued" -> "processing" -> "done" | "failed"
    error TEXT                -- non-NULL only when status == "failed"; cleared
                                  (set back to NULL) by finish(..., "done")
    created_at REAL           -- time.time() at creation, used for FIFO ordering

Concurrency: ``connect()`` puts the connection in autocommit mode
(``isolation_level = None``) so that transaction boundaries are exactly the
explicit ``BEGIN`` / ``COMMIT`` / ``ROLLBACK`` statements we issue ourselves
-- sqlite3's legacy (non-``None``) isolation mode only opens an implicit
transaction lazily on the first DML statement, which is too late to prevent
two connections from both SELECTing the same "queued" row before either has
written its UPDATE. ``claim_next`` therefore opens the transaction with
``BEGIN IMMEDIATE`` *before* the SELECT (acquiring the write lock up front).
A second connection's own ``BEGIN IMMEDIATE`` does **not** block
indefinitely: it waits up to sqlite3's busy timeout (the connection's
``timeout=`` constructor argument / ``PRAGMA busy_timeout``, default 5s)
for the first connection to commit, and if that elapses first, raises
``sqlite3.OperationalError: database is locked`` instead of proceeding.
Callers polling ``claim_next`` (the Task 15 worker loop) must therefore
catch ``sqlite3.OperationalError`` and treat it as a transient condition
(retry on the next poll), not a fatal error. ``claim_next`` additionally
guards the UPDATE with ``WHERE id=? AND status='queued'`` plus a
``rowcount`` check, so even a misconfigured/older-sqlite connection that
lost the row-lock race is caught rather than silently double-claiming.

Threading: connections are not shared across threads (``sqlite3.connect``'s
default ``check_same_thread=True`` is left as-is, deliberately). Each thread
that touches the job queue -- including the Task 15 worker loop -- must call
``connect()`` itself and keep the resulting connection thread-local; do not
pass a connection between threads.
"""
from __future__ import annotations

import sqlite3
import time
import uuid
import warnings
from os import PathLike
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    status TEXT NOT NULL,
    error TEXT,
    created_at REAL NOT NULL
);
"""

_wal_fallback_warned = False  # module-level flag: warn about WAL fallback once per process


def connect(path: str | PathLike) -> sqlite3.Connection:
    global _wal_fallback_warned
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p))
    conn.row_factory = sqlite3.Row
    # Autocommit mode: we manage transaction boundaries explicitly (see
    # module docstring). Do not rely on sqlite3's implicit-transaction
    # legacy behavior here.
    conn.isolation_level = None
    mode_row = conn.execute("PRAGMA journal_mode=WAL").fetchone()
    mode = mode_row[0] if mode_row else None
    if (not mode or str(mode).lower() != "wal") and not _wal_fallback_warned:
        # Some environments (e.g. certain network filesystems) can't honor
        # WAL and sqlite silently falls back to another journal mode. That's
        # not fatal -- claim_next's correctness comes from BEGIN IMMEDIATE's
        # locking, not specifically from WAL -- but it's worth surfacing.
        # Only warn once per process (every connect() call would otherwise
        # be as noisy as the underlying condition is persistent, e.g. one
        # warning per job claimed).
        _wal_fallback_warned = True
        warnings.warn(
            f"sqlite journal_mode={mode!r} (WAL unavailable, falling back)",
            RuntimeWarning,
            stacklevel=2,
        )
    conn.execute(_SCHEMA)
    return conn


def create_job(conn: sqlite3.Connection, filename: str) -> str:
    job_id = uuid.uuid4().hex
    conn.execute(
        "INSERT INTO jobs (id, filename, status, error, created_at) "
        "VALUES (?, ?, 'queued', NULL, ?)",
        (job_id, filename, time.time()),
    )
    return job_id


def claim_next(conn: sqlite3.Connection) -> sqlite3.Row | None:
    conn.execute("BEGIN IMMEDIATE")
    try:
        row = conn.execute(
            "SELECT * FROM jobs WHERE status='queued' "
            "ORDER BY created_at, rowid LIMIT 1"
        ).fetchone()
        if row is None:
            conn.execute("COMMIT")
            return None
        cur = conn.execute(
            "UPDATE jobs SET status='processing' WHERE id=? AND status='queued'",
            (row["id"],),
        )
        if cur.rowcount == 0:
            # Another connection claimed this row between our SELECT and
            # UPDATE (shouldn't happen given BEGIN IMMEDIATE's write lock,
            # but guarded belt-and-braces per the transaction semantics
            # documented above).
            conn.execute("COMMIT")
            return None
        conn.execute("COMMIT")
    except BaseException:
        try:
            conn.execute("ROLLBACK")
        except sqlite3.Error:
            # Don't let a failed ROLLBACK (e.g. connection already broken,
            # no transaction actually open) mask the original error.
            pass
        raise
    # Return the post-update row so callers never observe status='queued'
    # on a job that is actually already 'processing'.
    return get_job(conn, row["id"])


def finish(
    conn: sqlite3.Connection,
    job_id: str,
    status: str,
    error: str | None = None,
) -> None:
    if status not in ("done", "failed"):
        raise ValueError(f"invalid status: {status!r}; must be 'done' or 'failed'")
    # error is non-NULL only when status == "failed" (schema contract);
    # finishing "done" always clears any previously recorded error, e.g.
    # from an earlier failed attempt at the same job id.
    err = error if status == "failed" else None
    cur = conn.execute(
        "UPDATE jobs SET status=?, error=? WHERE id=?", (status, err, job_id)
    )
    if cur.rowcount == 0:
        raise ValueError(f"unknown job id: {job_id!r}")


def get_job(conn: sqlite3.Connection, job_id: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
