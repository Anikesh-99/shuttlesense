"""SQLite-backed job queue for the analysis worker.

Schema (single table, ``jobs``):
    id TEXT PRIMARY KEY       -- uuid4 hex
    filename TEXT             -- original uploaded filename
    status TEXT               -- "queued" -> "processing" -> "done" | "failed"
    error TEXT                -- set only when status == "failed"
    created_at REAL           -- time.time() at creation, used for FIFO ordering

Concurrency: ``connect()`` puts the connection in autocommit mode
(``isolation_level = None``) so that transaction boundaries are exactly the
explicit ``BEGIN`` / ``COMMIT`` / ``ROLLBACK`` statements we issue ourselves
-- sqlite3's legacy (non-``None``) isolation mode only opens an implicit
transaction lazily on the first DML statement, which is too late to prevent
two connections from both SELECTing the same "queued" row before either has
written its UPDATE. ``claim_next`` therefore opens the transaction with
``BEGIN IMMEDIATE`` *before* the SELECT (acquiring the write lock up front,
so a second connection blocks at its own ``BEGIN IMMEDIATE`` until the first
commits) and additionally guards the UPDATE with
``WHERE id=? AND status='queued'`` plus a ``rowcount`` check, so even a
misconfigured/older-sqlite connection that lost the row-lock race is caught
rather than silently double-claiming.

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


def connect(path: str | PathLike) -> sqlite3.Connection:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p))
    conn.row_factory = sqlite3.Row
    # Autocommit mode: we manage transaction boundaries explicitly (see
    # module docstring). Do not rely on sqlite3's implicit-transaction
    # legacy behavior here.
    conn.isolation_level = None
    conn.execute("PRAGMA journal_mode=WAL")
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
        conn.execute("ROLLBACK")
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
    if status == "failed":
        cur = conn.execute(
            "UPDATE jobs SET status=?, error=? WHERE id=?", (status, error, job_id)
        )
    else:
        # Never clobber a previously recorded error with NULL on success.
        cur = conn.execute("UPDATE jobs SET status=? WHERE id=?", (status, job_id))
    if cur.rowcount == 0:
        raise ValueError(f"unknown job id: {job_id!r}")


def get_job(conn: sqlite3.Connection, job_id: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
