"""SQLite-backed job queue for the analysis worker.

Schema (single table, ``jobs``):
    id TEXT PRIMARY KEY       -- uuid4 hex
    filename TEXT             -- original uploaded filename
    status TEXT               -- "queued" -> "processing" -> "done" | "failed"
    error TEXT                -- set only when status == "failed"
    created_at REAL           -- time.time() at creation, used for FIFO ordering

``claim_next`` uses ``BEGIN IMMEDIATE`` (via the connection's context manager)
so that under concurrent workers only one caller can move a given job from
"queued" to "processing".
"""
from __future__ import annotations

import sqlite3
import time
import uuid
from os import PathLike

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
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute(_SCHEMA)
    conn.commit()
    return conn


def create_job(conn: sqlite3.Connection, filename: str) -> str:
    job_id = uuid.uuid4().hex
    with conn:
        conn.execute(
            "INSERT INTO jobs (id, filename, status, error, created_at) "
            "VALUES (?, ?, 'queued', NULL, ?)",
            (job_id, filename, time.time()),
        )
    return job_id


def claim_next(conn: sqlite3.Connection) -> sqlite3.Row | None:
    with conn:
        row = conn.execute(
            "SELECT * FROM jobs WHERE status='queued' ORDER BY created_at LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        conn.execute("UPDATE jobs SET status='processing' WHERE id=?", (row["id"],))
    return row


def finish(
    conn: sqlite3.Connection,
    job_id: str,
    status: str,
    error: str | None = None,
) -> None:
    with conn:
        cur = conn.execute(
            "UPDATE jobs SET status=?, error=? WHERE id=?", (status, error, job_id)
        )
        if cur.rowcount == 0:
            raise ValueError(f"unknown job id: {job_id!r}")


def get_job(conn: sqlite3.Connection, job_id: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
