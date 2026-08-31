import sqlite3
import uuid

import pytest

from backend.app import db


def test_job_lifecycle(tmp_path):
    conn = db.connect(tmp_path / "jobs.sqlite")
    jid = db.create_job(conn, "match.mp4")
    assert db.get_job(conn, jid)["status"] == "queued"
    row = db.claim_next(conn)
    assert row["id"] == jid and db.get_job(conn, jid)["status"] == "processing"
    assert db.claim_next(conn) is None          # nothing left to claim
    db.finish(conn, jid, "done")
    assert db.get_job(conn, jid)["status"] == "done"


def test_claim_next_returns_post_update_row(tmp_path):
    conn = db.connect(tmp_path / "p.sqlite")
    jid = db.create_job(conn, "match.mp4")
    row = db.claim_next(conn)
    # Caller must never observe status='queued' on the row it just claimed.
    assert row["status"] == "processing"


def test_finish_with_error(tmp_path):
    conn = db.connect(tmp_path / "j.sqlite")
    jid = db.create_job(conn, "x.mp4")
    db.claim_next(conn)
    db.finish(conn, jid, "failed", error="not a badminton video")
    row = db.get_job(conn, jid)
    assert row["status"] == "failed" and "badminton" in row["error"]


def test_finish_done_clears_recorded_error(tmp_path):
    conn = db.connect(tmp_path / "q.sqlite")
    jid = db.create_job(conn, "y.mp4")
    db.claim_next(conn)
    db.finish(conn, jid, "failed", error="transient failure")
    # Schema contract: error is non-NULL only when status == "failed". A
    # later finish("done") call (e.g. a retried job that then succeeds)
    # must clear the stale error back to NULL, not leave it dangling
    # alongside a "done" status.
    db.finish(conn, jid, "done")
    row = db.get_job(conn, jid)
    assert row["status"] == "done"
    assert row["error"] is None


def test_finish_rejects_invalid_status(tmp_path):
    conn = db.connect(tmp_path / "r.sqlite")
    jid = db.create_job(conn, "z.mp4")
    with pytest.raises(ValueError):
        db.finish(conn, jid, "queued")


def test_get_job_unknown_returns_none(tmp_path):
    conn = db.connect(tmp_path / "k.sqlite")
    assert db.get_job(conn, "does-not-exist") is None


def test_finish_unknown_job_raises(tmp_path):
    conn = db.connect(tmp_path / "m.sqlite")
    with pytest.raises(ValueError):
        db.finish(conn, "does-not-exist", "done")


def test_claim_next_sees_other_connections_claims(tmp_path):
    """Two independent connections, sequential (non-overlapping) claims.

    c1 and c2 are independent sqlite3 connections opened against the same
    database file (as two worker processes/threads would each hold their
    own connection). Exactly one of them claims the single queued job; the
    other must see it as already gone.

    NOTE: this test alone does not discriminate a correctly-locking
    implementation from a buggy one (the old, unguarded `with conn:`
    implementation also passes it, since the two claim_next() calls never
    actually overlap in time here). See
    test_claim_next_holds_write_lock_during_its_select below for the
    test that actually forces the overlap and proves the write lock is
    held during claim_next's own SELECT.
    """
    path = tmp_path / "n.sqlite"
    setup_conn = db.connect(path)
    jid = db.create_job(setup_conn, "a.mp4")
    setup_conn.close()

    c1 = db.connect(path)
    c2 = db.connect(path)
    first = db.claim_next(c1)
    second = db.claim_next(c2)

    assert first is not None and first["id"] == jid and first["status"] == "processing"
    assert second is None
    c1.close()
    c2.close()


def test_claim_next_holds_write_lock_during_its_select(tmp_path):
    """Proves the write lock is held DURING a real db.claim_next(c1) call,
    not just in a hand-written stand-in for its statements.

    c1's sqlite trace callback fires for every SQL statement c1 actually
    executes inside claim_next(). When it sees claim_next's own "queued"
    SELECT fire, it has c2 (busy_timeout=0, so it fails fast instead of
    waiting) probe for the write lock right then, mid-call. If claim_next
    has already taken the write lock (BEGIN IMMEDIATE before the SELECT),
    c2's probe raises sqlite3.OperationalError; if not, it succeeds. This
    is a red/green test of the real function, not of hand-copied SQL --
    see "Fix round 3" in the task report for the required red-check
    (temporarily reverting claim_next to the old, unguarded `with conn:`
    form and confirming this test fails against it).
    """
    path = tmp_path / "lock.sqlite"
    s = db.connect(path); jid = db.create_job(s, "a.mp4"); s.close()
    c1 = db.connect(path)
    c2 = db.connect(path); c2.execute("PRAGMA busy_timeout=0")
    seen = {}
    def probe(stmt):
        if "status='queued'" in stmt and stmt.lstrip().upper().startswith("SELECT"):
            try:
                c2.execute("BEGIN IMMEDIATE"); c2.execute("ROLLBACK"); seen["locked"] = False
            except sqlite3.OperationalError:
                seen["locked"] = True
    c1.set_trace_callback(probe)
    assert db.claim_next(c1)["id"] == jid
    c1.set_trace_callback(None)
    assert seen.get("locked") is True


def test_create_job_id_is_uuid4_hex(tmp_path):
    conn = db.connect(tmp_path / "o.sqlite")
    jid = db.create_job(conn, "match.mp4")
    # round-trips through uuid.UUID and is 32 hex chars, no dashes
    assert len(jid) == 32
    uuid.UUID(hex=jid)
