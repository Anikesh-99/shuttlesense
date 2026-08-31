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


def test_finish_done_does_not_clobber_recorded_error(tmp_path):
    conn = db.connect(tmp_path / "q.sqlite")
    jid = db.create_job(conn, "y.mp4")
    db.claim_next(conn)
    db.finish(conn, jid, "failed", error="transient failure")
    # A later finish("done") call (e.g. a retried job) must not wipe the
    # previously recorded error back to NULL.
    db.finish(conn, jid, "done")
    row = db.get_job(conn, jid)
    assert row["status"] == "done"
    assert row["error"] == "transient failure"


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


def test_claim_next_race_across_two_connections(tmp_path):
    """Genuine two-connection claim, not just two calls on one connection.

    c1 and c2 are independent sqlite3 connections opened against the same
    database file (as two worker processes/threads would each hold their
    own connection). Exactly one of them claims the single queued job; the
    other must see it as already gone.
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


def test_create_job_id_is_uuid4_hex(tmp_path):
    conn = db.connect(tmp_path / "o.sqlite")
    jid = db.create_job(conn, "match.mp4")
    # round-trips through uuid.UUID and is 32 hex chars, no dashes
    assert len(jid) == 32
    uuid.UUID(hex=jid)
