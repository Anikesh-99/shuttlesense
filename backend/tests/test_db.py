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


def test_finish_with_error(tmp_path):
    conn = db.connect(tmp_path / "j.sqlite")
    jid = db.create_job(conn, "x.mp4")
    db.claim_next(conn)
    db.finish(conn, jid, "failed", error="not a badminton video")
    row = db.get_job(conn, jid)
    assert row["status"] == "failed" and "badminton" in row["error"]


def test_get_job_unknown_returns_none(tmp_path):
    conn = db.connect(tmp_path / "k.sqlite")
    assert db.get_job(conn, "does-not-exist") is None


def test_finish_unknown_job_raises(tmp_path):
    conn = db.connect(tmp_path / "m.sqlite")
    import pytest
    with pytest.raises(ValueError):
        db.finish(conn, "does-not-exist", "done")


def test_claim_next_is_atomic_across_sequential_claims(tmp_path):
    conn = db.connect(tmp_path / "n.sqlite")
    j1 = db.create_job(conn, "a.mp4")
    j2 = db.create_job(conn, "b.mp4")
    first = db.claim_next(conn)
    second = db.claim_next(conn)
    assert {first["id"], second["id"]} == {j1, j2}
    assert first["id"] != second["id"]
    assert db.claim_next(conn) is None


def test_create_job_id_is_uuid4_hex(tmp_path):
    import uuid
    conn = db.connect(tmp_path / "o.sqlite")
    jid = db.create_job(conn, "match.mp4")
    # round-trips through uuid.UUID and is 32 hex chars, no dashes
    assert len(jid) == 32
    uuid.UUID(hex=jid)
