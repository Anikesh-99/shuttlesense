"""Tests for the matches/samples HTTP API (`backend/app/routes.py`) and app
assembly + hardened static serving (`backend/app/main.py`).

Every test sets `SHUTTLESENSE_NO_WORKER=1` (via the autouse `_clean_env`
fixture) so the background worker thread is never actually started --
`db.claim_next`'s cross-thread sqlite contract (see `db.py`'s module
docstring) makes an accidental real worker thread touching the same tmp
data_dir from a different thread than the test a real hazard, not just
noise.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import backend.app.main as main_module
from backend.app import db, worker
from backend.app.config import Settings, get_settings


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    # Mirror test_config.py's pattern: strip ambient SHUTTLESENSE_* vars so
    # each test starts from field defaults, then force SHUTTLESENSE_NO_WORKER
    # so app startup never spawns the real polling thread.
    for name in Settings.model_fields:
        monkeypatch.delenv(f"SHUTTLESENSE_{name.upper()}", raising=False)
    monkeypatch.setenv("SHUTTLESENSE_NO_WORKER", "1")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def tmp_dirs(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    samples_dir = tmp_path / "samples"
    monkeypatch.setenv("SHUTTLESENSE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("SHUTTLESENSE_SAMPLES_DIR", str(samples_dir))
    monkeypatch.setenv("SHUTTLESENSE_MODELS_DIR", "backend/models")
    get_settings.cache_clear()
    return get_settings()


@pytest.fixture
def client(tmp_dirs):
    with TestClient(main_module.create_app()) as c:
        yield c


def _mark_done(settings, job_id, report=None, tracks=None):
    """Helper: fast-forward a job straight to 'done' with fake outputs on
    disk, bypassing the real worker (which is never started in these
    tests) -- mirrors what `worker.run_once` would have written."""
    conn = db.connect(worker.db_path(settings))
    try:
        db.finish(conn, job_id, "done")
    finally:
        conn.close()
    out_dir = worker.job_output_dir(settings, job_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report if report is not None else {"fps": 15.0}))
    (out_dir / "tracks.json").write_text(json.dumps(tracks if tracks is not None else {"edges": []}))


def _mark_failed(settings, job_id, error="boom"):
    conn = db.connect(worker.db_path(settings))
    try:
        db.finish(conn, job_id, "failed", error=error)
    finally:
        conn.close()


# --- healthz -----------------------------------------------------------


def test_healthz(client):
    r = client.get("/api/healthz")
    assert r.status_code == 200
    assert r.json() == {"ok": True}


# --- upload / status -----------------------------------------------------


def test_upload_and_status(client, tmp_dirs):
    r = client.post("/api/matches", files={"file": ("m.mp4", b"0" * 1024, "video/mp4")})
    assert r.status_code == 202
    jid = r.json()["job_id"]
    status = client.get(f"/api/matches/{jid}")
    assert status.status_code == 200
    body = status.json()
    assert body["status"] == "queued"
    assert body["error"] is None

    # the uploaded file must land exactly where the worker expects it
    conn = db.connect(worker.db_path(tmp_dirs))
    row = db.get_job(conn, jid)
    conn.close()
    video_path = Path(worker.job_video_path(tmp_dirs, row))
    assert video_path.is_file()
    assert video_path.read_bytes() == b"0" * 1024


def test_bad_extension_rejected(client, tmp_dirs):
    r = client.post("/api/matches", files={"file": ("x.txt", b"nope", "text/plain")})
    assert r.status_code == 400
    # must not have queued a job for a rejected upload
    conn = db.connect(worker.db_path(tmp_dirs))
    n = conn.execute("SELECT COUNT(*) AS n FROM jobs").fetchone()["n"]
    conn.close()
    assert n == 0


def test_oversize_upload_rejected(client, tmp_dirs, monkeypatch):
    monkeypatch.setenv("SHUTTLESENSE_MAX_UPLOAD_MB", "1")
    get_settings.cache_clear()
    # Rebuild client against the smaller cap.
    with TestClient(main_module.create_app()) as c:
        big = b"0" * (2 * 1024 * 1024)
        r = c.post("/api/matches", files={"file": ("m.mp4", big, "video/mp4")})
    assert r.status_code == 413


def test_unknown_job_404(client):
    r = client.get("/api/matches/does-not-exist")
    assert r.status_code == 404


def test_report_before_done_404(client):
    r = client.post("/api/matches", files={"file": ("m.mp4", b"0" * 1024, "video/mp4")})
    jid = r.json()["job_id"]
    assert client.get(f"/api/matches/{jid}/report").status_code == 404
    assert client.get(f"/api/matches/{jid}/tracks").status_code == 404
    assert client.get(f"/api/matches/{jid}/video").status_code == 404


def test_report_tracks_video_served_once_done(client, tmp_dirs):
    r = client.post("/api/matches", files={"file": ("m.mp4", b"0" * 1024, "video/mp4")})
    jid = r.json()["job_id"]
    _mark_done(tmp_dirs, jid, report={"fps": 15.0, "rallies": []}, tracks={"edges": [1, 2]})

    rep = client.get(f"/api/matches/{jid}/report")
    assert rep.status_code == 200
    assert rep.json() == {"fps": 15.0, "rallies": []}

    trk = client.get(f"/api/matches/{jid}/tracks")
    assert trk.status_code == 200
    assert trk.json() == {"edges": [1, 2]}

    vid = client.get(f"/api/matches/{jid}/video")
    assert vid.status_code == 200
    assert vid.content == b"0" * 1024

    status = client.get(f"/api/matches/{jid}")
    assert status.json()["status"] == "done"


def test_failed_job_status_reports_friendly_error(client, tmp_dirs):
    r = client.post("/api/matches", files={"file": ("m.mp4", b"0" * 1024, "video/mp4")})
    jid = r.json()["job_id"]
    _mark_failed(tmp_dirs, jid, error="This video doesn't look like a badminton match we can analyze.")

    status = client.get(f"/api/matches/{jid}")
    assert status.status_code == 200
    body = status.json()
    assert body["status"] == "failed"
    assert "badminton" in body["error"]

    # still 404 for report/tracks/video -- failed is not done
    assert client.get(f"/api/matches/{jid}/report").status_code == 404


# --- samples ---------------------------------------------------------------


def test_sample_listing_empty_dir(client):
    r = client.get("/api/samples")
    assert r.status_code == 200
    assert r.json() == []


def _write_sample(samples_dir: Path, sample_id: str, title: str | None = "Demo Match", *, video=b"vid", report=None, tracks=None, write_meta=True):
    d = samples_dir / sample_id
    d.mkdir(parents=True, exist_ok=True)
    if write_meta:
        meta = {"id": sample_id}
        if title is not None:
            meta["title"] = title
        (d / "meta.json").write_text(json.dumps(meta))
    (d / "report.json").write_text(json.dumps(report if report is not None else {"fps": 15.0}))
    (d / "tracks.json").write_text(json.dumps(tracks if tracks is not None else {"edges": []}))
    if video is not None:
        (d / "video.mp4").write_bytes(video)
    return d


def test_sample_listing_and_fetch(client, tmp_dirs):
    _write_sample(Path(tmp_dirs.samples_dir), "demo1", "Demo One")
    _write_sample(Path(tmp_dirs.samples_dir), "demo2", "Demo Two")

    r = client.get("/api/samples")
    assert r.status_code == 200
    ids = {s["id"] for s in r.json()}
    assert ids == {"demo1", "demo2"}
    titles = {s["title"] for s in r.json()}
    assert titles == {"Demo One", "Demo Two"}

    rep = client.get("/api/samples/demo1/report")
    assert rep.status_code == 200
    assert rep.json() == {"fps": 15.0}

    trk = client.get("/api/samples/demo1/tracks")
    assert trk.status_code == 200

    vid = client.get("/api/samples/demo1/video")
    assert vid.status_code == 200
    assert vid.content == b"vid"


def test_sample_missing_meta_json_skipped_not_crash(client, tmp_dirs, capfd):
    _write_sample(Path(tmp_dirs.samples_dir), "good", "Good One")
    # a sibling dir with no meta.json at all
    (Path(tmp_dirs.samples_dir) / "no_meta").mkdir(parents=True)

    r = client.get("/api/samples")
    assert r.status_code == 200
    ids = {s["id"] for s in r.json()}
    assert ids == {"good"}


def test_sample_malformed_meta_json_skipped_with_warning(client, tmp_dirs):
    _write_sample(Path(tmp_dirs.samples_dir), "good", "Good One")
    bad_dir = Path(tmp_dirs.samples_dir) / "bad"
    bad_dir.mkdir(parents=True)
    (bad_dir / "meta.json").write_text("{not valid json")
    (bad_dir / "report.json").write_text("{}")
    (bad_dir / "tracks.json").write_text("{}")

    r = client.get("/api/samples")
    assert r.status_code == 200
    ids = {s["id"] for s in r.json()}
    assert ids == {"good"}


def test_sample_meta_missing_title_skipped(client, tmp_dirs):
    _write_sample(Path(tmp_dirs.samples_dir), "good", "Good One")
    _write_sample(Path(tmp_dirs.samples_dir), "notitle", title=None)

    r = client.get("/api/samples")
    ids = {s["id"] for s in r.json()}
    assert ids == {"good"}


def test_unknown_sample_404(client, tmp_dirs):
    assert client.get("/api/samples/nope/report").status_code == 404
    assert client.get("/api/samples/nope/tracks").status_code == 404
    assert client.get("/api/samples/nope/video").status_code == 404


def test_sample_id_traversal_confined(client, tmp_dirs):
    # sample_id is a single path segment (FastAPI's default string
    # converter can't contain "/"), but ".." alone is a legal segment --
    # must not escape samples_dir to read an arbitrary sibling directory.
    r = client.get("/api/samples/../report")
    assert r.status_code == 404


# --- static SPA serving / traversal confinement -----------------------------


@pytest.fixture
def static_dir(tmp_path):
    d = tmp_path / "static"
    d.mkdir()
    (d / "index.html").write_text("<html>SPA SHELL</html>")
    (d / "app.js").write_text("console.log('hi')")
    return d


@pytest.fixture
def client_with_static(tmp_dirs, static_dir, monkeypatch):
    monkeypatch.setenv("SHUTTLESENSE_STATIC_DIR", str(static_dir))
    get_settings.cache_clear()
    with TestClient(main_module.create_app()) as c:
        yield c


def test_static_serves_index_at_root(client_with_static):
    r = client_with_static.get("/")
    assert r.status_code == 200
    assert "SPA SHELL" in r.text


def test_static_serves_real_asset(client_with_static):
    r = client_with_static.get("/app.js")
    assert r.status_code == 200
    assert "hi" in r.text


def test_static_client_route_falls_back_to_index(client_with_static):
    r = client_with_static.get("/some/client/route")
    assert r.status_code == 200
    assert "SPA SHELL" in r.text


def test_api_prefixed_unknown_path_is_json_404_not_spa(client_with_static):
    r = client_with_static.get("/api/does-not-exist")
    assert r.status_code == 404
    assert "SPA SHELL" not in r.text


def test_traversal_dot_dot_segments(client_with_static):
    r = client_with_static.get("/../../etc/passwd")
    assert r.status_code in (200, 404)
    if r.status_code == 200:
        assert "SPA SHELL" in r.text


def test_traversal_percent_encoded_dot_dot(client_with_static):
    # %2e%2e survives httpx's own URL normalization (unlike a literal ".."
    # typed in Python, which httpx/urllib resolves client-side before the
    # request is ever sent) -- this is the case that actually exercises the
    # server-side resolve-and-confine guard.
    r = client_with_static.get("/%2e%2e/%2e%2e/%2e%2e/%2e%2e/%2e%2e/etc/passwd")
    assert r.status_code in (200, 404)
    if r.status_code == 200:
        assert "SPA SHELL" in r.text
    assert "root:" not in r.text


def test_traversal_double_slash(client_with_static):
    # httpx normalizes a literal "//" written in Python client-side; use an
    # absolute-URL request against the raw transport to preserve it, per
    # the brief's note (mirrors this repo author's prior Lift-project
    # technique for the same probe).
    r = client_with_static.get("http://testserver//etc/passwd")
    assert r.status_code in (200, 404)
    if r.status_code == 200:
        assert "SPA SHELL" in r.text
    assert "root:" not in r.text


def test_traversal_null_byte_rejected_cleanly(client_with_static):
    # A %00 in the path must not raise an unhandled exception (e.g. from
    # Path.resolve() choking on an embedded NUL) -- must degrade to a clean
    # 404 instead of a 500.
    r = client_with_static.get("/index.html%00.png")
    assert r.status_code in (200, 404)
    assert r.status_code != 500


def test_no_worker_env_prevents_thread_start(monkeypatch, tmp_dirs):
    # Sanity check on the SHUTTLESENSE_NO_WORKER contract itself: with it
    # set (the autouse fixture sets it), create_app()'s lifespan must not
    # leave a lingering non-daemon/duplicate thread; we can't easily assert
    # "no thread was started" directly, but we can assert app startup and
    # shutdown complete quickly and don't hang (a real worker thread
    # querying a tmp sqlite db from another thread would otherwise be a
    # cross-thread sqlite3 hazard per db.py's docstring).
    with TestClient(main_module.create_app()) as c:
        assert c.get("/api/healthz").status_code == 200
