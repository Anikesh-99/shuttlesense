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

import asyncio
import json
import threading
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


@pytest.mark.parametrize("evil_filename", ["../pwn.mp4", "/tmp/pwn.mp4"])
def test_upload_filename_traversal_rejected(client, tmp_dirs, evil_filename):
    # C1 regression: the raw Content-Disposition filename must never be
    # trusted as a filesystem path. Both a relative-escape ("../pwn.mp4")
    # and an absolute path ("/tmp/pwn.mp4") must be rejected outright (400),
    # not silently reduced to "pwn.mp4" and accepted.
    r = client.post(
        "/api/matches", files={"file": (evil_filename, b"0" * 1024, "video/mp4")}
    )
    assert r.status_code == 400

    data_dir = Path(tmp_dirs.data_dir)
    uploads_root = data_dir / "uploads"

    # Nothing should exist outside <data_dir>/uploads/<job_id>/ as a result
    # of this request: no job row, and no "pwn.mp4" written anywhere
    # (specifically not at /tmp/pwn.mp4, nor as a sibling of data_dir).
    conn = db.connect(worker.db_path(tmp_dirs))
    n = conn.execute("SELECT COUNT(*) AS n FROM jobs").fetchone()["n"]
    conn.close()
    assert n == 0

    assert not Path("/tmp/pwn.mp4").exists()
    assert not (data_dir.parent / "pwn.mp4").exists()
    # If anything WAS staged under uploads_root, it must be confined to a
    # staging-* directory that gets cleaned up -- assert no stray file
    # named exactly "pwn.mp4" survives anywhere under uploads_root either.
    if uploads_root.is_dir():
        leaked = list(uploads_root.rglob("pwn.mp4"))
        assert leaked == []


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


def _write_sample(samples_dir: Path, sample_id: str, title: str | None = "Demo Match", *, video=b"vid", report=None, tracks=None, write_meta=True, players=None):
    d = samples_dir / sample_id
    d.mkdir(parents=True, exist_ok=True)
    if write_meta:
        meta = {"id": sample_id}
        if title is not None:
            meta["title"] = title
        if players is not None:
            meta["players"] = players
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


def test_sample_listing_includes_players_when_present(client, tmp_dirs):
    _write_sample(
        Path(tmp_dirs.samples_dir), "demo1", "Demo One",
        players=["Chou Tien Chen", "Anders Antonsen"],
    )
    _write_sample(Path(tmp_dirs.samples_dir), "demo2", "Demo Two")  # no players

    r = client.get("/api/samples")
    by_id = {s["id"]: s for s in r.json()}
    assert by_id["demo1"]["players"] == ["Chou Tien Chen", "Anders Antonsen"]
    assert "players" not in by_id["demo2"]


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
    assert client.get("/api/samples/nope/meta").status_code == 404


# --- /samples/{id}/meta (Fix round 1) ---------------------------------------


def test_sample_meta_endpoint_with_players(client, tmp_dirs):
    _write_sample(
        Path(tmp_dirs.samples_dir), "demo1", "Demo One",
        players=["Chou Tien Chen", "Anders Antonsen"],
    )
    r = client.get("/api/samples/demo1/meta")
    assert r.status_code == 200
    assert r.json() == {
        "id": "demo1",
        "title": "Demo One",
        "players": ["Chou Tien Chen", "Anders Antonsen"],
    }


def test_sample_meta_endpoint_without_players_omits_key(client, tmp_dirs):
    _write_sample(Path(tmp_dirs.samples_dir), "demo1", "Demo One")
    r = client.get("/api/samples/demo1/meta")
    assert r.status_code == 200
    assert r.json() == {"id": "demo1", "title": "Demo One"}


def test_sample_meta_endpoint_malformed_players_ignored(client, tmp_dirs):
    # A malformed "players" (wrong length, non-string entries, etc.) must
    # not fail the whole meta fetch -- id/title still come through, the
    # bad "players" key is just dropped (mirrors list_samples' defensive
    # posture for meta.json as a whole).
    _write_sample(Path(tmp_dirs.samples_dir), "demo1", "Demo One", players=["OnlyOneName"])
    r = client.get("/api/samples/demo1/meta")
    assert r.status_code == 200
    assert r.json() == {"id": "demo1", "title": "Demo One"}


def test_sample_meta_endpoint_missing_meta_json_404(client, tmp_dirs):
    _write_sample(Path(tmp_dirs.samples_dir), "demo1", "Demo One", write_meta=False)
    r = client.get("/api/samples/demo1/meta")
    assert r.status_code == 404


def test_sample_id_traversal_confined(client, tmp_dirs):
    # sample_id is a single path segment (FastAPI's default string
    # converter can't contain "/"), but ".." alone is a legal segment --
    # must not escape samples_dir to read an arbitrary sibling directory.
    r = client.get("/api/samples/../report")
    assert r.status_code == 404


def test_sample_id_null_byte_rejected_cleanly(client, tmp_dirs):
    # C2 regression: `_confined_sample_dir` used to call `.resolve()`
    # outside any try/except, so a %00-encoded sample_id raised an
    # unhandled ValueError -> 500. Must degrade to a clean 404 instead.
    for path in (
        "/api/samples/%00/report",
        "/api/samples/%00/tracks",
        "/api/samples/%00/video",
        "/api/samples/good%00evil/report",
    ):
        r = client.get(path)
        assert r.status_code == 404, f"{path} -> {r.status_code}"


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
    # httpx/urllib resolves a LITERAL ".." typed in Python client-side
    # before the request is ever sent (you can't navigate above an
    # absolute URL's root), so this specific probe never reaches the
    # server as a real ".." segment -- included per the brief, but
    # `test_traversal_raw_asgi_literal_dot_dot` below (which bypasses
    # httpx's URL parsing entirely) is what actually proves the server-side
    # guard handles this case; this one just confirms no leak either way.
    r = client_with_static.get("/../../etc/passwd")
    assert r.status_code in (200, 404)
    if r.status_code == 200:
        assert "SPA SHELL" in r.text
    assert "root:" not in r.text


def test_traversal_percent_encoded_dot_dot(client_with_static):
    # %2e%2e survives httpx's own URL normalization (unlike a literal ".."
    # typed in Python, which httpx/urllib resolves client-side before the
    # request is ever sent) -- this IS the case that actually exercises the
    # server-side resolve-and-confine guard over the wire, so it must be a
    # strict 404, not merely "not a leak".
    r = client_with_static.get("/%2e%2e/%2e%2e/%2e%2e/%2e%2e/%2e%2e/etc/passwd")
    assert r.status_code == 404
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


def test_traversal_double_slash_api_prefix_is_json_404_not_spa(client_with_static):
    # I3 regression: the /api guard used to check ONLY the raw
    # (pre-slash-collapse) request path, so "//api/..." -- which does not
    # start with the literal substring "/api/" -- fell through to the SPA
    # fallback and returned 200 HTML instead of a JSON 404.
    r = client_with_static.get("http://testserver//api/does-not-exist")
    assert r.status_code == 404
    assert "SPA SHELL" not in r.text


def test_traversal_null_byte_rejected_cleanly(client_with_static):
    # A %00 in the path must not raise an unhandled exception (e.g. from
    # Path.resolve() choking on an embedded NUL) -- must degrade to a clean
    # 404 instead of a 500.
    r = client_with_static.get("/index.html%00.png")
    assert r.status_code in (200, 404)
    assert r.status_code != 500


def _raw_asgi_get_status(app, raw_path: str) -> int:
    """Drive `app` as a bare ASGI callable with a hand-built HTTP scope
    whose `path` is exactly `raw_path` -- bypassing httpx/urllib's own
    client-side URL normalization entirely (which would otherwise resolve
    a literal ".." before the request is ever "sent"). This is the only way
    to actually deliver a raw, un-collapsed "/../../etc/passwd" to the
    server and observe what the resolve-and-confine guard does with it."""

    async def _run() -> int:
        scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "method": "GET",
            "scheme": "http",
            "path": raw_path,
            "raw_path": raw_path.encode("utf-8"),
            "query_string": b"",
            "root_path": "",
            "headers": [(b"host", b"testserver")],
            "server": ("testserver", 80),
            "client": ("testclient", 50000),
        }
        status_box: dict = {}

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message):
            if message["type"] == "http.response.start":
                status_box["status"] = message["status"]

        await app(scope, receive, send)
        return status_box["status"]

    return asyncio.run(_run())


def test_traversal_raw_asgi_literal_dot_dot(tmp_dirs, static_dir, monkeypatch):
    # Deliver a literal, un-collapsed "/../../etc/passwd" straight to the
    # ASGI app (bypassing httpx's client-side URL normalization -- see
    # `_raw_asgi_get_status`'s docstring) -- this is the real proof the
    # resolve-and-confine guard rejects an actual ".."-escaping path, not
    # just a client-side-neutered version of one.
    monkeypatch.setenv("SHUTTLESENSE_STATIC_DIR", str(static_dir))
    get_settings.cache_clear()
    app = main_module.create_app()
    status = _raw_asgi_get_status(app, "/../../etc/passwd")
    assert status == 404


def test_static_symlink_escape_confined(client_with_static, static_dir):
    # The assertion that actually distinguishes real resolve()-based
    # confinement from a naive string-prefix check: a symlink INSIDE the
    # static root whose TARGET lives outside it. A prefix check on the
    # unresolved path would see "static/leak.txt" (looks confined); only
    # resolving symlinks before the is_relative_to() check catches this.
    target = Path("/etc/passwd")
    if not target.is_file():
        pytest.skip("no /etc/passwd on this platform to symlink to")
    link = static_dir / "leak.txt"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlink creation not permitted in this environment")

    r = client_with_static.get("/leak.txt")
    assert r.status_code == 404
    assert "root:" not in r.text


def test_no_worker_env_returns_none_and_starts_no_thread(monkeypatch, tmp_dirs):
    # I6: with SHUTTLESENSE_NO_WORKER=1 (set by the autouse _clean_env
    # fixture), the helper must return None -- not merely "the app doesn't
    # hang" -- and must not have started a thread at all.
    before = {t.ident for t in threading.enumerate()}
    result = main_module._start_worker_thread(tmp_dirs)
    after = {t.ident for t in threading.enumerate()}
    assert result is None
    assert after == before


def test_worker_thread_starts_when_not_disabled(monkeypatch, tmp_dirs):
    # I6: without SHUTTLESENSE_NO_WORKER, _start_worker_thread must return
    # a live Thread. run_forever is monkeypatched to a fast no-op so the
    # thread exits almost immediately rather than polling a tmp db forever
    # (avoids the exact db.py cross-thread hazard this whole env var exists
    # to prevent, while still proving the thread-start path itself works).
    monkeypatch.delenv("SHUTTLESENSE_NO_WORKER", raising=False)

    ran = threading.Event()

    def fake_run_forever(settings):
        ran.set()

    monkeypatch.setattr(main_module, "run_forever", fake_run_forever)

    thread = main_module._start_worker_thread(tmp_dirs)
    assert thread is not None
    assert isinstance(thread, threading.Thread)
    assert thread.daemon is True
    thread.join(timeout=2)
    assert ran.is_set()
