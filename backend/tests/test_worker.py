import json
import sqlite3

import pytest

from backend.app import db, worker
from backend.app.config import Settings
from shuttlesense_core.schemas import MatchReport, RallyInterval, StrokeEvent


def _settings(tmp_path):
    return Settings(
        data_dir=str(tmp_path / "data"),
        models_dir="backend/models",
        samples_dir=str(tmp_path / "samples"),
    )


def _queue_job(settings, filename="m.mp4"):
    conn = db.connect(worker.db_path(settings))
    jid = db.create_job(conn, filename)
    conn.close()
    return jid


def _write_fake_upload(settings, jid, filename="m.mp4"):
    path = worker.job_video_path(settings, {"id": jid, "filename": filename})
    from pathlib import Path

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(b"not a real video")
    return path


def test_run_once_returns_false_when_queue_empty(tmp_path):
    settings = _settings(tmp_path)
    assert worker.run_once(settings) is False


def test_run_once_success_writes_outputs_and_marks_done(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    jid = _queue_job(settings)
    _write_fake_upload(settings, jid)

    fake_report = MatchReport(
        fps=15.0,
        width=1280,
        height=720,
        n_frames=100,
        rallies=[RallyInterval(0, 50)],
        strokes=[StrokeEvent(frame=10, player=0, stroke="smash", confidence=0.9)],
    )
    fake_tracks = {"fps": 15.0, "edges": [], "kpts": [], "scores": []}

    def fake_analyze(video_path, models_dir, target_fps=15.0):
        assert video_path.endswith("m.mp4")
        assert models_dir == settings.models_dir
        return fake_report, fake_tracks

    monkeypatch.setattr(worker, "analyze", fake_analyze)

    assert worker.run_once(settings) is True

    conn = db.connect(worker.db_path(settings))
    row = db.get_job(conn, jid)
    assert row["status"] == "done"
    assert row["error"] is None

    out_dir = worker.job_output_dir(settings, jid)
    report_on_disk = json.loads((out_dir / "report.json").read_text())
    tracks_on_disk = json.loads((out_dir / "tracks.json").read_text())
    assert report_on_disk == fake_report.to_dict()
    assert tracks_on_disk == fake_tracks


def test_run_once_failure_records_friendly_message_for_no_rallies(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    jid = _queue_job(settings)
    _write_fake_upload(settings, jid)

    def fake_analyze(video_path, models_dir, target_fps=15.0):
        raise ValueError("no rallies detected")

    monkeypatch.setattr(worker, "analyze", fake_analyze)

    assert worker.run_once(settings) is True

    conn = db.connect(worker.db_path(settings))
    row = db.get_job(conn, jid)
    assert row["status"] == "failed"
    assert row["error"] == worker.FRIENDLY_NO_RALLY_MSG


def test_run_once_failure_records_raw_message_for_other_errors(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    jid = _queue_job(settings)
    _write_fake_upload(settings, jid)

    def fake_analyze(video_path, models_dir, target_fps=15.0):
        raise RuntimeError("onnxruntime blew up")

    monkeypatch.setattr(worker, "analyze", fake_analyze)

    assert worker.run_once(settings) is True

    conn = db.connect(worker.db_path(settings))
    row = db.get_job(conn, jid)
    assert row["status"] == "failed"
    assert row["error"] == "onnxruntime blew up"


def test_run_once_output_write_failure_marks_job_failed(tmp_path, monkeypatch):
    # Fix round 1 item 5: report/tracks writes moved INSIDE the try -- a
    # write failure (disk full, permissions, whatever) must land the job in
    # "failed" with the real error, not leave it stuck in "processing"
    # forever with no trace on disk.
    settings = _settings(tmp_path)
    jid = _queue_job(settings)
    _write_fake_upload(settings, jid)

    fake_report = MatchReport(fps=15.0, width=1280, height=720, n_frames=1)
    monkeypatch.setattr(
        worker, "analyze", lambda *a, **k: (fake_report, {"fps": 15.0})
    )

    def boom(path, obj):
        raise OSError("disk full")

    monkeypatch.setattr(worker, "_atomic_write_json", boom)

    assert worker.run_once(settings) is True

    conn = db.connect(worker.db_path(settings))
    row = db.get_job(conn, jid)
    assert row["status"] == "failed"
    assert "disk full" in row["error"]


def test_run_forever_stops_looping_when_stop_flag_raised(tmp_path, monkeypatch):
    # run_forever loops forever by contract; drive it via a side-effecting
    # run_once stub and abort the loop with a sentinel exception after N calls
    # rather than actually blocking the test on time.sleep.
    settings = _settings(tmp_path)
    calls = {"n": 0}

    class _Stop(BaseException):
        # BaseException (not Exception): run_forever now deliberately catches
        # `Exception` broadly (fix round 1 item 5, so one bad job can't kill
        # the poll thread) -- a plain Exception-based sentinel here would be
        # silently swallowed by that same broad handler instead of stopping
        # the loop, hanging the test.
        pass

    def fake_run_once(s):
        calls["n"] += 1
        if calls["n"] >= 3:
            raise _Stop
        return False

    monkeypatch.setattr(worker, "run_once", fake_run_once)
    monkeypatch.setattr(worker.time, "sleep", lambda _s: None)

    with pytest.raises(_Stop):
        worker.run_forever(settings, poll_interval=0.0)
    assert calls["n"] == 3


def test_run_forever_treats_operational_error_from_claim_next_as_transient(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    calls = {"n": 0}

    class _Stop(BaseException):
        # see the sibling test above for why this is BaseException, not Exception.
        pass

    def flaky_run_once(s):
        calls["n"] += 1
        if calls["n"] == 1:
            raise sqlite3.OperationalError("database is locked")
        raise _Stop

    monkeypatch.setattr(worker, "run_once", flaky_run_once)
    monkeypatch.setattr(worker.time, "sleep", lambda _s: None)

    with pytest.raises(_Stop):
        worker.run_forever(settings, poll_interval=0.0)
    assert calls["n"] == 2


def test_run_forever_survives_unexpected_exception_from_run_once(tmp_path, monkeypatch, capsys):
    # Fix round 1 item 5: run_once raising something other than
    # sqlite3.OperationalError (a real bug, a disk-full error, whatever) must
    # NOT kill the poll thread -- run_forever logs it (traceback.print_exc())
    # and keeps polling.
    settings = _settings(tmp_path)
    calls = {"n": 0}

    class _Stop(BaseException):
        pass

    def flaky_run_once(s):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("totally unexpected")
        raise _Stop

    monkeypatch.setattr(worker, "run_once", flaky_run_once)
    monkeypatch.setattr(worker.time, "sleep", lambda _s: None)

    with pytest.raises(_Stop):
        worker.run_forever(settings, poll_interval=0.0)
    assert calls["n"] == 2
    assert "totally unexpected" in capsys.readouterr().err
