"""HTTP surface for ShuttleSense: the matches (upload/analysis job) API and
the pre-baked samples API.

Endpoints (all under the `/api` prefix):
    GET  /api/healthz                       -> {"ok": true}
    POST /api/matches           (multipart) -> 202 {"job_id"}; 400 bad
                                                extension/unreadable video
                                                that's over the duration cap;
                                                413 over the size cap.
    GET  /api/matches/{job_id}              -> {"status", "error"}; 404 if
                                                unknown.
    GET  /api/matches/{job_id}/report       -> report.json (FileResponse)
    GET  /api/matches/{job_id}/tracks       -> tracks.json (FileResponse)
    GET  /api/matches/{job_id}/video        -> uploaded video (FileResponse)
    GET  /api/samples                       -> [{"id", "title"}]
    GET  /api/samples/{id}/meta             -> {"id", "title", "players"?}
                                                (Fix round 1: "players" is
                                                [name0, name1] real names,
                                                present only when the sample
                                                has ShuttleSet-labeled
                                                ground truth -- see
                                                `_load_sample_meta`)
    GET  /api/samples/{id}/report|tracks|video

Read-path gate (binding, see plan controller carry-overs): report/tracks/
video for a *job* are 404 until `db.get_job(...)["status"] == "done"` --
never served off a bare "does report.json exist on disk" check. This matters
because `worker.run_once` can, in principle, leave a valid `report.json`/
`tracks.json` beside a job row that a *later* write step still failed on
(see `worker.py`'s module docstring: both `analyze()` and the JSON writes
happen inside one try/except, but a crash between the two `_atomic_write_json`
calls -- e.g. disk-full on `tracks.json` after `report.json` already landed
-- would leave a good `report.json` next to a `status="failed"` row). Gating
on the DB row's status, not file existence, is what makes that safe.

Trust boundary: `report.json`/`tracks.json` under a job's or sample's output
directory are files THIS SYSTEM wrote (worker.py / Task 19's
`scripts/build_samples.py`) -- they are served as opaque files, never
re-parsed/re-validated here. `samples/<id>/meta.json`, by contrast, is read
from disk at LIST time and is not something this codebase writes at request
time in the same trusted sense (Task 19's script writes it once, offline,
and a future deploy could plausibly hand-edit or mis-generate one) -- so
`list_samples` treats it defensively: missing/malformed/incomplete meta.json
skips that one sample directory with a warning, never a 500 for the whole
listing (see `_load_sample_meta`).
"""
from __future__ import annotations

import shutil
import uuid
import warnings
from json import JSONDecodeError, loads as json_loads
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from backend.app import db, worker
from backend.app.config import Settings, get_settings

ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv"}
SAMPLE_VIDEO_FILENAME = "video.mp4"  # fixed convention, see task-19-brief.md
UPLOAD_CHUNK_BYTES = 1024 * 1024

router = APIRouter(prefix="/api")


def video_duration_s(path: str) -> float:
    """Probe a saved video file's duration in seconds via OpenCV's frame
    count / fps, WITHOUT decoding the whole file. Deliberately tolerant of
    garbage/unreadable input (frame_count and/or fps both read back as 0
    from an invalid file) -- returns 0.0 rather than raising, since an
    upload that isn't actually a readable video should fail later, in
    `analyze()`, with its real error, not be misclassified as "too long"
    here."""
    import cv2

    cap = cv2.VideoCapture(path)
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    return float(n / fps) if fps else 0.0


def _db_connect(settings: Settings):
    return db.connect(worker.db_path(settings))


def _get_job_row(settings: Settings, job_id: str):
    conn = _db_connect(settings)
    try:
        return db.get_job(conn, job_id)
    finally:
        conn.close()


def _require_done_job(settings: Settings, job_id: str):
    """404s unless the job exists AND its DB row says `status == "done"` --
    see module docstring's "Read-path gate" note for why this must be the
    DB row's status, not on-disk file existence."""
    row = _get_job_row(settings, job_id)
    if row is None or row["status"] != "done":
        raise HTTPException(status_code=404, detail="job not found or not finished")
    return row


@router.get("/healthz")
def healthz() -> dict:
    return {"ok": True}


def _sanitize_upload_filename(raw_name: str) -> str:
    """Reduce a client-supplied `Content-Disposition` filename to a bare,
    single-component filename safe to join under a staging directory --
    or raise 400.

    `Path(...).name` alone strips any directory components ("../x.mp4" ->
    "x.mp4", "/abs/x.mp4" -> "x.mp4"), but silently ACCEPTING that
    stripped-down name would still be surprising/wrong behavior for a
    security-relevant validator (the client asked to write "../x.mp4" and
    we'd silently write "x.mp4" instead without telling them) -- so instead
    we REJECT outright whenever the raw name contains any directory
    component at all (`safe_name != raw_name`), on top of the emptiness
    checks. This is what makes `POST .../matches` with
    `filename="../pwn.mp4"` or `filename="/tmp/pwn.mp4"` a 400, not a
    silent-success-as-"pwn.mp4"."""
    safe_name = Path(raw_name).name
    if not safe_name or safe_name in (".", "..") or safe_name != raw_name:
        raise HTTPException(status_code=400, detail="invalid filename")
    return safe_name


@router.post("/matches", status_code=202)
async def upload_match(
    file: UploadFile = File(...), settings: Settings = Depends(get_settings)
) -> dict:
    raw_name = file.filename or "upload"
    safe_name = _sanitize_upload_filename(raw_name)

    suffix = Path(safe_name).suffix.lower()
    if suffix not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type {suffix!r}; allowed: "
            + ", ".join(sorted(ALLOWED_VIDEO_EXTENSIONS)),
        )

    # Save to a throwaway staging directory FIRST, validate, and only THEN
    # create the DB job row (with the id used for the final path) -- a
    # rejected upload (bad size/duration) must never leave a "queued" row
    # with no video behind it for the worker to trip over.
    uploads_root = Path(settings.data_dir) / "uploads"
    staging_dir = uploads_root / f"staging-{uuid.uuid4().hex}"
    staging_dir.mkdir(parents=True, exist_ok=True)
    staged_path = staging_dir / safe_name

    # Belt-and-braces confinement check: `safe_name` is already a bare,
    # single-component filename (validated above), so this should never
    # actually trip -- but re-deriving the guarantee from `resolve()` +
    # `is_relative_to()`, rather than trusting the string-level checks
    # above alone, costs nothing and doesn't rely on getting every future
    # edge case of `_sanitize_upload_filename` right.
    resolved_staged = staged_path.resolve()
    resolved_staging_dir = staging_dir.resolve()
    if not resolved_staged.is_relative_to(resolved_staging_dir):
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail="invalid filename")

    max_bytes = settings.max_upload_mb * 1024 * 1024

    try:
        size = 0
        oversize = False
        with open(staged_path, "wb") as out:
            while True:
                chunk = await file.read(UPLOAD_CHUNK_BYTES)
                if not chunk:
                    break
                size += len(chunk)
                if size > max_bytes:
                    oversize = True
                    break
                out.write(chunk)
        if oversize:
            # Backstop for clients that lie about/omit Content-Length --
            # the primary check is `main.py`'s request middleware, which
            # rejects an oversize request before Starlette ever reads the
            # body (see that module's docstring, I4).
            raise HTTPException(
                status_code=413,
                detail=f"File exceeds the {settings.max_upload_mb}MB upload limit",
            )

        try:
            duration = video_duration_s(str(staged_path))
        except Exception:  # noqa: BLE001 -- an unreadable file surfaces its real error later, in analyze()
            duration = 0.0
        if duration > settings.max_duration_s:
            raise HTTPException(
                status_code=400,
                detail=f"Video is ~{duration:.0f}s, exceeds the "
                f"{settings.max_duration_s}s limit",
            )
    except Exception:
        # ANY failure past this point -- not just the HTTPExceptions raised
        # above, but e.g. an unexpected OSError while writing -- must not
        # leak the staging directory.
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise
    finally:
        await file.close()

    # Final-review fix: db.create_job + the staging->final move must be
    # inside the SAME staging-cleanup guard as the write/validate steps
    # above -- an unhandled failure in either (e.g. a DB error, or
    # shutil.move hitting a permissions/disk error) must not leak a
    # `staging-<uuid>/` directory under uploads_root the way it would if
    # this were left as a bare, unguarded sequence after the try/except
    # above.
    try:
        conn = _db_connect(settings)
        try:
            # Store the sanitized name, never the raw client-supplied one --
            # a future reader of this row (e.g. `get_match_video` below)
            # must be able to trust `row["filename"]` is a bare filename
            # with no directory components, even though we additionally
            # re-confine it defensively there too (belt-and-braces for any
            # PRE-EXISTING row written before this fix landed).
            job_id = db.create_job(conn, safe_name)
        finally:
            conn.close()

        final_dir = uploads_root / job_id
        uploads_root.mkdir(parents=True, exist_ok=True)
        shutil.move(str(staging_dir), str(final_dir))
        # final_dir / safe_name now matches worker.job_video_path's fixed
        # convention (`<data_dir>/uploads/<job_id>/<filename>`).
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise

    return {"job_id": job_id}


@router.get("/matches/{job_id}")
def get_match_status(job_id: str, settings: Settings = Depends(get_settings)) -> dict:
    row = _get_job_row(settings, job_id)
    if row is None:
        raise HTTPException(status_code=404, detail="unknown job id")
    return {"status": row["status"], "error": row["error"]}


@router.get("/matches/{job_id}/report")
def get_match_report(job_id: str, settings: Settings = Depends(get_settings)) -> FileResponse:
    _require_done_job(settings, job_id)
    path = worker.job_output_dir(settings, job_id) / "report.json"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="report not found")
    return FileResponse(path, media_type="application/json")


@router.get("/matches/{job_id}/tracks")
def get_match_tracks(job_id: str, settings: Settings = Depends(get_settings)) -> FileResponse:
    _require_done_job(settings, job_id)
    path = worker.job_output_dir(settings, job_id) / "tracks.json"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="tracks not found")
    return FileResponse(path, media_type="application/json")


@router.get("/matches/{job_id}/video")
def get_match_video(job_id: str, settings: Settings = Depends(get_settings)) -> FileResponse:
    row = _require_done_job(settings, job_id)
    # `worker.job_video_path` builds `<data_dir>/uploads/<job_id>/<filename>`
    # from the DB row's stored `filename`. That column is sanitized to a
    # bare filename at write time by `upload_match` (see
    # `_sanitize_upload_filename`), but a row written before that fix
    # landed -- or written by any future code path that doesn't go through
    # `upload_match` -- could still carry a `filename` with directory
    # components in it. Re-confine here too so a pre-existing/bad row can
    # never become a read primitive outside this job's own upload
    # directory.
    job_dir = (Path(settings.data_dir) / "uploads" / row["id"]).resolve()
    path = Path(worker.job_video_path(settings, row)).resolve()
    try:
        confined = path.is_relative_to(job_dir)
    except ValueError:
        confined = False
    if not confined or not path.is_file():
        raise HTTPException(status_code=404, detail="video not found")
    return FileResponse(path)


def _load_sample_meta(meta_path: Path, sample_id: str) -> dict | None:
    """Defensively parse a sample's `meta.json`. Returns `None` (with a
    warning, never an exception) for anything short of a well-formed
    `{"title": str, ...}` object -- untrusted-JSON boundary per module
    docstring.

    Optional `"players"` field (Task 19 Fix round 1): a `[name0, name1]`
    pair of real competitor names, present only for samples built from
    ShuttleSet-labeled ground truth (see `scripts/build_samples.py`). This
    is DELIBERATELY a different identity axis than the on-screen skeleton
    colors -- `name0`/`name1` here index the same match-scoped "eventual
    match winner"/"loser" identity `report.rallies[].winner` (and therefore
    the frontend's score-race chart) uses, per ShuttleSet's `player`
    convention (`training/notes/shuttleset-format.md` (f)), NOT the
    pipeline's per-frame court-side skeleton slot -- there is no reliable
    correspondence between the two (empirically checked, see
    task-19-report.md's "Fix round 1"), so the frontend must never present
    `players[i]` as "the person drawn in player-color `i`"; it's only ever
    used as the score-race series label. Included in the returned dict only
    when well-formed (exactly 2 non-empty strings); malformed/absent
    `"players"` degrades to simply omitting the key, never failing the
    whole meta parse (title/id are the only required fields)."""
    try:
        raw = meta_path.read_text()
    except OSError as exc:
        warnings.warn(f"sample {sample_id!r}: could not read meta.json ({exc})")
        return None
    try:
        meta = json_loads(raw)
    except JSONDecodeError as exc:
        warnings.warn(f"sample {sample_id!r}: malformed meta.json ({exc})")
        return None
    if not isinstance(meta, dict):
        warnings.warn(f"sample {sample_id!r}: meta.json is not a JSON object")
        return None
    title = meta.get("title")
    if not isinstance(title, str) or not title:
        warnings.warn(f"sample {sample_id!r}: meta.json missing a non-empty 'title'")
        return None
    out = {"id": sample_id, "title": title}
    players = meta.get("players")
    if (
        isinstance(players, list)
        and len(players) == 2
        and all(isinstance(p, str) and p for p in players)
    ):
        out["players"] = players
    elif players is not None:
        warnings.warn(
            f"sample {sample_id!r}: meta.json has a malformed 'players' field "
            "(expected [name0, name1], 2 non-empty strings) -- ignoring it, "
            "not failing the whole meta"
        )
    return out


@router.get("/samples")
def list_samples(settings: Settings = Depends(get_settings)) -> list[dict]:
    root = Path(settings.samples_dir)
    if not root.is_dir():
        # Task 19 populates samples_dir; it may legitimately not exist yet
        # (or exist-but-empty) at this point in the branch -- both are a
        # normal empty listing, not an error.
        return []
    out = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        meta_path = entry / "meta.json"
        if not meta_path.is_file():
            continue
        parsed = _load_sample_meta(meta_path, entry.name)
        if parsed is not None:
            out.append(parsed)
    return out


def _confined_sample_dir(settings: Settings, sample_id: str) -> Path:
    """Resolve `sample_id` against `samples_dir` and confine it there.
    FastAPI's default (non-`:path`) string converter already rejects any
    segment containing "/", but ".." alone is a legal single path segment
    -- reusing the resolve-and-confine pattern here (see `main.py`'s static
    mount for the fuller version/rationale) closes that off defensively.

    A literal NUL byte (from a `%00`-encoded segment) must degrade to a
    clean 404, not an unhandled `ValueError` out of `Path.resolve()` --
    checked up front AND the `resolve()` call itself is wrapped, mirroring
    `main.py`'s static-mount guard."""
    if "\x00" in sample_id:
        raise HTTPException(status_code=404, detail="unknown sample id")
    root = Path(settings.samples_dir).resolve()
    try:
        candidate = (root / sample_id).resolve()
    except (ValueError, OSError):
        raise HTTPException(status_code=404, detail="unknown sample id")
    try:
        confined = candidate.is_relative_to(root)
    except ValueError:
        confined = False
    if not confined or not candidate.is_dir():
        raise HTTPException(status_code=404, detail="unknown sample id")
    return candidate


@router.get("/samples/{sample_id}/meta")
def get_sample_meta(sample_id: str, settings: Settings = Depends(get_settings)) -> dict:
    """Task 19 Fix round 1: a per-sample meta fetch (`list_samples` above
    only surfaces `{id, title}` for every sample at LIST time; the Report
    page needs one sample's optional `players` field too, which isn't part
    of that list contract). Deliberately returns the VALIDATED dict from
    `_load_sample_meta` (id/title/optional players), not a raw
    `FileResponse` of meta.json -- unlike report.json/tracks.json (files
    THIS SYSTEM wrote and serves opaquely, per the module docstring's trust
    boundary), meta.json gets the same defensive parse here it already gets
    at list time, so a malformed/hand-edited meta.json degrades to 404
    rather than handing the frontend garbage."""
    d = _confined_sample_dir(settings, sample_id)
    meta_path = d / "meta.json"
    if not meta_path.is_file():
        raise HTTPException(status_code=404, detail="meta not found")
    parsed = _load_sample_meta(meta_path, sample_id)
    if parsed is None:
        raise HTTPException(status_code=404, detail="meta not found")
    return parsed


@router.get("/samples/{sample_id}/report")
def get_sample_report(sample_id: str, settings: Settings = Depends(get_settings)) -> FileResponse:
    d = _confined_sample_dir(settings, sample_id)
    path = d / "report.json"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="report not found")
    return FileResponse(path, media_type="application/json")


@router.get("/samples/{sample_id}/tracks")
def get_sample_tracks(sample_id: str, settings: Settings = Depends(get_settings)) -> FileResponse:
    d = _confined_sample_dir(settings, sample_id)
    path = d / "tracks.json"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="tracks not found")
    return FileResponse(path, media_type="application/json")


@router.get("/samples/{sample_id}/video")
def get_sample_video(sample_id: str, settings: Settings = Depends(get_settings)) -> FileResponse:
    d = _confined_sample_dir(settings, sample_id)
    path = d / SAMPLE_VIDEO_FILENAME
    if not path.is_file():
        raise HTTPException(status_code=404, detail="video not found")
    return FileResponse(path)
