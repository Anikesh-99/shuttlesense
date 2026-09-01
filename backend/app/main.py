"""FastAPI app assembly: wires `routes.router` (the matches/samples API),
starts the background analysis worker (`worker.run_forever`) as a daemon
thread on startup, and -- when `settings.static_dir` is configured -- mounts
the built frontend with a hardened resolve-and-confine SPA static handler.

Worker thread: `SHUTTLESENSE_NO_WORKER=1` (set by tests/CI) skips starting
the thread entirely. This matters beyond "don't waste a thread in tests": a
real worker thread opens its OWN `db.connect()` (per `db.py`'s documented
cross-thread sqlite contract) and would otherwise poll whatever tmp
`data_dir` a given test happens to be using at the moment the thread was
started, which is exactly the kind of test-isolation hazard `db.py`'s module
docstring warns about.

Static serving -- resolve-and-confine pattern (binding, see plan controller
carry-overs): every request path is resolved against the (pre-resolved)
static root and REQUIRED to be `.is_relative_to(root)`, else 404. This is
deliberately NOT a `StaticFiles` mount (which serves 404 for a missing file
today, but is not the audited surface here) -- it's one catch-all route so
the SAME code path handles (a) real static assets, (b) the SPA
`index.html` fallback for client-side routes, AND (c) the traversal guard,
with no risk of the three diverging. Three things this must get right,
each with a probe in `test_api.py`:
  1. `/api`-prefixed paths NEVER fall through to the SPA fallback (checked
     against the raw, not path-param-decoded... actually same value --
     Starlette percent-decodes before route matching either way --
     `request.url.path` before any of our own normalization, so a request
     for an unmatched `/api/*` route gets a JSON 404, never `index.html`).
  2. Repeated slashes are collapsed before filesystem resolution (`//etc/passwd`
     must not be treated as an absolute-from-root escape hatch).
  3. A literal NUL byte in the decoded path (`%00`) must degrade to a clean
     404, not an unhandled `ValueError`/`OSError` out of `Path.resolve()`.
"""
from __future__ import annotations

import os
import re
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse

from backend.app.config import Settings, get_settings
from backend.app.routes import router
from backend.app.worker import run_forever

_REPEATED_SLASHES = re.compile(r"/+")


def _worker_disabled() -> bool:
    return os.environ.get("SHUTTLESENSE_NO_WORKER") == "1"


def _start_worker_thread(settings: Settings) -> threading.Thread | None:
    if _worker_disabled():
        return None
    thread = threading.Thread(target=run_forever, args=(settings,), daemon=True)
    thread.start()
    return thread


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    _start_worker_thread(settings)
    yield


def _mount_static(app: FastAPI, static_dir: str) -> None:
    root = Path(static_dir).resolve()

    @app.get("/{full_path:path}", include_in_schema=False)
    async def spa_fallback(full_path: str, request: Request):
        # /api/* paths are handled entirely by `routes.router`, included
        # BEFORE this catch-all, so only requests for undefined /api/*
        # routes ever reach here -- those must get a JSON 404, never the
        # SPA shell (carry-over #2).
        raw_path = request.url.path
        if raw_path == "/api" or raw_path.startswith("/api/"):
            raise HTTPException(status_code=404, detail="not found")

        # Collapse repeated slashes (e.g. "//etc/passwd") before treating
        # the remainder as a relative filesystem path under `root`.
        normalized = _REPEATED_SLASHES.sub("/", full_path).lstrip("/")

        if "\x00" in normalized:
            # An embedded NUL would raise ValueError out of Path.resolve()
            # on some platforms -- reject up front rather than relying on
            # the except clause below to catch it after the fact.
            raise HTTPException(status_code=404, detail="not found")

        try:
            candidate = (root / normalized).resolve()
        except (ValueError, OSError):
            raise HTTPException(status_code=404, detail="not found")

        try:
            confined = candidate.is_relative_to(root)
        except ValueError:
            confined = False
        if not confined:
            raise HTTPException(status_code=404, detail="not found")

        if candidate.is_file():
            return FileResponse(candidate)

        # Not a real static asset -- treat as a client-side SPA route and
        # fall back to index.html, if present.
        index = root / "index.html"
        if index.is_file():
            return FileResponse(index)
        raise HTTPException(status_code=404, detail="not found")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="ShuttleSense", lifespan=lifespan)
    app.include_router(router)

    if settings.static_dir:
        _mount_static(app, settings.static_dir)

    return app


app = create_app()
