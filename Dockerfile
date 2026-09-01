# ShuttleSense -- single-container image: static React frontend served by
# the FastAPI backend, which serves the pre-baked demo samples and runs the
# upload -> analysis pipeline (pose extraction + stroke/rally ONNX models).
#
# NOTE on Python version (deviation from task-20-brief.md's literal
# `python:3.11-slim` skeleton, documented per controller carry-over #5):
# this repo's dev venv is Python 3.12.8 (`.venv/bin/python --version`) and
# `core/pyproject.toml` only requires `>=3.11` (no upper pin), so 3.12 is a
# strict superset of what's required and matches what's actually been
# tested against locally. Using `python:3.12-slim` here rather than 3.11.

# ---- Stage 1: build the frontend static bundle -----------------------
FROM node:20-slim AS ui
WORKDIR /ui
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ---- Stage 2: runtime image --------------------------------------------
FROM python:3.12-slim

# ffmpeg: required by the analysis pipeline (core/backend video handling).
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy just the bits needed to resolve dependencies first, so dependency
# layers cache independently of application-code edits.
COPY core/ core/
COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt -e ./core

# rtmlib GUI-opencv conflict (binding, see plan controller carry-overs #1):
# rtmlib is intentionally NOT listed in backend/requirements.txt because its
# default dependency metadata pulls in opencv-python/opencv-contrib-python
# (GUI-linked builds that conflict with the headless-only opencv stack this
# image needs). backend/requirements.txt above already installs
# opencv-python-headless + onnxruntime + numpy, which satisfy rtmlib's real
# runtime imports -- so install rtmlib itself with --no-deps to pull in
# only its own code, not its (unwanted) declared dependencies.
RUN pip install --no-cache-dir --no-deps rtmlib

# Build-time self-check: fail the image build (not a later `docker run`) if
# rtmlib doesn't import cleanly, or if a non-headless opencv distribution
# somehow ended up installed alongside opencv-python-headless (the two
# packages both provide the `cv2` import name and silently shadow one
# another, which is exactly the GUI-library conflict this whole carry-over
# exists to prevent in a headless container with no GUI libs).
RUN python -c "import rtmlib; import cv2; print('rtmlib OK, cv2 at', cv2.__file__)" \
    && ( pip show opencv-python >/dev/null 2>&1 && \
         (echo 'FATAL: non-headless opencv-python is installed alongside opencv-python-headless' && exit 1) \
         || echo 'OK: opencv-python (non-headless) is not installed' )

COPY backend/ backend/
COPY --from=ui /ui/dist static/

RUN useradd -m -u 1000 appuser && mkdir -p /app/data && chown -R appuser /app/data
USER appuser

# NOTE (deviation from task-20-brief.md's literal `ENV STATIC_DIR=...`,
# discovered via this task's container run verification): `backend/app/
# config.py`'s `Settings` uses `env_prefix="SHUTTLESENSE_"`
# (pydantic-settings), so the actual env var the app reads is
# `SHUTTLESENSE_STATIC_DIR`, not a bare `STATIC_DIR` -- confirmed against
# `backend/tests/test_api.py`'s own fixtures, which set
# `SHUTTLESENSE_STATIC_DIR`. A bare `STATIC_DIR` is silently ignored by
# pydantic-settings (no error, `settings.static_dir` just stays `None`),
# which made the SPA fallback route never get registered at all -- caught
# by this task's `docker run` + `curl /` verification returning FastAPI's
# generic 404 instead of index.html.
ENV SHUTTLESENSE_STATIC_DIR=/app/static PYTHONUNBUFFERED=1
EXPOSE 8000

CMD ["sh", "-c", "uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
