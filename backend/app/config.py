"""Backend runtime settings.

Values are sourced from environment variables prefixed ``SHUTTLESENSE_``
(e.g. ``SHUTTLESENSE_DATA_DIR=/srv/data``), falling back to the field
defaults below. Use :func:`get_settings` to obtain a process-wide cached
instance rather than constructing :class:`Settings` directly, so the whole
app agrees on one configuration snapshot.
"""
from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SHUTTLESENSE_")

    data_dir: str = "data"
    models_dir: str = "backend/models"
    samples_dir: str = "backend/samples"
    static_dir: str | None = None
    max_upload_mb: int = 100
    max_duration_s: int = 95
    target_fps: float = 15.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
