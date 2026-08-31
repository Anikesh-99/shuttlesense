import pytest

from backend.app.config import Settings, get_settings


@pytest.fixture(autouse=True)
def _clean_settings_env(monkeypatch):
    # get_settings() is lru_cache'd; make sure each test starts and ends
    # with a clean cache so env-var monkeypatches in one test can't leak
    # a stale Settings instance into another. Also strip any ambient
    # SHUTTLESENSE_* env vars from the real environment (e.g. set by a
    # developer's shell or CI) so tests -- especially
    # test_defaults_match_binding_list -- see the field defaults rather
    # than whatever happens to be set outside the test.
    for name in Settings.model_fields:
        monkeypatch.delenv(f"SHUTTLESENSE_{name.upper()}", raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_defaults_match_binding_list():
    s = Settings()
    assert s.data_dir == "data"
    assert s.models_dir == "backend/models"
    assert s.samples_dir == "backend/samples"
    assert s.static_dir is None
    assert s.max_upload_mb == 100
    assert s.max_duration_s == 95
    assert s.target_fps == 15.0


def test_env_prefix_is_shuttlesense(monkeypatch):
    monkeypatch.setenv("SHUTTLESENSE_DATA_DIR", "/srv/data")
    monkeypatch.setenv("SHUTTLESENSE_MAX_UPLOAD_MB", "250")
    monkeypatch.setenv("SHUTTLESENSE_TARGET_FPS", "30")
    s = Settings()
    assert s.data_dir == "/srv/data"
    assert s.max_upload_mb == 250
    assert s.target_fps == 30.0


def test_unprefixed_env_var_is_ignored(monkeypatch):
    # Wrong/missing prefix must not affect the field -- guards against a
    # typo'd env-prefix (e.g. "SHUTTLESHENSE_") silently doing nothing.
    monkeypatch.setenv("DATA_DIR", "/should/not/apply")
    monkeypatch.setenv("MAX_UPLOAD_MB", "999")
    s = Settings()
    assert s.data_dir == "data"
    assert s.max_upload_mb == 100


def test_get_settings_is_cached(monkeypatch):
    first = get_settings()
    second = get_settings()
    assert first is second

    monkeypatch.setenv("SHUTTLESENSE_MAX_UPLOAD_MB", "42")
    # Still the cached (pre-monkeypatch) instance until cleared.
    assert get_settings() is first
    assert get_settings().max_upload_mb == 100

    get_settings.cache_clear()
    refreshed = get_settings()
    assert refreshed is not first
    assert refreshed.max_upload_mb == 42
