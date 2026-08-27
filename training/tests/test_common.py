import json

import pytest

from training.common import resolve_splits


def _write(tmp_path, obj):
    p = tmp_path / "splits.json"
    p.write_text(json.dumps(obj))
    return str(p)


def test_resolve_splits_missing_key_raises_and_names_it(tmp_path):
    # "test" key entirely absent (not even an explicit []) -- must be rejected
    # distinctly from an explicitly-empty split.
    path = _write(tmp_path, {"train": ["m01"], "val": ["m02"]})
    with pytest.raises(ValueError, match=r"missing required key.*test"):
        resolve_splits({"m01", "m02"}, path)


def test_resolve_splits_all_keys_missing_lists_all(tmp_path):
    path = _write(tmp_path, {})
    with pytest.raises(ValueError, match="train"):
        resolve_splits(set(), path)


def test_resolve_splits_explicit_empty_list_is_not_missing_key(tmp_path):
    # An explicit [] for "test" must be accepted -- it is a deliberate choice,
    # not a missing key.
    path = _write(tmp_path, {"train": ["m01"], "val": ["m02"], "test": []})
    out = resolve_splits({"m01", "m02"}, path)
    assert out["test"] == []


def test_resolve_splits_rejects_overlap(tmp_path):
    path = _write(tmp_path, {"train": ["m01", "m02"], "val": ["m02"], "test": []})
    with pytest.raises(ValueError, match="disjoint"):
        resolve_splits({"m01", "m02"}, path)


def test_resolve_splits_warns_on_unknown_id(tmp_path, capsys):
    path = _write(tmp_path, {"train": ["m01", "m99"], "val": ["m02"], "test": []})
    out = resolve_splits({"m01", "m02"}, path)
    err = capsys.readouterr().err
    assert "m99" in err and "zero rows" in err
    # unknown id is still returned (filtering is the caller's job)
    assert out["train"] == ["m01", "m99"]
