"""Tests for docking engine profiles loading and validation."""

from __future__ import annotations

import pytest

from smiles2select.preparability.engines import (
    EngineFileError,
    available_engines,
    load_engine,
    load_engine_file,
)

pytestmark = pytest.mark.unit


def test_builtin_engines_load():
    engines = available_engines()
    assert "vina" in engines
    assert "gold" in engines
    assert "glide" in engines

    for engine_id in ("vina", "gold", "glide"):
        profile = load_engine(engine_id)
        assert profile.id == engine_id
        assert profile.name
        assert "C" in profile.common_elements
        assert len(profile.flags) == 5
        flag_ids = {f.id for f in profile.flags}
        assert "UNCOMMON_ELEMENT" in flag_ids
        assert "UNDEFINED_STEREO" in flag_ids


def test_unknown_engine_raises_error():
    with pytest.raises(EngineFileError, match="unknown docking engine 'unknown_xyz'"):
        load_engine("unknown_xyz")


def test_malformed_json_rejected(tmp_path):
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{invalid json", encoding="utf-8")
    with pytest.raises(EngineFileError, match="invalid JSON"):
        load_engine_file(bad_json)

    missing_field = tmp_path / "missing.json"
    missing_field.write_text('{"id": "test"}', encoding="utf-8")
    with pytest.raises(EngineFileError, match="missing required field"):
        load_engine_file(missing_field)


def test_load_engine_case_insensitive():
    profile_upper = load_engine("Vina")
    assert profile_upper.id == "vina"
    profile_mixed = load_engine("  Glide  ")
    assert profile_mixed.id == "glide"
