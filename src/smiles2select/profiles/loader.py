"""Loading profiles from JSON files.

Built-in profiles live in ``profiles/builtins`` and user profiles in any
directory the caller points at, so a new rule set needs a JSON file and no
change to the engine.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from smiles2select.profiles.registry import Profile, ProfileRegistry
from smiles2select.rules.engine import rule_from_dict

BUILTIN_DIR = Path(__file__).parent / "builtins"
CUSTOM_DIR = Path(__file__).parent / "custom"


class ProfileFileError(ValueError):
    """Raised when a profile file is missing fields or malformed."""


def profile_from_dict(payload: dict[str, Any], *, source: str = "<dict>") -> Profile:
    try:
        profile_id = payload["id"]
        name = payload["name"]
        rules_payload = payload["rules"]
    except KeyError as exc:
        raise ProfileFileError(f"{source}: missing required field {exc}") from exc

    if not isinstance(rules_payload, list) or not rules_payload:
        raise ProfileFileError(f"{source}: 'rules' must be a non-empty list")

    try:
        rules = tuple(rule_from_dict(rule, profile_id) for rule in rules_payload)
    except (ValueError, KeyError) as exc:
        raise ProfileFileError(f"{source}: {exc}") from exc

    try:
        return Profile(
            id=profile_id,
            name=name,
            category=payload.get("category", "custom"),
            version=payload.get("version", "1.0.0"),
            rules=rules,
            pass_policy=payload.get("pass_policy", {"type": "all_rules"}),
            short_name=payload.get("short_name", ""),
            logp_method=payload.get("logp_method", ""),
            atom_count_definition=payload.get("atom_count_definition", ""),
            reference=payload.get("reference", ""),
            notes=payload.get("notes", ""),
        )
    except ValueError as exc:
        raise ProfileFileError(f"{source}: {exc}") from exc


def load_profile_file(path: str | Path) -> Profile:
    file_path = Path(path)
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ProfileFileError(f"{file_path.name}: invalid JSON ({exc})") from exc
    except OSError as exc:
        raise ProfileFileError(f"{file_path}: cannot be read ({exc})") from exc
    return profile_from_dict(payload, source=file_path.name)


def load_directory(path: str | Path) -> list[Profile]:
    """Load every ``*.json`` profile in a directory (missing directory -> empty)."""
    directory = Path(path)
    if not directory.is_dir():
        return []
    return [load_profile_file(file_path) for file_path in sorted(directory.glob("*.json"))]


def builtin_registry(extra_directories: Iterable[str | Path] = ()) -> ProfileRegistry:
    """Registry with the shipped profiles plus any user directories."""
    registry = ProfileRegistry(load_directory(BUILTIN_DIR))
    for directory in (CUSTOM_DIR, *extra_directories):
        for profile in load_directory(directory):
            registry.add(profile, overwrite=True)
    return registry


def save_profile(profile: Profile, path: str | Path) -> Path:
    """Write a profile back to JSON (used when the GUI saves a custom preset)."""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(
        json.dumps(profile.as_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return file_path
