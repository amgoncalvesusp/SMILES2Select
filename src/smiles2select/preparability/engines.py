"""Docking engine profiles and compatibility specifications.

Compatibility rules and element sets are declared in JSON profiles,
never hardcoded into evaluation algorithms.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BUILTIN_DIR = Path(__file__).parent / "builtins"


class EngineFileError(ValueError):
    """Raised when an engine profile file is missing fields or malformed."""


@dataclass(frozen=True)
class EngineFlag:
    id: str
    label: str
    severity: str
    threshold: float | None = None


@dataclass(frozen=True)
class EngineProfile:
    id: str
    name: str
    version: str
    notes: str
    common_elements: frozenset[str]
    flags: tuple[EngineFlag, ...]


def engine_from_dict(payload: dict[str, Any], *, source: str = "<dict>") -> EngineProfile:
    try:
        engine_id = payload["id"]
        name = payload["name"]
        version = payload.get("version", "1.0.0")
        notes = payload.get("notes", "")
        raw_elements = payload["common_elements"]
        raw_flags = payload["flags"]
    except KeyError as exc:
        raise EngineFileError(f"{source}: missing required field {exc}") from exc

    if not isinstance(raw_elements, list):
        raise EngineFileError(f"{source}: 'common_elements' must be a list")
    common_elements = frozenset(str(elem) for elem in raw_elements)

    if not isinstance(raw_flags, list):
        raise EngineFileError(f"{source}: 'flags' must be a list")

    flags: list[EngineFlag] = []
    for item in raw_flags:
        if not isinstance(item, dict):
            raise EngineFileError(f"{source}: flag item must be a dictionary")
        try:
            flag_id = item["id"]
            label = item["label"]
            severity = item["severity"]
        except KeyError as exc:
            raise EngineFileError(f"{source}: flag missing required field {exc}") from exc
        raw_thresh = item.get("threshold")
        threshold = float(raw_thresh) if raw_thresh is not None else None
        flags.append(EngineFlag(id=flag_id, label=label, severity=severity, threshold=threshold))

    return EngineProfile(
        id=str(engine_id),
        name=str(name),
        version=str(version),
        notes=str(notes),
        common_elements=common_elements,
        flags=tuple(flags),
    )


def load_engine_file(path: str | Path) -> EngineProfile:
    file_path = Path(path)
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EngineFileError(f"{file_path.name}: invalid JSON ({exc})") from exc
    except OSError as exc:
        raise EngineFileError(f"{file_path}: cannot be read ({exc})") from exc
    return engine_from_dict(payload, source=file_path.name)


def load_engine(engine_id: str, directory: Path | None = None) -> EngineProfile:
    normalized_id = engine_id.strip().lower()
    dir_path = directory if directory is not None else BUILTIN_DIR
    target = dir_path / f"{normalized_id}.json"
    if not target.is_file():
        raise EngineFileError(
            f"unknown docking engine '{engine_id}'; available: {available_engines(directory)}"
        )
    return load_engine_file(target)


def available_engines(directory: Path | None = None) -> tuple[str, ...]:
    dir_path = directory if directory is not None else BUILTIN_DIR
    if not dir_path.is_dir():
        return ()
    return tuple(sorted(p.stem for p in dir_path.glob("*.json")))
