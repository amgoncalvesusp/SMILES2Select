"""Import verified ONNX model packages into the user's local catalog."""

import hashlib
import importlib.util
import os
import shutil
import stat
import tempfile
from pathlib import Path

from .artifacts import file_hash
from .decision import list_models

_REQUIRED = frozenset(
    {"manifest.json", "model.onnx", "references/manifest.json", "references/records.jsonl"}
)
_OPTIONAL = frozenset({"MODEL_CARD.md"})

# Names retained from the three baseline choices frozen before test evaluation.
_VALIDATION_SELECTED = frozenset({"Q72547_WT_IC50", "P0DMS8_WT_Ki", "Q07869_WT_EC50"})


def bundled_model_root() -> Path:
    """Read-only model packages distributed with the application."""
    return Path(__file__).resolve().parent / "bundled_models"


def user_model_root() -> Path:
    """Keep imported packages in app-specific, user-writable data storage."""
    from PySide6.QtCore import QStandardPaths

    location = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.GenericDataLocation)
    if not location:
        raise RuntimeError("No user-writable application data location is available")
    return Path(location) / "SMILES2Select" / "models"


def _regular(path: Path, *, directory: bool) -> bool:
    details = path.lstat()
    if path.is_symlink() or (
        getattr(details, "st_file_attributes", 0)
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    ):
        return False
    return stat.S_ISDIR(details.st_mode) if directory else stat.S_ISREG(details.st_mode)


def _package_files(package: Path) -> tuple[str, ...]:
    if not package.is_dir() or not _regular(package, directory=True):
        raise ValueError("Model package must be a regular directory, without links")
    names = set()
    for child in package.iterdir():
        if child.name == "references":
            if not _regular(child, directory=True):
                raise ValueError("Model references must be a regular directory, without links")
            for member in child.iterdir():
                if not _regular(member, directory=False):
                    raise ValueError("Model package members must be regular files, without links")
                names.add(f"references/{member.name}")
        else:
            if not _regular(child, directory=False):
                raise ValueError("Model package members must be regular files, without links")
            names.add(child.name)
    if not _REQUIRED.issubset(names):
        raise ValueError("Model package is missing required files")
    if names - (_REQUIRED | _OPTIONAL):
        raise ValueError("Model package contains unexpected files")
    return tuple(sorted(names))


def _package_digest(package: Path, names: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for name in names:
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(file_hash(package / name)))
    return digest.hexdigest()


def list_user_models(
    root: str | Path | None = None,
    *,
    target: str | None = None,
    endpoint: str | None = None,
) -> list[dict]:
    """Show installed packages, including incompatible models and reasons."""
    catalog = Path(root) if root is not None else user_model_root()
    if not catalog.exists():
        return []
    return list_models(catalog, target=target, endpoint=endpoint)


def list_bundled_models(*, target: str | None = None, endpoint: str | None = None) -> list[dict]:
    catalog = bundled_model_root()
    if not catalog.is_dir():
        return []
    return [
        {**item, "origin": "bundled", "validation_selected": item["name"] in _VALIDATION_SELECTED}
        for item in list_models(catalog, target=target, endpoint=endpoint)
    ]


def list_catalog_models(
    root: str | Path | None = None,
    *,
    target: str | None = None,
    endpoint: str | None = None,
) -> list[dict]:
    """Keep bundled packages and user imports visible with explicit origin."""
    catalog = list_bundled_models(target=target, endpoint=endpoint) + [
        {**item, "origin": "user"}
        for item in list_user_models(root, target=target, endpoint=endpoint)
    ]
    if importlib.util.find_spec("onnxruntime") is None:
        return [
            {**item, "compatible": False,
             "reason": item["reason"] or "Install smiles2select[inference] for ONNX inference"}
            for item in catalog
        ]
    return catalog


def import_model_package(source: str | Path, *, root: str | Path | None = None) -> Path:
    """Copy a fixed data-only package and publish it after integrity checks."""
    package = Path(source)
    names = _package_files(package)
    source_digest = _package_digest(package, names)
    catalog = Path(root) if root is not None else user_model_root()
    catalog.mkdir(parents=True, exist_ok=True)
    destination = catalog / f"model-{source_digest}"
    if os.path.lexists(destination):
        if _package_files(destination) != names or _package_digest(destination, names) != source_digest:
            raise ValueError("Model catalog destination conflicts with imported package")
        inspection = list_models(destination)[0]
        if not inspection["compatible"]:
            raise ValueError(f"Model package is incompatible: {inspection['reason']}")
        return destination
    with tempfile.TemporaryDirectory(prefix=".model-", dir=catalog) as temporary:
        staged = Path(temporary) / "package"
        staged.mkdir()
        (staged / "references").mkdir()
        for name in names:
            shutil.copyfile(package / name, staged / name)
        if _package_files(staged) != names or _package_digest(staged, names) != source_digest:
            raise ValueError("Model package changed during import")
        inspection = list_models(staged)[0]
        if not inspection["compatible"]:
            raise ValueError(f"Model package is incompatible: {inspection['reason']}")
        try:
            staged.rename(destination)
        except OSError:
            if not os.path.lexists(destination):
                raise
            if _package_files(destination) != names or _package_digest(destination, names) != source_digest:
                raise ValueError("Model catalog destination conflicts with imported package") from None
            inspection = list_models(destination)[0]
            if not inspection["compatible"]:
                raise ValueError(f"Model package is incompatible: {inspection['reason']}") from None
    return destination
