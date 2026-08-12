"""Verify that a PyInstaller Windows bundle contains its native dependencies.

This checks every EXE, DLL and Python extension in the bundle. Imports that are
not in the bundle must be Windows system libraries; third-party runtime DLLs
such as Qt, RDKit, NumPy and the MSVC runtime must be present in the bundle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

try:
    import pefile
except ImportError as exc:  # pragma: no cover - exercised in incomplete build envs
    raise SystemExit("Bundle verification requires pefile; install the build extra.") from exc


SYSTEM_DLLS = {
    "AUTHZ.DLL",
    "ADVAPI32.DLL",
    "BCRYPT.DLL",
    "COMCTL32.DLL",
    "COMBASE.DLL",
    "COMDLG32.DLL",
    "CRYPT32.DLL",
    "D3D11.DLL",
    "D3D12.DLL",
    "D3D9.DLL",
    "D2D1.DLL",
    "DWrite.DLL",
    "DNSAPI.DLL",
    "DWMAPI.DLL",
    "DXGI.DLL",
    "GDI32.DLL",
    "IMAGEHLP.DLL",
    "IMM32.DLL",
    "ICUUC.DLL",
    "IPHLPAPI.DLL",
    "KERNEL32.DLL",
    "KERNELBASE.DLL",
    "MSVCRT.DLL",
    "NTDLL.DLL",
    "OLE32.DLL",
    "OLEAUT32.DLL",
    "MPR.DLL",
    "MSIMG32.DLL",
    "NCRYPT.DLL",
    "NETAPI32.DLL",
    "PDH.DLL",
    "PSAPI.DLL",
    "POWRPROF.DLL",
    "PROPSYS.DLL",
    "RPCRT4.DLL",
    "SETUPAPI.DLL",
    "SHELL32.DLL",
    "SHCORE.DLL",
    "SHLWAPI.DLL",
    "UCRTBASE.DLL",
    "UIAUTOMATIONCORE.DLL",
    "USER32.DLL",
    "USERENV.DLL",
    "UXTHEME.DLL",
    "VERSION.DLL",
    "WINMM.DLL",
    "WINHTTP.DLL",
    "WINDOWSCODECS.DLL",
    "WS2_32.DLL",
    "WINSPOOL.DRV",
    "WTSAPI32.DLL",
    "SECUR32.DLL",
}
SYSTEM_DLLS = {name.upper() for name in SYSTEM_DLLS}


def _is_system_dll(name: str) -> bool:
    upper = name.upper()
    return upper in SYSTEM_DLLS or upper.startswith(("API-MS-WIN-", "EXT-MS-WIN-"))


def _imports(path: Path) -> list[str]:
    pe = pefile.PE(str(path), fast_load=False)
    names: set[str] = set()
    for entry in getattr(pe, "DIRECTORY_ENTRY_IMPORT", []):
        if entry.dll:
            names.add(entry.dll.decode(errors="replace"))
    for entry in getattr(pe, "DIRECTORY_ENTRY_DELAY_IMPORT", []):
        if entry.dll:
            names.add(entry.dll.decode(errors="replace"))
    return sorted(names, key=str.casefold)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(bundle: Path) -> dict[str, object]:
    if not bundle.is_dir():
        raise ValueError(f"bundle directory does not exist: {bundle}")
    executable = bundle / "SMILES2Select.exe"
    if not executable.is_file():
        raise ValueError(f"missing main executable: {executable}")

    files = [path for path in bundle.rglob("*") if path.is_file()]
    by_name: dict[str, list[str]] = {}
    for path in files:
        by_name.setdefault(path.name.casefold(), []).append(str(path.relative_to(bundle)))

    native = [path for path in files if path.suffix.casefold() in {".exe", ".dll", ".pyd"}]
    dependencies: dict[str, list[str]] = {}
    missing: dict[str, list[str]] = {}
    for path in native:
        try:
            imported = _imports(path)
        except pefile.PEFormatError:
            continue
        relative = str(path.relative_to(bundle))
        dependencies[relative] = imported
        for name in imported:
            if name.casefold() in by_name or _is_system_dll(name):
                continue
            missing.setdefault(relative, []).append(name)

    profile_roots = [
        bundle / "smiles2select" / "profiles" / "builtins",
        bundle / "_internal" / "smiles2select" / "profiles" / "builtins",
    ]
    required_profiles = [
        path
        for root in profile_roots
        if root.is_dir()
        for path in root.glob("*.json")
    ]
    required_names = {"qwindows.dll", "qt6core.dll", "qt6gui.dll", "qt6widgets.dll"}
    present_names = set(by_name)
    missing_runtime = sorted(name for name in required_names if name not in present_names)
    if not required_profiles:
        missing_runtime.append("smiles2select/profiles/builtins/*.json")
    if not any(path.name == "BaseFeatures.fdef" for path in files):
        missing_runtime.append("rdkit/Data/BaseFeatures.fdef")

    result: dict[str, object] = {
        "bundle": str(bundle),
        "file_count": len(files),
        "native_file_count": len(native),
        "files": [
            {
                "path": str(path.relative_to(bundle)),
                "size": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in sorted(files)
        ],
        "dependencies": dependencies,
        "missing_dependencies": missing,
        "missing_runtime_requirements": missing_runtime,
    }
    if missing or missing_runtime:
        raise RuntimeError(json.dumps(result, indent=2))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args(argv)
    result = verify(args.bundle)
    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        f"Verified {result['file_count']} files ({result['native_file_count']} native files); "
        "all non-system imports are bundled."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
