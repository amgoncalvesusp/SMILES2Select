import json
import runpy
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from tools import verify_windows_bundle
from tools.verify_windows_bundle import _is_system_dll


def test_windows_system_dll_classification() -> None:
    assert _is_system_dll("dbghelp.dll")
    assert _is_system_dll("DBGHELP.DLL")
    assert not _is_system_dll("onnxruntime.dll")


def test_native_imports_include_normal_and_delayed_dependencies(tmp_path, monkeypatch):
    native = SimpleNamespace(
        DIRECTORY_ENTRY_IMPORT=[SimpleNamespace(dll=b"MSCMS.DLL"), SimpleNamespace(dll=None)],
        DIRECTORY_ENTRY_DELAY_IMPORT=[SimpleNamespace(dll=b"onnxruntime.dll"),
                                      SimpleNamespace(dll=b"MSCMS.DLL"), SimpleNamespace(dll=None)],
    )

    def parse(path, fast_load):
        assert Path(path).name == "qwindows.dll"
        assert fast_load is False, "All PE import tables must be available for verification"
        return native

    monkeypatch.setattr(verify_windows_bundle.pefile, "PE", parse)
    assert verify_windows_bundle._imports(tmp_path / "qwindows.dll") == ["MSCMS.DLL", "onnxruntime.dll"]


@pytest.mark.parametrize("dependency, allowed", [
    ("mscms.dll", True), ("MSCMS.DLL", True),
    ("mscms_vendor.dll", False), ("onnxruntime.dll", False),
])
def test_qt_color_management_import_is_system_only(tmp_path, monkeypatch, dependency, allowed):
    platforms = (
        "_internal/PySide6/plugins/platforms/qwindows.dll",
        "_internal/PySide6/plugins/platforms/qdirect2d.dll",
    )
    resources = (
        "SMILES2Select.exe", "_internal/Qt6Core.dll", "_internal/Qt6Gui.dll",
        "_internal/Qt6Widgets.dll", *platforms, "_internal/rdkit/Data/BaseFeatures.fdef",
        "_internal/smiles2select/profiles/builtins/default.json",
        "_internal/smiles2select/preparability/builtins/default.json",
    )
    for name in resources:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
    monkeypatch.setattr(verify_windows_bundle, "_imports", lambda path:
                        [dependency] if path.name in {"qwindows.dll", "qdirect2d.dll"} else [])
    manifest = tmp_path.parent / f"{tmp_path.name}-manifest.json"
    arguments = [str(tmp_path), "--manifest", str(manifest)]
    if allowed:
        assert verify_windows_bundle.main(arguments) == 0
        report = json.loads(manifest.read_text(encoding="utf-8"))
        assert report["missing_dependencies"] == {}
        assert report["missing_runtime_requirements"] == []
        assert all(report["dependencies"][str(Path(name))] == [dependency] for name in platforms)
    else:
        with pytest.raises(RuntimeError) as failure:
            verify_windows_bundle.main(arguments)
        report = json.loads(str(failure.value))
        assert report["missing_dependencies"] == {str(Path(name)): [dependency] for name in platforms}
        assert not manifest.exists(), "A failed bundle must not produce a success manifest"


def test_windows_bundle_rejects_shadowed_system_icu(tmp_path, monkeypatch) -> None:
    resources = (
        "SMILES2Select.exe", "_internal/Qt6Core.dll", "_internal/Qt6Gui.dll",
        "_internal/Qt6Widgets.dll", "_internal/qwindows.dll", "_internal/icuuc.dll",
        "_internal/rdkit/Data/BaseFeatures.fdef",
        "_internal/smiles2select/profiles/builtins/default.json",
        "_internal/smiles2select/preparability/builtins/default.json",
    )
    for name in resources:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
    monkeypatch.setattr(verify_windows_bundle, "_imports", lambda path: [])
    with pytest.raises(RuntimeError, match="icuuc.dll must use the Windows system copy"):
        verify_windows_bundle.verify(tmp_path)


@pytest.mark.parametrize("platform, expected", [("win32", False), ("linux", True)])
def test_spec_excludes_windows_system_icu_only_on_windows(monkeypatch, platform, expected):
    # Exercise configuration without making the build extra a test dependency.
    hooks = ModuleType("PyInstaller.utils.hooks")
    hooks.collect_data_files = lambda name: []
    hooks.collect_dynamic_libs = lambda name: []
    hooks.collect_submodules = lambda name: []
    monkeypatch.setitem(sys.modules, "PyInstaller", ModuleType("PyInstaller"))
    monkeypatch.setitem(sys.modules, "PyInstaller.utils", ModuleType("PyInstaller.utils"))
    monkeypatch.setitem(sys.modules, "PyInstaller.utils.hooks", hooks)
    monkeypatch.setattr(sys, "platform", platform)
    collected = {}
    native = [("ICUUC.DLL", "unrelated/poppler/icuuc.dll", "BINARY"),
              ("PySide6/Qt6Core.dll", "qt/Qt6Core.dll", "BINARY")]
    analysis = SimpleNamespace(binaries=native, pure=[], zipped_data=[], scripts=[],
                               zipfiles=[], datas=[])
    spec = Path(__file__).resolve().parents[1] / "packaging/smiles2select.spec"
    runpy.run_path(str(spec), init_globals={
        "SPECPATH": str(spec.parent), "Analysis": lambda *args, **kwargs: analysis,
        "PYZ": lambda *args, **kwargs: None, "EXE": lambda *args, **kwargs: None,
        "COLLECT": lambda *args, **kwargs: collected.update(binaries=args[1]),
    })
    assert any(Path(item[0]).name.casefold() == "icuuc.dll"
               for item in collected["binaries"]) is expected
    assert native[1] in collected["binaries"]
