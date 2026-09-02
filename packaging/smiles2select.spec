# PyInstaller spec for SMILES2Select.
#
# Author: Adriano Marques Goncalves - Universidade de Araraquara (UNIARA)
#
# One spec for both platforms: PyInstaller cannot cross-compile, so the Windows
# binary is built on Windows and the Linux one on Linux (see the release
# workflow). Everything platform-specific here is derived from sys.platform
# rather than hard-coded.

import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

PROJECT_ROOT = Path(SPECPATH).parent
SOURCE_ROOT = PROJECT_ROOT / "src"
ASSET_ROOT = SOURCE_ROOT / "smiles2select" / "assets"

# Profile definitions are read from disk at runtime, so they must travel with
# the binary; without them the application starts with no rules at all.
datas = [
    (
        str(SOURCE_ROOT / "smiles2select" / "profiles" / "builtins"),
        "smiles2select/profiles/builtins",
    ),
    (
        str(SOURCE_ROOT / "smiles2select" / "preparability" / "builtins"),
        "smiles2select/preparability/builtins",
    ),
    (str(ASSET_ROOT / "SMILES2Select.png"), "smiles2select/assets"),
    (str(ASSET_ROOT / "SMILES2Select.ico"), "smiles2select/assets"),
]

# RDKit ships data files (filter catalogues, the SA/NP contrib models) outside
# the importable package tree.
datas += collect_data_files("rdkit")

# Keep native runtime libraries that RDKit and NumPy load dynamically. The
# normal import graph catches most of these, but explicit collection prevents
# a clean machine without Python from missing a transitive DLL.
binaries = [
    *collect_dynamic_libs("rdkit"),
    *collect_dynamic_libs("numpy"),
]

# Python's extension modules may depend on runtime DLLs that PyInstaller does
# not discover when the interpreter comes from conda. Include the active
# environment's copies so the bundle also works on a clean Windows machine.
if sys.platform == "win32":
    runtime_dll_names = {
        "ffi.dll",
        "libmpdec-4.dll",
        "msvcp140.dll",
        "msvcp140_1.dll",
        "msvcp140_2.dll",
        "msvcp140_atomic_wait.dll",
        "msvcp140_codecvt_ids.dll",
        "vcruntime140.dll",
        "vcruntime140_1.dll",
    }
    runtime_roots = [
        Path(sys.prefix) / "DLLs",
        Path(sys.prefix) / "Library" / "bin",
        Path(sys.prefix),
    ]
    runtime_dlls = {
        path.name.casefold(): path
        for root in runtime_roots
        if root.is_dir()
        for path in root.glob("*.dll")
        if path.name.casefold() in {name.casefold() for name in runtime_dll_names}
    }
    binaries += [(str(path), ".") for path in runtime_dlls.values()]

hiddenimports = [
    *collect_submodules("rdkit.Chem"),
    "smiles2select.gui.workspace.workspace_window",
    "pyqtgraph",
]

analysis = Analysis(
    [str(SOURCE_ROOT / "smiles2select" / "gui" / "app.py")],
    pathex=[str(SOURCE_ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    # Excluded on purpose: pulled in transitively but never used at runtime,
    # and each adds tens of megabytes to the bundle.
    excludes=["tkinter", "pytest", "IPython", "notebook"],
    noarchive=False,
)

pyz = PYZ(analysis.pure, analysis.zipped_data)

executable = EXE(
    pyz,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="SMILES2Select",
    icon=str(
        ASSET_ROOT / ("SMILES2Select.ico" if sys.platform == "win32" else "SMILES2Select.png")
    ),
    debug=False,
    strip=False,
    upx=False,
    # No console window on Windows; on Linux the terminal stays useful for logs.
    console=sys.platform != "win32",
)

COLLECT(
    executable,
    analysis.binaries,
    analysis.zipfiles,
    analysis.datas,
    strip=False,
    upx=False,
    name="SMILES2Select",
)
