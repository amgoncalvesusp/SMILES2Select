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

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

PROJECT_ROOT = Path(SPECPATH).parent
SOURCE_ROOT = PROJECT_ROOT / "src"

# Profile definitions are read from disk at runtime, so they must travel with
# the binary; without them the application starts with no rules at all.
datas = [
    (
        str(SOURCE_ROOT / "smiles2select" / "profiles" / "builtins"),
        "smiles2select/profiles/builtins",
    ),
]

# RDKit ships data files (filter catalogues, the SA/NP contrib models) outside
# the importable package tree.
datas += collect_data_files("rdkit")

hiddenimports = [
    *collect_submodules("rdkit.Chem"),
    "smiles2select.gui.workspace.workspace_window",
    "pyqtgraph",
]

analysis = Analysis(
    [str(SOURCE_ROOT / "smiles2select" / "gui" / "app.py")],
    pathex=[str(SOURCE_ROOT)],
    binaries=[],
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
