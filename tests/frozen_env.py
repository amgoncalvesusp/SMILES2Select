"""Keep source-test runtime paths out of frozen application processes."""

from __future__ import annotations

import os


def frozen_environment() -> dict[str, str]:
    excluded = {
        "RDBASE", "PYTHONPATH", "PYTHONHOME", "QT_PLUGIN_PATH", "QT_QPA_PLATFORM_PLUGIN_PATH",
        "QT_API", "PYQTGRAPH_QT_LIB", "QML_IMPORT_PATH", "QML2_IMPORT_PATH",
        "LD_LIBRARY_PATH", "LD_LIBRARY_PATH_ORIG", "LD_PRELOAD", "DYLD_LIBRARY_PATH",
        "DYLD_FALLBACK_LIBRARY_PATH", "DYLD_FRAMEWORK_PATH",
    }
    return {
        **{name: value for name, value in os.environ.items() if name.upper() not in excluded},
        "QT_QPA_PLATFORM": "offscreen",
    }
