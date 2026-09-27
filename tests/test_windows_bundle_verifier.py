from tools.verify_windows_bundle import _is_system_dll


def test_windows_system_dll_classification() -> None:
    assert _is_system_dll("dbghelp.dll")
    assert _is_system_dll("DBGHELP.DLL")
    assert not _is_system_dll("onnxruntime.dll")
