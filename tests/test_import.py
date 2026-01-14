def test_import_main():
    """
    Minimal smoke test: ensure main.py can be imported.
    This catches syntax errors and missing dependencies.
    """
    import importlib

    module = importlib.import_module("src.main")
    assert module is not None
