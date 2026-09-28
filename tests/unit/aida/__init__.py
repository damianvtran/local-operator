"""Aida's unit tests: her state files, her engine, and the session hooks.

One package-level fixture (``isolated_root``) points HOME and
``LOCAL_OPERATOR_CONFIG_DIR`` at a per-test scratch root, so every module here
can call the code under test the way production does — through
``paths.config_dir()`` — without touching the operator's real store.
"""
