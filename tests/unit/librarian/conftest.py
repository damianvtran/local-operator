"""Fixtures shared by the Librarian tests."""

from __future__ import annotations

from pathlib import Path

import pytest


def isolated_root_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A scratch config root, reachable BOTH explicitly and via ``paths``.

    The same shape the Aida suite uses, and for the same reason: production
    code reads the root two ways — callers pass it in (the boot hooks hand
    ``config_dir`` to ``ensure_session``) and the session hooks resolve it
    through ``local_operator.paths.config_dir()`` — so the fixture sets the
    override AND HOME and hands back the same path it set.

    ``LOCAL_OPERATOR_NO_LIBRARIAN`` is cleared explicitly: a sibling suite or a
    developer shell that exported it must not silently turn every test here
    into the disabled case. ``LOCAL_OPERATOR_NO_AIDA`` is set so an ensure
    under test never competes with Aida's engine for the same wake index.
    """
    root = tmp_path / "config"
    home = tmp_path / "home"
    root.mkdir(parents=True, exist_ok=True)
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    monkeypatch.delenv("LOCAL_OPERATOR_NO_LIBRARIAN", raising=False)
    monkeypatch.setenv("LOCAL_OPERATOR_NO_AIDA", "1")
    return root


@pytest.fixture()
def isolated_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    return isolated_root_path(tmp_path, monkeypatch)
