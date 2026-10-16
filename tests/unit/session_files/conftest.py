"""Isolation for the session-files tests: a private config dir, empty process caches."""

from __future__ import annotations

from pathlib import Path

import pytest

from local_operator.session_files import ledger, service


@pytest.fixture(autouse=True)
def _isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Config + HOME under ``tmp_path`` so ``cache_media`` and the cache write nowhere real.

    ``HOME`` is redirected too: ``Path.home()`` and ``~`` expansion feed the service's home
    argument and the denylist, and a test must never read the operator's real dotfiles.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "cfg"))
    ledger.forget_resident()
    service.forget_stats()
    return tmp_path
