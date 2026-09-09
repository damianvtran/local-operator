"""Isolation for every secret-store test.

`AGENTS.md` is explicit that ``LOCAL_OPERATOR_CONFIG_DIR`` alone is not
isolation, and that a redirected READ path is not automatically a redirected
WRITE path — the analytics backfill wrote 612 rows into the operator's real
database from a run that looked sandboxed. This store writes to
``config_dir()/secrets``, so the fixture redirects ``HOME`` as well and the
tests below assert the store landed under ``tmp_path`` rather than trusting it.

The broker/runtime-dir sweep that used to live here now sits in
``tests/conftest.py``: the TUI registers a session at startup, so tests well
outside this directory spawn brokers too, and a fixture scoped here could not
see them (QA Q7).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from local_operator.paths import CONFIG_DIR_ENV
from local_operator.secrets.crypto import generate_master_key
from local_operator.secrets.store import SecretStore


@pytest.fixture
def config_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A private config dir; both HOME and the override point inside tmp_path."""
    root = tmp_path / "config"
    root.mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv(CONFIG_DIR_ENV, str(root))
    (tmp_path / "home").mkdir()
    return root


@pytest.fixture
def master_key() -> bytes:
    return generate_master_key()


@pytest.fixture
def store(config_root: Path, master_key: bytes) -> SecretStore:
    """An initialised store under the isolated config dir."""
    instance = SecretStore(master_key, base=config_root)
    instance.initialize()
    assert instance.path.is_relative_to(config_root), "store escaped the sandbox"
    return instance
