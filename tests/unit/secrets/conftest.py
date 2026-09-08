"""Isolation for every secret-store test.

`AGENTS.md` is explicit that ``LOCAL_OPERATOR_CONFIG_DIR`` alone is not
isolation, and that a redirected READ path is not automatically a redirected
WRITE path — the analytics backfill wrote 612 rows into the operator's real
database from a run that looked sandboxed. This store writes to
``config_dir()/secrets``, so the fixture redirects ``HOME`` as well and the
tests below assert the store landed under ``tmp_path`` rather than trusting it.
"""

from __future__ import annotations

import os
import signal
import time
from contextlib import suppress
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


@pytest.fixture(autouse=True)
def _stop_brokers_started_by_this_test(tmp_path: Path):
    """Kill any broker a test caused to start, at that test's teardown.

    Not optional hygiene. Retrieval lazily starts a daemon (design §13), and
    the CLI tests drive real ``lop secret`` subprocesses, so a full run of this
    directory left ~90 broker processes alive — each holding a master key in
    memory and idling for 30 minutes. On the shared machine this repo is worked
    on, with many concurrent worktrees, that is a resource leak an agent
    inflicts on the operator rather than a harmless artifact.

    Keyed on ``tmp_path`` rather than on ``config_root``: ``test_cli.py`` builds
    its own isolated config dir under the same ``tmp_path`` instead of using
    that fixture, and requesting ``config_root`` here would both miss those
    brokers and re-create a directory the CLI fixture already made.

    Teardown, not setup: each test gets a fresh ``tmp_path``, so there is
    nothing to clean up beforehand.
    """
    yield

    from local_operator.secrets import client

    # Every config dir this test could have used lives under tmp_path; a broker
    # is a per-config-dir singleton, so ask each one whether it has a daemon.
    for candidate in {tmp_path / "config", tmp_path / "home" / ".local-operator"}:
        if not candidate.exists():
            continue
        status = client.broker_status(candidate)
        if status is None:
            continue
        pid = status.get("pid")
        if not isinstance(pid, int):
            continue
        with suppress(OSError):
            os.kill(pid, signal.SIGTERM)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and client.is_running(candidate):
            time.sleep(0.05)
