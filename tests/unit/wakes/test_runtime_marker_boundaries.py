"""Ownership must be cleared at the actual boot/spawn boundaries, not just helpers."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from local_operator.wakes import owned_runtimes


@pytest.fixture(autouse=True)
def _no_host_control(monkeypatch: pytest.MonkeyPatch) -> None:
    # HOME does not isolate the uid's services or listeners. These boundary
    # probes must stop before ANY process launch/signal, including a service
    # manager invoked indirectly by a future boot/spawn refactor.
    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("marker boundary probe attempted host process control")

    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(os, "kill", forbidden)


class _BoundaryReached(BaseException):
    """Stop before boot can load credentials or a phone spawn can start a process."""


def _inherited_marker(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(owned_runtimes.OWNER_ENV, "synthetic-inherited-owner")
    monkeypatch.setenv(owned_runtimes.SUPERVISOR_PID_ENV, "99999999")
    monkeypatch.setenv(owned_runtimes.LEGACY_ACTIVE_ENV, "1")


def test_real_runtime_main_strips_marker_before_first_boot_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from local_operator.compaction import tokens
    from local_operator.session.runtime import process

    _inherited_marker(monkeypatch)
    observed: list[dict[str, str]] = []

    def first_boot_work() -> None:
        observed.append(dict(os.environ))
        raise _BoundaryReached

    monkeypatch.setattr(tokens, "warm_tokenizer_in_background", first_boot_work)
    # If the first-work checkpoint moves/disappears, fail before the rest of
    # boot rather than silently running providers, sockets or lease ownership.
    monkeypatch.setattr(
        "local_operator.logger.configure_file_logging",
        lambda **kwargs: pytest.fail("runtime main passed the boot checkpoint"),
    )
    with pytest.raises(_BoundaryReached):
        process.main()
    assert len(observed) == 1
    assert not set(owned_runtimes.MARKER_ENVS).intersection(observed[0])

    # Counterfactual: this same real-entry probe must detect removal of the boot
    # strip. It deliberately stops before logging, runtime leases or providers.
    _inherited_marker(monkeypatch)
    monkeypatch.setattr(owned_runtimes, "forget_inherited_marker", lambda: None)
    with pytest.raises(_BoundaryReached):
        process.main()
    assert set(owned_runtimes.MARKER_ENVS).issubset(observed[-1])
    print(
        "runtime main: marker absent before first boot work; no-strip control detects all markers"
    )


@pytest.mark.asyncio
async def test_phone_spawn_does_not_inherit_another_supervisors_marker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from local_operator import procname
    from local_operator.mobile import attach_client, daemon

    _inherited_marker(monkeypatch)
    monkeypatch.setattr(attach_client, "find_runtime_record", lambda *a: (None, "synthetic"))
    monkeypatch.setattr(procname, "spawn_identity", lambda *a, **kw: (sys.executable, None))
    environments: list[dict[str, str]] = []

    async def intercept_spawn(*args: Any, **kwargs: Any) -> None:
        environments.append(dict(kwargs["env"]))
        raise _BoundaryReached

    # Intercept the actual subprocess boundary, not stamp_child_env: no child,
    # provider call, service-manager command or listener discovery can occur.
    monkeypatch.setattr(daemon.asyncio, "create_subprocess_exec", intercept_spawn)
    phone = daemon.MobileDaemon(password="synthetic-password")
    with pytest.raises(_BoundaryReached):
        await phone._spawn_session(str(tmp_path), None, None, "synthetic-phone-session")
    assert len(environments) == 1
    assert environments[0]["LOP_MOBILE_CHILD_RESUME"] == "synthetic-phone-session"
    assert not set(owned_runtimes.MARKER_ENVS).intersection(environments[0])
    assert os.environ[owned_runtimes.OWNER_ENV] == "synthetic-inherited-owner"

    # Pin the instrument as well: bypassing just the stamp must reproduce the
    # inherited exec-time marker, still without starting any child process.
    monkeypatch.setattr(owned_runtimes, "stamp_child_env", lambda env: None)
    with pytest.raises(_BoundaryReached):
        await phone._spawn_session(str(tmp_path), None, None, "synthetic-phone-control")
    assert len(environments) == 2
    assert set(owned_runtimes.MARKER_ENVS).issubset(environments[-1])
    print("phone spawn: marker absent; no-stamp control detects all markers; no children started")
