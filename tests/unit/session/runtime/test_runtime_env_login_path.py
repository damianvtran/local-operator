"""A runtime is born with the operator's login PATH, whichever daemon engaged it.

launchd starts the ``python -m`` daemons that engage runtimes (team-bridge
connector, mesh relay, mobile service) with ``PATH=/usr/bin:/bin:/usr/sbin:/sbin``,
and a systemd user manager hands its units a similarly minimal PATH. Before
``launch._runtime_env`` primed the PATH itself, a runtime those daemons engaged
inherited that bare value and its bash tool could not find ``gh`` or anything
else under ``/opt/homebrew/bin`` (Aida's runtime, 2026-11-15). These cells pin
the prime at the one builder every spawn route shares, its per-lineage skip, and
its retry-on-failure contract.

``tests/conftest.py::neutralise_runtime_path_prime`` stubs the prime for the rest
of the suite; every cell here restores the real function captured at import.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from local_operator import helpers
from local_operator.session.runtime import launch, session_host

# Captured at collection, before the autouse fixture swaps in its no-op.
_REAL_PRIME = launch._prime_login_path

BARE_LAUNCHD_PATH = "/usr/bin:/bin:/usr/sbin:/sbin"
LOGIN_PATH = "/opt/homebrew/bin:/opt/homebrew/sbin:/usr/local/bin:" + BARE_LAUNCHD_PATH


@pytest.fixture
def bare_daemon(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """An unprimed daemon lineage with launchd's PATH; returns the probe call log."""
    monkeypatch.setattr(launch, "_prime_login_path", _REAL_PRIME)
    monkeypatch.setenv("PATH", BARE_LAUNCHD_PATH)
    monkeypatch.delenv(helpers.PATH_PRIMED_ENV, raising=False)
    monkeypatch.setattr(helpers.platform, "system", lambda: "Darwin")
    calls: list[int] = []

    def probe() -> str | None:
        calls.append(1)
        return LOGIN_PATH

    monkeypatch.setattr(helpers, "get_posix_shell_path", probe)
    return calls


def _path_entries(env: dict[str, str]) -> list[str]:
    return env["PATH"].split(":")


def test_a_bare_launchd_path_is_replaced_by_the_login_path(
    bare_daemon: list[int], tmp_path: Path
) -> None:
    env = launch._runtime_env("loginpath01", str(tmp_path), defer_materialise=False)

    assert "/opt/homebrew/bin" in _path_entries(env)
    assert bare_daemon == [1]
    # The daemon itself is now primed, so its next engage is free.
    assert env[helpers.PATH_PRIMED_ENV] == "1"


def test_the_hosted_route_ships_the_primed_path_to_the_session_host(
    bare_daemon: list[int], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The host forks with the mapping it is SENT, so the prime must precede it."""
    seen: dict[str, Any] = {}

    def fake_request(root: Path, **kw: Any) -> Any:
        seen.update(kw["env"])
        return session_host.HostedProcess(7, coalesced=True, connection=None)

    monkeypatch.setattr(session_host, "request_spawn", fake_request)
    launch._spawn_runtime_hosted("loginpath02", str(tmp_path), defer_materialise=False)

    assert "/opt/homebrew/bin" in seen["PATH"].split(":")
    assert bare_daemon == [1]


def test_a_primed_lineage_pays_no_probe(
    bare_daemon: list[int], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The c50a075ac27f startup work: an inherited marker skips the round-trip."""
    monkeypatch.setenv("PATH", LOGIN_PATH)
    monkeypatch.setenv(helpers.PATH_PRIMED_ENV, "1")

    for n in range(3):
        env = launch._runtime_env(f"loginpath1{n}", str(tmp_path), defer_materialise=False)
        assert env["PATH"] == LOGIN_PATH

    assert bare_daemon == []


def test_only_the_first_engage_from_an_unprimed_daemon_probes(
    bare_daemon: list[int], tmp_path: Path
) -> None:
    for n in range(3):
        launch._runtime_env(f"loginpath2{n}", str(tmp_path), defer_materialise=False)

    assert bare_daemon == [1]


def test_a_failed_capture_keeps_the_inherited_path_and_retries_next_engage(
    bare_daemon: list[int], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[int] = []

    def failing_probe() -> str | None:
        calls.append(1)
        return None

    monkeypatch.setattr(helpers, "get_posix_shell_path", failing_probe)

    env = launch._runtime_env("loginpath30", str(tmp_path), defer_materialise=False)
    assert env["PATH"] == BARE_LAUNCHD_PATH
    assert helpers.PATH_PRIMED_ENV not in env
    # Not pinned: a transient failure must not freeze the bare PATH on the tree.
    launch._runtime_env("loginpath31", str(tmp_path), defer_materialise=False)
    assert calls == [1, 1]


def test_a_systemd_user_unit_path_is_primed_on_linux(
    bare_daemon: list[int], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The mesh relay on a Linux node runs under ``systemd --user`` with the
    unit's fixed minimal PATH (``supervisors.py``); the same builder primes it."""
    unit_path = "/usr/local/bin:/usr/local/sbin:/usr/bin:/usr/sbin"
    linux_login = "/home/op/.local/bin:/home/linuxbrew/.linuxbrew/bin:" + unit_path
    monkeypatch.setenv("PATH", unit_path)
    monkeypatch.setattr(helpers.platform, "system", lambda: "Linux")
    monkeypatch.setattr(helpers, "get_posix_shell_path", lambda: linux_login)

    env = launch._runtime_env("loginpath40", str(tmp_path), defer_materialise=False)

    assert "/home/linuxbrew/.linuxbrew/bin" in _path_entries(env)
    assert env[helpers.PATH_PRIMED_ENV] == "1"


def test_the_mobile_daemons_hand_built_start_spawn_is_primed_too(
    bare_daemon: list[int], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``MobileDaemon._spawn_session`` builds its child env by hand, outside
    ``_runtime_env``; a phone-started session from the launchd mobile service
    must get the login PATH as well."""
    import asyncio

    import local_operator.mobile.daemon as daemon_module

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    daemon = daemon_module.MobileDaemon(port=0, password="fixture-password")
    envs: list[dict[str, str]] = []

    class Spawned(Exception):
        pass

    def capture(*_args: Any, env: dict[str, str], **_kwargs: Any) -> Any:
        envs.append(dict(env))
        raise Spawned

    monkeypatch.setattr(daemon_module, "spawn_disclaimed", capture)
    with pytest.raises(Spawned):
        asyncio.run(daemon._spawn_session(str(tmp_path), None, None, "abcdef123456"))

    assert "/opt/homebrew/bin" in envs[0]["PATH"].split(":")
    assert bare_daemon == [1]
