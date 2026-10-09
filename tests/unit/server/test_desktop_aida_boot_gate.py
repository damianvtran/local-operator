"""The server-lifespan auto-activation gate (R17/R21, slice B).

``lop serve`` schedules her first-run ensure on EVERY boot (the task is kept on
``app.state`` and never awaited — see the lifespan). What slice B adds is the
predicate that decides whether that task may CREATE her: an interactive
install (a terminal, or a daemon the desktop app spawned) still gets her with
no configuration, and a cloud/automation daemon (agent-runtime-svc pipes, no
desktop plane) pays nothing — no session, no state file, no wake cost. The
explicit paths (``POST /v1/desktop/aida``, ``/aida``, a runtime claim) are
deliberately NOT gated: not auto-activated is not locked out.

The signal itself is pinned branch-by-branch in
``tests/unit/aida/test_aida_activation.py``; here it is wired to the boot.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from local_operator.server.app import app


async def _settle(task: asyncio.Task[None]) -> None:
    await task


#: The spy fixture's journal: one ``(args, kwargs)`` pair per ensure call.
Calls = list[tuple[tuple[Any, ...], dict[str, Any]]]


def _boot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    token: str | None,
    terminal: bool,
    home_is_users: bool = True,
) -> Path:
    """Point the app at a scratch root and choose the boot's surface signals.

    ``home_is_users`` pins the THIRD signal (R17's home check): every test runs
    under a redirected HOME, so the real predicate would refuse every boot.
    The default patches the INPUT (``supervisors.real_home``) to agree with
    ``$HOME`` — the shape of a user's own machine. The redirected-home cell
    passes ``False`` and leaves the real predicate in place.
    """
    root = tmp_path / ".local-operator"
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    monkeypatch.delenv("LOCAL_OPERATOR_NO_AIDA", raising=False)
    if token is None:
        monkeypatch.delenv("LOCAL_OPERATOR_DESKTOP_TOKEN", raising=False)
    else:
        monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", token)
    from local_operator.aida import activation

    monkeypatch.setattr(activation, "_has_terminal", lambda: terminal)
    if home_is_users:
        # Patch the PREDICATE, not its input: patching ``real_home`` would make
        # the tmp root resolve as addressable to the supervisor-install guard,
        # and the explicit-ensure cell below runs the REAL bootstrap. The
        # predicate's own branches are pinned in test_aida_activation.py.
        monkeypatch.setattr(activation, "home_is_the_users", lambda: True)
    return root


def _settle_boot(client: TestClient) -> None:
    task = getattr(app.state, "aida_boot_task", None)
    assert task is not None, "the lifespan must schedule the boot ensure"
    portal = client.portal
    assert portal is not None
    portal.call(_settle, task)


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> Calls:
    """Record every ensure the boot task makes, without doing the real work."""
    seen: Calls = []

    async def spy(*args, **kwargs):
        seen.append((args, kwargs))
        return "spied"

    monkeypatch.setattr("local_operator.aida.ensure_session", spy)
    return seen


def test_a_cloud_boot_is_not_auto_activated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, calls: Calls
) -> None:
    root = _boot(tmp_path, monkeypatch, token=None, terminal=False)
    with TestClient(app) as client:
        _settle_boot(client)
    assert calls == [], "a cloud/automation boot must not auto-create her"
    assert not (root / "aida").exists()
    assert not (root / "sessions").exists()


def test_a_desktop_managed_boot_is_auto_activated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, calls: Calls
) -> None:
    _boot(tmp_path, monkeypatch, token="desktop-token", terminal=False)
    with TestClient(app) as client:
        _settle_boot(client)
    assert len(calls) == 1, "a daemon the desktop app spawned auto-activates her"


def test_a_terminal_boot_is_auto_activated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, calls: Calls
) -> None:
    _boot(tmp_path, monkeypatch, token=None, terminal=True)
    with TestClient(app) as client:
        _settle_boot(client)
    assert len(calls) == 1, "a daemon run from a terminal auto-activates her"


def test_a_redirected_home_boot_is_not_auto_activated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, calls: Calls
) -> None:
    """T9: a pty is a RUN, not a person, when HOME is not the user's.

    The gap this closes is the TERMINAL signal's: it answers True for ANY
    pty, and a pty-allocating ``lop serve`` under a rig/container HOME is
    such a run, not a seat — the toast gate already refuses such a process,
    and the boot must agree before it creates a session, a cadence and a wake
    supervisor for a store nobody owns. The DESKTOP arm is deliberately not
    gated here — see the token cell below, and
    ``activation.terminal_under_a_foreign_home`` for the reasoning.

    Mutation: drop the HOME term from the boot hook — ``calls`` becomes 1.
    """
    root = _boot(tmp_path, monkeypatch, token=None, terminal=True, home_is_users=False)
    with TestClient(app) as client:
        _settle_boot(client)
    assert calls == [], "a redirected-home boot must not auto-create her"
    assert not (root / "aida").exists()
    assert not (root / "sessions").exists()


def test_a_desktop_token_boot_under_a_redirected_home_still_activates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, calls: Calls
) -> None:
    """The desktop arm is NOT gated by HOME (T9's exact scope).

    T9 closes the TERMINAL gap: a pty under a foreign HOME is a run. The
    desktop token is different in kind — the app spawns its daemon as the
    user, and the platform's desktop simulations (the desktop e2e suites) run
    token-plus-isolated-HOME by design, so gating this arm would recast the
    shipped desktop contract. What still bounds such a boot is per-process
    and unchanged: the toast and announce gates refuse a foreign-HOME
    process, so it pays no banner and no retry ladder.

    Mutation: widen the HOME gate back over both arms (suppress any boot
    whose HOME is not the user's) → ``calls`` becomes [] → red.
    """
    _boot(tmp_path, monkeypatch, token="desktop-token", terminal=False, home_is_users=False)
    with TestClient(app) as client:
        _settle_boot(client)
    assert len(calls) == 1, "the desktop plane's token is an app-issued human signal"


def test_an_explicit_ensure_still_creates_her_on_a_cloud_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Not auto-activated is not locked out: the explicit paths still ensure her.

    Deliberately does NOT take the ``calls`` spy: the point is the real
    ``ensure_session`` doing its real work on a root the boot left untouched.
    """
    from local_operator import aida as aida_pkg

    root = _boot(tmp_path, monkeypatch, token=None, terminal=False)
    with TestClient(app) as client:
        _settle_boot(client)
    assert not (root / "aida").exists()

    her_id = asyncio.run(aida_pkg.ensure_session(root))
    assert her_id
    assert (root / "aida" / "state.json").exists()
