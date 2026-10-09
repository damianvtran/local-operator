"""The desktop's post-sign-in re-home: who gets asked, who is left alone, what it reports.

The decision matrix is the whole point of this module. A re-home moves a
conversation the user did not ask to move, so every guard here is a way for the
repair to do nothing rather than something wrong: another device owns the
session, the mirror is stale, the session is working, its provider is fine, or
the credential store cannot be read. The owner's compare-and-set re-checks all
of it under its own state (``session/runtime/serving.py``); what this file pins
is which sessions the desktop even ASKS about, and the receipt shape the routes
return.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from local_operator.server.utils.desktop_rehome import (
    REHOMED_KEY,
    rehome_after_login,
    with_rehome_count,
)
from local_operator.server.utils.desktop_sessions import DesktopSessions


class _FakeClient:
    """Stands in for the AttachClient the bridge dials."""

    def __init__(self, answer: str = "rehomed: radient/auto → deepseek/deepseek-flash") -> None:
        self.answer = answer
        self.asked: list[tuple[str, str, str]] = []

    async def rehome_if_current(self, expected: str, provider: str, model_id: str) -> str:
        self.asked.append((expected, provider, model_id))
        return self.answer


class _FakeRemote:
    """The slice of ``AttachedSession`` the pool's filters read."""

    def __init__(
        self,
        *,
        provider: str = "radient",
        model_id: str = "auto",
        streaming: bool = False,
        pending_gate: Any = None,
        jobs: tuple[Any, ...] = (),
        loop: dict[str, Any] | None = None,
        cold: bool = False,
        canonical_current: bool = True,
    ) -> None:
        self.is_cold = cold
        self.canonical_current = canonical_current
        self.is_streaming = streaming
        self.pending_gate = pending_gate
        self._selected = SimpleNamespace(provider=provider, model_id=model_id)
        self.frontend_state = SimpleNamespace(selected_model=self._selected, jobs=jobs, loop=loop)

    def activity_phase_clock(self) -> tuple[str, float]:
        return ("", 0.0)


class _FakeBridge:
    """Duck-typed bridge: the pool reads exactly these four members."""

    def __init__(self, client: _FakeClient, remote: _FakeRemote | None, *, remote_row: Any = None):
        self.session_id = "s1"
        self.remote_row = remote_row
        self.remote = remote
        self._client = client

    def _owner_connection(self) -> _FakeClient:
        return self._client


def _pool_with(*bridges: _FakeBridge) -> DesktopSessions:
    """A real pool holding duck-typed bridges.

    The pool's method reads four members off a bridge (``remote_row``,
    ``remote``, ``_owner_connection``, ``session_id``), so the fakes above are
    the honest double for WOULD IT ASK — the bridge's own construction and
    attach lifecycle are pinned in ``test_desktop_sessions.py``.
    """
    pool = DesktopSessions(Path("/nonexistent-rehome-root"))
    resident = cast("dict[str, Any]", pool.bridges)
    for bridge in bridges:
        resident[bridge.session_id] = bridge
    return pool


@pytest.mark.asyncio
async def test_a_stranded_idle_bound_session_is_asked_and_counted() -> None:
    client = _FakeClient()
    pool = _pool_with(_FakeBridge(client, _FakeRemote()))

    moved = await pool.rehome_stranded_sessions({"deepseek"}, "deepseek", "deepseek-flash")

    assert client.asked == [("radient/auto", "deepseek", "deepseek-flash")]
    assert moved == [("radient/auto", "deepseek/deepseek-flash")]


@pytest.mark.asyncio
async def test_the_guards_each_keep_a_session_untouched() -> None:
    """One pool, five reasons not to ask — and not one request sent."""
    other_device = _FakeClient()
    remote_row = _FakeBridge(other_device, _FakeRemote(), remote_row=SimpleNamespace(device="b"))
    cold = _FakeClient()
    cold_bridge = _FakeBridge(cold, _FakeRemote(cold=True))
    stale_mirror = _FakeClient()
    stale = _FakeBridge(stale_mirror, _FakeRemote(canonical_current=False))
    busy = _FakeClient()
    busy_bridge = _FakeBridge(busy, _FakeRemote(streaming=True))
    signed_in = _FakeClient()
    signed_in_bridge = _FakeBridge(signed_in, _FakeRemote(provider="deepseek", model_id="flash"))
    pool = _pool_with(remote_row, cold_bridge, stale, busy_bridge, signed_in_bridge)

    moved = await pool.rehome_stranded_sessions({"deepseek"}, "deepseek", "deepseek-flash")

    assert moved == []
    for client in (other_device, cold, stale_mirror, busy, signed_in):
        assert client.asked == []


@pytest.mark.asyncio
async def test_nothing_is_asked_when_the_target_is_not_accessible() -> None:
    """The toggle: an unreadable store or an unreachable target asks NOBODY."""
    client = _FakeClient()
    pool = _pool_with(_FakeBridge(client, _FakeRemote()))

    assert await pool.rehome_stranded_sessions(None, "deepseek", "deepseek-flash") == []
    assert await pool.rehome_stranded_sessions({"openai"}, "deepseek", "deepseek-flash") == []
    assert client.asked == []


@pytest.mark.asyncio
async def test_an_owner_refusal_is_not_counted_and_never_raises() -> None:
    """The owner re-checks under its own state; its "kept: …" is a normal answer."""
    refused = _FakeClient(answer="kept: the session is working right now")
    raising = _FakeClient()
    pool = _pool_with(_FakeBridge(refused, _FakeRemote()))

    async def _boom(*args: Any, **kwargs: Any) -> str:
        raise ConnectionError("owner went away")

    raising.rehome_if_current = _boom  # type: ignore[method-assign]
    second = _FakeBridge(raising, _FakeRemote())
    second.session_id = "s2"
    cast("dict[str, Any]", pool.bridges)[second.session_id] = second

    moved = await pool.rehome_stranded_sessions({"deepseek"}, "deepseek", "deepseek-flash")

    assert moved == []
    assert refused.asked and len(raising.asked) == 0


def test_the_receipt_shape_is_additive_and_says_what_happened() -> None:
    """``defaults_applied`` gains one field; a re-home-only login gets its own sentence.

    ``None`` means "the config default was left alone and there is nothing to
    say", and a login that moved live sessions is not that — so the count and a
    sentence appear even when the planner wrote nothing.
    """
    one = [("radient/auto", "deepseek/deepseek-flash")]
    applied = {"hosting": "deepseek", "model": "flash", "model_name": "F", "receipt": "Written."}
    assert with_rehome_count(applied, []) == applied
    assert with_rehome_count(None, []) is None
    assert with_rehome_count(applied, one) == {**applied, REHOMED_KEY: 1}

    only = with_rehome_count(None, one)
    assert only is not None
    assert only[REHOMED_KEY] == 1
    assert only["hosting"] is None, "nothing was written, so no hosting is claimed"
    assert only["receipt"] == "Switched to deepseek/deepseek-flash — not signed in to radient."

    many = with_rehome_count(None, one + [("radient/auto", "deepseek/deepseek-flash")])
    assert many is not None and many[REHOMED_KEY] == 2
    assert many["receipt"] == (
        "Moved 2 open sessions to deepseek/deepseek-flash — not signed in to radient."
    )


@pytest.mark.asyncio
async def test_a_host_without_a_pool_or_manager_moves_nothing() -> None:
    """An embedded app (or a test client) has no sessions to repair, and that is
    not an error — the sign-in it rides on has already succeeded."""
    empty = SimpleNamespace(state=SimpleNamespace())
    assert await rehome_after_login(empty) == []

    manager = SimpleNamespace(
        get_config=lambda: SimpleNamespace(values={"hosting": "", "model_name": ""})
    )
    no_target = SimpleNamespace(
        state=SimpleNamespace(config_manager=manager, desktop_sessions=object())
    )
    assert await rehome_after_login(no_target) == [], "no default pair, nothing to move to"
