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

from local_operator.providers.model_access import REHOME_BUSY_REPLY
from local_operator.server.utils.desktop_rehome import (
    DEFERRED_KEY,
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

    moved, deferred = await pool.rehome_stranded_sessions(
        {"deepseek"}, "deepseek", "deepseek-flash"
    )

    assert client.asked == [("radient/auto", "deepseek", "deepseek-flash")]
    assert moved == [("radient/auto", "deepseek/deepseek-flash")]
    assert deferred == []


@pytest.mark.asyncio
async def test_the_guards_each_keep_a_session_untouched() -> None:
    """One pool, four reasons not to ask — and not one request sent to them.

    Busy is deliberately NOT among them any more (UX review U1): the MID-TURN
    session is ASKED so its owner can refuse in its own words and tell the
    conversation itself (see the refusal test below), which is what turns a
    silent non-repair into a deferral.
    """
    other_device = _FakeClient()
    remote_row = _FakeBridge(other_device, _FakeRemote(), remote_row=SimpleNamespace(device="b"))
    cold = _FakeClient()
    cold_bridge = _FakeBridge(cold, _FakeRemote(cold=True))
    stale_mirror = _FakeClient()
    stale = _FakeBridge(stale_mirror, _FakeRemote(canonical_current=False))
    signed_in = _FakeClient()
    signed_in_bridge = _FakeBridge(signed_in, _FakeRemote(provider="deepseek", model_id="flash"))
    pool = _pool_with(remote_row, cold_bridge, stale, signed_in_bridge)

    moved, deferred = await pool.rehome_stranded_sessions(
        {"deepseek"}, "deepseek", "deepseek-flash"
    )

    assert moved == [] and deferred == []
    for client in (other_device, cold, stale_mirror, signed_in):
        assert client.asked == []


@pytest.mark.asyncio
async def test_a_busy_session_is_asked_and_counts_as_deferred() -> None:
    """U1: the mid-turn conversation is not skipped — it is asked, refused, and counted.

    The owner is the one that answers ``kept: the session is working right now``
    (and speaks the deferral sentence into the conversation); the pool's half of
    the repair is then to report the count, so a sign-in where every session was
    busy is not silent.
    """
    busy = _FakeClient(answer=REHOME_BUSY_REPLY)
    pool = _pool_with(_FakeBridge(busy, _FakeRemote(streaming=True)))

    moved, deferred = await pool.rehome_stranded_sessions(
        {"deepseek"}, "deepseek", "deepseek-flash"
    )

    assert busy.asked == [("radient/auto", "deepseek", "deepseek-flash")]
    assert moved == []
    assert deferred == ["radient/auto"]


@pytest.mark.asyncio
async def test_nothing_is_asked_when_the_target_is_not_accessible() -> None:
    """The toggle: an unreadable store or an unreachable target asks NOBODY."""
    client = _FakeClient()
    pool = _pool_with(_FakeBridge(client, _FakeRemote()))

    assert await pool.rehome_stranded_sessions(None, "deepseek", "deepseek-flash") == ([], [])
    assert await pool.rehome_stranded_sessions({"openai"}, "deepseek", "deepseek-flash") == ([], [])
    assert client.asked == []


@pytest.mark.asyncio
async def test_an_owner_refusal_is_not_counted_and_never_raises() -> None:
    """The owner re-checks under its own state; its "kept: …" is a normal answer."""
    refused = _FakeClient(answer="kept: radient is signed in again")
    raising = _FakeClient()
    pool = _pool_with(_FakeBridge(refused, _FakeRemote()))

    async def _boom(*args: Any, **kwargs: Any) -> str:
        raise ConnectionError("owner went away")

    raising.rehome_if_current = _boom  # type: ignore[method-assign]
    second = _FakeBridge(raising, _FakeRemote())
    second.session_id = "s2"
    cast("dict[str, Any]", pool.bridges)[second.session_id] = second

    moved, deferred = await pool.rehome_stranded_sessions(
        {"deepseek"}, "deepseek", "deepseek-flash"
    )

    assert moved == [] and deferred == [], "a non-busy refusal is neither moved nor deferred"
    assert refused.asked and len(raising.asked) == 0


def test_the_receipt_shape_is_additive_and_says_what_happened() -> None:
    """``defaults_applied`` gains fields; a session-only login gets its own sentence.

    ``None`` means "the config default was left alone and there is nothing to
    say", and a login that moved live sessions (or deferred busy ones) is not
    that — so a count and a sentence appear even when the planner wrote nothing.
    """
    one = [("radient/auto", "deepseek/deepseek-flash")]
    applied = {"hosting": "deepseek", "model": "flash", "model_name": "F", "receipt": "Written."}
    assert with_rehome_count(applied, [], []) == applied
    assert with_rehome_count(None, [], []) is None
    assert with_rehome_count(applied, one, []) == {**applied, REHOMED_KEY: 1}

    only = with_rehome_count(None, one, [])
    assert only is not None
    assert only[REHOMED_KEY] == 1
    assert only["hosting"] is None, "nothing was written, so no hosting is claimed"
    assert only["receipt"] == "Switched to deepseek/deepseek-flash — not signed in to radient."

    many = with_rehome_count(None, one + [("radient/auto", "deepseek/deepseek-flash")], [])
    assert many is not None and many[REHOMED_KEY] == 2
    # The old provider is not named (review NIT-1): a two-provider move must not
    # read as if one provider had been involved.
    assert many["receipt"] == "Moved 2 open sessions to deepseek/deepseek-flash."

    # Every session busy: nothing moved, but the receipt says so — with the same
    # sentence the conversations themselves were told.
    busy_only = with_rehome_count(None, [], ["radient/auto"])
    assert busy_only is not None
    assert busy_only[DEFERRED_KEY] == 1
    assert busy_only["receipt"] == (
        "This conversation stays on radient/auto until the turn ends — /model switches it now."
    )

    # Mixed: the move's sentence stands, both counts ride along.
    mixed = with_rehome_count(None, one, ["deepseek/auto"])
    assert mixed is not None
    assert mixed[REHOMED_KEY] == 1 and mixed[DEFERRED_KEY] == 1
    assert mixed["receipt"] == "Switched to deepseek/deepseek-flash — not signed in to radient."

    # A config write that happened keeps ITS receipt; the counts still join.
    both = with_rehome_count(applied, one, ["radient/auto"])
    assert both == {**applied, REHOMED_KEY: 1, DEFERRED_KEY: 1}


@pytest.mark.asyncio
async def test_a_host_without_a_pool_or_manager_moves_nothing() -> None:
    """An embedded app (or a test client) has no sessions to repair, and that is
    not an error — the sign-in it rides on has already succeeded."""
    empty = SimpleNamespace(state=SimpleNamespace())
    assert await rehome_after_login(empty, "openai") == ([], [])

    manager = SimpleNamespace(
        get_config=lambda: SimpleNamespace(values={"hosting": "", "model_name": ""})
    )
    no_target = SimpleNamespace(
        state=SimpleNamespace(config_manager=manager, desktop_sessions=object())
    )
    assert await rehome_after_login(no_target, "openai") == (
        [],
        [],
    ), "no default pair, nothing to move to"


@pytest.mark.asyncio
async def test_the_first_login_rule_gates_the_whole_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the FIRST provider login moves sessions (operator refinement, round 1).

    With a second provider credentialed — or a prior Radient sign-in, which
    counts the same — the login is a user adding a provider to switch models
    with; the pool must not be asked at all, however stranded its sessions look.
    """
    from local_operator.server.utils import desktop_rehome

    client = _FakeClient()
    pool = _pool_with(_FakeBridge(client, _FakeRemote()))
    manager = SimpleNamespace(
        get_config=lambda: SimpleNamespace(
            values={"hosting": "deepseek", "model_name": "deepseek-flash"}
        )
    )
    app = SimpleNamespace(state=SimpleNamespace(config_manager=manager, desktop_sessions=pool))

    monkeypatch.setattr(
        desktop_rehome,
        "credentialed_chat_providers_here",
        lambda **kwargs: {"radient", "deepseek"},
    )
    assert await desktop_rehome.rehome_after_login(app, "deepseek") == ([], [])
    assert client.asked == [], "a second login must not touch conversations"

    monkeypatch.setattr(
        desktop_rehome, "credentialed_chat_providers_here", lambda **kwargs: {"deepseek"}
    )
    moved, deferred = await desktop_rehome.rehome_after_login(app, "deepseek")
    assert moved == [("radient/auto", "deepseek/deepseek-flash")]
    assert deferred == []
