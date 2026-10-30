"""A gate that PARKS a card raises a row in the operator's inbox.

THE GAP THESE PIN (operator report, 2026-10-30). An approval that parks is
visible on every surface except the inbox: ``Session._notify_inbox`` is the
inbox's only writer — the door behind the agent-callable ``notify`` tool — and
nothing on the gate path called it. The desktop showed the card, the reel
showed nothing, and an operator away from the screen learned about the gate only
from a lock-screen banner that is gone the moment it is dismissed
(``docs/design/notifications-inbox.md`` §1: that is the whole reason the store
exists).

The tests drive the REAL park — the gate closures a serving run installs on its
session (``ServingSessionHandle._install_gates``), not the raiser directly. A
raiser nothing calls is the defect being fixed, so the call path is the thing
under test.

TWO CASES, ONE GAP. An approval (a tool call held for a decision) and an ask (a
question held for an answer) are the same event from the inbox's side: the turn
cannot advance until a person returns. ``notifications/compose.gate_body``
already composes both; neither raised a row.

THE OWNER RULE (``Session._notification_owner``). The agent tool's gate answers
"can this session notify the operator" and refuses a delegated child, a hidden
session, a one-shot host, an output-contract session and ``LOP_NOTIFY=0``
(§3.2). A parked gate is the event that must not be silenced by the first two of
those: a child parked on an approval is a human-blocking condition in the
conversation the operator opened, so the item belongs in THAT conversation's
inbox — the parent's. The last three still mean no inbox, and the ownerless case
is pinned below as a deliberate, named drop rather than an accident.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from local_operator.harness.types import AskOption, AskQuestion
from local_operator.notifications.items import list_page
from local_operator.session.runtime.serving import ServingSessionHandle
from tests.unit.session.test_aside import RecordingStream, make_session

#: How long a raise may take to land. It goes through ``asyncio.to_thread`` to
#: SQLite (the ``_notify_inbox`` shape every other writer uses), so it is never
#: synchronous with the park.
_RAISE_BUDGET_S = 10.0


def _items(config_root: Path) -> list[dict[str, Any]]:
    """The operator's inbox as the desktop reads it (``list_page``, §5.2)."""
    return list_page(config_root, source="notify", limit=10)["items"]


async def _settle_items(config_root: Path) -> list[dict[str, Any]]:
    """The rows once the raise has had its chance to land, or the empty list."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + _RAISE_BUDGET_S
    while True:
        items = _items(config_root)
        if items or loop.time() >= deadline:
            return items
        await asyncio.sleep(0.01)


def _inbox_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the inbox at the test's own root, before any session is built.

    ``paths.config_dir()`` reads the environment on every call and
    ``Session.__init__`` resolves ``_config_dir`` through it, so the item a
    session raises lands here — never in the developer's real inbox.
    """
    config_root = tmp_path / "config"
    config_root.mkdir(exist_ok=True)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config_root))
    return config_root


def _handle(tmp_path: Path, session: Any) -> ServingSessionHandle:
    return ServingSessionHandle(session, asyncio.get_running_loop(), cwd=str(tmp_path))


@pytest.mark.asyncio
async def test_a_parked_approval_raises_an_inbox_item(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reported bug: the gate parks, the inbox stays empty."""
    config_root = _inbox_root(tmp_path, monkeypatch)
    session = make_session(tmp_path / "top", RecordingStream())
    handle = _handle(tmp_path, session)

    parked = asyncio.ensure_future(handle._approval_gate("write", "notes.txt"))
    await asyncio.sleep(0)
    pending = handle._fold.projection.pending
    assert pending is not None, "the approval never parked; the precondition is broken"

    items = await _settle_items(config_root)
    assert len(items) == 1, f"a parked approval raised no inbox item: {items}"
    item = items[0]
    # The item names the ACTION, the way the banner beside it already does
    # (``gate_body``) — "write: notes.txt", never "write: write: notes.txt".
    assert item["title"] == "write: notes.txt", item
    assert item["priority"] == "high", item
    # Attributed to the conversation the operator would open, not to the tool.
    assert item["session"]["id"] == session.session_id
    assert item["read"] is False and item["dismissed"] is False, item

    # Answering the card settles the CARD; the item stays until the operator
    # reads or dismisses it, exactly like every other row in the store.
    await handle.approval_answer(pending.request_id, True, False)
    assert await parked is True
    assert len(_items(config_root)) == 1

    await handle.dispose()


@pytest.mark.asyncio
async def test_a_parked_ask_raises_an_inbox_item(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Case 2, the same gap: a held decision raises no row either."""
    config_root = _inbox_root(tmp_path, monkeypatch)
    session = make_session(tmp_path / "top", RecordingStream())
    handle = _handle(tmp_path, session)

    question = AskQuestion(
        id="q0",
        question="Ship the release?",
        options=[AskOption(label="Ship it"), AskOption(label="Wait")],
    )
    answering = asyncio.ensure_future(handle._ask_gate([question]))
    await asyncio.sleep(0)
    pending = handle._fold.projection.pending
    assert pending is not None, "the ask never parked; the precondition is broken"

    items = await _settle_items(config_root)
    assert len(items) == 1, f"a parked ask raised no inbox item: {items}"
    assert items[0]["title"] == "Ship the release?", items[0]
    assert items[0]["priority"] == "high", items[0]
    assert items[0]["session"]["id"] == session.session_id

    answering.cancel()
    await asyncio.gather(answering, return_exceptions=True)
    await handle.dispose()


@pytest.mark.asyncio
async def test_a_delegated_child_parks_into_the_parents_inbox(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """THE DESIGN TRAP: the child's own gate refuses it, the parent's must not.

    A subagent inherits its parent's approval gate (``harness/subagent.py``), so
    a card it parks is a card the operator must answer in the parent
    conversation. The child's ``notify`` gate is ``None`` by design — asserting
    that first is what makes this a test of the OWNER walk rather than of the
    parent happening to be the session in hand.
    """
    config_root = _inbox_root(tmp_path, monkeypatch)
    parent = make_session(tmp_path / "parent", RecordingStream())
    child = make_session(
        tmp_path / "child", RecordingStream(), job_id="job-7", parent_session=parent
    )
    assert child._notify_inbox_callable() is None, "the child must not own an inbox"

    # WHICH door was used, not merely that one was: both sessions in this test
    # are built from a directory called ``sess`` (``make_session``'s shape), so
    # the ids collide and a row's ``session_id`` cannot tell them apart. The
    # child's own door is armed as a bomb and the parent's records the call.
    raised: list[dict[str, Any]] = []
    parent_door = parent._notify_inbox

    async def _parent_door(**kwargs: Any) -> str:
        raised.append(kwargs)
        return await parent_door(**kwargs)

    async def _child_must_not_raise(**kwargs: Any) -> str:
        raise AssertionError(f"the delegated child raised its own item: {kwargs}")

    monkeypatch.setattr(parent, "_notify_inbox", _parent_door)
    monkeypatch.setattr(child, "_notify_inbox", _child_must_not_raise)

    handle = _handle(tmp_path, child)
    parked = asyncio.ensure_future(handle._approval_gate("bash", "rm -rf build/"))
    await asyncio.sleep(0)
    assert handle._fold.projection.pending is not None, "the child's gate never parked"

    items = await _settle_items(config_root)
    assert len(items) == 1, f"a child's parked gate raised no item: {items}"
    assert raised, "the item did not go through the PARENT's inbox door"
    assert raised[0]["title"] == "bash: rm -rf build/", raised[0]

    parked.cancel()
    await asyncio.gather(parked, return_exceptions=True)
    await handle.dispose()


@pytest.mark.asyncio
async def test_an_ownerless_session_parks_without_raising(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A one-shot host has no inbox to send anyone back to — and says so.

    The host's product is the final text (``headless_print.py``), there is no
    conversation listing to open, and a row naming a session nothing surfaces
    can reach would be a notification the operator cannot act on or even find.
    The DROP is the decision; what this pins is that it is deliberate, that the
    card still parks, and that it is not the agent tool's gate being reused by
    accident.
    """
    config_root = _inbox_root(tmp_path, monkeypatch)
    session = make_session(tmp_path / "one-shot", RecordingStream())
    session.declare_one_shot_exit()
    handle = _handle(tmp_path, session)

    parked = asyncio.ensure_future(handle._approval_gate("write", "notes.txt"))
    await asyncio.sleep(0)
    assert handle._fold.projection.pending is not None, "the card must still park"

    # Give a would-be raise the same budget the positive tests get: absence has
    # to be as deliberate as presence, so the wait is the same one.
    loop = asyncio.get_running_loop()
    deadline = loop.time() + 1.0
    while loop.time() < deadline:
        await asyncio.sleep(0.01)
    assert _items(config_root) == [], "a one-shot host raised an item nobody can open"

    parked.cancel()
    await asyncio.gather(parked, return_exceptions=True)
    await handle.dispose()
