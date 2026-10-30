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

ONE PARK IS ONE BANNER (review round 1, R1-1). The row is the gap this change
fills; the TOAST is not — ``_announce_pending`` already owns a park's OS banner
and decides it on evidence the store write does not have. So the raise goes
through the door with ``raise_banner=False``, and the cell below parks a real
gate on a detached handle and waits out ``BANNER_GRACE_MS`` (the interval the
item's own banner would have waited) before counting banners. Without that
leg standing down, one park posts two lock-screen cards for one event, which is
invisible to a test that calls ``_announce_pending`` in isolation.

AND ONE WAIT IS ONE ROW (QA round 1, Q1/Q2). The row's ``key`` is
``gate:{kind}:{wait_id}`` — the id the gate minted for that park — so a repeat
signal inside one wait updates one row, while a second wait gets a row of its
own. Keying on the TOOL instead made every later park of that tool invisible
after one dismissal (the store revives a dismissed row only on a priority
escalation, and a gate row is always ``high``), and made two concurrent cards
for one tool share a single row. Both cells below fail on that older key.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from local_operator.harness.types import AskOption, AskQuestion
from local_operator.notifications.delivery import BANNER_GRACE_MS
from local_operator.notifications.items import list_page
from local_operator.notifications.store import NotificationStore, NotifyRefused
from local_operator.session.runtime.serving import ServingSessionHandle
from local_operator.tools import builtin
from tests.unit.session.test_aside import RecordingStream, make_session

#: How long a raise may take to land. It goes through ``asyncio.to_thread`` to
#: SQLite (the ``_notify_inbox`` shape every other writer uses), so it is never
#: synchronous with the park.
_RAISE_BUDGET_S = 10.0


@pytest.fixture(autouse=True)
def _banners(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Every OS banner these cells would raise, and none of them on a real screen.

    ``notifications_enabled()`` is TRUE under pytest — the suite scrubs the
    ``LOP_NOTIFY`` kill switch and ``display.notifications`` defaults on — and
    ``desktop_belongs_to_this_process()`` is true for a developer's own HOME, so
    a park with nothing watching falls straight through to ``detached_notify``:
    without this recorder every park in this file posts a real lock-screen card
    from the machine running the suite, which is the exact failure
    ``suppress_notifications_for_process`` exists to prevent.

    ONE recorder covers BOTH legs, because each imports the function from
    ``local_operator.tui.notify`` at call time: ``_announce_pending`` (the
    park's toast, at park) and ``notifications.delivery.deliver_item`` (the
    runtime rung of an item's banner, after its grace). Counting them is how
    the one-banner-per-park rule becomes assertable — the duplicate only ever
    appears between the two.
    """
    raised: list[dict[str, Any]] = []

    def _record(*args: Any, **kwargs: Any) -> bool:
        raised.append({"args": args, **kwargs})
        return True

    monkeypatch.setattr("local_operator.tui.notify.detached_notify", _record)
    return raised


async def _until(predicate: Any, budget: float = _RAISE_BUDGET_S) -> bool:
    """Whether ``predicate`` held before ``budget`` elapsed (it is re-read at the end)."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + budget
    while loop.time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.01)
    return bool(predicate())


def _items(config_root: Path) -> list[dict[str, Any]]:
    """The operator's UNREAD inbox as the desktop reads it (``list_page``, §5.2)."""
    return list_page(config_root, source="notify", limit=10)["items"]


def _all_items(config_root: Path) -> list[dict[str, Any]]:
    """Every row the reel shows (``all`` = every row that is NOT dismissed)."""
    return list_page(config_root, source="notify", state="all", limit=10)["items"]


def _dismissed_items(config_root: Path) -> list[dict[str, Any]]:
    """The dismissed rows — the tombstones a repeat used to write into.

    ``list_items`` spells the three states as ``unread`` / ``all`` (not
    dismissed) / ``dismissed``, so a dismissal test has to ask for the third to
    see the row it dismissed.
    """
    return list_page(config_root, source="notify", state="dismissed", limit=10)["items"]


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


@pytest.mark.asyncio
async def test_a_parked_gate_raises_exactly_one_banner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _banners: list[dict[str, Any]]
) -> None:
    """ONE PARK, ONE EVENT, ONE CARD — the row must not banner it a second time.

    ``_announce_pending`` toasts a park the moment it happens. The inbox write
    that follows holds its own banner behind ``BANNER_GRACE_MS``, so a row that
    banner'd too put a SECOND lock-screen card, ~2.5 s later, on the same event
    and with the same body — in exactly the away-from-screen case this change
    exists for. ``_announce_pending`` already states the rule for its own
    Electron/detached pair ("falling through would post both Electron and
    detached-runtime OS toasts"); a park's row is the same pairing one store
    write later.

    The existing toast test in ``test_parked_gates.py`` cannot see this: it calls
    ``_announce_pending`` in isolation and never gives the item's banner its
    grace. This parks the real gate and waits it out.
    """
    config_root = _inbox_root(tmp_path, monkeypatch)
    # `tests/conftest` arms the kill switch for every cell ("safer ON"), and BOTH
    # legs of a park's banner ask it first — so the duplicate this cell pins is
    # unreachable while it is set. Dropping it is what puts this park in the
    # operator-away case; the recorder above then keeps the banners off this
    # machine's own screen, which is the price of measuring them otherwise.
    monkeypatch.delenv("LOCAL_OPERATOR_NO_NOTIFICATIONS", raising=False)
    # AND THE SUITE IS NOT THIS SCENARIO. `desktop_belongs_to_this_process`
    # refuses every banner under the test environment's redirected HOME
    # ("foreign HOME (rig guard)"), which would gate the item's route before any
    # count could be taken — the cell would pass while measuring nothing. The
    # case being pinned is the operator's OWN machine, away from the screen, so
    # the guard answers as it does there; the recorder above is what keeps the
    # cards off the screen that runs the suite.
    monkeypatch.setattr("local_operator.tui.notify.desktop_belongs_to_this_process", lambda: True)
    session = make_session(tmp_path / "top", RecordingStream())
    handle = _handle(tmp_path, session)

    # DETACHED AND ALONE: nothing is watching to paint the card, no desktop app
    # holds the lease, and no TUI reports itself, so both legs see an operator
    # who is not at the screen — the case the row exists for.
    monkeypatch.setattr(handle, "_watching_surfaces", lambda: frozenset())
    monkeypatch.setattr(handle, "_desktop_notification_available", lambda: False)
    monkeypatch.setattr(session, "_notify_tui_evidence", lambda: None)

    parked = asyncio.ensure_future(handle._approval_gate("bash", "rm -rf build/"))
    await asyncio.sleep(0)
    assert handle._fold.projection.pending is not None, "the gate never parked"
    assert len(_banners) == 1, f"the park itself must announce exactly once: {_banners}"

    # THE ROW IS STILL WRITTEN, and asserted beside the banner count so "one
    # banner" can never be satisfied by the write having been skipped instead.
    items = await _settle_items(config_root)
    assert len(items) == 1, f"a parked approval raised no inbox item: {items}"

    # WAIT OUT THE GRACE the item's banner would have held, and a margin past
    # it: the duplicate only exists after that expiry, which is why this is the
    # only shape of test that can catch it.
    await asyncio.sleep(BANNER_GRACE_MS / 1000 + 0.6)
    assert len(_banners) == 1, f"one park raised {len(_banners)} OS banners: {_banners}"
    assert len(_items(config_root)) == 1, "the row must survive the grace unmoved"

    parked.cancel()
    await asyncio.gather(parked, return_exceptions=True)
    await handle.dispose()


@pytest.mark.asyncio
async def test_one_wait_signalling_twice_updates_one_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The wait is the dedupe unit: a repeat signal updates its own row.

    This is the half of the key's job that must NOT change — two signals for
    ONE park (the same ``wait_id``) are one item, so a noisy re-raise costs the
    hourly ``NEW_ITEMS_PER_HOUR`` budget nothing. The keyed branch of
    ``NotificationStore.notify`` is what does it; what this pins is that the gate
    still hands it a key worth having.
    """
    config_root = _inbox_root(tmp_path, monkeypatch)
    session = make_session(tmp_path / "top", RecordingStream())

    for _ in range(2):
        await session.raise_gate_item("approval", "bash", "rm -rf build/", "wait-abc123")

    items = _all_items(config_root)
    assert len(items) == 1, f"one wait produced {len(items)} rows: {items}"
    assert items[0]["key"] == "gate:approval:wait-abc123", items[0]["key"]
    # IDENTICAL CONTENT IS ABSORBED WITHOUT A REVISION BUMP (``notify`` returns
    # "unchanged"), which is the store's own rule — what matters here is that the
    # second signal found its row instead of creating one.
    assert items[0]["repeat_count"] == 1, items[0]["repeat_count"]


@pytest.mark.asyncio
async def test_a_new_wait_after_a_dismissal_is_visible_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """QA ROUND 1, Q1: dismissing one card must not mute that tool.

    The key names the WAIT. With the tool in the key, the second park wrote into
    the dismissed row — and ``NotificationStore.notify``'s keyed branch revives a
    dismissed row only on a priority escalation, which a ``high`` gate row can
    never produce. The operator dismissed one approval and every later ``write``
    approval became invisible: the same "notification dies with its dismissal"
    gap this PR exists to close, still open for repeats.
    """
    config_root = _inbox_root(tmp_path, monkeypatch)
    session = make_session(tmp_path / "top", RecordingStream())
    handle = _handle(tmp_path, session)

    async def _park(tool: str, description: str) -> Any:
        parked = asyncio.ensure_future(handle._approval_gate(tool, description))
        await asyncio.sleep(0)
        assert handle._fold.projection.pending is not None, "the gate never parked"
        return parked

    first = await _park("write", "notes.txt")
    items = await _settle_items(config_root)
    assert len(items) == 1, f"the first park raised no item: {items}"
    dismissed_id, revision = items[0]["id"], items[0]["revision"]

    # DISMISSED, through the store's own token-bound write — the same one the
    # reel's dismiss button runs.
    NotificationStore(config_root).apply("dismiss", ids=[(dismissed_id, revision)])
    assert _items(config_root) == [], "the dismissed row must leave the unread reel"
    first.cancel()
    await asyncio.gather(first, return_exceptions=True)

    second = await _park("write", "after-dismiss.txt")
    assert await _until(
        lambda: len(_items(config_root)) == 1
    ), f"a park after the dismissal raised nothing visible: {_all_items(config_root)}"
    visible = _items(config_root)
    assert visible[0]["id"] != dismissed_id, "the new wait wrote into the dismissed row"
    assert visible[0]["key"] != "gate:approval:write", "the key still names the tool"
    # THE TOMBSTONE SURVIVES, deliberately: the old row stays dismissed (the
    # store's own absorption rule) while the new wait gets a row of its own.
    assert [row["id"] for row in _dismissed_items(config_root)] == [dismissed_id]
    assert [row["dismissed"] for row in visible] == [False]

    second.cancel()
    await asyncio.gather(second, return_exceptions=True)
    await handle.dispose()


@pytest.mark.asyncio
async def test_two_concurrent_parks_of_one_tool_are_two_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """QA ROUND 1, Q2: a parallel batch opens two cards, so it needs two rows.

    Two approvals for the same tool are two decisions the operator must make,
    and the fold already keeps them independent (``push_pending`` / ``pop_pending``
    by ``request_id``). Keying the item on the tool made them share one row, so
    answering the first looked like acknowledging both.
    """
    config_root = _inbox_root(tmp_path, monkeypatch)
    session = make_session(tmp_path / "top", RecordingStream())
    handle = _handle(tmp_path, session)

    # THE SAME TOOL, which is the case QA measured: a parallel batch can hold
    # two ``bash`` calls at once, and the card that names the tool must not be
    # the identity of the wait.
    first = asyncio.ensure_future(handle._approval_gate("bash", "rm -rf build/"))
    second = asyncio.ensure_future(handle._approval_gate("bash", "rm -rf /tmp/old"))
    await asyncio.sleep(0)
    assert handle._fold.projection.pending is not None, "the gates never parked"
    assert await _until(lambda: len(_items(config_root)) == 2), (
        f"two concurrent parks raised {len(_items(config_root))} row(s): "
        f"{_all_items(config_root)}"
    )
    keys = {row["key"] for row in _all_items(config_root)}
    assert len(keys) == 2, f"the two waits shared a key: {keys}"

    for task in (first, second):
        task.cancel()
    await asyncio.gather(first, second, return_exceptions=True)
    await handle.dispose()


@pytest.mark.asyncio
async def test_a_notify_disabled_process_parks_without_raising(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``LOP_NOTIFY=0`` removes the tool, so it removes the gate's row too.

    The kill switch is the operator declaring this process offers no inbox at
    all; a park that wrote a row anyway would be the one thing the switch
    promised not to do. The card still parks — the switch is about the inbox,
    never about the wait.
    """
    config_root = _inbox_root(tmp_path, monkeypatch)
    # Read through the function (``builtin.notify_enabled``), not the resolved
    # constant, so a cell can flip it: the constant is sampled at import.
    monkeypatch.setattr(builtin, "notify_enabled", lambda: False)
    session = make_session(tmp_path / "top", RecordingStream())
    assert session._notify_inbox_callable() is None, "the kill switch must close the door"
    handle = _handle(tmp_path, session)

    parked = asyncio.ensure_future(handle._approval_gate("write", "notes.txt"))
    await asyncio.sleep(0)
    assert handle._fold.projection.pending is not None, "the card must still park"

    loop = asyncio.get_running_loop()
    deadline = loop.time() + 1.0
    while loop.time() < deadline:
        await asyncio.sleep(0.01)
    assert _items(config_root) == [], "a notify-disabled process raised an item"

    parked.cancel()
    await asyncio.gather(parked, return_exceptions=True)
    await handle.dispose()


@pytest.mark.asyncio
async def test_a_composer_failure_costs_the_row_and_not_the_park(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The promise is the WHOLE raise, its imports included.

    ``raise_gate_item`` promises the card is never the thing lost, and the
    composer import sits on the same path — an ``ImportError`` there (a stripped
    deployment, a partially installed tree) escaped the previously guarded store
    write and would have broken the park, which is the one outcome the docstring
    says cannot happen. The card parks, the gate still answers, and the row is
    what the failure costs.
    """
    config_root = _inbox_root(tmp_path, monkeypatch)

    def _unimportable(*args: Any, **kwargs: Any) -> str:
        raise ImportError("no composer in this tree")

    monkeypatch.setattr("local_operator.notifications.gate_body", _unimportable)
    session = make_session(tmp_path / "top", RecordingStream())
    handle = _handle(tmp_path, session)

    # TWO PROBES, because absence alone proves nothing here: ``walked`` shows the
    # raise really ran (and resolved an owner) before the composer blew up, and
    # ``writes`` shows it never reached the store.
    walked: list[str] = []
    writes: list[dict[str, Any]] = []
    real_owner = session._notification_owner
    door = session._notify_inbox

    def _owner_probe() -> Any:
        walked.append("walked")
        return real_owner()

    async def _door_probe(**kwargs: Any) -> str:
        writes.append(kwargs)
        return await door(**kwargs)

    monkeypatch.setattr(session, "_notification_owner", _owner_probe)
    monkeypatch.setattr(session, "_notify_inbox", _door_probe)

    parked = asyncio.ensure_future(handle._approval_gate("bash", "rm -rf build/"))
    await asyncio.sleep(0)
    pending = handle._fold.projection.pending
    assert pending is not None, "a composer failure must not break the park"

    loop = asyncio.get_running_loop()
    deadline = loop.time() + 1.0
    while loop.time() < deadline:
        await asyncio.sleep(0.01)
    assert walked, "the raise was never attempted, so this cell would prove nothing"
    assert writes == [], "the failed composer must not reach the store"
    assert _items(config_root) == [], "the row is what a composer failure costs"

    # THE PARK IS INTACT, which is the point: the operator can still decide.
    await handle.approval_answer(pending.request_id, True, False)
    await asyncio.gather(parked, return_exceptions=True)
    await handle.dispose()


@pytest.mark.asyncio
async def test_an_unreadable_inbox_costs_the_row_and_not_the_park(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The store leg of the same promise: a busy or broken store loses the row.

    ``notify`` raises ``NotifyRefused`` past the hourly cap and
    ``StoreBusy``/``OperationalError`` on a held or unreadable database — the
    paths a parked runtime meets on a busy machine. None of them may reach the
    gate: the card is already parked, and an unparked approval is the worse
    failure.
    """
    _inbox_root(tmp_path, monkeypatch)
    session = make_session(tmp_path / "top", RecordingStream())
    handle = _handle(tmp_path, session)

    def _store_is_down(*args: Any, **kwargs: Any) -> Any:
        raise NotifyRefused("store busy", "refused: another process holds the lock")

    monkeypatch.setattr(NotificationStore, "notify", _store_is_down)

    parked = asyncio.ensure_future(handle._approval_gate("write", "notes.txt"))
    await asyncio.sleep(0)
    pending = handle._fold.projection.pending
    assert pending is not None, "a failed inbox write must not break the park"

    await handle.approval_answer(pending.request_id, True, False)
    await asyncio.gather(parked, return_exceptions=True)
    await handle.dispose()
