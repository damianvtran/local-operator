"""``display.hide_cross_session``: cross-session rows hidden from the TUI views.

The flag (default OFF) hides exactly two row kinds when on: inbound peer
receipts (``PeerMessageBlock``) and tool rows named ``send``. Every gate is a
return/continue BEFORE its object's construction, so the tests here drive the
REAL handlers — live receipt, composing/start/end cards, the replay fold, the
skipped-live painter, the drill-in page — rather than asserting on builders,
because an assertion on a constructor would miss exactly the class of
regression this file exists to catch: a gate moved below its construction, or
one that runs on a settle path.

Three properties carry the feature, and each has an opposite-case control so a
test cannot pass by suppressing too much:

* **OFF is today's rendering.** Receipts mount and record their de-dup id;
  send cards mint; the replay feeder skips a live call exactly as before.
* **ON mounts nothing AND records nothing.** A hidden live receipt must not
  enter ``_live_peer_receipts``, or a later replay under flag-off would skip a
  row that was never painted.
* **A mid-session flip is FORWARD ONLY.** A card created under flag-off
  settles normally after the flip, and the restore path is a re-projection
  under the current value — the ``/resume`` semantics ``tui/settings.py``
  documents — which the restore test drives through ``_project_settled_rows``.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator import settings_io
from local_operator.config import ConfigManager
from local_operator.harness.message_types import PEER_MESSAGE_MESSAGE_TYPE
from local_operator.harness.types import (
    AgentToolUpdate,
    Message,
    NoticeEvent,
    PeerMessageDeliveredEvent,
    TextContent,
    ToolCall,
    ToolCallComposeEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolExecutionUpdateEvent,
    ToolResult,
)
from local_operator.tui.app import OperatorApp
from local_operator.tui.events import ToolComposing, ToolEnded, ToolStarted, ToolUpdated
from local_operator.tui.widgets.tool_card import ToolCard
from local_operator.tui.widgets.transcript import (
    NoticeBlock,
    PeerMessageBlock,
    TranscriptView,
    UserBlock,
)

from .test_app_pilot import FakeSession, _factory
from .test_band_panels import FakeSession as _BandSession
from .test_band_panels import _async_factory, _fake_jobs
from .test_subagent_view import _call, _job_with, _open, _result

SENDER = {"pid": 4242, "conversation_name": "peer-session"}


@pytest.fixture()
def hide_cross_session(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """The flag, through the real config file and reader.

    Written through ``settings_io`` rather than by patching ``settings_get``,
    so the flat-dotted key the reader actually looks up is the one exercised —
    a nested write would pass a patched test and fail a user (the discipline
    ``test_narration_toggle``'s fixture records). Absent means OFF: tests that
    need the default leave ``set_hidden`` uncalled.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    from local_operator.tui.settings import settings_reload

    settings_reload()

    def set_hidden(on: bool) -> None:
        settings_io.write_setting(
            ConfigManager(tmp_path), settings_io.BY_KEY["display.hide_cross_session"], on
        )

    yield set_hidden
    settings_reload()


async def _boot(pilot, app: OperatorApp) -> None:
    for _ in range(200):
        await pilot.pause()
        if app._session is not None:
            return
    raise AssertionError("session did not finish booting")


async def _wait_for(pilot, predicate, *, what: str, frames: int = 200) -> None:
    for _ in range(frames):
        await pilot.pause()
        if predicate():
            return
    raise AssertionError(f"timed out waiting for {what}")


def _blocks(app: OperatorApp) -> list[Any]:
    return app.query_one(TranscriptView).blocks()


def _peer_blocks(app: OperatorApp) -> list[PeerMessageBlock]:
    return [b for b in _blocks(app) if isinstance(b, PeerMessageBlock)]


def _send_cards(app: OperatorApp) -> list[ToolCard]:
    return [b for b in _blocks(app) if isinstance(b, ToolCard) and b.tool_name == "send"]


def _peer_history_row() -> Any:
    """The persisted shape a ``/resume`` reads back: a custom peer row."""
    return SimpleNamespace(
        role=None,
        custom_type=PEER_MESSAGE_MESSAGE_TYPE,
        id="peer-1",
        text="",
        tool_calls=None,
        content=[],
        details={"body": "replayed note", "sender": {"pid": 9, "conversation_name": "other"}},
    )


def _send_call_tail(call_id: str) -> list[Any]:
    """A user turn plus an assistant call that has not returned — the shape
    whose call the replay treats as live when the session reports it running."""
    return [
        Message.user("run the thing"),
        Message(
            role="assistant",
            content=[TextContent(text="")],
            tool_calls=[ToolCall(id=call_id, name="send", arguments={"text": "hi"})],
            stop_reason="toolUse",
        ),
    ]


class _LiveSendSession(FakeSession):
    """A session whose ``send`` call is executing right now."""

    def executing_display_tool_ids(self) -> set[str]:
        return {"call-send"}


# ---------------------------------------------------------------------------
# The inbound peer receipt
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_opt_in_default_mounts_the_receipt_and_records_it(hide_cross_session) -> None:
    """Absent config = flag off = today's rendering, de-dup bookkeeping included."""
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        session.emit(
            PeerMessageDeliveredEvent(body="gates are green", sender=SENDER, message_id="peer-1")
        )
        await _wait_for(pilot, lambda: _peer_blocks(app), what="the peer receipt")
        blocks = _peer_blocks(app)
        assert len(blocks) == 1
        assert blocks[0].text() == "gates are green"
        # The double-paint guard is fed too, so a replay in this session skips it.
        assert "peer-1" in app._live_peer_receipts


@pytest.mark.asyncio
async def test_a_hidden_receipt_mounts_nothing_and_records_nothing(hide_cross_session) -> None:
    """ON: no block, and NO de-dup id — nothing was painted, nothing to de-dup."""
    hide_cross_session(True)
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        session.emit(
            PeerMessageDeliveredEvent(body="gates are green", sender=SENDER, message_id="peer-1")
        )
        # A SECOND event rides behind the peer note, and its block is waited
        # for: the queue provably drained past the peer event, so the negative
        # below is "delivered and suppressed" rather than "not delivered yet".
        # The user row retires the splash so the notice lands as a TRANSCRIPT
        # row — on the empty state, notices paint on the splash instead.
        app._append_block(UserBlock("hello"))
        session.emit(NoticeEvent(text="after the peer note", kind="info"))
        await _wait_for(
            pilot,
            lambda: any(isinstance(b, NoticeBlock) for b in _blocks(app)),
            what="the notice queued behind the peer note",
        )
        assert _peer_blocks(app) == []
        assert "peer-1" not in app._live_peer_receipts


@pytest.mark.asyncio
async def test_a_reprojection_under_flag_off_restores_the_receipt(hide_cross_session) -> None:
    """The restore path: flip off, re-project — the persisted row mounts.

    Booted under ON, the history replay consumes the peer row without
    painting OR recording it; the same row must then mount on a
    re-projection (the ``/resume`` semantics) once the flag is off. This
    fails if the replay gate ever records the id it did not paint.
    """
    hide_cross_session(True)
    session = FakeSession()
    session._history = [_peer_history_row()]
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        # The claim set is updated at the END of a projection pass, so its
        # membership proves the boot replay ran and consumed the row.
        await _wait_for(
            pilot, lambda: "peer-1" in app._resume_mounted_ids, what="the boot replay pass"
        )
        assert _peer_blocks(app) == []

        hide_cross_session(False)
        app._project_settled_rows(list(session._history))
        await _wait_for(pilot, lambda: _peer_blocks(app), what="the re-projected receipt")
        blocks = _peer_blocks(app)
        assert len(blocks) == 1
        assert blocks[0].text() == "replayed note"


# ---------------------------------------------------------------------------
# The send tool row
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hidden_send_never_mints_a_card(hide_cross_session) -> None:
    """Both creation points — the dictation and the start — are refused."""
    hide_cross_session(True)
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        app.post_message(
            ToolComposing(
                ToolCallComposeEvent(
                    tool_call_id="call-send", tool_name="send", intent="messaging a peer"
                )
            )
        )
        app.post_message(
            ToolStarted(
                ToolExecutionStartEvent(tool_call_id="call-send", tool_name="send", args={})
            )
        )
        for _ in range(10):
            await pilot.pause()
        assert _send_cards(app) == []
        assert "call-send" not in app._composing_cards
        assert "call-send" not in app._tool_cards


@pytest.mark.asyncio
async def test_send_events_still_mint_cards_when_not_hidden(hide_cross_session) -> None:
    """The control for the test above: the same two events DO mint when off."""
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        app.post_message(
            ToolComposing(
                ToolCallComposeEvent(
                    tool_call_id="call-send", tool_name="send", intent="messaging a peer"
                )
            )
        )
        await pilot.pause()
        assert "call-send" in app._composing_cards
        app.post_message(
            ToolStarted(
                ToolExecutionStartEvent(tool_call_id="call-send", tool_name="send", args={})
            )
        )
        await pilot.pause()
        assert "call-send" in app._tool_cards
        cards = _send_cards(app)
        assert len(cards) == 1


@pytest.mark.asyncio
async def test_a_card_created_before_the_flip_still_settles(hide_cross_session) -> None:
    """FORWARD ONLY: the flip never suppresses an existing card's settle.

    A card created under flag-off (compose + start) must settle through
    ``on_tool_ended`` after a mid-session flip to ON — one row, settled, not
    torn down and not left running.
    """
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        app.post_message(
            ToolComposing(ToolCallComposeEvent(tool_call_id="call-send", tool_name="send"))
        )
        app.post_message(
            ToolStarted(
                ToolExecutionStartEvent(tool_call_id="call-send", tool_name="send", args={})
            )
        )
        await pilot.pause()
        assert "call-send" in app._tool_cards

        hide_cross_session(True)  # the mid-session flip
        app.post_message(
            ToolEnded(
                ToolExecutionEndEvent(
                    tool_call_id="call-send",
                    tool_name="send",
                    result=ToolResult(
                        tool_call_id="call-send",
                        tool_name="send",
                        content=[TextContent(text="delivered")],
                    ),
                )
            )
        )
        await pilot.pause()
        cards = _send_cards(app)
        assert len(cards) == 1, "the flip must neither remove nor double the row"
        assert cards[0]._state == "success"


@pytest.mark.asyncio
async def test_hidden_update_and_end_with_no_card_are_noops(hide_cross_session) -> None:
    """The tolerance half: events for a row that was never minted do nothing.

    All four live arms run against a hidden ``send`` whose compose was
    suppressed; update and end must not crash, not mount, and not resurrect
    the row the flag refused.
    """
    hide_cross_session(True)
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        app.post_message(
            ToolComposing(ToolCallComposeEvent(tool_call_id="call-send", tool_name="send"))
        )
        app.post_message(
            ToolUpdated(
                ToolExecutionUpdateEvent(
                    tool_call_id="call-send",
                    tool_name="send",
                    partial_result=AgentToolUpdate(),
                )
            )
        )
        app.post_message(
            ToolEnded(
                ToolExecutionEndEvent(
                    tool_call_id="call-send",
                    tool_name="send",
                    result=ToolResult(
                        tool_call_id="call-send",
                        tool_name="send",
                        content=[TextContent(text="delivered")],
                    ),
                )
            )
        )
        for _ in range(10):
            await pilot.pause()
        assert _send_cards(app) == []
        assert "call-send" not in app._tool_cards
        assert "call-send" not in app._composing_cards


@pytest.mark.asyncio
async def test_a_hidden_send_never_reaches_the_skipped_live_painter(hide_cross_session) -> None:
    """The replay gate sits at the TOP of ``replay_tool_call``.

    A live ``send`` call would normally be diverted into
    ``_projection_skipped_live`` and painted by the skipped-live painter;
    hidden, nothing may be painted — observable as no card and no registry
    entry. The control below proves the same history and session DO paint
    when the flag is off, so this cannot pass vacuously.
    """
    hide_cross_session(True)
    session = _LiveSendSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        app._session = session
        app._project_settled_rows(_send_call_tail("call-send"))
        for _ in range(5):
            await pilot.pause()
        assert _send_cards(app) == []
        assert "call-send" not in app._tool_cards


@pytest.mark.asyncio
async def test_a_visible_live_send_is_still_painted_by_the_skipped_live_painter(
    hide_cross_session,
) -> None:
    """The control: with the flag off the same live call is fed and painted."""
    session = _LiveSendSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        app._session = session
        app._project_settled_rows(_send_call_tail("call-send"))
        for _ in range(5):
            await pilot.pause()
        cards = _send_cards(app)
        assert len(cards) == 1
        assert cards[0]._state == "running"
        assert "call-send" in app._tool_cards


@pytest.mark.asyncio
async def test_the_live_painter_refuses_a_hidden_send_second_door(hide_cross_session) -> None:
    """Defence in depth: even a call smuggled into the skipped list is refused.

    ``_replay_tool_call``'s gate means the feeder never receives a send today;
    this calls the painter directly with one so the second door is exercised
    rather than assumed.
    """
    hide_cross_session(True)
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        call = SimpleNamespace(id="call-send", name="send", arguments={})
        painted = app._paint_skipped_live_tool_rows(
            app.query_one(TranscriptView), {}, [call], session=app._session
        )
        assert painted == []
        assert _send_cards(app) == []


# ---------------------------------------------------------------------------
# The subagent drill-in
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_child_page_mounts_no_send_card_when_hidden(hide_cross_session) -> None:
    """A child's ``send`` row is dropped; its neighbours still mount.

    ``send`` is in ``DEFAULT_TOOL_NAMES``, so a child really can emit this
    row, and the drill-in builds one ``ToolCard`` per tool entry — the gate
    drops the entry before that construction. The flag-off control in the
    same test proves the trajectory reaches the mount.
    """
    hide_cross_session(True)
    trajectory = [
        _call("cs1", "send", text="hi"),
        _result("cs1", "send", "delivered"),
        _call("r1", "read", path="a.py"),
        _result("r1", "read", "contents"),
    ]

    session = _BandSession()
    session.jobs = _fake_jobs(_job_with(trajectory))
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        view = await _open(pilot, app, _job_with(trajectory))
        cards = [b for b in view._body.blocks() if isinstance(b, ToolCard)]
        assert [card.tool_name for card in cards] == ["read"]

    hide_cross_session(False)
    session = _BandSession()
    session.jobs = _fake_jobs(_job_with(trajectory))
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        view = await _open(pilot, app, _job_with(trajectory))
        cards = [b for b in view._body.blocks() if isinstance(b, ToolCard)]
        assert [card.tool_name for card in cards] == ["send", "read"]


# ---------------------------------------------------------------------------
# The one-place contract
# ---------------------------------------------------------------------------


def test_the_hidden_set_is_decided_in_one_place() -> None:
    """The key and the predicate are spelled ONCE, so the gates cannot drift.

    ``display.hide_cross_session`` may appear as a string literal in exactly
    three files: the predicate module that reads it, the settings registry
    that declares it, and the display-flag notes that document it. Every gate
    — TUI, phone, and the in-thread find pipeline — imports
    ``cross_session_hidden`` / ``SEND_TOOL_NAME`` instead, from the six files
    listed. A new spelling, a new gate site, or a gate that re-reads the key
    itself fails here rather than silently growing a second decision point;
    mirrors the repo's "exactly one caller" AST-guard style.
    """
    import ast

    package = Path(__file__).resolve().parents[3] / "local_operator"
    key = "display.hide_cross_session"
    spellings: set[str] = set()
    predicate_users: set[str] = set()
    for path in sorted(package.rglob("*.py")):
        rel = path.relative_to(package.parent).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=rel)
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and node.value == key:
                spellings.add(rel)
            elif isinstance(node, ast.Name) and node.id in (
                "cross_session_hidden",
                "SEND_TOOL_NAME",
            ):
                predicate_users.add(rel)
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name in ("cross_session_hidden", "SEND_TOOL_NAME"):
                        predicate_users.add(rel)

    assert spellings == {
        "local_operator/cross_session.py",
        "local_operator/settings_io.py",
        "local_operator/tui/settings.py",
    }, sorted(spellings)
    assert predicate_users == {
        "local_operator/cross_session.py",
        "local_operator/mobile/projection.py",
        "local_operator/session/transcript_find.py",
        "local_operator/tui/app.py",
        "local_operator/tui/session_presentation.py",
        "local_operator/tui/widgets/subagent_view.py",
    }, sorted(predicate_users)
