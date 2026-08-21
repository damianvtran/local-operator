"""Wake surfaces: the transcript delivery receipt and the composer band panel.

A wake fires with no keystroke, which used to leave two gaps: the transcript
showed the agent starting to work with no record of WHY (no delivery line),
and the composer band had no standing answer to "does this session wake on its
own". These cover the two blocks that close them — ``WakeBlock`` (the
expandable receipt) and ``WakePanel`` (one row per schedule, hidden when
empty) — plus the session event that carries a live fire to the front end.
"""

from __future__ import annotations

import pytest
from rich.text import Text
from textual.app import App

from local_operator.harness.types import WakeDeliveredEvent
from local_operator.harness.wake import WakeSchedule
from local_operator.tui.widgets.tool_card import COLLAPSE_HINT, EXPAND_HINT
from local_operator.tui.widgets.transcript import (
    GAP_CLASS,
    NoticeBlock,
    TranscriptView,
    WakeBlock,
)
from local_operator.tui.widgets.wake_panel import WakePanel
from tests.unit.tui.conftest import StyledTranscriptApp

LIVE_TEXT = (
    "(alarm) Scheduled wake w3 (3/8, every 1h) — "
    'cancel with wake({op:"cancel",id:"w3"}) once its goal is met.\n\ncheck the build'
)
CATCHUP_TEXT = (
    "(alarm) The session resumed after being closed; the following scheduled wake(s) "
    "came due while it was down.\n\n- w1 (due 09:00): missed while the session was down.\n"
    "  Message: check the backup"
)


def _wake_text(block: WakeBlock) -> Text:
    """The applied Rich text, narrowed once for pyright and every assertion."""
    rendered = block.renderable
    assert isinstance(rendered, Text)
    return rendered


class TestWakeBlock:
    def test_collapsed_line_names_the_wake_and_hides_the_cancel_howto(self) -> None:
        block = WakeBlock(LIVE_TEXT)
        rendered = block._build_row(80).plain
        assert "w3" in rendered
        assert "wake" in rendered  # the name column, matching the tool ledger
        assert "check the build" not in rendered  # the body stays collapsed
        # The cancel instruction is for the model, not the user reading the line.
        assert "cancel with wake(" not in rendered
        # The envelope prefix is the model's framing; the card's fill already
        # says this is a delivery, so repeating "(alarm)" on the summary is
        # the dim single-line the user could not find.
        assert "(alarm)" not in rendered
        # At rest the expand hint is silent — the fill and the icon are the
        # affordance, the same contract as a settled tool row.
        assert EXPAND_HINT not in rendered
        assert "message" not in rendered

    def test_expand_reveals_the_full_message(self) -> None:
        block = WakeBlock(LIVE_TEXT)
        assert block.expanded is False
        assert block.toggle_expanded() is True
        expanded = block._build_content(80).plain
        assert "check the build" in expanded
        assert "w3" in expanded  # the summary row stays

    def test_catchup_bullet_wraps_with_a_hanging_indent(self) -> None:
        """A wrapped continuation hangs two cells deeper than the next
        bullet's dash, so at narrow widths a fold never reads as a new
        schedule line (design review round 1, D3)."""
        long_message = "report the state of the build and every failing test"
        text = (
            "(alarm) Scheduled wake w1 (1).\n\n"
            "- w1 (due 09:00): " + long_message + "\n"
            "- w2 (due 10:00): next"
        )
        block = WakeBlock(text, catchup=True)
        block.toggle_expanded()
        lines = block._build_content(40).plain.splitlines()
        bullet_rows = [i for i, line in enumerate(lines) if line.strip().startswith("- ")]
        assert len(bullet_rows) == 2
        continuation = lines[bullet_rows[0] + 1]
        assert not continuation.strip().startswith("- ")
        assert continuation.startswith(" " * 4)

    def test_catchup_line_marks_the_folded_misses(self) -> None:
        block = WakeBlock(CATCHUP_TEXT, catchup=True)
        rendered = block._build_row(80).plain
        assert "catch-up" in rendered
        assert "1 missed wake" in rendered
        assert "w1" in rendered
        # The model-facing "(alarm) The session resumed…" preamble must NOT
        # leak into the user-facing headline (review round 3, m3).
        assert "(alarm)" not in rendered
        block.toggle_expanded()
        expanded = block._build_content(80).plain
        assert "check the backup" in expanded
        assert "- w1" in expanded

    def test_activate_toggles_like_the_tool_ledger(self) -> None:
        """``activate`` returns True when it toggled, matching ToolCard —
        both expand and collapse are the row's one action, so both report
        success. The old return was the new expanded state, which made a
        collapse look like a no-op to any caller that checked the bool."""
        block = WakeBlock(LIVE_TEXT)
        assert block.activate() is True
        assert block.expanded is True
        assert block.activate() is True
        assert block.expanded is False

    def test_hint_appears_only_when_pointed_at_or_focused(self) -> None:
        """Same two-pointer contract as ToolCard: at rest the fill is the
        whole affordance; the hint lights under the pointer or the keyboard."""
        block = WakeBlock(LIVE_TEXT)
        assert EXPAND_HINT not in block._build_row(80).plain

        block._set_hovered(True)
        assert EXPAND_HINT in block._build_row(80).plain

        block._set_hovered(False)
        block._set_focused(True)
        assert EXPAND_HINT in block._build_row(80).plain

        block.toggle_expanded()
        row = block._build_row(80).plain
        assert COLLAPSE_HINT in row and EXPAND_HINT not in row

    def test_the_pointer_leaving_does_not_put_out_a_focused_rows_hint(self) -> None:
        block = WakeBlock(LIVE_TEXT)
        block._set_focused(True)
        block._set_hovered(True)
        block._set_hovered(False)
        assert EXPAND_HINT in block._build_row(80).plain

    def test_collapsed_card_is_one_row_and_expanded_is_taller(self) -> None:
        block = WakeBlock(LIVE_TEXT)
        assert block.spans_multiple_rows() is False
        block.toggle_expanded()
        assert block.spans_multiple_rows() is True


@pytest.mark.asyncio
async def test_real_pointer_hover_and_click_use_the_tool_trace_contract() -> None:
    """Exercise the actual Textual event path under the production sheet.

    Unit calls to ``_set_hovered`` prove the row builder; this proves the
    terminal receives a hand pointer, the real hover event reveals the hint,
    and a click grows/collapses the card without losing the one-row gap below.
    """
    app = StyledTranscriptApp()
    async with app.run_test(size=(100, 24)) as pilot:
        view = app.query_one(TranscriptView)
        wake = WakeBlock(LIVE_TEXT)
        below = NoticeBlock("trying another account", "warning")
        view.append_block(wake)
        view.append_block(below)
        await pilot.pause()
        await pilot.pause()

        assert wake.size.height == 1
        assert below.has_class(GAP_CLASS)
        assert below.region.y - wake.region.y == 2
        assert EXPAND_HINT not in _wake_text(wake).plain

        landed = await pilot.hover(wake)
        assert landed, "hover missed the wake card"
        await pilot.pause()
        assert app.screen._pointer_shape == "pointer"
        assert EXPAND_HINT in _wake_text(wake).plain

        await pilot.click(wake)
        await pilot.pause()
        assert wake.expanded is True
        assert wake.size.height == 2
        assert COLLAPSE_HINT in _wake_text(wake).plain
        assert below.region.y - wake.region.y == 3

        await pilot.click(wake)
        await pilot.pause()
        assert wake.expanded is False
        assert wake.size.height == 1
        assert below.region.y - wake.region.y == 2


@pytest.mark.asyncio
async def test_focused_wake_expands_and_collapses_on_enter_and_space() -> None:
    """The keyboard half of the affordance is not optional: a terminal may
    have no mouse reporting, and focus has to reveal what Enter will do."""
    app = StyledTranscriptApp()
    async with app.run_test(size=(100, 24)) as pilot:
        view = app.query_one(TranscriptView)
        wake = WakeBlock(LIVE_TEXT)
        view.append_block(wake)
        await pilot.pause()

        wake.focus()
        await pilot.pause()
        assert wake.has_focus
        assert EXPAND_HINT in _wake_text(wake).plain

        await pilot.press("enter")
        await pilot.pause()
        assert wake.expanded is True
        assert COLLAPSE_HINT in _wake_text(wake).plain

        await pilot.press("space")
        await pilot.pause()
        assert wake.expanded is False
        assert EXPAND_HINT in _wake_text(wake).plain


def _schedule(wake_id: str, message: str, every_ms: int | None = None, **kw) -> WakeSchedule:
    import time as _time

    now = int(_time.time() * 1000)
    return WakeSchedule(
        id=wake_id,
        message=message,
        next_due_at=now + 3_600_000,
        every_ms=every_ms,
        created_at=now,
        **kw,
    )


class _FakeScheduler:
    def __init__(self, schedules: list[WakeSchedule]) -> None:
        self.schedules = tuple(schedules)


class _FakeSession:
    def __init__(self, schedules: list[WakeSchedule]) -> None:
        self.wake_scheduler = _FakeScheduler(schedules)


class _PanelHost(App[None]):
    """A minimal app that mounts only a WakePanel, so ``Static.update`` and the
    screen-geometry probes run against a live Textual app (they raise
    ``NoActiveAppError`` / ``NoScreen`` off-app, which is what the panel's
    guards are for)."""

    def compose(self):  # type: ignore[override]
        yield WakePanel()


async def _paint(schedules: list[WakeSchedule]) -> tuple[bool, str]:
    """Sync a mounted panel; return (was_displayed, painted_text).

    ``display`` is read INSIDE the running app and returned as a plain bool:
    it is a Textual reactive that app shutdown resets, so reading it on the
    panel after ``run_test`` exits reports False even for a panel that painted.
    """
    app = _PanelHost()
    async with app.run_test(size=(100, 30)) as pilot:
        panel = app.query_one(WakePanel)
        panel.sync(_FakeSession(schedules))
        await pilot.pause()
        return bool(panel.display), str(panel._body.content)


class TestWakePanel:
    @pytest.mark.asyncio
    async def test_hidden_when_no_wakes(self) -> None:
        displayed, out = await _paint([])
        assert displayed is False
        assert out == ""

    @pytest.mark.asyncio
    async def test_one_row_per_schedule_not_per_occurrence(self) -> None:
        """A wake that fires every hour for a week is still ONE schedule, so
        it gets one line — the recurrence is stated once on it, never re-listed
        per trigger."""
        displayed, out = await _paint(
            [
                _schedule("w1", "check the backup", every_ms=3_600_000),
                _schedule("w2", "poll the queue"),
            ]
        )
        assert displayed is True
        assert "Wakes · 2 scheduled" in out
        assert "w1" in out and "every 1h" in out
        assert "w2" in out and "once" in out
        # Each schedule appears exactly once even though w1 recurs.
        assert out.count("w1") == 1

    @pytest.mark.asyncio
    async def test_message_snippet_is_shown(self) -> None:
        _, out = await _paint([_schedule("w1", "check the backup")])
        assert "check the backup" in out

    @pytest.mark.asyncio
    async def test_equality_guard_skips_identical_repaints(self) -> None:
        app = _PanelHost()
        async with app.run_test(size=(100, 30)):
            panel = app.query_one(WakePanel)
            session = _FakeSession([_schedule("w1", "x")])
            panel.sync(session)
            first = panel._shown
            panel.sync(session)
            assert panel._shown is first  # same object — no repaint happened

    @pytest.mark.asyncio
    async def test_survives_a_session_without_a_scheduler(self) -> None:
        app = _PanelHost()
        async with app.run_test(size=(100, 30)):
            panel = app.query_one(WakePanel)
            panel.sync(object())  # no wake_scheduler attribute
            assert panel.display is False

    @pytest.mark.asyncio
    async def test_overflow_marker_survives_a_short_screen(self) -> None:
        """On a screen short enough to floor the row budget, the "… N more"
        marker must still paint (review round 2, m1): silently dropping it
        leaves one visible wake beside a hidden count. The fix reserves the
        marker row even when that costs the last visible wake."""
        app = _PanelHost()
        async with app.run_test(size=(100, 12)):
            panel = app.query_one(WakePanel)
            session = _FakeSession([_schedule(f"w{i}", f"wake {i}") for i in range(1, 6)])
            panel.sync(session)
            out = str(panel._body.content)
            # Five schedules can never fit a floored budget; the marker must
            # report the hidden ones rather than vanish.
            assert "more wakes" in out


class TestWakeDeliveredEvent:
    def test_live_fire_and_catchup_are_distinguished(self) -> None:
        live = WakeDeliveredEvent(text=LIVE_TEXT, catchup=False)
        catchup = WakeDeliveredEvent(text=CATCHUP_TEXT, catchup=True)
        assert live.type == "wake_delivered" and live.catchup is False
        assert catchup.catchup is True

    def test_carries_the_full_text_for_the_expansion(self) -> None:
        event = WakeDeliveredEvent(text=LIVE_TEXT)
        assert "check the build" in event.text


@pytest.mark.asyncio
async def test_live_wake_fire_emits_a_delivery_receipt(tmp_path) -> None:
    """The session emits wake_delivered BEFORE spawning the turn, so the front
    end can paint the expandable line ahead of the work the wake triggered."""
    import time as _time

    from local_operator.harness.types import StreamEndEvent, StreamTextDelta
    from local_operator.harness.wake import DueWake, WakeSchedule
    from local_operator.session.session import Session
    from local_operator.session.transcript import Transcript
    from tests.unit.session.test_session import MODEL, ScriptedStream

    stream = ScriptedStream([[StreamTextDelta(delta="ack"), StreamEndEvent(stop_reason="stop")]])
    session = Session(
        model=MODEL,
        stream_fn=stream,
        tools=[],
        transcript=Transcript(tmp_path / "sess"),
        system_blocks_provider=lambda: [],
    )
    events = []
    session.subscribe(events.append)

    schedule = WakeSchedule(
        id="w1", message="wake up", next_due_at=int(_time.time() * 1000), created_at=0
    )
    due = DueWake(schedule=schedule, occurrence=1, planned_total=1, final=True)
    await session._deliver_wake(due)

    receipts = [e for e in events if getattr(e, "type", None) == "wake_delivered"]
    assert len(receipts) == 1
    assert receipts[0].catchup is False
    assert "wake up" in receipts[0].text
    await session.dispose()
