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
from textual.app import App

from local_operator.harness.types import WakeDeliveredEvent
from local_operator.harness.wake import WakeSchedule
from local_operator.tui.widgets.transcript import WakeBlock
from local_operator.tui.widgets.wake_panel import WakePanel

LIVE_TEXT = (
    "(alarm) Scheduled wake w3 (3/8, every 1h) — "
    'cancel with wake({op:"cancel",id:"w3"}) once its goal is met.\n\ncheck the build'
)
CATCHUP_TEXT = (
    "(alarm) The session resumed after being closed; the following scheduled wake(s) "
    "came due while it was down.\n\n- w1 (due 09:00): missed while the session was down.\n"
    "  Message: check the backup"
)


class TestWakeBlock:
    def test_collapsed_line_names_the_wake_and_hides_the_cancel_howto(self) -> None:
        block = WakeBlock(LIVE_TEXT)
        rendered = str(block._build())
        assert "w3" in rendered
        assert "check the build" not in rendered  # the body stays collapsed
        # The cancel instruction is for the model, not the user reading the line.
        assert "cancel with wake(" not in rendered
        assert "message" in rendered  # the collapsed line advertises the expansion

    def test_expand_reveals_the_full_message(self) -> None:
        block = WakeBlock(LIVE_TEXT)
        assert block.expanded is False
        assert block.toggle_expanded() is True
        expanded = str(block._build())
        assert "check the build" in expanded

    def test_catchup_line_marks_the_folded_misses(self) -> None:
        block = WakeBlock(CATCHUP_TEXT, catchup=True)
        rendered = str(block._build())
        assert "Wake catch-up" in rendered
        assert "missed wake(s)" in rendered

    def test_activate_toggles_like_the_tool_ledger(self) -> None:
        block = WakeBlock(LIVE_TEXT)
        assert block.activate() is True
        assert block.activate() is False


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
