"""The ``/schedule`` full-page page: real app, real keys, a fake session's scheduler.

Every state is driven through the shipped page. The session is the app's own
protocol fake with a scheduler bolted on, because the page's data path is the
LIVE scheduler (``session.wake_scheduler.schedules``) and its write path is
``session.set_wake_schedules`` — the session's one writer. Nothing here writes a
transcript or an index file.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest
from rich.text import Text

from local_operator.harness.wake_types import WakeSchedule
from local_operator.tui import schedule as schedule_model
from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets import schedule_view as schedule_view_mod
from local_operator.tui.widgets.schedule_view import ScheduleView
from tests.unit.tui.test_app_pilot import FakeSession, _factory

NOW = 1_790_000_000_000
MINUTE = 60_000
DAY = 86_400_000


def _schedule(wake_id: str, *, message: str = "check the build", **fields: Any) -> WakeSchedule:
    return WakeSchedule(
        id=wake_id,
        message=message,
        next_due_at=fields.pop("next_due_at", NOW + 30 * MINUTE),
        created_at=NOW - DAY,
        **fields,
    )


class ScheduleFakeSession(FakeSession):
    """The app's protocol fake, plus the scheduler the page reads.

    ``set_wake_schedules`` is the ONLY mutation seam the page is allowed to
    use, so the fake records every call and applies it to the scheduler's rows
    exactly as the real session's writer re-arms them.
    """

    def __init__(self, rows: list[WakeSchedule] | None = None, *, owns_runtime: bool = True):
        super().__init__()
        self.owns_runtime = owns_runtime
        self.wake_scheduler = SimpleNamespace(schedules=list(rows or ()))
        self.wake_writes: list[list[WakeSchedule]] = []

    async def set_wake_schedules(self, schedules: list[WakeSchedule]) -> None:
        self.wake_writes.append(list(schedules))
        self.wake_scheduler.schedules = list(schedules)


def _plain(static) -> str:
    content = static.content
    assert isinstance(content, Text)
    return content.plain


async def _settle(pilot, predicate, *, turns: int = 300) -> None:
    for _ in range(turns):
        if predicate():
            return
        await pilot.pause()
        await asyncio.sleep(0.01)
    raise AssertionError("condition did not settle")


async def _boot(pilot, app: OperatorApp) -> None:
    await _settle(pilot, lambda: app._session is not None)


async def _open(pilot, app: OperatorApp) -> ScheduleView:
    app._run_slash_command("/schedule")
    await pilot.pause()
    view = app.query_one(ScheduleView)
    await pilot.pause()
    return view


@pytest.mark.asyncio
async def test_schedule_lists_the_sessions_wakes_and_esc_restores_the_draft() -> None:
    session = ScheduleFakeSession(
        [
            _schedule("w1", message="check the deploy", every_ms=DAY, limit=3, fired_count=1),
            _schedule("w2", message="water the plants", notify=True),
        ]
    )
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(150, 40)) as pilot:
        await _boot(pilot, app)
        app._editor().load_text("half-typed prompt")
        await pilot.pause()

        view = await _open(pilot, app)
        assert app.screen.has_class("settings")
        assert not app._transcript_view().display
        body = _plain(view._body_text)
        assert "w1  check the deploy" in body
        assert "w2  water the plants" in body
        assert "every 1d" in body and "1 of 3 fired" in body
        assert "notify" in body
        assert view.selected is not None and view.selected.id == "w1"
        assert "check the deploy" in _plain(view._pane_body)

        await pilot.press("down")
        await pilot.pause()
        assert view.selected is not None and view.selected.id == "w2"
        assert "water the plants" in _plain(view._pane_body)

        # `enter` is the second edit key (the list's own vocabulary).
        await pilot.press("enter")
        await pilot.pause()
        assert view.view_mode == "form"
        await pilot.press("escape")
        await pilot.pause()
        assert view.view_mode == "list"

        await pilot.press("escape")
        await pilot.pause()
        assert len(app.query(ScheduleView)) == 0
        assert app._transcript_view().display
        assert not app.screen.has_class("settings")
        assert app._editor().text == "half-typed prompt"


@pytest.mark.asyncio
async def test_the_empty_state_names_the_new_key_and_opens_the_form() -> None:
    app = OperatorApp(lambda: _factory(ScheduleFakeSession()))
    async with app.run_test(size=(150, 40)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app)
        body = _plain(view._body_text)
        assert "No scheduled wakes" in body
        assert "Press n to schedule one" in body
        # The empty page still advertises its exit and its re-read.
        assert view._hint_new.display is True
        assert view._hint_refresh.display is True
        assert view._hint_escape.display is True

        await pilot.press("n")
        await pilot.pause()
        assert view.view_mode == "form"
        assert view.form.display is True

        # Esc pops ONE level: the form closes, the page stays.
        await pilot.press("escape")
        await pilot.pause()
        assert view.view_mode == "list"
        assert len(app.query(ScheduleView)) == 1


@pytest.mark.asyncio
async def test_create_through_the_form_writes_the_full_list_to_the_session() -> None:
    session = ScheduleFakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(150, 40)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app)

        await pilot.press("n")
        await pilot.pause()
        form = view.form
        form.message_field.focus()
        await pilot.press(*"hello")
        await pilot.pause()
        form.when_field.value = "in 30m"
        await pilot.pause()
        await pilot.press("ctrl+s")
        await pilot.pause()

        assert len(session.wake_writes) == 1
        [written] = session.wake_writes[0]
        assert written.message == "hello"
        assert written.next_due_at > NOW
        assert view.view_mode == "list"
        assert "w1  hello" in _plain(view._body_text)
        assert "Scheduled w1" in _plain(view._notice_text)


@pytest.mark.asyncio
async def test_edit_prefills_and_keeps_id_time_and_fired_count() -> None:
    row = _schedule(
        "w3",
        message="check the build",
        next_due_at=NOW + 2 * DAY,
        every_ms=DAY,
        limit=3,
        fired_count=2,
        notify=True,
    )
    session = ScheduleFakeSession([row])
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(150, 40)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app)

        await pilot.press("e")
        await pilot.pause()
        form = view.form
        assert form.message_field.text == "check the build"
        assert form.when_field.value == ""  # blank keeps the instant
        assert form.repeat_field.value == "1d"
        assert form.ends_field.value == "limit 3"
        assert form.notify_field.value == "on"

        form.message_field.text = "check the build twice"
        await pilot.pause()
        await pilot.press("ctrl+s")
        await pilot.pause()

        [written] = session.wake_writes[-1]
        assert written.id == "w3"
        assert written.message == "check the build twice"
        assert written.next_due_at == row.next_due_at
        assert written.fired_count == 2
        assert written.limit == 3
        assert "Saved w3" in _plain(view._notice_text)

        # ...and filling When DOES move the row's instant, keeping its id.
        await pilot.press("e")
        await pilot.pause()
        view.form.when_field.value = "in 1h"
        view.form.notify_field.set_value("off")
        await pilot.pause()
        await pilot.press("ctrl+s")
        await pilot.pause()
        [moved] = session.wake_writes[-1]
        assert moved.id == "w3"
        assert moved.fired_count == 2
        assert moved.next_due_at != row.next_due_at
        assert moved.notify is False


@pytest.mark.asyncio
async def test_cancel_asks_then_removes_and_n_keeps() -> None:
    session = ScheduleFakeSession([_schedule("w4", message="check the build")])
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(150, 40)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app)

        await pilot.press("d")
        await pilot.pause()
        assert view.confirming is True
        assert "cancel w4?" in _plain(view._notice_text)

        await pilot.press("n")
        await pilot.pause()
        assert view.confirming is False
        assert session.wake_writes == []
        assert "w4  check the build" in _plain(view._body_text)

        await pilot.press("d")
        await pilot.pause()
        await pilot.press("y")
        await pilot.pause()
        assert session.wake_writes == [[]]
        assert "Cancelled w4" in _plain(view._notice_text)
        assert "No scheduled wakes" in _plain(view._body_text)


@pytest.mark.asyncio
async def test_refusals_stay_in_the_form_with_the_builders_sentence() -> None:
    session = ScheduleFakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(150, 40)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app)
        await pilot.press("n")
        await pilot.pause()
        form = view.form

        async def save_and_read() -> str:
            await pilot.press("ctrl+s")
            await pilot.pause()
            assert view.view_mode == "form", "a refusal must not close the form"
            return form.refusal

        assert "non-empty 'message'" in await save_and_read()

        form.message_field.text = "check the build"
        form.when_field.value = "soonish"
        await pilot.pause()
        assert "invalid time 'soonish'" in await save_and_read()

        form.when_field.value = "2000-01-01 09:00"
        await pilot.pause()
        assert "in the past" in await save_and_read()

        form.when_field.value = "in 30m"
        form.repeat_field.value = "every 30s"
        await pilot.pause()
        assert "at least 60s" in await save_and_read()

        form.repeat_field.value = ""
        form.ends_field.value = "limit 3"
        await pilot.pause()
        assert "bound a repeat" in await save_and_read()

        # The cap is the live table's, not a snapshot the form took on open.
        form.ends_field.value = ""
        session.wake_scheduler.schedules = [
            _schedule(f"w{index}", message=f"row {index}") for index in range(1, 17)
        ]
        await pilot.pause()
        assert "at most 16 wake schedules" in await save_and_read()
        assert session.wake_writes == []


class RefusingWriteSession(ScheduleFakeSession):
    async def set_wake_schedules(self, schedules: list[WakeSchedule]) -> None:
        raise RuntimeError("the store is unwritable")


@pytest.mark.asyncio
async def test_a_failed_write_stays_in_the_form_with_the_reason() -> None:
    session = RefusingWriteSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(150, 40)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app)
        await pilot.press("n")
        await pilot.pause()
        view.form.message_field.text = "check the build"
        view.form.when_field.value = "in 30m"
        await pilot.pause()
        await pilot.press("ctrl+s")
        await _settle(pilot, lambda: "not saved" in view.form.refusal)
        assert view.view_mode == "form"
        assert "the store is unwritable" in view.form.refusal


@pytest.mark.asyncio
async def test_reread_picks_up_a_row_armed_outside_the_page() -> None:
    session = ScheduleFakeSession([_schedule("w1", message="check the build")])
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(150, 40)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app)
        assert "w2" not in _plain(view._body_text)

        session.wake_scheduler.schedules.append(_schedule("w2", message="water the plants"))
        await pilot.press("r")
        await pilot.pause()
        assert "w2  water the plants" in _plain(view._body_text)


@pytest.mark.asyncio
async def test_a_follower_gets_a_read_only_page_and_no_mutation_keys() -> None:
    session = ScheduleFakeSession([_schedule("w1", message="check the build")], owns_runtime=False)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(150, 40)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app)
        assert view.writable is False
        assert "Read-only" in _plain(view._notice_text)

        await pilot.press("n")
        await pilot.pause()
        assert view.view_mode == "list"
        await pilot.press("e")
        await pilot.pause()
        assert view.view_mode == "list"
        await pilot.press("d")
        await pilot.pause()
        assert view.confirming is False
        assert session.wake_writes == []


@pytest.mark.asyncio
async def test_a_narrow_page_hides_the_detail_pane() -> None:
    app = OperatorApp(lambda: _factory(ScheduleFakeSession([_schedule("w1")])))
    async with app.run_test(size=(80, 30)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app)
        assert view._narrow is True
        assert view._pane.display is False

    app = OperatorApp(lambda: _factory(ScheduleFakeSession([_schedule("w1")])))
    async with app.run_test(size=(150, 40)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app)
        assert view._narrow is False
        assert view._pane.display is True


@pytest.mark.asyncio
async def test_the_supervisor_line_is_one_best_effort_warning(monkeypatch) -> None:
    monkeypatch.setattr(
        schedule_view_mod,
        "supervisor_state",
        lambda config_root: SimpleNamespace(
            loaded=False, running=False, verifiable=False, detail="not loaded"
        ),
    )
    app = OperatorApp(lambda: _factory(ScheduleFakeSession()))
    async with app.run_test(size=(150, 40)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app)
        await _settle(pilot, lambda: view.supervisor_checked)
        assert "supervisor not running" in _plain(view._notice_text)
        # ...and it is a NOTE, not a gate: the page still opens the form.
        await pilot.press("n")
        await pilot.pause()
        assert view.view_mode == "form"


@pytest.mark.asyncio
async def test_a_running_supervisor_paints_no_warning(monkeypatch) -> None:
    monkeypatch.setattr(
        schedule_view_mod,
        "supervisor_state",
        lambda config_root: SimpleNamespace(loaded=True, running=True, verifiable=True, detail=""),
    )
    app = OperatorApp(lambda: _factory(ScheduleFakeSession()))
    async with app.run_test(size=(150, 40)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app)
        await _settle(pilot, lambda: view.supervisor_checked)
        assert "supervisor" not in _plain(view._notice_text)


def test_schedule_is_a_registered_frontend_local_slash_command() -> None:
    from local_operator.session.frontend_state import (
        _FRONTEND_LOCAL_SLASHES,
        CommandScope,
        _slash_capabilities,
    )
    from local_operator.slash_commands import SLASH_COMMANDS, slash_command_for

    command = slash_command_for("/schedule")
    assert command is not None
    assert command.name == "schedule"
    # TUI-only: no palette row can forward it to a desktop destination.
    assert command.desktop_destination == ""
    assert command.arguments.name == "NONE"
    assert "schedule" in _FRONTEND_LOCAL_SLASHES
    assert "schedule" in {entry.name for entry in SLASH_COMMANDS}
    capabilities = _slash_capabilities()
    assert (
        next(c for c in capabilities if c.command == "schedule").scope
        == CommandScope.FRONTEND_LOCAL
    )


def test_the_model_and_the_form_agree_on_what_edit_sends() -> None:
    """The form's prefill is what a blank field means: untouched values ride
    back exactly, so an edit that changes nothing changes nothing."""
    row = _schedule("w5", message="check the build", every_ms=DAY, limit=2, fired_count=1)
    values = schedule_model.prefilled(row)
    result = schedule_model.edit(values, [row], "w5", NOW)
    assert result.ok and result.schedule is not None
    assert result.schedule.model_dump() == row.model_dump()
