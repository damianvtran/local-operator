"""A warm re-selection reconciles its retained view by IDENTITY, never by count.

The reported defect: going back to a long conversation intermittently froze the
terminal app for 25-30 s before the first frame. Measured on the recheck rig
(``first-paint-evidence/TUI-RECHECK.md``, ``attr-main-*``): the retained
presentation had been prepared on the SAVED-PREVIEW facade, whose window is the
120-message excerpt, so it recorded ``history_size=120``. By the time the user
came back the same facade had bound the full journal (27,001 messages), and the
commit read ``total - history_size`` = 26,881 as "rows appended since the view
was prepared" and projected them synchronously — 20,148 blocks before the first
frame. None of them were new: all 51 mounted ids were inside that slice, and the
newest mounted row was the journal's last row. The slice was the gap between an
excerpt and the canonical history, and it was painted BELOW the tail.

These cells pin the fix as the operator meets it, on the real app and a real
saved-preview facade:

* the commit projects at most a screenful synchronously and the revealed frame
  carries no more blocks than a cold open of the same conversation;
* the history the excerpt did not hold is still reachable by scrolling up, in
  order, exactly once, with the reader's anchor held while a page lands.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest

from local_operator.harness.types import Message, TextContent
from local_operator.session.attached import AttachedSession
from local_operator.session.history_window import DISPLAY_HISTORY_MESSAGES
from local_operator.tui.app import (
    RESUME_PAGE_MESSAGES,
    OperatorApp,
    _viewport_message_budget,
)
from local_operator.tui.session_interaction import SessionInteraction
from local_operator.tui.widgets.transcript import UserBlock
from tests.e2e.harness import (
    ScriptedStream,
    seed_transcript,
    text_turn,
    wait_for_adoption,
)
from tests.unit.session.test_remote import _never_take_over
from tests.unit.tui.test_first_paint_reveal import FRAME, LONG_PROSE, _viewer

#: Well past the preview's excerpt, so the canonical history the facade binds
#: later is many screenfuls larger than the window the view was prepared from.
ROWS = DISPLAY_HISTORY_MESSAGES * 4


def _message(index: int) -> Message:
    return Message(
        id=f"warm-row-{index:04}", role="assistant", content=[TextContent(text=LONG_PROSE)]
    )


async def _preview(tmp_path, name: str, rows: int = ROWS) -> tuple[AttachedSession, list[Message]]:
    config = tmp_path / "config"
    config.mkdir(exist_ok=True)
    (config / "config.yml").write_text(
        "version: 0.0.0\nvalues:\n  hosting: test\n  model_name: mock\n"
    )
    directory = config / "sessions" / f"synthetic-{name}"
    messages = [_message(index) for index in range(rows)]
    await seed_transcript(directory, messages)
    remote = await AttachedSession.saved_preview(
        directory.name, config_dir=config, cwd=str(tmp_path), takeover_factory=_never_take_over
    )
    return remote, messages


def _lease(app: OperatorApp, remote: AttachedSession) -> SessionInteraction:
    source = SessionInteraction(remote)
    app._sidebar_sources[remote.session_id] = source
    app._interactions[id(remote)] = source

    async def lease(session_id, *, speculative=False):  # noqa: ANN001, ARG001
        source.preparations += 1
        return source

    app._lease_sidebar_source = lease  # type: ignore[method-assign]
    source.display_only = True
    source.draft.following_tail = True
    return source


def _projections(calls: list[int]):
    real = OperatorApp._project_settled_rows

    def spy(self, history, **kwargs):  # noqa: ANN001
        calls.append(len(history))
        return real(self, history, **kwargs)

    return real, spy


def _anchor_ids(app: OperatorApp) -> list[str]:
    return [
        block.navigation_anchor_id
        for block in app._transcript_view().blocks()
        if getattr(block, "navigation_anchor_id", "")
    ]


async def _settle(pilot, frames: int = 8) -> None:
    for _ in range(frames):
        await pilot.pause()


@pytest.mark.asyncio
async def test_a_retained_preview_commits_a_screenful_after_the_full_history_binds(
    tmp_path,
) -> None:
    async with _viewer(tmp_path, "warm-home") as home:

        async def factory():
            return home

        app = OperatorApp(factory)
        async with app.run_test(size=FRAME) as pilot:
            for _ in range(100):
                await pilot.pause()
                if app._session is home:
                    break
            budget = _viewport_message_budget(app.size.height)

            # The cold-open baseline: the same conversation, revealed from its
            # preview with nothing bound behind it.
            cold, _ = await _preview(tmp_path, "warm-cold")
            _lease(app, cold)
            prepared = await app._prepare_sidebar_session(cold.session_id)
            app._commit_sidebar_session(
                cold.session_id, prepared, app._sidebar_navigation.generation
            )
            await _settle(pilot)
            cold_blocks = len(app._transcript_view().blocks())
            await cold.dispose()

            remote, messages = await _preview(tmp_path, "warm-target")
            _lease(app, remote)
            prepared = await app._prepare_sidebar_session(remote.session_id)
            assert prepared[1].history_size == DISPLAY_HISTORY_MESSAGES
            # The facade binds the canonical journal while the view sits
            # retained — the hydration the sidebar's own connection performs.
            remote._bind_history(list(messages), None, drop_history_duplicates=True)
            assert remote.history_message_count == ROWS

            calls: list[int] = []
            real, spy = _projections(calls)
            OperatorApp._project_settled_rows = spy  # type: ignore[method-assign]
            try:
                app._commit_sidebar_session(
                    remote.session_id, prepared, app._sidebar_navigation.generation
                )
            finally:
                OperatorApp._project_settled_rows = real  # type: ignore[method-assign]
            committed_blocks = len(app._transcript_view().blocks())
            assert all(n <= budget for n in calls), (calls, budget)
            assert committed_blocks <= cold_blocks, (committed_blocks, cold_blocks)

            # The revealed rows are the conversation's TAIL, in order: nothing
            # older was painted below it.
            await _settle(pilot)
            ids = _anchor_ids(app)
            assert ids[-1] == messages[-1].id, ids[-3:]
            assert ids == sorted(ids), ids
            # Everything the excerpt did not hold is pending, not lost.
            assert len(app._resume_pending_head) + len(set(ids)) == ROWS
            await remote.dispose()


@pytest.mark.asyncio
async def test_scrolling_up_reaches_the_history_the_preview_did_not_hold(tmp_path) -> None:
    async with _viewer(tmp_path, "warm-home-2") as home:

        async def factory():
            return home

        app = OperatorApp(factory)
        async with app.run_test(size=FRAME) as pilot:
            for _ in range(100):
                await pilot.pause()
                if app._session is home:
                    break
            remote, messages = await _preview(tmp_path, "warm-scroll")
            _lease(app, remote)
            prepared = await app._prepare_sidebar_session(remote.session_id)
            remote._bind_history(list(messages), None, drop_history_duplicates=True)
            app._commit_sidebar_session(
                remote.session_id, prepared, app._sidebar_navigation.generation
            )
            await _settle(pilot, 20)
            view = app._transcript_view()

            # ANCHOR HELD: the row at the top of the viewport stays where the
            # reader saw it while an older page is inserted above it.
            view.scroll_to(y=0, animate=False)
            await _settle(pilot)
            top = view.scroll_y
            anchor: Any = next(
                block
                for block in view.blocks()
                if getattr(block, "navigation_anchor_id", "") and block.virtual_region.bottom > top
            )
            offset = anchor.virtual_region.y - view.scroll_y
            before = len(app._resume_pending_head)
            assert before > 0
            app._mount_older_resume_page()
            await _settle(pilot, 12)
            assert len(app._resume_pending_head) < before
            assert anchor.virtual_region.y - view.scroll_y == offset

            # Drain the head the way the documented chord does; every row of
            # the canonical history lands once, oldest first.
            for _ in range(ROWS):
                if not app._resume_pending_head:
                    break
                app._mount_older_resume_page()
                await _settle(pilot, 4)
            assert not app._resume_pending_head
            assert _anchor_ids(app) == [message.id for message in messages]
            await remote.dispose()


def _host(*, mounted, head=(), tail=(), notice=True, height=48):
    """The reconciliation's own state, without an app: ids, pending rows, notice."""
    projected: list[list[Any]] = []
    host = SimpleNamespace(
        _resume_mounted_ids={str(m.id) for m in mounted},
        _resume_pending_head=list(head),
        _resume_pending_tail=list(tail),
        _resume_results={},
        _resume_head_notice=object() if notice else None,
        size=SimpleNamespace(height=height),
        _project_settled_rows=lambda rows, **_: projected.append(list(rows)),
        _transcript_view=lambda: SimpleNamespace(blocks=lambda: []),
    )
    return host, projected


def test_rows_appended_to_an_anchored_window_join_its_pending_tail() -> None:
    """A view windowed around a saved anchor is not at the tail: nothing is painted below it."""
    rows = [_message(index) for index in range(40)]
    host, projected = _host(mounted=rows[10:20], head=rows[:10], tail=rows[20:30])
    assert OperatorApp._reconcile_retained_history(host, rows)  # type: ignore[arg-type]
    assert projected == []
    assert [m.id for m in host._resume_pending_tail] == [m.id for m in rows[20:]]
    assert [m.id for m in host._resume_pending_head] == [m.id for m in rows[:10]]


def test_more_than_a_screenful_appended_falls_back_to_a_bounded_reprojection() -> None:
    rows = [_message(index) for index in range(200)]
    host, projected = _host(mounted=rows[:20])
    assert not OperatorApp._reconcile_retained_history(host, rows)  # type: ignore[arg-type]
    assert projected == []


def test_older_rows_without_a_head_notice_fall_back_to_a_bounded_reprojection() -> None:
    """A view that never deferred anything has no row to page older history from."""
    rows = [_message(index) for index in range(60)]
    host, _ = _host(mounted=rows[50:], notice=False)
    assert not OperatorApp._reconcile_retained_history(host, rows)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_a_watched_turn_is_not_painted_twice_after_a_parked_append(
    tmp_path, monkeypatch
) -> None:
    """Review round 1, B1: rows painted LIVE are rows the view already knows.

    A turn watched on screen paints its prompt and reply through the live path,
    which never touched ``_resume_mounted_ids``. The id reconciliation then took
    the last PROJECTED row as the newest known one, classed the live rows as
    "appended", and painted the watched turn a second time when a turn that
    finished while parked brought the count gap. Real owners behind a real
    socket, the production sidebar select, as the reviewer reproduced it.
    """
    from tests.unit.tui.test_sidebar_peer_receipt import _switch, live_owners

    for key in tuple(__import__("os").environ):
        if key.startswith("CMUX_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("LOCAL_OPERATOR_NO_NOTIFICATIONS", "1")
    monkeypatch.setenv("LOCAL_OPERATOR_NO_TERMINAL_TITLE", "1")
    for name in (
        "_check_for_update",
        "_start_terminal_title",
        "_start_multiplexer_broadcast",
        "_start_herdr_reporter",
    ):
        monkeypatch.setattr(OperatorApp, name, lambda _self: None)
    config = tmp_path / "config"
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config))
    async with live_owners(config, ["home00", "busy01"], monkeypatch) as resume:
        handle = resume.servers["busy01"]._handle  # type: ignore[attr-defined]
        # The fixture's owners carry a one-turn tape; this cell runs two
        # turns, plus spares for the session's own extra calls (naming).
        handle._session._stream_fn = ScriptedStream(
            [text_turn("answer one"), text_turn("answer two")]
            + [text_turn("spare") for _ in range(6)]
        )
        app = OperatorApp(lambda: resume("home00"), resume_factory=resume)
        async with app.run_test(size=(120, 36)) as pilot:
            await wait_for_adoption(app, pilot)
            app._set_sidebar_open(True)
            if app._sidebar_timer is not None:
                app._sidebar_timer.pause()

            async def turn(text: str) -> None:
                viewer = cast(Any, app._sidebar_sources["busy01"].session)
                count = viewer.history_message_count
                await handle.prompt(text)
                for _ in range(800):
                    await pilot.pause()
                    if (
                        viewer.history_message_count >= count + 2
                        and not viewer.frontend_state.streaming
                    ):
                        break
                else:
                    raise AssertionError(f"the turn {text!r} never settled on the viewer")
                for _ in range(10):
                    await pilot.pause()

            await _switch(app, "busy01", pilot)
            await turn("live question one")  # watched: painted by the live path
            await _switch(app, "home00", pilot)
            await turn("parked question two")  # lands while the view is parked
            await _switch(app, "busy01", pilot)

            users = [
                block.navigation_anchor_id
                for block in app._transcript_view().blocks()
                if isinstance(block, UserBlock)
            ]
            assert len(users) == len(set(users)) == 3, users


@pytest.mark.asyncio
async def test_a_retained_view_too_far_behind_falls_back_to_a_bounded_tail(tmp_path) -> None:
    """Review round 1, M1: the fallback re-projection is itself bounded.

    More than a budget of rows appended past the retained tail cannot be
    reconciled in place, so the commit re-projects from scratch — and that
    pass must carry the screenful bound, or the fallback is the same unbounded
    synchronous projection the reconciliation exists to remove.
    """
    async with _viewer(tmp_path, "warm-home-3") as home:

        async def factory():
            return home

        app = OperatorApp(factory)
        async with app.run_test(size=FRAME) as pilot:
            for _ in range(100):
                await pilot.pause()
                if app._session is home:
                    break
            budget = _viewport_message_budget(app.size.height)
            remote, messages = await _preview(tmp_path, "warm-behind")
            _lease(app, remote)
            prepared = await app._prepare_sidebar_session(remote.session_id)
            later = [_message(ROWS + index) for index in range(budget * 3)]
            remote._bind_history(list(messages) + later, None, drop_history_duplicates=True)

            calls: list[tuple[int, int | None]] = []
            real = OperatorApp._project_settled_rows

            def spy(self, history, **kwargs):  # noqa: ANN001
                calls.append((len(history), kwargs.get("bound")))
                return real(self, history, **kwargs)

            OperatorApp._project_settled_rows = spy  # type: ignore[method-assign]
            try:
                app._commit_sidebar_session(
                    remote.session_id, prepared, app._sidebar_navigation.generation
                )
            finally:
                OperatorApp._project_settled_rows = real  # type: ignore[method-assign]
            assert calls, "the commit projected nothing"
            # The fallback's own pass carries the bound; anything else the
            # commit runs is an ordinary older-history page (the reveal's
            # prefill), which is RESUME_PAGE_MESSAGES by construction.
            assert calls[0][1] == budget, (calls, budget)
            assert all(
                bound == budget or n <= max(budget, RESUME_PAGE_MESSAGES) for n, bound in calls
            ), (calls, budget)
            # At commit: at most two budgets from the projection (its start
            # snaps back to a turn edge) plus the ONE older page the commit's
            # own under-fill pass mounts synchronously — measured 22 + 24.
            committed = set(_anchor_ids(app))
            assert len(committed) <= 2 * budget + RESUME_PAGE_MESSAGES, len(committed)
            await _settle(pilot)
            assert _anchor_ids(app)[-1] == later[-1].id
            await remote.dispose()
