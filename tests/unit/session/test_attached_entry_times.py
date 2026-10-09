"""``AttachedSession``'s entry-time join: how it is accumulated, and its lifetime.

The viewer keeps the owner's ``{entry id: true entry ts}`` join so a reader
downstream can stamp a wire row with the moment it was WRITTEN rather than the
moment it was served. That join arrives one display page at a time, so a facade
has to MERGE it everywhere the rows it describes are merged — the initial window,
an older page prepended by a scroll, a materialized replay — and DROP it wherever
those rows are dropped. A method that merely returned ``{}`` (or that stopped
merging on one of those paths) leaves the desktop bridge serving ``unstated`` for
rows the owner could stamp, which is precisely the honesty gap the feature closes,
with nothing failing anywhere (agent review round 1, R-3).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from local_operator.harness.types import Message
from local_operator.session.attached import AttachedSession
from local_operator.session.frontend_state import (
    FrontendModelSpec,
    FrontendSessionState,
    FrontendSync,
)
from local_operator.session.history_window import DisplayHistoryWindow
from local_operator.session.transcript import Transcript


async def _no_takeover() -> None:
    raise AssertionError("a display facade never takes over")


def _model() -> FrontendModelSpec:
    return FrontendModelSpec(provider="p", model_id="m")


def _window(
    *rows: Message,
    entry_times: dict[str, float],
    generation: int = 1,
    start: int = 0,
    total: int | None = None,
    before_token: str | None = None,
    snapshot_token: str | None = "snap",
) -> DisplayHistoryWindow:
    return DisplayHistoryWindow(
        status="ok",
        conversation_id="s1",
        owner_epoch="e1",
        history_generation=generation,
        through_id=None,
        messages=list(rows),
        start=start,
        total_message_count=total if total is not None else len(rows),
        before_token=before_token,
        snapshot_token=snapshot_token,
        entry_times=dict(entry_times),
    )


def _frontend(window: DisplayHistoryWindow | None, *, generation: int = 1) -> FrontendSync:
    state = FrontendSessionState(
        session_id="s1",
        epoch="e1",
        cwd="/tmp",
        selected_model=_model(),
        effective_model=_model(),
        history_generation=generation,
    )
    return FrontendSync(
        epoch="e1", sequence=1, snapshot=state, live_cursor=None, display_history=window
    )


def _viewer(tmp_path: Path) -> AttachedSession:
    viewer = AttachedSession(config_dir=tmp_path, session_id="s1", takeover_factory=_no_takeover)
    # The window is only requested for a REMOTE placement (see the constructor's
    # own note), and a windowed page is what carries the join at all.
    viewer._display_window_requested = True  # noqa: SLF001 — the flag under test
    return viewer


@pytest.mark.asyncio
async def test_the_initial_window_installs_the_owners_join(tmp_path: Path) -> None:
    viewer = _viewer(tmp_path)
    first, second = Message.user("one"), Message.assistant("two")
    window = _window(first, second, entry_times={first.id: 10.0, second.id: 20.0})

    await viewer._load_frontend_history(_frontend(window))  # noqa: SLF001

    assert viewer.history_entry_times() == {first.id: 10.0, second.id: 20.0}


@pytest.mark.asyncio
async def test_an_older_page_prepends_its_own_join(tmp_path: Path) -> None:
    """A scroll backwards must extend the join, not replace or forget it.

    ``load_older_display_page`` prepends the older rows to ``_history``; the map
    has to grow with them or every row below the first window reads ``unstated``
    for a facade that has its instant in hand.
    """
    viewer = _viewer(tmp_path)
    newer = Message.user("newer")
    window = _window(
        newer,
        entry_times={newer.id: 20.0},
        start=1,
        total=2,
        before_token="tok",
        snapshot_token="snap",
    )
    await viewer._load_frontend_history(_frontend(window))  # noqa: SLF001

    older = Message.user("older")
    older_page = _window(
        older,
        entry_times={older.id: 10.0},
        start=0,
        total=2,
        before_token=None,
        snapshot_token=None,
    )

    async def fake_fetch(before: str, epoch: str, cursor: str | None, anchor: str = ""):
        del before, epoch, cursor, anchor
        return older_page

    viewer._fetch_history_page = fake_fetch  # type: ignore[method-assign]  # noqa: SLF001
    rows = await viewer.load_older_display_page()

    assert [row.id for row in rows] == [older.id]
    assert viewer.history_entry_times() == {older.id: 10.0, newer.id: 20.0}


@pytest.mark.asyncio
async def test_materialize_accumulates_every_page_it_walks(tmp_path: Path) -> None:
    """The explicit full replay walks several pages; the map covers all of them."""
    viewer = _viewer(tmp_path)
    tail_row = Message.user("tail")
    window = _window(
        tail_row,
        entry_times={tail_row.id: 30.0},
        start=1,
        total=2,
        before_token=None,
        snapshot_token="snap",
    )
    await viewer._load_frontend_history(_frontend(window))  # noqa: SLF001
    viewer._history_hydrated = False  # noqa: SLF001 — the replay's own precondition

    # The token names the TAIL cut, so the walk fetches the tail page first and
    # then follows ITS ``before_token`` — the real order, not a flattened one.
    head_row = Message.user("head")
    pages = {
        "snap": _window(
            tail_row,
            entry_times={tail_row.id: 30.0},
            start=1,
            total=2,
            before_token="older",
            snapshot_token="snap",
        ),
        "older": _window(
            head_row,
            entry_times={head_row.id: 10.0},
            start=0,
            total=2,
            before_token=None,
            snapshot_token="older",
        ),
    }

    async def fake_fetch(before: str, epoch: str, cursor: str | None, anchor: str = ""):
        del epoch, cursor, anchor
        return pages[before]

    viewer._fetch_history_page = fake_fetch  # type: ignore[method-assign]  # noqa: SLF001
    await viewer.materialize_history()

    assert viewer.history_entry_times() == {head_row.id: 10.0, tail_row.id: 30.0}


@pytest.mark.asyncio
async def test_a_non_window_install_drops_the_join_with_the_rows(tmp_path: Path) -> None:
    """A reset installs rows the owner did NOT stamp, so the map goes with them.

    ``_bind_history`` is the non-window source (a legacy full replay, a
    transcript bind): keeping the previous window's join here would describe ids
    from a page these rows no longer are — the same "join over absent rows" the
    two oversized fallbacks refuse.
    """
    viewer = _viewer(tmp_path)
    row = Message.user("row")
    await viewer._load_frontend_history(  # noqa: SLF001
        _frontend(_window(row, entry_times={row.id: 10.0}))
    )
    assert viewer.history_entry_times() == {row.id: 10.0}

    viewer._bind_history(
        [Message.user("replayed")], None, drop_history_duplicates=True
    )  # noqa: SLF001

    assert viewer.history_entry_times() == {}


@pytest.mark.asyncio
async def test_a_row_appended_live_is_absent_from_the_join(tmp_path: Path) -> None:
    """The negative half, and it is the honest answer rather than a gap.

    A live row reached this facade on the event stream, not on a display page, so
    no entry time exists here for it. It must be ABSENT — a reader turns that into
    ``unstated`` — rather than being dated with the facade's own clock.
    """
    viewer = _viewer(tmp_path)
    durable = Message.user("durable")
    await viewer._load_frontend_history(  # noqa: SLF001
        _frontend(_window(durable, entry_times={durable.id: 10.0}))
    )

    live = Message.user("live")
    viewer._live_history[live.id] = live  # noqa: SLF001
    served = {row.id for row in viewer.display_history_window()}
    assert live.id in served, "the fixture's live row was not served, so nothing is pinned"
    assert viewer.history_entry_times() == {durable.id: 10.0}
    assert live.id not in viewer.history_entry_times()


@pytest.mark.asyncio
async def test_the_accessor_hands_back_a_copy(tmp_path: Path) -> None:
    """A consumer must not be able to mutate this facade's state through the map."""
    viewer = _viewer(tmp_path)
    row = Message.user("row")
    await viewer._load_frontend_history(  # noqa: SLF001
        _frontend(_window(row, entry_times={row.id: 10.0}))
    )

    handed = viewer.history_entry_times()
    handed["injected"] = 99.0

    assert viewer.history_entry_times() == {row.id: 10.0}


@pytest.mark.asyncio
async def test_an_owner_that_ships_nothing_yields_the_empty_map(tmp_path: Path) -> None:
    """An older owner (or an unnegotiated attach) proves nothing, and says so."""
    viewer = _viewer(tmp_path)
    row = Message.user("row")
    await viewer._load_frontend_history(_frontend(_window(row, entry_times={})))  # noqa: SLF001

    assert viewer.history_entry_times() == {}


#: Kept so the module imports ``Transcript`` (the type the join is keyed by) and
#: fails loudly if this file is ever turned into a rig that writes a real journal
#: without saying so.
_ = Transcript
