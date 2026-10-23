"""The OPEN-ask dismiss doors and the card context line (R5, SA-D7).

Three claims, none of which a still can make:

* **`x` on an open ask asks WHY first.** A timed-out row keeps its one-key door
  (the agent already walked past that deadline, §14); an open one is still owed
  an answer, and the reason is what the agent acts on — so the key opens the
  two-choice card and sends nothing until a reason is chosen.
* **The reason reaches every door.** Own session, fleet live and fleet cold all
  forward it; a `None` on a lapsed row is the queue's own default, never a
  reason this surface invented.
* **The context line belongs to its row.** SA-D7's line is painted only on a
  stale OPEN ask, and the pointer hit test treats it as part of the ask above
  it — the same invariant round 3's wrap defect broke for the rows themselves.

The app-level paths are driven through the real ``OperatorApp`` (pilot), because
the failure modes here are interaction ones: a chooser that outlives its row, a
key that fires twice, an Escape that dismisses when it should have cancelled.
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from local_operator.asks import policy, store
from local_operator.asks.render import NOT_IN_QUEUE
from local_operator.tui import ask_open_policy
from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.ask_queue import (
    CONTEXT_CUE,
    DISMISS_CHOICES,
    FILTER_ALL,
    FILTER_SETTLED,
    SUB_LINE_INDENT,
    AskQueueList,
    ask_rows,
    context_text,
)
from tests.unit.tui.test_app_pilot import FakeSession, _factory

pytestmark = pytest.mark.asyncio

_NOW = int(time.time() * 1000)


def _row(ask_id: str, *, status: str = "open", **extra: Any) -> dict[str, Any]:
    """One fold-shaped ask record, the shape ``ask_rows`` flattens."""
    return {
        "ask_id": ask_id,
        "created_at": _NOW - 60_000,
        "expires_at": _NOW + 3_600_000,
        "timeout_s": 3600,
        "urgent": False,
        "status": status,
        "delivered": False,
        "questions": [
            {
                "id": "q1",
                "question": f"Question for {ask_id}?",
                "options": [{"label": "yes"}, {"label": "no"}],
                "multi": False,
                "recommended": None,
                "secret": False,
                "persist": False,
            }
        ],
        **extra,
    }


class _RecordingSession(FakeSession):
    """The owner contract, with the dismissal path recorded."""

    def __init__(self) -> None:
        super().__init__()
        #: ``(ask_id, surface, reason)`` per dismissal — the reason is the point.
        self.dismissed: list[tuple[str, str, str | None]] = []
        #: A sentence to refuse with, per ask id.
        self.dismiss_refusals: dict[str, str] = {}

    def dismiss_ask(self, ask_id, *, by="unknown", reason=None):
        error = self.dismiss_refusals.get(ask_id)
        if error:
            return {"ok": False, "error": error}
        self.dismissed.append((ask_id, by, reason))
        return {"ok": True}


def _app(session: FakeSession | None = None) -> OperatorApp:
    return OperatorApp(lambda: _factory(session or _RecordingSession()))


async def _settle(pilot, turns: int = 3) -> None:
    for _ in range(turns):
        await pilot.pause()


@pytest.fixture
def enabled(monkeypatch):
    """The queued-ask feature ON, and the open-by-default policy OFF.

    Every case here opens the surface itself (``_expand_asks``), so a policy that
    put it up on its own would make the frame under test ambiguous.
    """
    monkeypatch.setattr(policy, "NONBLOCKING_ASK", True)
    monkeypatch.setattr(ask_open_policy, "AUTO_OPEN", False)


async def _open_list(pilot, app: OperatorApp, rows: list[dict[str, Any]]) -> AskQueueList:
    app._sync_ask_surface(ask_rows(rows))
    await _settle(pilot)
    app._expand_asks()
    await _settle(pilot)
    return app.query_one(AskQueueList)


# -- `x` on an OPEN row: the two-choice reason card --------------------------


async def test_x_on_an_open_row_opens_the_reason_card_and_sends_nothing(enabled):
    """The key asks WHY before anything is written (R5).

    The old behaviour is the thing this replaces: `x` over an open row was inert
    (dismiss was offered on a timed-out ask alone), so a stale question waited
    for its deadline with no way to put it away from the terminal.
    """
    session = _RecordingSession()
    app = _app(session)
    async with app.run_test(size=(120, 30)) as pilot:
        await _settle(pilot)
        # TWO open rows, so the door mounts the LIST: one answerable ask expands
        # straight to its card (design D11), and the list is what `x` belongs to.
        listing = await _open_list(pilot, app, [_row("a1"), _row("a2")])
        await pilot.press("x")
        await _settle(pilot)
        assert listing.choosing, "x over an open row must open the reason card"
        # NOTHING IS SENT UNTIL A REASON IS CHOSEN — a dismissal the user did not
        # name a reason for is not one they gave.
        assert session.dismissed == []
        painted = listing.render().plain
        assert "Dismiss this ask" in painted
        assert "No longer needed" in painted
        assert "Answered in chat" in painted
        assert "enter dismiss" in painted and "esc cancel" in painted
        # THE BLOCK'S FOOTER NAMES THE KEYS, ONCE (design N2/n2): while the chooser
        # is up the header paints no hints at all, so the same row is not printed
        # twice within seven lines.
        assert listing.header_hints() == ()


async def test_enter_forwards_the_highlighted_reason(enabled):
    """Enter sends the reason the cursor is on, and `moot` is where it starts."""
    session = _RecordingSession()
    app = _app(session)
    async with app.run_test(size=(120, 30)) as pilot:
        await _settle(pilot)
        listing = await _open_list(pilot, app, [_row("a1"), _row("a2")])
        await pilot.press("x")
        await _settle(pilot)
        await pilot.press("enter")
        await _settle(pilot)
        assert session.dismissed == [("a1", "terminal", "moot")]
        assert not listing.choosing


async def test_down_then_enter_forwards_answered_in_chat(enabled):
    session = _RecordingSession()
    app = _app(session)
    async with app.run_test(size=(120, 30)) as pilot:
        await _settle(pilot)
        await _open_list(pilot, app, [_row("a1"), _row("a2")])
        await pilot.press("x")
        await _settle(pilot)
        await pilot.press("down")
        await pilot.press("enter")
        await _settle(pilot)
        assert session.dismissed == [("a1", "terminal", "answered_in_chat")]


async def test_escape_puts_the_reason_card_away_and_dismisses_nothing(enabled):
    """Esc cancels the chooser; it must not fall through to the list's collapse
    or, worse, leave a reason half-chosen for the next Enter to send."""
    session = _RecordingSession()
    app = _app(session)
    async with app.run_test(size=(120, 30)) as pilot:
        await _settle(pilot)
        listing = await _open_list(pilot, app, [_row("a1"), _row("a2")])
        await pilot.press("x")
        await _settle(pilot)
        await pilot.press("escape")
        await _settle(pilot)
        assert not listing.choosing
        assert session.dismissed == []
        # The LIST is still up: Esc closed the card, not the surface.
        assert app.query_one(AskQueueList) is listing
        assert "enter answer" in listing.header_hints()


async def test_a_lapsed_row_still_dismisses_in_one_key_with_no_reason(enabled):
    """§14's door is unchanged: a timed-out ask goes at once, and the queue
    records its own default reason rather than a word this surface invented."""
    session = _RecordingSession()
    app = _app(session)
    async with app.run_test(size=(120, 30)) as pilot:
        await _settle(pilot)
        listing = await _open_list(pilot, app, [_row("t1", status="timed_out")])
        assert not listing.choosing
        await pilot.press("x")
        await _settle(pilot)
        assert not listing.choosing, "a lapsed row must not open the reason card"
        assert session.dismissed == [("t1", "terminal", None)]


async def test_the_reason_card_closes_when_its_row_leaves_the_wire(enabled):
    """A chooser for a row nobody draws is a door to nothing: the next Enter
    would post a dismissal for an ask that is already settled elsewhere."""
    session = _RecordingSession()
    app = _app(session)
    async with app.run_test(size=(120, 30)) as pilot:
        await _settle(pilot)
        listing = await _open_list(pilot, app, [_row("a1"), _row("a2")])
        await pilot.press("x")
        await _settle(pilot)
        assert listing.choosing
        app._sync_ask_surface(ask_rows([_row("a2")]))
        await _settle(pilot)
        listing = app.query_one(AskQueueList)
        assert not listing.choosing
        assert session.dismissed == []


async def test_a_refused_dismiss_reports_and_clears_the_mark(enabled):
    """The reason door is the same door: a refusal is its own sentence, and the
    row comes back to life rather than staying inert."""
    session = _RecordingSession()
    session.dismiss_refusals["a1"] = "this ask was already answered."
    app = _app(session)
    async with app.run_test(size=(120, 30)) as pilot:
        await _settle(pilot)
        await _open_list(pilot, app, [_row("a1"), _row("a2")])
        seen: list[str] = []
        app._ask_gesture_refusal = lambda text: seen.append(text)
        await pilot.press("x")
        await pilot.press("enter")
        await _settle(pilot)
        assert seen == ["this ask was already answered."]
        assert "a1" not in app._ask_in_flight


def test_the_offered_reasons_are_the_queue_vocabulary_in_chooser_order():
    """The LABEL is this surface's; the RECORDED WORD is the log's. They are
    paired here once, and this is the cell that keeps the pairing honest."""
    reasons = [reason for reason, _label, _note in DISMISS_CHOICES]
    assert reasons == [store.DISMISS_REASON_DEFAULT, "answered_in_chat"]
    assert set(reasons) == set(store.DISMISS_REASONS)


# -- SA-D7: the card context line -------------------------------------------


def test_the_context_line_renders_the_wire_pair():
    rows = ask_rows([_row("a1", user_messages_since=3, last_user_message_at=_NOW - 42 * 60_000)])
    line = context_text(rows[0])
    assert line.startswith("3 messages from you since this was asked (last ")
    assert line.endswith(")")
    assert SUB_LINE_INDENT == 4


@pytest.mark.parametrize(
    "extra, status",
    [
        ({}, "open"),  # an older runtime: no field at all
        (
            {"user_messages_since": 0, "last_user_message_at": _NOW},
            "open",
        ),  # newer than every message
        ({"user_messages_since": 3}, "open"),  # a count with no clock
        ({"user_messages_since": 3, "last_user_message_at": _NOW}, "answered"),
        ({"user_messages_since": 3, "last_user_message_at": _NOW}, "timed_out"),
    ],
)
def test_no_context_line_without_a_stale_open_ask(extra, status):
    """ABSENCE IS THE CONTRACT: no field, a zero, half a pair, or a settled row
    all draw no line — the surface states nothing it cannot stand behind."""
    rows = ask_rows([_row("a1", status=status, **extra)])
    assert context_text(rows[0]) == ""


async def test_the_context_line_paints_under_its_own_row_and_nowhere_else(enabled):
    """A MOUNTED list, because the line is CLIPPED against the painted width: a
    detached widget has no content box, and every line would come out as `…`."""
    session = _RecordingSession()
    app = _app(session)
    async with app.run_test(size=(120, 30)) as pilot:
        await _settle(pilot)
        listing = await _open_list(
            pilot,
            app,
            [
                _row("a1", user_messages_since=3, last_user_message_at=_NOW - 42 * 60_000),
                _row("a2"),
            ],
        )
        painted = listing.render().plain.splitlines()
        # header + (row a1 + its context line) + row a2
        assert len(painted) == 1 + 2 + 1, painted
        assert "Question for a1?" in painted[1]
        assert "3 messages from you since this was asked" in painted[2]
        # It is a SUB-line: indented, and it never replaces the ask's own row.
        assert painted[2].startswith(" " * SUB_LINE_INDENT)
        assert "Question for a2?" in painted[3]


async def test_the_context_line_is_part_of_its_row_for_the_pointer(enabled):
    """Clicking the context line selects the ask it explains, not the one below
    it — the hit test reads the same line map the painter writes."""
    session = _RecordingSession()
    app = _app(session)
    async with app.run_test(size=(120, 30)) as pilot:
        await _settle(pilot)
        listing = await _open_list(
            pilot,
            app,
            [
                _row("a1", user_messages_since=3, last_user_message_at=_NOW - 60_000),
                _row("a2"),
            ],
        )
        top = listing.content_region.y - listing.region.y
        # y of the context line: header, then a1's row, then the sub-line.
        await pilot.click(AskQueueList, offset=(4, top + listing.HEADER_ROWS + 1))
        await _settle(pilot)
        # A click on a row opens its card; the ask it opened is the one above the
        # sub-line, never the one below.
        assert app._ask_mounted_id == "a1"


# -- round 1 remediation: the surfaces ---------------------------------------


def test_an_open_row_names_x_dismiss_at_the_default_widths(enabled) -> None:
    """D2 + UX U1: the new door must be named where the repo is actually used.

    The clause used to outbid the hint set, so a 100-column terminal named one
    key and a 78-cell one named two. The clause now yields to the whole live set,
    and the separator compacts before a hint is dropped.
    """
    listing = AskQueueList(ask_rows([_row("a1"), _row("a2")]))
    for width in (78, 98, 100):
        painted = listing.header_text(width)
        assert "x dismiss" in painted, (width, painted)
        assert "d decline" in painted, (width, painted)
        assert "enter answer" in painted, (width, painted)
    # THE MEASURED LIMIT, not a defect to fix here: with the sidebar up the list
    # is 56 cells, and `? ` + the three filter segments + `enter answer` already
    # spends 52 of them. Naming `x` too needs 63, and the only way to find the
    # room is to shed a segment the ladder ranks above the hints.
    narrow = listing.header_text(56)
    assert "enter answer" in narrow
    assert "x dismiss" not in narrow


async def test_the_context_line_leads_with_a_connector_a_row_never_has(enabled) -> None:
    """Design D6: the indent is also where a lapsed ask's row sits, so the
    context line needs a cue of its own or the slot means two things."""
    session = _RecordingSession()
    app = _app(session)
    async with app.run_test(size=(120, 30)) as pilot:
        await _settle(pilot)
        listing = await _open_list(
            pilot,
            app,
            [_row("a1"), _row("a2")],
        )
        listing.set_rows(
            ask_rows(
                [
                    _row("a1", user_messages_since=3, last_user_message_at=_NOW - 60_000),
                    _row("a2"),
                ]
            )
        )
        await _settle(pilot)
        painted = listing.render().plain.splitlines()
        context = next(line for line in painted if "messages from you" in line)
        assert context.lstrip().startswith(CONTEXT_CUE), context
        assert CONTEXT_CUE not in painted[0]


def test_the_context_line_wraps_instead_of_losing_its_clock() -> None:
    """UX U4: clipped to one row it lost "(last 14:14)" — the half that says how
    stale the ask is. It wraps on a space instead."""
    from local_operator.tui.widgets.ask_queue import _wrap_cells

    text = f"{CONTEXT_CUE}3 messages from you since this was asked (last 14:14)"
    wide = _wrap_cells(text, 72)
    assert len(wide) == 1
    narrow = _wrap_cells(text, 50)
    assert len(narrow) == 2, narrow
    assert "(last 14:14)" in narrow[1]
    assert all("…" not in line for line in narrow), narrow


def test_a_chooser_note_never_clips_mid_word() -> None:
    """UX U4: the note on the second reason used to be cut inside the word that
    carried the distinction ("conversati…"). It gets its own full-width line."""
    from local_operator.tui.widgets.ask_queue import _wrap_cells

    note = "you already replied in the conversation"
    assert _wrap_cells(note, 54) == [note]
    wrapped = _wrap_cells(note, 26)
    assert len(wrapped) >= 1
    assert wrapped[0].endswith(" ") is False
    # Every line is a whole word boundary: the first line does not end mid-word.
    assert note.startswith(wrapped[0])
    assert note[len(wrapped[0]) :].lstrip().startswith(wrapped[1].split(" ")[0])


async def test_a_successful_dismissal_states_its_receipt(enabled) -> None:
    """UX U2: the happy path used to say nothing at all — the row left the list
    and nothing reported what the queue had recorded."""
    session = _RecordingSession()
    app = _app(session)
    async with app.run_test(size=(120, 30)) as pilot:
        await _settle(pilot)
        await _open_list(pilot, app, [_row("a1"), _row("a2")])
        seen: list[str] = []
        app._ask_gesture_notice = lambda text: seen.append(text)
        await pilot.press("x")
        await pilot.press("enter")
        await _settle(pilot)
        assert seen == ["Dismissed — no longer needed"], seen


async def test_the_second_reason_reports_its_own_receipt(enabled) -> None:
    session = _RecordingSession()
    app = _app(session)
    async with app.run_test(size=(120, 30)) as pilot:
        await _settle(pilot)
        await _open_list(pilot, app, [_row("a1"), _row("a2")])
        seen: list[str] = []
        app._ask_gesture_notice = lambda text: seen.append(text)
        await pilot.press("x")
        await pilot.press("down")
        await pilot.press("enter")
        await _settle(pilot)
        assert seen == ["Dismissed — answered in chat"], seen


async def test_the_reason_card_closes_when_the_row_stops_being_answerable(enabled) -> None:
    """Agent review m1: an id that stays in the list is not enough — a row that
    settles to `answered` or `timed_out` has left the door the reasons belong to."""
    for status in ("answered", "timed_out"):
        session = _RecordingSession()
        app = _app(session)
        async with app.run_test(size=(120, 30)) as pilot:
            await _settle(pilot)
            listing = await _open_list(pilot, app, [_row("a1"), _row("a2")])
            await pilot.press("x")
            await _settle(pilot)
            assert listing.choosing
            listing.set_rows(
                ask_rows([_row("a1", status=status), _row("a2")]),
                open_count=1,
            )
            await _settle(pilot)
            assert not listing.choosing, status
            # And the next Enter cannot post a dismissal for a state nobody saw.
            await pilot.press("enter")
            await _settle(pilot)
            assert session.dismissed == [], status


async def test_the_reason_card_closes_when_the_filter_hides_its_row(enabled) -> None:
    """m1's second half: the Pending filter drops a row the moment it settles, so
    a chooser left up would act on something the list no longer draws."""
    session = _RecordingSession()
    app = _app(session)
    async with app.run_test(size=(120, 30)) as pilot:
        await _settle(pilot)
        listing = await _open_list(pilot, app, [_row("a1"), _row("a2")])
        await pilot.press("x")
        await _settle(pilot)
        assert listing.choosing
        listing.set_filter(FILTER_SETTLED)
        await _settle(pilot)
        assert not listing.choosing


async def test_the_list_keys_and_filter_segments_are_inert_while_choosing(enabled) -> None:
    """Agent review m2: the chooser owns the keyboard. `d` must not decline the
    row the reasons are about, and a filter must not silently cancel the choice
    the reader has not made yet."""
    session = _RecordingSession()
    session.dismiss_refusals.clear()
    app = _app(session)
    async with app.run_test(size=(120, 30)) as pilot:
        await _settle(pilot)
        listing = await _open_list(pilot, app, [_row("a1"), _row("a2")])
        await pilot.press("x")
        await _settle(pilot)
        await pilot.press("d")
        await pilot.press("2")
        await pilot.press("]")
        await _settle(pilot)
        assert listing.choosing
        assert listing.filter_value == FILTER_ALL
        assert session.dismissed == []
        # A click on a filter segment is inert for the same reason.
        spans = listing._segment_spans(max(1, int(listing.content_size.width)))
        value, start, _end = spans[1]
        await pilot.click(listing, offset=(start + 1, listing._top_inset()))
        await _settle(pilot)
        assert listing.filter_value == FILTER_ALL, value
        assert listing.choosing


async def test_a_click_on_a_reason_sends_that_reason(enabled) -> None:
    """The pointer path m2 asked to pin: a click on a choice's line sends it."""
    session = _RecordingSession()
    app = _app(session)
    async with app.run_test(size=(120, 30)) as pilot:
        await _settle(pilot)
        listing = await _open_list(pilot, app, [_row("a1"), _row("a2")])
        await pilot.press("x")
        await _settle(pilot)
        width = max(1, int(listing.content_size.width))
        owners = listing._line_owners(width)
        second = owners.index(("reason", 1))
        y = listing._top_inset() + listing.HEADER_ROWS + second
        await pilot.click(listing, offset=(10, y))
        await _settle(pilot)
        assert session.dismissed == [("a1", "terminal", "answered_in_chat")]


async def test_the_painted_lines_and_the_owner_map_agree_with_a_chooser_up(enabled) -> None:
    """The invariant the hit test rests on: one owner per painted line below the
    header, including the reason block's label and note rows."""
    session = _RecordingSession()
    app = _app(session)
    async with app.run_test(size=(120, 30)) as pilot:
        await _settle(pilot)
        listing = await _open_list(pilot, app, [_row("a1"), _row("a2")])
        await pilot.press("x")
        await _settle(pilot)
        width = max(1, int(listing.content_size.width))
        painted = listing.render().plain.splitlines()
        owners = listing._line_owners(width)
        assert len(painted) - listing.HEADER_ROWS == len(owners), (painted, owners)
        # The block's shape: heading, two lines per reason, hints.
        assert owners[1:2] == [None]
        assert owners[2:4] == [("reason", 0), ("reason", 0)]
        assert owners[4:6] == [("reason", 1), ("reason", 1)]
        assert owners[6] is None


async def test_a_fleet_dismissal_says_the_conversation_has_no_such_ask(
    enabled, monkeypatch
) -> None:
    """UX U5: on the all-conversations list, "it may belong to another session"
    contradicts the screen. The refusal names the log the row came from."""
    session = _RecordingSession()
    app = _app(session)

    async def _refuse(*_args, **_kwargs):
        return {"ok": False, "error": NOT_IN_QUEUE}

    monkeypatch.setattr("local_operator.asks.dismiss.dismiss_cold", _refuse)
    async with app.run_test(size=(120, 30)) as pilot:
        await _settle(pilot)
        row = ask_rows([_row("a1")])[0]
        seen: list[str] = []
        app._ask_gesture_refusal = lambda text: seen.append(text)
        app._ask_scope = "fleet"
        await app._fleet_ask_worker(row, "dismiss", None, "moot")
        await _settle(pilot)
        assert seen and "that conversation's log" in seen[0], seen
