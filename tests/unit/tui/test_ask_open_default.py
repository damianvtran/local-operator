"""The queued-ask surface opens BY ITSELF when a conversation with pending asks opens.

The operator's request: the minimized bar above the composer is easy to miss, so a
first-time user who opens a conversation with questions waiting may never find them.
Every ask surface therefore opens its primary interaction by default, under one
six-clause contract (``tui/ask_open_policy`` states it; the pure decision matrix is
pinned in ``test_ask_open_policy.py``). THIS file pins what the real ``OperatorApp``
does with it: what mounts, where the caret goes, and what a switch, a re-render and
a deliberate close do to it.

THE FOUR STATES THE OPERATOR NAMED, and the cells that own each:

* no asks on open            -> closed         ``test_state_no_asks_*``
* pending on open            -> open (once)    ``test_state_pending_on_open_*``
* all addressed on open      -> closed, and
                                never reopens  ``test_state_all_addressed_*``
* user closed while pending  -> stays closed
  across a re-render, a queue refresh, a new
  ask, and a switch away and back              ``test_state_dismissed_*``

WHY THIS FILE NEVER IMPORTS THE POLICY MODULE. It drives the app through the same
seams a user does (a published frame, a key, a click, a conversation switch) and reads
only what a user could see (what is mounted, where the caret is, what a key typed
into the composer did). That is what lets the same file run against ``origin/main``,
where nothing ever opens on its own, and fail there for a BEHAVIOURAL reason — the
surface never appeared — rather than an ``ImportError``. The three negative states
cannot fail on code that never opens anything, so each is written as the CLOSED arm
of a contrast whose OPEN arm does fail there: the closed result is then shown to be a
consequence of the data, not of a dead feature.

FRAMES COME THROUGH A REAL ``FrontendStateStore``, not ``_sync_ask_surface`` called by
hand. The decision is taken at the end of the frame fold, after the app has reconciled
the rows, and a hand-fed call skips the subscription, the coalescer and the swap edge
that decide WHEN the first frame is seen — which is exactly where the "frame lands
inside the adopt, before the incoming draft is loaded" case lives.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Sequence
from typing import Any

import pytest

from local_operator.asks import policy
from local_operator.session.frontend_state import (
    FrontendModelSpec,
    FrontendSessionState,
    FrontendStateStore,
)
from local_operator.tui.app import OperatorApp
from local_operator.tui.session_interaction import SessionInteraction
from local_operator.tui.widgets.ask_picker import TAB_HINT_KEY, AskPickerScreen
from local_operator.tui.widgets.ask_queue import ASK_TOGGLE_KEY, AskBar, AskQueueList
from local_operator.tui.widgets.editor import Editor
from local_operator.tui.widgets.transcript import TranscriptView
from tests.unit.tui.test_app_pilot import _factory
from tests.unit.tui.test_ask_queue_surface import _AskSession, _row
from tests.unit.tui.test_sidebar_swap_reset import SidebarRemote, _switch

pytestmark = pytest.mark.asyncio

#: An ask that existed a minute before the view began (so it is PENDING on open), and
#: one stamped a minute after (so it ARRIVED while the user was already looking). The
#: policy's own ``created_at`` tolerance is a few seconds either way; a minute clears it.
_BEFORE_MS = -60_000
_AFTER_MS = 60_000


def _ask(
    ask_id: str, question: str = "Deploy now?", *, when_ms: int = _BEFORE_MS, **extra: Any
) -> dict[str, Any]:
    """One wire row whose ``created_at`` is relative to NOW, which is the view's start.

    The shared fixture rows are dated 1970 and so are always "pending on open"; a row
    that has to be an ARRIVAL needs a clock that moves with the test.
    """
    created = int(time.time() * 1000) + when_ms
    return _row(
        ask_id,
        question,
        created_at=created,
        expires_at=created + 3_600_000,
        **extra,
    )


def _wire(rows: Sequence[dict[str, Any]] | None) -> dict[str, Any]:
    """The two frame fields a runtime publishes, in the shapes ``ask_wire`` produces.

    * ``None`` — a runtime that does not publish the queue at all: BOTH fields absent.
    * ``[]`` — a live queue with nothing in it: the rows absent, ``asks_open: 0``.
    * rows — the rows and the tally of the ones still outstanding.

    The split matters (the #864 lesson): absence is the capability proxy, and only an
    explicit ``0`` is an answer.
    """
    if rows is None:
        return {"asks": None, "asks_open": None}
    outstanding = sum(1 for row in rows if row["status"] in ("open", "timed_out"))
    return {"asks": list(rows) or None, "asks_open": outstanding}


class _Conversation(SidebarRemote):
    """An owner-backed session that carries a queue, on a REAL frontend store.

    ``SidebarRemote`` already gives the real store (so a frame travels the production
    subscription) and the navigation surface the sidebar's prepare/commit pair needs;
    this adds the queue's three ops, recorded so an assertion can name what a key
    actually did to an ask.
    """

    def __init__(
        self,
        session_id: str,
        rows: Sequence[dict[str, Any]] | None = None,
        *,
        wire: bool = False,
    ) -> None:
        super().__init__(session_id)
        self.answered: list[tuple[str, dict[str, Any]]] = []
        self.declined: list[str] = []
        self.dismissed: list[str] = []
        #: ask id -> the queue's refusal sentence. The row stays outstanding, which is what
        #: a refused answer leaves behind in real life.
        self.refusals: dict[str, str] = {}
        #: With ``wire`` the frames reach the app the way an ATTACHED viewer's do: the
        #: owner's store bounds the rows to the wire's text budget, and this store folds
        #: the resulting DELTAS. The sidebar switches onto exactly such sessions, and
        #: the bound is where ``asks_truncated`` comes from — an in-process owner never
        #: sets it (measured), so a cell that skips the wire cannot see the flag at all.
        self._owner: FrontendStateStore | None = None
        if wire:
            self._owner = FrontendStateStore(
                FrontendSessionState(
                    session_id=session_id,
                    epoch=f"epoch-{session_id}",
                    selected_model=FrontendModelSpec(provider="test", model_id="model"),
                )
            )
            # ``apply_update`` returns the new state and a subscriber's callback must return
            # None, so the viewer's fold is wrapped rather than handed over directly.
            self._owner.subscribe(lambda update: self._store.apply_update(update) and None)
        if rows is not None:
            self.publish(rows)

    def publish(self, rows: Sequence[dict[str, Any]] | None, **extra: Any) -> None:
        """Publish a frame, the way the runtime does on every queue change."""
        (self._owner or self._store).mutate(**{**_wire(rows), **extra})

    def respond_ask(self, ask_id, answers, *, by="unknown"):
        self.answered.append((ask_id, dict(answers)))
        error = self.refusals.get(ask_id)
        return {"ok": not error, **({"error": error} if error else {})}

    def decline_ask(self, ask_id, *, by="unknown"):
        self.declined.append(ask_id)
        return {"ok": True}

    def dismiss_ask(self, ask_id, *, by="unknown"):
        self.dismissed.append(ask_id)
        return {"ok": True}


class _OwnerConversation(_AskSession):
    """An in-process OWNER (``owns_runtime``) carrying a queue on a real frontend store.

    ``_Conversation`` is an ATTACHED viewer, and a viewer answers an approval through the
    non-owner route (``preserve_viewer_gate_reply``). The approval-over-card hazard lives
    on the owner's route, which is the one a user at their own terminal is on, so it needs
    the owner shape. ``_AskSession`` already is one and records the queue's ops; this adds
    the real store so frames still travel the production subscription.
    """

    def __init__(self, session_id: str, rows: Sequence[dict[str, Any]] | None = None) -> None:
        super().__init__()
        self._id = session_id
        self._store = FrontendStateStore(
            FrontendSessionState(
                session_id=session_id,
                epoch=f"epoch-{session_id}",
                selected_model=FrontendModelSpec(provider="test", model_id="model"),
            )
        )
        if rows is not None:
            self.publish(rows)

    @property
    def session_id(self) -> str:
        return self._id

    @property
    def frontend_state(self):
        return self._store.state

    def subscribe_frontend(self, handler):
        return self._store.subscribe(handler)

    def publish(self, rows: Sequence[dict[str, Any]] | None, **extra: Any) -> None:
        self._store.mutate(**{**_wire(rows), **extra})


@pytest.fixture(autouse=True)
def enabled(monkeypatch, tmp_path):
    """The queued-ask feature ON, and the app hermetic — the sibling suites' isolation.

    ``AttachedSession`` is swapped for the fake so the sidebar's commit path accepts
    it (the same patch ``test_sidebar_swap_reset`` applies), and the multiplexer / title /
    update hooks that would reach outside the process are stubbed.
    """
    monkeypatch.setattr(policy, "NONBLOCKING_ASK", True)
    monkeypatch.setattr("local_operator.session.attached.AttachedSession", _Conversation)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("LOCAL_OPERATOR_NO_NOTIFICATIONS", "1")
    monkeypatch.setenv("LOCAL_OPERATOR_NO_TERMINAL_TITLE", "1")
    monkeypatch.setattr(OperatorApp, "_check_for_update", lambda _self: None)
    monkeypatch.setattr(OperatorApp, "_start_terminal_title", lambda _self: None)
    monkeypatch.setattr(OperatorApp, "_start_multiplexer_broadcast", lambda _self: None)
    monkeypatch.setattr(OperatorApp, "_start_herdr_reporter", lambda _self: None)


def _app(session: _Conversation | _OwnerConversation) -> OperatorApp:
    return OperatorApp(lambda: _factory(session))


async def _until(pilot, predicate: Callable[[], bool], *, timeout: float = 4.0) -> bool:
    """Pump the app until ``predicate`` holds — waiting on the EVENT, never a fixed sleep."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        await pilot.pause(0.03)
    return predicate()


async def _pump(pilot, turns: int = 12) -> None:
    for _ in range(turns):
        await pilot.pause(0.02)


def _cards(app: OperatorApp) -> list[AskPickerScreen]:
    # `AskPickerScreen` is also the approval prompt's base class; the queued card is
    # the one carrying the queue's own id prefix.
    return [
        card
        for card in app.query(AskPickerScreen)
        if str(card.id or "").startswith("ask-queue-card-")
    ]


def _surface(app: OperatorApp) -> str:
    """``closed`` / ``card`` / ``list`` — read from what is MOUNTED, and cross-checked.

    The mode flag alone would pass over a surface that is flagged up with nothing
    painted (the stranded-ask-mode defect), and the mounts alone over a flag that
    says closed with a card still on screen; a frame where they disagree is returned as
    its own readable string so the failure says which half is wrong.
    """
    cards, lists = _cards(app), list(app.query(AskQueueList))
    mounted = "card" if cards else "list" if lists else "closed"
    if (mounted == "closed") == app._ask_mode:
        return f"inconsistent(mode={app._ask_mode}, cards={len(cards)}, lists={len(lists)})"
    return mounted


def _decided(app: OperatorApp) -> bool:
    """Whether the current view has taken its one decision (open OR leave closed).

    Read with a fallback so this file also runs against a tree with no policy at all
    (``origin/main``): there every view is "decided" at once, the closed assertions
    hold, and a cell fails where it should — at the arm that expects a surface to appear.
    """
    decider = getattr(app, "_ask_open_policy", None)
    return decider is None or not decider.awaiting(app._conversation_id())


def _composer_has_the_caret(app: OperatorApp) -> bool:
    return isinstance(app.focused, Editor)


async def _boot(pilot, app: OperatorApp) -> None:
    """Let the first frame land. Boot is a view like any other (clause 2)."""
    await _pump(pilot, 20)


# -- state 2: pending on open -> open, once ----------------------------------------


async def test_state_pending_on_open_one_ask_opens_its_card_and_the_caret_stays(enabled):
    """THE FEATURE: a conversation that opens with one question waiting shows it.

    Fails on ``origin/main`` because nothing there ever mounts a surface on its own —
    the bar is shown and the card is left for the user to find, which is the report.
    The caret assertion is the other half of clause 5: a user who has only just
    arrived and starts typing must land in the composer, not in a question.
    """
    session = _Conversation("conv-a", [_ask("a1")])
    app = _app(session)
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        assert await _until(pilot, lambda: _surface(app) == "card"), _surface(app)
        assert _composer_has_the_caret(app), f"the auto-open moved the caret to {app.focused!r}"
        assert [card.id for card in _cards(app)] == ["ask-queue-card-a1"]


async def test_state_pending_on_open_several_asks_open_the_list_and_the_caret_stays(enabled):
    """With more than one the LIST opens (choosing WHICH to answer is the point)."""
    session = _Conversation("conv-a", [_ask("a1", "Which region?"), _ask("a2", "Which tier?")])
    app = _app(session)
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        assert await _until(pilot, lambda: _surface(app) == "list"), _surface(app)
        assert _composer_has_the_caret(app), f"the auto-open moved the caret to {app.focused!r}"


async def test_state_pending_on_open_a_switch_to_a_conversation_with_pending_asks_opens_it(
    enabled,
):
    """ "Opened/switched-to" is the second half of clause 2, and the harder one.

    The sidebar's commit lands the first frame INSIDE the adopt, before the incoming
    conversation's own draft is loaded into the composer, and no further frame is
    promised for an idle conversation. A decision taken on that frame is taken blind,
    and a decision that waited needs its own clock — so this cell passes only if the
    app both WAITS for the switch and asks AGAIN when it ends.
    """
    home = _Conversation("conv-home")
    target = _Conversation("conv-a", [_ask("a1")])
    app = _app(home)
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        assert _surface(app) == "closed"
        await _switch(app, pilot, target)
        assert await _until(pilot, lambda: _surface(app) == "card"), _surface(app)
        assert app._conversation_id() == "conv-a"
        assert _composer_has_the_caret(app)


# -- state 1: no asks -> closed ------------------------------------------------------


async def test_state_no_asks_stay_closed_and_an_unresolved_frame_is_not_pending_asks(enabled):
    """Nothing to show -> nothing shown; and a frame that CANNOT say is not "pending".

    Three frames that carry no rows, then the contrast. A runtime that does not publish
    the queue (both fields absent) and a frame the wire bound reduced to its tally
    (``asks_open: 3`` with the rows dropped) are both "I cannot tell" — opening on the
    tally alone would mount an empty list over a queue that has asks (clause 5). An
    explicit ``asks_open: 0`` is an answer, and the answer is "nothing".

    The open arm is the point: the SAME view, once its rows are really there, opens.
    That arm fails on ``origin/main``; the closed arms before it cannot, which is why
    they are one cell and not three.
    """
    session = _Conversation("conv-a", None)
    app = _app(session)
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        # Unsupported: both fields absent.
        assert _surface(app) == "closed"
        # Tally-only: the count is known, the rows are not. Still waiting, still shut.
        session.publish(None, asks_open=3)
        await _pump(pilot)
        assert _surface(app) == "closed"
        # Rows resolve inside the window: NOW it is pending asks, and it opens. That it can
        # is the proof the two frames above did not spend the view's one decision.
        session.publish([_ask("a1"), _ask("a2")])
        assert await _until(pilot, lambda: _surface(app) == "list"), _surface(app)


async def test_state_no_asks_an_explicit_empty_queue_decides_closed(enabled):
    """``asks_open: 0`` is an answer: the view is decided, and a later ask only arrives."""
    session = _Conversation("conv-a", [])
    app = _app(session)
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        assert await _until(pilot, lambda: _decided(app))
        assert _surface(app) == "closed"
        # An ask the agent raises while the user is looking is covered by the bar, not
        # by a panel that appears over what they were reading.
        session.publish([_ask("a1", when_ms=_AFTER_MS)])
        await _pump(pilot)
        assert _surface(app) == "closed"
        assert app.query_one(AskBar).display, "the bar is how a new ask announces itself"


# -- state 3: all addressed -> closed, and never reopens -------------------------------


async def test_state_all_addressed_on_open_stays_closed_and_never_reopens(enabled):
    """Only settled rows -> closed, and the view's one decision is spent on that.

    The contrast: one outstanding ask among the settled ones opens (fails on
    ``origin/main``). The closed arm then proves the second half of clause 3 — a
    settled queue does not reopen when an ask arrives later, because the view already
    decided.
    """
    settled = [
        _ask("s1", status="answered", delivered=True),
        _ask("s2", status="declined"),
        _ask("s3", status="withdrawn"),
        _ask("s4", status="late"),
    ]
    session = _Conversation("conv-a", settled)
    app = _app(session)
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        assert await _until(pilot, lambda: _decided(app))
        assert _surface(app) == "closed"
        # An ask arriving later is the bar's business, not a reopening.
        session.publish([*settled, _ask("n1", when_ms=_AFTER_MS)])
        await _pump(pilot)
        assert _surface(app) == "closed"

    contrast = _Conversation("conv-b", [*settled, _ask("p1")])
    app = _app(contrast)
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        assert await _until(pilot, lambda: _surface(app) == "card"), _surface(app)


async def test_state_all_addressed_a_settled_queue_stays_shut_once_the_surface_has_gone(enabled):
    """The surface follows the queue DOWN too, and a view that has decided never reopens.

    The existing rule (unchanged here) is that a card stays up until its ask LEAVES the
    fold: an ask answered elsewhere keeps its row, settled, until the wire drops it. So
    the cell publishes the drop, and then tries to bring the surface back two ways — an
    old-looking ask and a genuinely new one — because "never auto-reopens once settled"
    has to hold against both.
    """
    session = _Conversation("conv-a", [_ask("a1")])
    app = _app(session)
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        assert await _until(pilot, lambda: _surface(app) == "card"), _surface(app)
        session.publish([])
        assert await _until(pilot, lambda: _surface(app) == "closed"), _surface(app)
        session.publish([_ask("late-old")])  # looks pending: it predates the view
        await _pump(pilot)
        assert _surface(app) == "closed", "a settled view reopened"
        session.publish([_ask("late-new", when_ms=_AFTER_MS)])
        await _pump(pilot)
        assert _surface(app) == "closed", "an arrival reopened a view that had decided"
        assert app.query_one(AskBar).display, "the bar is how the new ask announces itself"


# -- state 4: the user closed it while asks were pending -> stays closed ---------------


async def _open_then_close_with_f4(pilot, app: OperatorApp) -> None:
    """Wait for the auto-open, then close it the way a user does (the door, pressed)."""
    assert await _until(pilot, lambda: _surface(app) != "closed"), _surface(app)
    await pilot.press(ASK_TOGGLE_KEY)
    assert await _until(pilot, lambda: _surface(app) == "closed"), _surface(app)


async def test_state_dismissed_a_deliberate_close_holds_across_every_way_back(enabled):
    """Clause 4, whole: re-render, queue refresh, a new ask, and a switch away and back.

    Each step is a thing that WOULD reopen a surface that kept no memory of the close, and
    each is asserted separately so a failure names the one that did. The first assertion
    (it opened at all) is what makes this cell fail on ``origin/main``; everything after
    it is what goes red when the close stops being recorded (see the mutation table in the
    PR).
    """
    rows = [_ask("a1", "Which region?"), _ask("a2", "Which tier?")]
    home, a, b = _Conversation("conv-home"), _Conversation("conv-a", rows), _Conversation("conv-b")
    app = _app(home)
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _switch(app, pilot, a)
        await _open_then_close_with_f4(pilot, app)

        # A re-render: a frame that moved no ask (a token count), which still runs the fold.
        a._store.mutate(cumulative_parent_cost=1.5)
        await _pump(pilot)
        assert _surface(app) == "closed", "a re-render reopened a surface the user closed"
        # A queue refresh: an ask changed state (it timed out and is still answerable).
        a.publish([dict(rows[0], status="timed_out"), rows[1]])
        await _pump(pilot)
        assert _surface(app) == "closed", "a queue refresh reopened it"
        # A genuinely new ask: the bar announces it; it does not force the surface open.
        a.publish([dict(rows[0], status="timed_out"), rows[1], _ask("a3", when_ms=_AFTER_MS)])
        await _pump(pilot)
        assert _surface(app) == "closed", "a new ask forced it open"
        assert app.query_one(AskBar).display

        # Away and back, in the same app lifetime: the close is still respected.
        await _switch(app, pilot, b)
        assert _surface(app) == "closed"
        await _switch(app, pilot, a)
        await _pump(pilot, 30)
        assert app._conversation_id() == "conv-a"
        assert _surface(app) == "closed", "switching back reopened a surface the user closed"


@pytest.mark.parametrize("how", ["escape", "bar"])
async def test_state_dismissed_every_deliberate_close_is_recorded(enabled, how):
    """Esc and a click on the bar are the other two ways a user closes it, and both count.

    A recorded close is exactly what a LATER view of the same conversation is asked
    about, so the check is the strongest one available: leave and come back. Without the
    record the surface would be up again.
    """
    rows = [_ask("a1", "Which region?"), _ask("a2", "Which tier?")]
    home, a, b = _Conversation("conv-home"), _Conversation("conv-a", rows), _Conversation("conv-b")
    app = _app(home)
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _switch(app, pilot, a)
        assert await _until(pilot, lambda: _surface(app) == "list"), _surface(app)
        if how == "escape":
            await pilot.press("escape")
        else:
            await pilot.click(AskBar)
        assert await _until(pilot, lambda: _surface(app) == "closed"), _surface(app)
        await _switch(app, pilot, b)
        await _switch(app, pilot, a)
        await _pump(pilot, 30)
        assert _surface(app) == "closed", f"a close by {how} was not remembered"


async def test_state_dismissed_only_a_deliberate_close_counts_not_the_surface_going_away(enabled):
    """Leaving a conversation with the surface open is NOT the user refusing it.

    The swap takes the surface down (it belongs to the conversation that was on screen),
    and if that were recorded as a dismissal the user would come back to a closed panel
    they never closed. The contrast with the cell above is the whole point: same
    conversation, same asks, the only difference is who took the surface down.
    """
    rows = [_ask("a1", "Which region?"), _ask("a2", "Which tier?")]
    home, a, b = _Conversation("conv-home"), _Conversation("conv-a", rows), _Conversation("conv-b")
    app = _app(home)
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _switch(app, pilot, a)
        assert await _until(pilot, lambda: _surface(app) == "list"), _surface(app)
        await _switch(app, pilot, b)
        assert _surface(app) == "closed"
        await _switch(app, pilot, a)
        assert await _until(
            pilot, lambda: _surface(app) == "list"
        ), "the surface was taken down by the swap and then treated as a refusal: " + _surface(app)


async def test_state_dismissed_closing_the_fleet_list_is_not_refusing_this_conversation(
    enabled, tmp_path
):
    """The fleet list is about OTHER conversations, so closing it refuses nothing here.

    The user opens the all-conversations list from the sidebar note while this
    conversation's own asks are pending, and closes it. Recorded as a dismissal it would
    leave this conversation closed on every later visit over a question they never looked
    at. The contrast is the same close on the session's own list, which IS recorded (the
    sibling cells), so the difference is the scope and nothing else.
    """
    from local_operator.asks import store

    other = tmp_path / "config" / "sessions" / "conv-other"
    other.mkdir(parents=True)
    store.write_entry(tmp_path / "config", "conv-other", cwd="/tmp/other", asks=[_ask("o1")])

    rows = [_ask("a1", "Which region?"), _ask("a2", "Which tier?")]
    home, a, b = _Conversation("conv-home"), _Conversation("conv-a", rows), _Conversation("conv-b")
    app = _app(home)
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _switch(app, pilot, a)
        assert await _until(pilot, lambda: _surface(app) == "list"), _surface(app)
        app.action_open_fleet_asks()
        assert await _until(pilot, lambda: app._ask_scope == "fleet"), "the fleet list never opened"
        await pilot.press("escape")
        assert await _until(pilot, lambda: _surface(app) == "closed"), _surface(app)
        await _switch(app, pilot, b)
        await _switch(app, pilot, a)
        assert await _until(
            pilot, lambda: _surface(app) == "list"
        ), "closing the FLEET list was recorded as refusing this conversation: " + _surface(app)


@pytest.mark.parametrize("count", [1, 2])
async def test_state_dismissed_esc_on_an_engaged_surface_is_a_deliberate_close(enabled, count):
    """An engaged card or list handles Esc ITSELF, and that close is respected too.

    The sibling Esc cell closes a PASSIVE surface, which never holds the caret, so its Esc
    is the composer's stop route. Once the user has engaged the surface (Tab) Esc goes to
    the card's own cancel or the list's own collapse instead, and each reaches the app by a
    message of its own: two more doors to the same refusal, and each one a place the record
    can be forgotten without any other cell noticing.
    """
    rows = [_ask(f"a{i}", f"Question {i}?") for i in range(1, count + 1)]
    kind = "card" if count == 1 else "list"
    home, a, b = _Conversation("conv-home"), _Conversation("conv-a", rows), _Conversation("conv-b")
    app = _app(home)
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _switch(app, pilot, a)
        assert await _until(pilot, lambda: _surface(app) == kind), _surface(app)
        await pilot.press("tab")
        await _pump(pilot)
        surface = (_cards(app) if count == 1 else list(app.query(AskQueueList)))[0]
        assert app.focused is surface, f"Tab left the caret on {app.focused!r}"
        await pilot.press("escape")
        assert await _until(pilot, lambda: _surface(app) == "closed"), _surface(app)
        await _switch(app, pilot, b)
        await _switch(app, pilot, a)
        await _pump(pilot, 30)
        assert _surface(app) == "closed", f"the {kind}'s own Esc was not remembered as a close"


async def test_state_dismissed_a_refused_answer_is_not_a_refusal_to_look(enabled):
    """Answering is the opposite of waving a question off, even when the queue says no.

    The card collapses after an answer attempt whatever the verdict, so a REFUSED answer
    (the ask expired a moment earlier, say) leaves the question outstanding and the surface
    gone. Recording that collapse as a dismissal would keep the question hidden from a user
    who did exactly what the surface asked of them; coming back, they must see it again.
    """
    rows = [_ask("a1", "Which region?")]
    home, a, b = _Conversation("conv-home"), _Conversation("conv-a", rows), _Conversation("conv-b")
    a.refusals["a1"] = "this ask has expired"
    app = _app(home)
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _switch(app, pilot, a)
        assert await _until(pilot, lambda: _surface(app) == "card"), _surface(app)
        await pilot.press("tab")
        await _pump(pilot)
        await pilot.press("1")
        await pilot.press("enter")
        assert await _until(pilot, lambda: bool(a.answered)), "the answer never reached the queue"
        assert await _until(pilot, lambda: _surface(app) == "closed"), _surface(app)
        await _switch(app, pilot, b)
        await _switch(app, pilot, a)
        assert await _until(
            pilot, lambda: _surface(app) == "card"
        ), "a refused answer was recorded as the user waving the question off: " + _surface(app)


async def test_a_door_that_got_there_first_is_not_replaced_by_the_policy(enabled):
    """The user opens the surface while the view is still waiting; a late frame leaves it be.

    The first frame is a queue of settled rows beside a positive tally, so the view cannot
    tell yet (clause 5) and waits. The user presses the door and gets a list of that
    history. Then the frame that NAMES the pending ask arrives, which is exactly the frame
    that would open the surface for a view that was still waiting. It must not swap the
    user's list for a card they did not ask for.

    Three things stand between this frame and that swap, and any ONE of them is enough:
    the door settles the view (``note_user_opened``), the policy is told the surface is up
    (``surface_open``), and a door-opened surface is a hold on the keyboard (``occupied``).
    The layers are redundant on purpose, so no single mutant of them can fail this cell;
    removing all three can, and the mutation table in the PR says so.
    """
    settled = [_ask(f"s{i}", status="answered", delivered=True) for i in range(3)]
    session = _Conversation("conv-a")
    session.publish(settled, asks_open=1)  # one ask outstanding, but its row was dropped
    app = _app(session)
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        assert _surface(app) == "closed"
        await pilot.press(ASK_TOGGLE_KEY)
        assert await _until(pilot, lambda: _surface(app) == "list"), _surface(app)
        listing = app.query_one(AskQueueList)
        session.publish([_ask("p1", "Roll back?")])
        await _pump(pilot, 30)
        assert _surface(app) == "list", "the policy swapped the user's list for a card"
        assert listing.is_attached and app.query_one(AskQueueList) is listing
        # ``getattr`` so the file also runs on a tree with no policy (``origin/main``): this
        # is a GUARD cell, not a fail-on-old one, because nothing there opens a surface on
        # its own that could race the door. Its job is to fail on a tree that LOST the guard.
        assert getattr(listing, "passive", False) is False, "a hand-opened list went passive"


async def test_state_dismissed_is_per_conversation(enabled):
    """Closing conversation A's surface says nothing about conversation B's."""
    a = _Conversation("conv-a", [_ask("a1", "Which region?"), _ask("a2", "Which tier?")])
    b = _Conversation("conv-b", [_ask("b1", "Rotate the key?")])
    home = _Conversation("conv-home")
    app = _app(home)
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _switch(app, pilot, a)
        await _open_then_close_with_f4(pilot, app)
        await _switch(app, pilot, b)
        assert await _until(pilot, lambda: _surface(app) == "card"), _surface(app)


async def test_state_dismissed_is_forgotten_once_every_ask_waved_off_is_gone(enabled):
    """The dismissal is about THOSE asks: answered while away, a fresh batch opens afresh.

    This is the rule the whole feature was reworked around (a flag keyed by conversation
    hid a brand-new pair of questions behind a refusal that was about the old ones), and
    it is judged on the first frame on return — while the user is away no frame for the
    conversation is ever seen, so "I saw the queue empty" can never be observed.
    """
    old = [_ask("a1", "Which region?"), _ask("a2", "Which tier?")]
    home, a, b = _Conversation("conv-home"), _Conversation("conv-a", old), _Conversation("conv-b")
    app = _app(home)
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _switch(app, pilot, a)
        await _open_then_close_with_f4(pilot, app)
        await _switch(app, pilot, b)
        # While away: both answered, and the agent asks a fresh pair.
        a.publish([_ask("a3", "Which shard?"), _ask("a4", "Roll back?")])
        await _switch(app, pilot, a)
        assert await _until(pilot, lambda: _surface(app) == "list"), _surface(app)


async def test_state_dismissed_is_held_while_any_ask_waved_off_is_still_outstanding(enabled):
    """The other half of the same rule: one of the two answered is not "they are gone"."""
    old = [_ask("a1", "Which region?"), _ask("a2", "Which tier?")]
    home, a, b = _Conversation("conv-home"), _Conversation("conv-a", old), _Conversation("conv-b")
    app = _app(home)
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _switch(app, pilot, a)
        await _open_then_close_with_f4(pilot, app)
        await _switch(app, pilot, b)
        # a1 was answered while away; a2 is still outstanding, beside a brand-new a3.
        a.publish(
            [_ask("a2", "Which tier?"), _ask("a3", "Which shard?"), _ask("a1", status="answered")]
        )
        await _switch(app, pilot, a)
        await _pump(pilot, 30)
        assert _surface(app) == "closed", "released while an ask they waved off was still there"


# -- the same rule against the REAL wire's bound (an attached viewer) -------------------


def _long_ask(ask_id: str, *, status: str = "open", when_ms: int = _BEFORE_MS) -> dict[str, Any]:
    """An ask long enough that the wire's 6,000-character text budget drops its neighbours.

    Eight of these is the shape the desktop lane captured on the shipped wire: the frame
    carries five rows beside ``asks_open: 8``.
    """
    created = int(time.time() * 1000) + when_ms
    prose = "Which rollout should the stale-row migration take tonight, and which shard? " * 3
    return _row(
        ask_id,
        prose,
        status=status,
        delivered=status == "answered",
        created_at=created,
        expires_at=created + 3_600_000,
        questions=[
            {
                "id": "q1",
                "question": prose,
                "options": [
                    {"label": f"Option {i} " + "x" * 120, "description": "d" * 160}
                    for i in range(4)
                ],
                "multi": False,
                "recommended": None,
                "secret": False,
                "persist": False,
            }
        ],
    )


async def test_state_dismissed_is_not_wedged_by_the_sticky_truncation_flag(enabled):
    """``asks_truncated`` stays set after every ask is answered; the dismissal must not.

    Measured on the shipped wire: eight long asks, then all eight answered, still reads
    ``rows=5, asks_open=0, asks_truncated=True`` on an attached viewer, for as long as the
    dropped settled rows live. A "forget the dismissal only on a frame that is not
    truncated" rule can therefore never fire again on that conversation. The rule here is
    the tally's: ``asks_open`` counts OUTSTANDING rows before the text bound drops any,
    so a frame names everything outstanding exactly when ``asks_open`` does not exceed the
    outstanding rows it carries.

    The cell: dismiss on a truncated frame, answer everything while away, come back to a
    fresh pair — which is itself behind eight settled rows the budget keeps dropping, so
    the flag is still set. It must open. (Fails on the flag rule; passes on the tally.)
    """
    eight = [_long_ask(f"o{i}", when_ms=_BEFORE_MS - i * 1000) for i in range(8)]
    home = _Conversation("conv-home")
    a = _Conversation("conv-a", eight, wire=True)
    b = _Conversation("conv-b")
    app = _app(home)
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _switch(app, pilot, a)
        await _open_then_close_with_f4(pilot, app)
        assert a.frontend_state.asks_truncated is True, "the cell no longer exercises the bound"
        await _switch(app, pilot, b)

        settled = [_long_ask(f"o{i}", status="answered") for i in range(8)]
        fresh = [_ask("n1", "Which shard?"), _ask("n2", "Roll back?")]
        a.publish([*fresh, *settled])
        # The flag is STILL set, with nothing outstanding that the user ever saw.
        assert a.frontend_state.asks_truncated is True
        assert a.frontend_state.asks_open == 2
        await _switch(app, pilot, a)
        assert await _until(
            pilot, lambda: _surface(app) == "list"
        ), "a sticky asks_truncated wedged the dismissal: " + _surface(app)


async def test_state_dismissed_is_held_when_the_frame_hides_an_outstanding_ask(enabled):
    """A frame whose tally exceeds the rows it names proves nothing, so the refusal holds.

    The user waved off ``o0``; while away seven newer long asks arrive, so ``o0`` (the
    oldest) is exactly the row the text budget drops, and ``o0`` is still outstanding. The
    frame names five rows, none of them ``o0``, beside ``asks_open: 8``. Releasing on the
    named ids alone would open a panel over an ask the user refused — the fail-open the
    tally test exists to prevent.
    """
    oldest = _long_ask("o0", when_ms=-600_000)
    home, b = _Conversation("conv-home"), _Conversation("conv-b")
    a = _Conversation("conv-a", [oldest], wire=True)
    app = _app(home)
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _switch(app, pilot, a)
        await _open_then_close_with_f4(pilot, app)
        await _switch(app, pilot, b)

        newer = [_long_ask(f"n{i}", when_ms=-300_000 + i * 1000) for i in range(7)]
        a.publish([*newer, oldest])
        state = a.frontend_state
        named = {row.ask_id for row in state.asks or []}
        assert "o0" not in named and state.asks_open == 8, "o0 was not dropped by the bound"
        await _switch(app, pilot, a)
        await _pump(pilot, 30)
        assert _surface(app) == "closed", "released over an ask the frame could not name"


async def test_state_all_addressed_is_not_claimed_over_a_frame_that_hides_a_pending_ask(enabled):
    """Settled rows beside a positive tally are not "everything addressed".

    The wire's text budget keeps NEWER settled rows over an older timed-out one, so a frame
    can carry three answered rows and ``asks_open: 1``: the outstanding ask is the one the
    bound dropped. Reading that as settled would spend the view's one decision on a claim
    the frame itself contradicts, and the ask would never be offered even after a later
    frame names it. The view waits instead. (Fails when the rows are trusted over the
    tally: the surface never opens, because the decision was already taken.)
    """
    settled = [_ask(f"s{i}", status="answered", delivered=True) for i in range(3)]
    session = _Conversation("conv-a", settled, wire=False)
    session.publish(settled, asks_open=1)  # the tally says one ask is outstanding, unnamed
    app = _app(session)
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        assert _surface(app) == "closed"
        # Room appears (the settled rows age out) and the pending ask is now named. That it
        # then opens is the proof the frame above did not spend the view's one decision.
        session.publish([_ask("p1", "Roll back?")])
        assert await _until(pilot, lambda: _surface(app) == "card"), _surface(app)


# -- clause 5: never steal the keyboard ---------------------------------------------------


async def test_the_caret_stays_in_the_composer_and_what_is_typed_is_text(enabled):
    """The first character after an auto-open is the user's, not an answer or a command.

    ``1`` on a card would answer option one; ``d`` on a list would DECLINE the head ask,
    which is irreversible. Both must land in the composer. (The sibling cells assert the
    surface opened; this one asserts what it did NOT do with a keystroke.)
    """
    one = _Conversation("conv-a", [_ask("a1", "Which region?")])
    app = _app(one)
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        assert await _until(pilot, lambda: _surface(app) == "card"), _surface(app)
        await pilot.press("1")
        await _pump(pilot)
        assert app._editor().text == "1", f"the digit went to {app.focused!r}"
        assert one.answered == [], "a bare digit answered a question nobody had looked at"

    several = _Conversation("conv-b", [_ask("b1", "Which region?"), _ask("b2", "Which tier?")])
    app = _app(several)
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        assert await _until(pilot, lambda: _surface(app) == "list"), _surface(app)
        await pilot.press("d")
        await _pump(pilot)
        assert app._editor().text == "d", f"the letter went to {app.focused!r}"
        assert several.declined == [], "a bare letter declined a question nobody had looked at"


async def test_a_draft_is_never_interrupted_and_the_view_does_not_come_back_for_it(enabled):
    """A busy composer vetoes the open, and the veto is for the view, not for the moment.

    The incoming conversation's draft is loaded into the composer by the switch itself, so
    "is the user typing" is a fact only once the switch has finished — the policy waits for
    that (a wait, not a refusal). Having waited and found a draft, it declines FOR GOOD:
    a panel that appeared when the user next paused would be the interruption in slow
    motion.
    """
    home = _Conversation("conv-home")
    a = _Conversation("conv-a", [_ask("a1", "Which region?"), _ask("a2", "Which tier?")])
    app = _app(home)
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        parked = SessionInteraction(a)
        parked.draft.text = "half a sentence"
        app._sidebar_sources["conv-a"] = parked
        await _switch(app, pilot, a)
        await _pump(pilot, 30)
        assert app._editor().text == "half a sentence", "the draft did not survive the switch"
        assert _surface(app) == "closed", "the surface opened over a user with a draft"
        # They send it (or clear it): the view has decided, and does not come back for them.
        app._editor().load_text("")
        a._store.mutate(cumulative_parent_cost=2.5)
        await _pump(pilot, 30)
        assert _surface(app) == "closed", "the surface appeared when the user stopped typing"

        # The contrast: the SAME asks, reached the same way, with nothing typed, open. So the
        # closed result above is the draft's doing and not a feature that never fires (the
        # arm that fails on ``origin/main``, where nothing opens on its own).
        b = _Conversation("conv-b", [_ask("b1", "Which region?"), _ask("b2", "Which tier?")])
        await _switch(app, pilot, b)
        assert await _until(pilot, lambda: _surface(app) == "list"), _surface(app)


@pytest.mark.parametrize("count", [1, 2])
async def test_an_auto_open_does_not_pull_the_caret_off_what_the_user_was_reading(enabled, count):
    """A late frame opens the surface over a user who has gone to the transcript to read.

    The view begins at boot with no queue published yet (an unresolved frame, so the
    policy waits). The user clicks into the transcript; THEN the rows arrive, still inside
    the window, and the surface opens. Opening it must move focus nowhere.

    A DOOR runs ``_sync_ask_composer`` as part of the gesture (the user asked for the
    surface, so the composer handing the caret over belongs to it). The policy's own open
    must not: the same call would carry the caret from the transcript to the composer,
    which is the keyboard being moved by something nobody asked for (clause 5). The
    sibling cells open the surface from a composer that already holds the caret, where
    that call is a no-op, so only a caret that starts ELSEWHERE can tell the two apart.
    """
    rows = [_ask(f"a{i}", f"Question {i}?") for i in range(1, count + 1)]
    kind = "card" if count == 1 else "list"
    session = _Conversation("conv-a")
    app = _app(session)
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        assert _surface(app) == "closed"
        view = app.query_one(TranscriptView)
        view.focus()
        await _pump(pilot)
        assert app.focused is view
        session.publish(rows)
        assert await _until(pilot, lambda: _surface(app) == kind), _surface(app)
        assert app.focused is view, f"the auto-open moved the caret to {app.focused!r}"


@pytest.mark.parametrize("count", [1, 2])
async def test_a_surface_that_opened_on_its_own_is_not_a_hold_on_the_keyboard(enabled, count):
    """It owns no key until the user hands it one, so the composer is never refused.

    ``_focus_is_claimed`` is what every "put me back in the input" gesture asks; counting
    an un-engaged surface as a claim would make each of them a no-op over a panel nobody
    opened. The door-opened surface IS a claim (it took the caret to be answered on), and
    that contrast is the control. One ask is a card, several are a list: the two take
    different branches of the claim, so each has its own arm.
    """
    rows = [_ask(f"a{i}", f"Question {i}?") for i in range(1, count + 1)]
    kind = "card" if count == 1 else "list"
    session = _Conversation("conv-a", rows)
    app = _app(session)
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        assert await _until(pilot, lambda: _surface(app) == kind), _surface(app)
        assert app._focus_is_claimed() is False
        await pilot.press(ASK_TOGGLE_KEY)  # close
        assert await _until(pilot, lambda: _surface(app) == "closed")
        await pilot.press(ASK_TOGGLE_KEY)  # the door, pressed: now the user asked for it
        assert await _until(pilot, lambda: _surface(app) == kind)
        assert app._focus_is_claimed() is True, "a door-opened surface must still own the keyboard"
        assert isinstance(app.focused, AskPickerScreen if count == 1 else AskQueueList)


@pytest.mark.parametrize("count", [1, 2])
async def test_a_passive_surface_that_holds_the_caret_anyway_is_still_a_claim(enabled, count):
    """Passive means "un-engaged AND not holding the caret" — both halves, not either.

    A user can land on the surface without the handover (Shift+Tab from the composer, or
    Textual lending it focus), and a composer-focus gesture must not then take the caret
    off a question they are looking at. Judging passivity on the flag alone would make that
    a no-op's opposite: the claim would vanish exactly while the keyboard is on it.
    """
    rows = [_ask(f"a{i}", f"Question {i}?") for i in range(1, count + 1)]
    kind = "card" if count == 1 else "list"
    app = _app(_Conversation("conv-a", rows))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        assert await _until(pilot, lambda: _surface(app) == kind), _surface(app)
        surface = (_cards(app) if count == 1 else list(app.query(AskQueueList)))[0]
        assert app._focus_is_claimed() is False
        surface.focus()  # focus without the engage gesture
        await _pump(pilot)
        assert app.focused is surface
        assert surface.passive is False, "a surface holding the caret was still called passive"
        assert app._focus_is_claimed() is True


async def test_an_approval_answered_over_the_auto_opened_card_hands_the_caret_to_the_composer(
    enabled,
):
    """Found by probe, not by argument: the card the user never touched must not inherit focus.

    Removing an approval makes Textual hand focus to the next focusable node, which here is
    the passive card; the app then moves it on to "the surviving prompt", and a card that
    treated receiving focus as the user engaging with it had already stopped being passive,
    so the next bare ``1`` ANSWERED a question nobody had looked at. Before the fix the
    probe read ``focus=AskPickerScreen`` and ``answered=1``.
    """
    one = _OwnerConversation("conv-a", [_ask("a1", "Which region?")])
    app = _app(one)
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        assert await _until(pilot, lambda: _surface(app) == "card"), _surface(app)
        assert await _until(pilot, lambda: getattr(one, "approval_handler", None) is not None)
        handler = getattr(one, "approval_handler", None)
        assert handler is not None, "the app never installed its approval gate"
        app._set_approve_all(False)
        app._approvals_default_auto = False
        approving = asyncio.ensure_future(handler("bash", "run: make"))
        await _pump(pilot, 20)
        await pilot.press("y")
        assert await asyncio.wait_for(approving, 2) is True
        await _pump(pilot, 20)
        assert isinstance(app.focused, Editor), f"the caret went to {app.focused!r}"
        await pilot.press("1")
        await _pump(pilot)
        assert one.answered == [], "a bare digit answered a question nobody had looked at"
        assert app._editor().text == "1"


# -- the user can always get in, and always get out ----------------------------------------


@pytest.mark.parametrize("count", [1, 2])
async def test_tab_hands_the_caret_to_the_surface_and_it_then_behaves_as_any_door_opened_one(
    enabled, count
):
    """Tab from the composer is the keyboard's way in; without it the surface is mouse-only.

    One ask opens its card, several open the list (the list is not a ``_live_prompt``, so
    the router's Tab branch never saw it). After the handover the card/list owns the
    caret and the keys the passive one refused.
    """
    rows = [_ask(f"a{i}", f"Question {i}?") for i in range(1, count + 1)]
    session = _Conversation("conv-a", rows)
    app = _app(session)
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        kind = "card" if count == 1 else "list"
        assert await _until(pilot, lambda: _surface(app) == kind), _surface(app)
        assert isinstance(app.focused, Editor)
        await pilot.press("tab")
        await _pump(pilot)
        surface = (_cards(app) if count == 1 else list(app.query(AskQueueList)))[0]
        assert app.focused is surface, f"Tab left the caret on {app.focused!r}"
        assert surface.passive is False
        assert app._focus_is_claimed() is True
        assert app._editor().text == "", "Tab typed whitespace into the composer"
        # Engagement is a gesture, not a focus event: with the caret back in the composer
        # the surface is still the one the user chose, exactly as a door-opened one is.
        app._editor().focus()
        await _pump(pilot)
        assert isinstance(app.focused, Editor)
        assert surface.passive is False, "taking the caret away undid the user's engagement"


@pytest.mark.parametrize("count", [1, 2])
async def test_a_click_on_the_surface_engages_it(enabled, count):
    """The pointer's way in, beside Tab's: a click on the surface is the user choosing it.

    The click lands on the TITLE / HEADER line, never a row, because a click on a card's row
    answers it and on a list's row opens that ask: this cell is about the engagement, not
    about what a row does. Like Tab, it must outlive the caret — the surface the user
    clicked stays the one they chose after they click back into the composer.
    """
    rows = [_ask(f"a{i}", f"Question {i}?") for i in range(1, count + 1)]
    kind = "card" if count == 1 else "list"
    app = _app(_Conversation("conv-a", rows))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        assert await _until(pilot, lambda: _surface(app) == kind), _surface(app)
        surface = (_cards(app) if count == 1 else list(app.query(AskQueueList)))[0]
        assert surface.passive is True
        await pilot.click(type(surface), offset=(2, 0))
        await _pump(pilot)
        assert app.focused is surface, f"the click left the caret on {app.focused!r}"
        assert surface.passive is False
        app._editor().focus()
        await _pump(pilot)
        assert surface.passive is False, "clicking back into the composer undid the engagement"


async def test_a_passive_list_paints_what_the_keyboard_can_do_and_nothing_it_cannot(enabled):
    """No ``❯`` and no ``enter answer`` over a caret that is in the composer; both after Tab.

    A list that looked live over a dead keyboard is the dead end UX round 1 (U2) removed
    from the card. The header names the one key that works (Tab) and the frame does not
    shift when the list takes the caret (the marker column is kept, blank).
    """
    session = _Conversation("conv-a", [_ask("a1", "Which region?"), _ask("a2", "Which tier?")])
    app = _app(session)
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        assert await _until(pilot, lambda: _surface(app) == "list"), _surface(app)
        listing = app.query_one(AskQueueList)
        passive = listing.render().plain
        assert "❯" not in passive, "a row is marked selected over a keyboard that is elsewhere"
        assert f"{TAB_HINT_KEY} answer here" in passive
        # Every key the list's own header could name, none of which can reach it from here.
        assert "enter answer" not in passive and "d decline" not in passive
        await pilot.press("tab")
        await _pump(pilot)
        live = listing.render().plain
        assert "❯" in live
        assert f"{TAB_HINT_KEY} answer here" not in live
        # `enter answer` leads the shed ladder since U3/D4, so at 100 columns the
        # hint the user just earned by pressing Tab is the one that survives.
        assert "enter answer" in live, live

        def column(text: str, question: str) -> int:
            return next(line for line in text.splitlines() if question in line).index(question)

        # The marker column is kept (blank) while passive, so taking the caret moves no row.
        for question in ("Which region?", "Which tier?"):
            assert column(passive, question) == column(live, question), question


async def test_a_passive_card_paints_what_the_keyboard_can_do_and_nothing_it_cannot(enabled):
    """The card mirror of the list's cell (D1/U1): no caret, no accent, no claim.

    The auto-opened card was for one round visually identical to an engaged one —
    caret, tint band and the accent-green label, all claiming "Enter takes this" —
    while ``answer_keys()`` was empty and the obvious keys went to the composer.
    UX walked it: ``1`` then Enter posted an unintended chat message. The ink is
    gated on passivity exactly as the list's is; this cell replays the walk.
    """
    session = _Conversation("conv-a", [_ask("a1", "Which region?")])
    app = _app(session)
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        assert await _until(pilot, lambda: _surface(app) == "card"), _surface(app)
        card = _cards(app)[0]
        assert card.passive is True

        from rich.color import Color

        from local_operator.tui import theme as theme_mod

        def ink_at(text, needle):
            start = text.plain.index(needle)
            for span in text.spans:
                if span.start <= start < span.end:
                    return span.style.color if span.style else None
            return None

        label = card.question.options[0].label
        # ``semantic_color`` hands back the hex; a span's style carries rich's parsed
        # ``Color``, and the two only compare equal through a common parse.
        accent = Color.parse(theme_mod.semantic_color("accent"))
        passive_lines = card.render_lines_for_test()
        assert all(
            "❯" not in line for line in passive_lines
        ), "the card marks a row while the keyboard is elsewhere"
        assert (
            ink_at(card._card_text(), label) != accent
        ), "the accent claims what Enter will take on a card that owns no key"
        assert card._row_ground(0) == card._row_ground(
            1
        ), "the selected row keeps the selection fill while no key can reach it"

        # UX's walk, replayed: the obvious keys go to the composer — where the
        # footer says the caret is — and no keystroke settles the ask.
        await pilot.press("1")
        await _pump(pilot)
        assert app._editor().text == "1", "the digit did not land in the composer"
        assert session.answered == [], "a digit answered a question nobody engaged"
        await pilot.press("enter")
        await _pump(pilot)
        assert session.prompts == ["1"], "Enter did not post the composer's text"
        assert _surface(app) == "card", "a keystroke the card refused settled the ask"

        # Engaged (the Tab handover), every cue comes back.
        await pilot.press("tab")
        await _pump(pilot)
        assert card.passive is False
        live_lines = card.render_lines_for_test()
        assert any("❯" in line for line in live_lines), "the caret never came back"
        assert ink_at(card._card_text(), label) == accent
        assert card._row_ground(0) != card._row_ground(1)


async def test_the_first_hint_outbids_the_drawer_clause_at_80_columns(enabled):
    """U3: at the repo's own default size the header must still name a key.

    At 80 columns the drawer sentence, the filter chips and any hint cannot fit
    together. The ladder spent the hints first, so the auto-opened surface named
    no key at all — the one thing this feature exists to teach. The clause now
    yields to the FIRST hint the way it already yields to the numbers: ``⇥``
    while passive, ``enter answer`` after Tab.
    """
    session = _Conversation("conv-a", [_ask("a1", "Which region?"), _ask("a2", "Which tier?")])
    app = _app(session)
    async with app.run_test(size=(80, 24)) as pilot:
        await _boot(pilot, app)
        assert await _until(pilot, lambda: _surface(app) == "list"), _surface(app)
        listing = app.query_one(AskQueueList)
        passive_header = listing.header_text(80)
        assert f"{TAB_HINT_KEY} answer here" in passive_header, passive_header
        await pilot.press("tab")
        await _pump(pilot)
        live_header = listing.header_text(80)
        assert "enter answer" in live_header, live_header
        assert (
            "questions waiting" not in live_header
        ), "the drawer clause should have yielded to the hint, not the other way"


async def test_the_surface_that_opened_on_its_own_can_always_be_closed_and_reopened(enabled):
    """Never a trap, and the door keeps its meaning (clause 6).

    Esc from the composer closes it (and does not stop a turn: the ask surface owns Esc
    first). ``f4`` is a TOGGLE, so pressed over the auto-opened surface it closes it; pressed
    again — now a user's own gesture — it opens with the caret on the card, which is what
    ``f4`` has always done.
    """
    session = _Conversation("conv-a", [_ask("a1", "Which region?")])
    app = _app(session)
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        assert await _until(pilot, lambda: _surface(app) == "card"), _surface(app)
        await pilot.press("escape")
        assert await _until(pilot, lambda: _surface(app) == "closed"), _surface(app)
        assert isinstance(app.focused, Editor)
        await pilot.press(ASK_TOGGLE_KEY)
        assert await _until(pilot, lambda: _surface(app) == "card"), _surface(app)
        assert isinstance(app.focused, AskPickerScreen), "the door did not take the caret"


@pytest.mark.parametrize("count", [1, 2])
async def test_closing_a_surface_nobody_engaged_does_not_move_the_caret(enabled, count):
    """It took no focus on the way up, so it gives none back on the way down.

    A user who clicked into the transcript to read, then pressed Esc to dismiss a panel they
    never touched, would otherwise be yanked into the composer. A card and a list give the
    caret back by different routes (the card restores the target it recorded at mount, the
    list never recorded one), so each has its own arm.
    """
    rows = [_ask(f"a{i}", f"Question {i}?") for i in range(1, count + 1)]
    kind = "card" if count == 1 else "list"
    app = _app(_Conversation("conv-a", rows))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        assert await _until(pilot, lambda: _surface(app) == kind), _surface(app)
        view = app.query_one(TranscriptView)
        view.focus()
        await _pump(pilot)
        assert app.focused is view
        await pilot.press("escape")
        assert await _until(pilot, lambda: _surface(app) == "closed"), _surface(app)
        assert app.focused is view, f"closing moved the caret to {app.focused!r}"
