"""The queued-ask surfaces: the minimized bar and the open-ask list.

Design: ``docs/design/ask-nonblocking.md`` §5.0 (R7) and §5.1 (PR B).

Two widgets for one feature, and the split is the interaction model rather than
a filing convention:

* :class:`AskBar` — the **MINIMIZED** state: one line above the composer that
  says how many asks are waiting and expands on a click or
  :data:`ASK_TOGGLE_KEY`. It is the
  default presentation of a queued ask, so it must exist while the answer
  surface is NOT mounted: it is a child of the composer's own panel, never of
  ``#prompt-host`` (which reserves rows only while a card is in it).
* :class:`AskQueueList` — the **list** the bar opens when more than one ask is
  open. One row per ask (status glyph, first question, expiry), a click or
  Enter mounts that ask's picker. It is mounted in ``#prompt-host`` beside the
  picker, because both are the *expanded* state and only one is ever up.

WHY THE BAR IS NOT THE NOTICE IT REPLACES. A queued ask used to surface as a
one-line transcript notice. A notice scrolls away, is not clickable, and
cannot say how many questions are behind it — and R7 needs exactly those three
things: persistent, clickable, countable. One affordance, not two: the app
replaces the notice with this bar rather than painting both.

COLOUR IS PERSISTENT, NEVER ANIMATED. §5.0 is explicit: the accent is a fixed
colour on the glyph. A pulse would be the focus-steal this design exists to
avoid, expressed in colour instead of keys, so nothing here owns a timer.

INK IS CHOSEN AGAINST THE GROUND IT REALLY SITS ON, and that is a review
outcome rather than a preference (design round 1, D2/D3). The first cut painted
the bar's own message in ``dim`` on ``raised`` (3.81:1 dark, 3.09:1 LIGHT —
under the palette gate's 3.4:1 dim floor) and the list's rows in ``dim`` on
``overlay`` (3.43:1 dark, 2.72:1 light). The gate checks ``bg``/``surface``
only, which is exactly how both slipped through: a new surface has to solve its
own pairs. So the bar's ground is the composer's ``surface`` (the gate's own
ground, where ``fg`` 13.76/13.86, ``muted`` 7.93/6.58 and ``dim`` 4.18/3.46 all
clear their floors, and the accent glyph reads 8.02/4.43) and the list's rows
avoid ``dim`` entirely on ``overlay`` (``fg`` 11.30/10.92, ``muted`` 6.51/5.18).

THE STATE HUES ON THAT PANEL ARE THE DERIVED ``chip-*`` INKS (design round 2,
D1). Round 1 solved ``fg``/``muted`` against ``overlay`` and stopped there: the
marker was still ``accent`` (3.49:1 on the light ramp's overlay), the delivering
glyph and the ``answered`` chip word still ``success`` (3.45:1), the urgency
word still ``warning`` — all under this repo's 4.0 state floor, all now under a
gate pair of their own (``tests/unit/tui/test_palette_contrast.py``).
``theme._fill_chip_live`` already derives exactly this: the hue when it clears
the panel's ground, the ramp's neutral ink when it does not, so every ramp keeps
its own palette and the dark ramp is byte-identical (there ``chip-live`` IS the
accent).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable, Mapping, Sequence

from rich.cells import cell_len
from rich.style import Style
from rich.text import Text
from textual import events
from textual.binding import Binding
from textual.message import Message
from textual.widget import Widget

from local_operator.asks import store

# i18n: the ask bar's and list header's counts are user-facing copy, extracted
# so a locale can carry them. `_t` is bound BELOW the whole import block (a
# binding between imports makes every later import read as not-at-top, E402).
from local_operator.i18n.keys import tui_asks as _asks
from local_operator.i18n.keys import tui_small as _x
from local_operator.i18n.text import tr
from local_operator.tui import theme as theme_mod
from local_operator.tui.widgets.ask_picker import TAB_HINT_KEY

_t = tr("tui.small")

#: The dismiss door's OWN namespace (R5, SA-D7). A new namespace rather than keys
#: added to ``tui.small``: a new key in a namespace a locale has not carried yet
#: GROWS the shrink-only coverage allowance (docs/I18N.md), while a complete new
#: namespace needs no allowance at all.
_a = tr("tui.asks")

#: The one glyph the ask surfaces share. ``?`` is the question mark the
#: composer chip vocabulary already leaves free (``!`` is the sidebar's
#: needs-you gate mark, ``❯`` the composer caret, ``$`` bang-mode), and using
#: ONE glyph for the bar and the sidebar mark is what lets a user connect the
#: two without a legend.
ASK_MARKER = "?"

#: The chevron at the bar's right edge. Two glyphs, one meaning each: the
#: direction the click will move the surface, exactly like the transcript's
#: expand affordance.
#: The key that reaches the ask surface without a mouse, named ONCE so the bar's
#: copy cannot advertise a key the app does not bind.
#:
#: A FUNCTION key, continuing the app's f8/f9/f10 row, because the composer owns
#: the letters: ``TextArea`` binds ctrl+a/e/w/d/x/k/u/z/y/c/v and, measured in
#: Textual 8.2.8, ``f6``/``f7`` as well — so f7, the obvious neighbour, is
#: swallowed by the focused composer and never reaches an app binding (found by
#: driving the key in a pilot, round 1 remediation). f8/f9/f10 are the app's.
#: f5 is avoided too: ``editor.py``'s note claims TextArea binds f5 as a
#: selection chord, and a dispute about one key is not worth winning.
ASK_TOGGLE_KEY = "f4"

ASK_BAR_CHEVRON_COLLAPSED = "⌄"
ASK_BAR_CHEVRON_EXPANDED = "⌃"

#: The statuses a row can carry, re-exported from ``asks/store.py``'s fold rather
#: than spelled here. The bar, the list and the transcript card used to keep
#: their OWN literals "spelled once" in this module, which is exactly how a
#: surface ends up disagreeing with the wire about what ``timed_out`` means:
#: the only spelling that is allowed to be authoritative is the fold's.
STATUS_OPEN = store.STATUS_OPEN
STATUS_TIMED_OUT = store.STATUS_TIMED_OUT
STATUS_LATE = store.STATUS_LATE
STATUS_ANSWERED = store.STATUS_ANSWERED
STATUS_DECLINED = store.STATUS_DECLINED
STATUS_DISMISSED = store.STATUS_DISMISSED
STATUS_WITHDRAWN = store.STATUS_WITHDRAWN
STATUS_EXPIRED = store.STATUS_EXPIRED

#: THE TERMINAL STATUSES — an ask in one of these has nothing left to answer.
#: ``answered`` and ``declined`` are settled; ``late`` joins them because it is
#: the same settlement one deadline later — the agent was told, the receipt is in
#: the transcript, and offering an answer box for it can only be refused (design
#: round 1, D6/U3: ``late`` was painted "timed out — still answerable" and stayed
#: in the answerable set). ``withdrawn`` joins them because the ASKER retracted
#: the question (design §12) — nobody is waiting on it, and a stale tap can only
#: be refused. ``expired`` injects nothing at all (§2.2). ``timed_out`` joins
#: them on this surface (decision D10/D11): its deadline fired and the agent went
#: ahead without an answer, so no surface offers one any more — the store no
#: longer counts it outstanding (#2388) but still takes a late answer through
#: chat, and the list reads it the way it reads a declined ask, one settled line with a status word.
#:
#: A SURFACE NO LONGER DROPS THEM (design §4): the list keeps every row and the
#: three-way filter picks the half, so what this set is FOR now is the question a
#: caller asks about ONE row — is this over? — while the row-level answer the list
#: reads is :attr:`AskRow.settled` (which also covers the DELIVERING half: an
#: answered-but-undelivered row is terminal as an ASK but still pending on the
#: user's attention).
SETTLED_STATUSES = frozenset(
    {
        STATUS_ANSWERED,
        STATUS_LATE,
        STATUS_DECLINED,
        STATUS_DISMISSED,
        STATUS_WITHDRAWN,
        STATUS_EXPIRED,
        STATUS_TIMED_OUT,
    }
)

STATUS_DELIVERING = frozenset({STATUS_ANSWERED, STATUS_LATE})

# ---------------------------------------------------------------------------
# The three-way filter (design §4, amendment A5)
# ---------------------------------------------------------------------------

FILTER_ALL = "all"
FILTER_OUTSTANDING = "outstanding"
FILTER_SETTLED = "settled"

#: Press order, left to right, and the cycle order for `[`/`]`.
FILTER_ORDER: tuple[str, ...] = (FILTER_ALL, FILTER_OUTSTANDING, FILTER_SETTLED)

#: EXACT labels, shared with the desktop's own filter control (``ask-panel.tsx``).
#: The middle one is "Pending", the code's own word for that half (``pending`` =
#: answerable or delivering). It used to be "Waiting or moved on" because a
#: timed-out ask sat in it; since D11 every timed-out ask is Settled, so the old
#: label named a population the half can no longer hold (#2481 design D1).
FILTER_LABELS: dict[str, str] = {
    FILTER_ALL: "All",
    FILTER_OUTSTANDING: "Pending",
    FILTER_SETTLED: "Settled",
}

#: Digit keys select a half outright; `[`/`]` cycle. None of these collide with
#: the list's existing bindings (escape, enter, up/down, k/j, d, x).
FILTER_KEYS: dict[str, str] = {
    "1": FILTER_ALL,
    "2": FILTER_OUTSTANDING,
    "3": FILTER_SETTLED,
}

#: The three empty sentences, copied VERBATIM from the desktop panel so the two
#: surfaces cannot drift: the whole queue is empty, the middle half is empty,
#: the settled half is empty. A filtered view must never read as an empty queue
#: (design round 1, D1 = UX U2), so each half states its OWN emptiness and names
#: the half that is holding the rows.
EMPTY_ALL = "No asks outstanding. The agent is not waiting on anything."
EMPTY_OUTSTANDING = "No asks are pending. They have all settled — see Settled."
EMPTY_SETTLED = "No asks have settled yet. They are all still pending."

#: The scope the list is reading: this conversation's queue, or every
#: conversation's. The TUI has no session-less screen, so the FLEET scope is
#: reachable only from a fleet surface (the sidebar's swarm note) — the door
#: decides the scope, and the list only paints the queue its door promised.
SCOPE_SESSION = "session"
SCOPE_FLEET = "fleet"
SCOPE_SUBJECTS: dict[str, str] = {
    SCOPE_SESSION: "This conversation",
    SCOPE_FLEET: "All conversations",
}

#: Settled rows wear a WORD rather than a glyph (design §4): the status is the
#: row's whole content once there is nothing left to act on, so it is spelled
#: and inked by severity. `answered` succeeded, `late` is the same answer one
#: deadline late (warning), and declined/dismissed/withdrawn/expired are the
#: quiet end of the scale — an expiry is not a failure (§5's copy contract),
#: and a withdrawal is not a refusal (the asker retracted its own question;
#: design §12 gives it the same muted register as `dismissed`).
SETTLED_CHIPS: dict[str, tuple[str, str]] = {
    STATUS_ANSWERED: ("answered", "success"),
    STATUS_LATE: ("late", "warning"),
    STATUS_DECLINED: ("declined", "muted"),
    STATUS_DISMISSED: ("dismissed", "muted"),
    STATUS_WITHDRAWN: ("withdrawn", "muted"),
    STATUS_EXPIRED: ("expired", "muted"),
    # Muted like `declined` (D11): a lapse is the quiet end of the scale — the
    # agent carried on, nothing failed. "Timed out" is Ben's word for it and the
    # bar's own register already uses it.
    STATUS_TIMED_OUT: ("timed out", "muted"),
}
#: ...which is to say: everything the reader must not be asked to answer. THE
#: OUTSTANDING SET, taken from the fold rather than re-listed here, MINUS the
#: lapsed set (D10): "answerable now" is outstanding AND NOT lapsed, and both
#: halves live in ``asks.store`` so no surface re-lists the statuses. Since
#: #2388 the two sets are disjoint, so this equals the outstanding set; the
#: subtraction stays so the offer never widens if they overlap again.
_ANSWERABLE = store.OUTSTANDING_STATUSES - store.LAPSED_STATUSES

#: THE DELIVERING HALF's word (§10): answered, response row not yet durable. It
#: belongs to the PENDING side — the user's queue is not finished until the agent
#: has been told — but it is not answerable, so it reads as a status word like a
#: settled one and takes the same `·` separator (round 2: F3/U4). Two spellings
#: because `late` is a different fact from `answered` and §5's copy contract
#: names both.
_DELIVERING_CHIPS: dict[str, str] = {
    STATUS_ANSWERED: "answered, delivering",
    STATUS_LATE: "answered late, delivering",
}

#: How many cells a fleet row's conversation handle may take. The handle is a
#: catalogue title — a sentence — and a single line has a question to show.
SESSION_HANDLE_CELLS = 24

#: Closed-set glyphs for the list's status column. Deliberately NOT a spinner
#: and not an animated set: a queued ask is not doing anything, it is waiting.
#: ``●`` open. A SETTLED row — a lapsed one included (D11) — paints no glyph and
#: no fallback bit of punctuation either: its status is a WORD in the tail
#: (``settled_chip``), because once there is nothing left to act on the status
#: IS the row.
STATUS_MARKS: dict[str, str] = {
    STATUS_OPEN: "●",
}

#: The glyph an ANSWERED-but-undelivered row wears in the list's status column
#: (design §10's delivering half): the answer is recorded and the response has
#: not reached the transcript yet, so the row is still PENDING — it is not
#: something to answer, but it is the reason the pending count is honest.
DELIVERING_MARK = "▸"

#: The glyph a row wears while an op for it is IN FLIGHT through the fleet
#: worker (fleet scope only — the current session's ops are synchronous). Sized
#: for the DIAL's own window (engage 30 s + ack 15 s), which a dismissal no longer
#: takes (it is one append against the log, fork A §14) — the mark is about the row
#: being spoken for, not about a clock. It is not a spinner: nothing repaints on a
#: clock here, and a row that looked busy would be the animated affordance §5.0
#: forbids. It exists so the row cannot be fired twice.
IN_FLIGHT_MARK = "…"


#: The dismiss chooser's two reasons, in the order the chooser offers them, each
#: with the keys that NAME it and EXPLAIN it. The first element is the queue's
#: own vocabulary (``asks/store.DISMISS_REASONS``), never a translated word — the
#: copy is this surface's, the recorded reason is the log's, and letting the
#: label decide what gets written is how a locale change becomes a data change.
#: Functions rather than resolved strings: this table is built at import, and a
#: module constant would freeze the locale it was imported under.
DISMISS_CHOICES: tuple[tuple[str, Any, Any], ...] = (
    (store.DISMISS_REASON_DEFAULT, _asks.dismiss_moot, _asks.dismiss_moot_note),
    ("answered_in_chat", _asks.dismiss_answered_in_chat, _asks.dismiss_answered_in_chat_note),
)

#: The cells the context line and the chooser rows are INDENTED by: the width of
#: the row's own two leading columns (``❯ ``) plus the status glyph cell, so a
#: sub-line reads as belonging to the ask above it rather than as another ask.
SUB_LINE_INDENT = 4

#: The connector a context line leads with (design round 1, D6). The same indent
#: is where a LAPSED ask's collapsed row sits, and that row carries no marker of
#: its own, so without a cue the slot read as two different things at once ("a
#: child line of the row above" and "another ask"). ``└`` is this surface's
#: existing parent/child glyph — the sidebar's tree sections and the analytics
#: panel both use it — and no ask row is ever painted with it.
CONTEXT_CUE = "└ "


@dataclass(frozen=True)
class AskRow:
    """One wire ask, flattened to what a surface actually draws.

    A view-model rather than the wire model on purpose: the app receives
    ``PendingAskState`` objects on the frontend snapshot, the capture scripts
    and tests feed plain dicts, and the list must render both without a second
    code path per source. Reading happens once, here, so a missing key is a
    default rather than an ``AttributeError`` inside a paint.
    """

    ask_id: str
    status: str
    created_at: int
    expires_at: int
    urgent: bool
    questions: tuple[Mapping[str, Any], ...]
    #: Whether the transcript already carries this ask's response/deadline row.
    #: Carried from the wire (``store.fold``'s ``delivered``) so the DELIVERING
    #: half §10 describes is expressible: an ``answered`` row whose response has
    #: not been delivered is still pending on the user's attention, and one whose
    #: row exists has settled.
    delivered: bool = False
    #: The session this row belongs to, when the list is reading the FLEET scope
    #: (``store.index_asks`` adds it). Empty in session scope, and empty is the
    #: honest answer there: the row belongs to whatever conversation is on screen.
    session_id: str = ""
    #: The row's own working directory, fleet scope only — what the scope line
    #: names when two conversations share a title.
    cwd: str = ""
    #: SA-D7's card context, straight off the wire (`asks/store.pending_row`):
    #: operator messages admitted after this ask was raised, and when the latest
    #: one was (epoch ms). ``None`` means the writing runtime could not count them
    #: — an older runtime, an ask older than its retained window — and ABSENCE IS
    #: THE CONTRACT: no line is drawn rather than a zero the surface would have to
    #: disbelieve. A zero is a fact (the ask is newer than every message) and draws
    #: no line either, because there is nothing stale to report.
    user_messages_since: int | None = None
    last_user_message_at: int | None = None

    @property
    def question_count(self) -> int:
        return len(self.questions)

    @property
    def context_stamp(self) -> tuple[int, int] | None:
        """``(count, last_ms)`` for an OPEN row with something to say, else ``None``.

        Open only: a settled ask has no staleness to report, and the line is the
        one thing this surface spends a row on that is not the ask itself. Both
        halves are required — the writer sends the pair together or not at all
        (`AskQueue._card_context`), so a count without a clock is a payload this
        surface has no sentence for.
        """
        if self.status != STATUS_OPEN or not self.user_messages_since:
            return None
        if not self.last_user_message_at:
            return None
        return int(self.user_messages_since), int(self.last_user_message_at)

    @property
    def head_question(self) -> str:
        """The first question's own text, or "" when the row carries none.

        Empty rather than a placeholder: the list falls back to the ask id in
        its own copy, because a surface that invented a question would be
        telling the reader something the model never said.
        """
        for question in self.questions:
            text = str(question.get("question") or "").strip()
            if text:
                return text
        return ""

    @property
    def answerable(self) -> bool:
        """Whether this surface OFFERS an answer for the row (open, not lapsed)."""
        return self.status in _ANSWERABLE

    @property
    def lapsed(self) -> bool:
        """Whether the row's deadline fired with no answer (``store.is_lapsed``).

        No longer outstanding to the store (#2388), though it still takes a late
        answer, and read-only here (D10): it sits in the Settled half behind its
        ``timed out`` chip, and the only gesture it takes is ``x`` dismiss — the
        one op the queue accepts on a timed-out ask, which answers nobody.
        """
        return store.is_lapsed(self.status)

    @property
    def delivering(self) -> bool:
        """Whether an answer is recorded and its response has not landed yet.

        §10's in-flight half: ``answered``/``late`` with ``delivered`` false.
        It belongs to the PENDING side of the filter (the desktop draws it as a
        pending card), because the user's queue is not finished until the agent
        has been told.
        """
        return self.status in STATUS_DELIVERING and not self.delivered

    @property
    def pending(self) -> bool:
        """The middle half of the filter: answerable OR still delivering."""
        return self.answerable or self.delivering

    @property
    def settled(self) -> bool:
        """The last half: nothing left for the user to do or to wait for."""
        return not self.pending

    @property
    def waiting(self) -> bool:
        """Whether anyone is still WAITING for this row's answer.

        The same population as :attr:`answerable` since D10 made a lapsed row
        read-only; kept as its own name because the copy asks the question in
        these words.
        """
        return self.status == STATUS_OPEN


def _value(row: Any, key: str) -> Any:
    """Read one field from a wire model OR a plain mapping."""
    if isinstance(row, Mapping):
        return row.get(key)
    return getattr(row, key, None)


def _optional_int(value: Any) -> int | None:
    """A count/epoch an older runtime omitted, kept ABSENT rather than zeroed.

    ``None`` is the wire's own spelling for "this writer could not count it"
    (``asks/store.pending_row`` only adds the key when it has one), so coercing
    it to ``0`` here would turn "unknown" into "nothing has happened since" and
    the surface would state the opposite of the truth.
    """
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def ask_rows(rows: Iterable[Any] | None) -> list[AskRow]:
    """Flatten the wire's asks into the surface view-model, settled rows included.

    ``None`` and ``[]`` both mean "no queued asks" — the wire is ABSENT, not
    empty, while this runtime runs the BLOCKING arm (the kill switch, or an
    older build), so a caller that mistook absence for a list would be
    rendering a feature the runtime does not have.

    SETTLED ROWS ARE KEPT (design §4): the list is the ONE expanded surface and
    it shows the whole queue — the filter, not this function, decides which half
    is on screen. Dropping them here is what made "no filter, no history"
    structural, and it is also what made the drawer's per-half counts impossible
    to state: a count of what was thrown away is not a count a surface can make.

    It is the CALLER's job to keep reading the ANSWERABLE set where the user
    owes something (``AskRow.answerable`` / ``pending``) — the bar, the sidebar
    mark and ``_open_ask_rows`` all do, and settling a row must never inflate a
    count of what is still owed.
    """
    out: list[AskRow] = []
    for row in rows or ():
        status = str(_value(row, "status") or STATUS_OPEN)
        questions = _value(row, "questions") or ()
        out.append(
            AskRow(
                ask_id=str(_value(row, "ask_id") or ""),
                status=status,
                created_at=int(_value(row, "created_at") or 0),
                expires_at=int(_value(row, "expires_at") or 0),
                urgent=bool(_value(row, "urgent")),
                questions=tuple(q for q in questions if isinstance(q, Mapping)),
                delivered=bool(_value(row, "delivered")),
                session_id=str(_value(row, "session_id") or ""),
                cwd=str(_value(row, "cwd") or ""),
                user_messages_since=_optional_int(_value(row, "user_messages_since")),
                last_user_message_at=_optional_int(_value(row, "last_user_message_at")),
            )
        )
    return out


def filter_counts(rows: Sequence[AskRow]) -> dict[str, int]:
    """``{filter value: count}`` for the three segments, over ``rows``.

    The halves PARTITION the rows by construction (``pending`` is the negation
    of ``settled``), so the segments can never sum to less than ``All`` — which
    is what makes an empty half a true statement about the other one rather
    than a gap in the arithmetic.
    """
    pending = sum(1 for row in rows if row.pending)
    return {FILTER_ALL: len(rows), FILTER_OUTSTANDING: pending, FILTER_SETTLED: len(rows) - pending}


def filter_label(value: str, count: int) -> str:
    """One segment's painted text: the exact label and this half's live count."""
    return f"{FILTER_LABELS[value]} · {count}"


def empty_sentence(rows: Sequence[AskRow], filter_value: str) -> str | None:
    """The sentence a filtered view owes when it would otherwise be blank.

    ``None`` means the view has rows to draw. A filtered view must never read
    as an empty queue (design round 1, D1 = UX U2): each half names itself and
    the half holding the rows, so "nothing here" is always paired with "they
    are over there".
    """
    if not rows:
        return EMPTY_ALL
    if filter_value == FILTER_OUTSTANDING and not any(row.pending for row in rows):
        return EMPTY_OUTSTANDING
    if filter_value == FILTER_SETTLED and all(row.pending for row in rows):
        return EMPTY_SETTLED
    return None


def settled_chip(row: AskRow) -> tuple[str, str]:
    """``(word, ink)`` for a settled row's status chip (design §4).

    An unknown status falls back to its own raw spelling in the muted ink — a
    status this surface cannot name is still a fact about the row, and inventing
    a friendlier word for it would be the surface lying about the fold.
    """
    return SETTLED_CHIPS.get(row.status, (row.status or "settled", "muted"))


def delivering_chip(row: AskRow) -> tuple[str, str] | None:
    """``(word, ink)`` for the DELIVERING half, or ``None`` when not delivering.

    §10's second state, and the one the first cut of this header had no words
    for: an ask the user has ANSWERED whose response row the runtime has not
    made durable yet. It is PENDING (it belongs in the middle segment — the
    queue is not finished until the agent has been told) and it is NOT
    answerable, so the row carries this status word where a settled row carries
    its chip (round 2: F3/Q1/U4).

    IT DOES NOT HOLD THE SESSION'S MARK (round 2: F13). The mark is the index's
    OUTSTANDING set (``open`` only since #2388, §4.8) unioned with the current
    session's answerable rows, so it clears the moment the user answers — which
    is right, because it tracks a debt the USER owes and this ask's debt was
    paid. The agent's own delivery lag is what this row states; the sidebar is
    not the surface for it.

    The ink is the same `success`/`warning` role a settled answer uses, so a
    reader cannot tell the two apart by colour — they are told apart by the
    WORDS, which is the point: one says the agent was told, the other says it
    has not been yet.
    """
    if not row.delivering:
        return None
    word = _DELIVERING_CHIPS.get(row.status)
    if word is None:
        return None
    return (word, SETTLED_CHIPS.get(row.status, (word, "success"))[1])


def _openable(row: AskRow) -> bool:
    """Whether Enter or a click on ``row`` opens a card.

    An ANSWERABLE row opens its answer card. A LAPSED row opens the same card
    READ-ONLY (#2481 UX U1): the list row is one truncated line, and the card is
    the only place the whole question and its options can be read. Seeing it is
    the explicit look Ben asked for, and the card sends nothing (D10). Every
    other settled row, and a delivering one, stays inert.
    """
    return row.answerable or row.lapsed


def compact_headline(open_count: int, timed_out_count: int) -> str:
    """The bar's count when :func:`queue_headline` does not fit (#2481 design D6).

    Whole clauses, never cells: ``N waiting · N timed out``. A cut inside the full
    sentence read ``… 1 question`` at 50 columns — a third count of something
    waiting — and took the chevron with it. Urgency has no clause here; the amber
    glyph still carries it.
    """
    parts: list[str] = []
    if open_count:
        parts.append(_t(_x.ask_queue_waiting_clause(shown=str(open_count))))
    if timed_out_count:
        parts.append(_t(_x.ask_queue_timed_out_clause(shown=str(timed_out_count))))
    return " · ".join(parts)


def queue_headline(open_count: int, timed_out_count: int, urgent_count: int = 0) -> str:
    """The queue's count, in ONE vocabulary for the bar and the list header.

    Two totals used to describe one queue: the bar counted the OPEN asks
    ("2 questions waiting") while the list header counted every row it drew
    ("3 open asks", the timed-out one included), and a reader with both on
    screen saw two numbers three rows apart (design D7 / UX U4 / QA Q4). The
    words are here, once, so the two surfaces cannot disagree about what they
    are counting — and the no-timed-out case keeps §5's own copy verbatim.

    THIS IS THE BAR'S REGISTER (the CHIP). The list header takes the DRAWER
    register instead (``drawer_headline``), and the two differ ON PURPOSE — see
    there for why one queue can honestly wear two sentences.
    """
    parts: list[str] = []
    if open_count:
        # The noun's number agreement moves into the message's own ICU plural
        # (`one `{shown} question waiting``/`other `{shown} questions waiting``),
        # with `count` selecting the branch and `shown` staying a RAW digit
        # string so the rendered bytes are unchanged (no grouping, no re-format).
        parts.append(_t(_x.ask_queue_question_waiting(count=open_count, shown=str(open_count))))
    if urgent_count:
        # URGENCY IN WORDS, because the bar's amber glyph is a hue and a hue is
        # not a channel a reader without colour has: D8's own principle, which
        # had reached the list row and not the bar (design round 2, D13). It
        # also removes the mismatch D13 found — the glyph is painted from
        # `any(row.urgent)`, so with no word for it the amber annotated the ONE
        # question the bar happened to name, which need not be the urgent one.
        parts.append(_t(_x.ask_queue_urgent_clause(shown=str(urgent_count))))
    if timed_out_count:
        # "question", the noun the waiting clause uses and the phone and desktop
        # use for the same state, so one bar never counts in two nouns (D5) —
        # now carried by the key's own plural branches.
        parts.append(
            _t(_x.ask_queue_question_timed_out(count=timed_out_count, shown=str(timed_out_count)))
        )
    return " · ".join(parts)


def drawer_headline(
    rows: Sequence[AskRow], *, open_count: int | None = None, truncated: bool = False
) -> str:
    """The list header's count, in the DRAWER register (design §4, amendment A6).

    WHY IT DIFFERS FROM THE BAR. ``queue_headline`` counts the rows the agent is
    still waiting on, which is the right register for a one-line chip in a row of
    tallies. As the drawer's own count it under-described the drawer's own
    contents: the drawer draws a row for every OUTSTANDING ask, moved-on ones
    included, so `1 question waiting` sat over two answerable rows and the row
    under it read "moved on; you can still answer" (the desktop's UX round 1,
    U5 — the same finding, transferred with the surface).

    A mixed queue used to spell a second "moved on" half here for timed-out
    rows. Since D11 a lapsed row is SETTLED — one line behind its ``timed out``
    chip — so the pending half holds only waiting and delivering rows, and this
    sentence counts those. The bar keeps the chip clause (it still names timed-out
    asks, because it is the door to their Dismiss); the two registers are named
    in §11.

    THE DELIVERING HALF IS PENDING AND MUST NEVER READ AS SETTLED (round 2:
    F3/Q1/U4). ``pending`` = answerable ∪ delivering (§10), so an
    answered-but-undelivered row sits in the "Pending" segment while
    the old fallback printed `N settled` over it — the header contradicting its
    own segment. It says so in §5's words instead, and without the desktop's
    "you can still change it": the TUI has no revise wire at all, and promising
    a door that does not exist is worse than the missing sentence. The
    all-settled case takes the desktop drawer's own constant (`All asks
    settled`) rather than inventing a second form for it.

    THE TRUNCATED FRAME STATES THE BACKEND'S TALLY (``open_count``) AND
    WITHHOLDS THE SPLIT: the wire caps the list, so a pending/settled split
    computed from the rows this frame carries would be a prefix passing for the
    whole queue. This is the desktop's `askSplitIsKnowable` rule ("the louder
    number wins"); `open_count is None` means the caller has no backend tally to
    state, and then the row-derived split is all there is — which is exactly the
    session-scope case, where the wire carries every ask the session has.
    """
    if truncated:
        return f"{open_count or 0} outstanding"
    waiting = sum(1 for row in rows if row.waiting)
    delivering = sum(1 for row in rows if row.delivering)
    # OPEN rows only, the same set the bar counts (review round 3, MINOR-1): an
    # urgent ask is one whose deadline is imminent, and a timed-out ask has no
    # imminent deadline left — counting it here made the list say `1 urgent · 1
    # ask timed out` over a bar showing only the amber hue.
    urgent = sum(1 for row in rows if row.waiting and row.urgent)
    # NO "MOVED ON" HALF ANY MORE (D11): a lapsed row is settled, so the pending
    # half holds only what is waiting and what is delivering, and the drawer
    # counts it the way the chip does.
    clause = queue_headline(waiting, 0, urgent)
    if delivering:
        noun = "answer" if delivering == 1 else "answers"
        # §10's own words for this half (design §5's copy contract: "Answered —
        # delivering"), and deliberately WITHOUT the desktop's "— you can still
        # change it": that sentence is true on a surface with a revise wire, and
        # this one has none (recorded as deferred on the PR).
        stating = f"{delivering} {noun} delivering — the agent will be told"
        clause = f"{clause} · {stating}" if clause else stating
    if clause:
        return clause
    if rows:
        # Every row settled, and the chip register has no words for that (it
        # counts what is owed). The drawer still owes a count — it is showing
        # exactly those rows — so it says the third half's own name, in the
        # desktop drawer's own constant (round 2: D4/F3).
        return "All asks settled"
    return ""


class AskBar(Widget):
    """The minimized ask affordance: one line, click or the toggle key to expand.

    Focusable so a keyboard can reach it, but it never TAKES focus on its own:
    §5.0's no-auto-mount/no-focus-steal rule covers this widget too. Clicking
    it focuses it as a side effect of the click (Textual focuses a focusable
    widget under a press), and Enter then toggles — so the pointer is the
    primary path and the keyboard is available without a second gesture. The
    app's own :data:`ASK_TOGGLE_KEY` binding is the route that needs no
    focus at all, and the tests pin that the key is one the composer does not
    claim (UX
    round 1, U5: the bar's own Enter binding was unreachable, because nothing
    focuses a bar the user cannot see is focusable).
    """

    can_focus = True

    class Toggled(Message):
        """The user asked to expand or collapse the answer surface."""

    #: NOT named ``toggle``: Textual's ``DOMNode`` already owns an
    #: ``action_toggle(attribute_name)`` (it toggles a CSS class), and pyright
    #: rejects the override as an incompatible signature. A name that says what
    #: this one does is clearer than a shadowing one anyway.
    BINDINGS = [Binding("enter", "expand_or_collapse", "Expand/collapse", show=False)]

    def __init__(self, widget_id: str = "ask-bar") -> None:
        super().__init__(id=widget_id)
        self._count = 0
        self._timed_out = 0
        self._head = ""
        self._expanded = False
        self._urgent = False
        self._urgent_count = 0
        #: Whether there is anything to open at all — see `set_state`.
        self._present = False
        self.display = False

    @property
    def count(self) -> int:
        return self._count

    @property
    def expanded(self) -> bool:
        return self._expanded

    def set_state(
        self,
        *,
        count: int,
        timed_out: int = 0,
        head: str = "",
        expanded: bool,
        urgent: bool = False,
        urgent_count: int = 0,
        present: bool | None = None,
    ) -> None:
        """Repaint for the current queue.

        ``head`` is the first question, drawn as a dim trailing hint so the bar
        names WHAT is waiting without becoming a second list. It is clipped by
        THIS widget, against the width the row really has (design D5: a fixed
        clip plus a tail appended after it overflowed a 76-cell bar at 80x24
        and lost the chevron entirely).

        ``count`` is what the user still OWES — the OPEN asks — while
        ``present`` says whether there is anything at all to open. The two
        differ for exactly one state, and it is the reason both exist: a queue
        of nothing but TIMED-OUT asks owes nobody anything (the agent went ahead
        without them, D10), but ``x`` dismiss lives behind the bar, so a bar
        hidden by ``count == 0`` would make them unreachable from the TUI.
        """
        if present is None:
            present = count > 0 or timed_out > 0
        changed = (
            count != self._count
            or timed_out != self._timed_out
            or head != self._head
            or expanded != self._expanded
            or urgent != self._urgent
            or urgent_count != self._urgent_count
            or present != self._present
        )
        self._count = count
        self._timed_out = timed_out
        self._head = head
        self._expanded = expanded
        self._urgent = urgent
        self._urgent_count = urgent_count
        self._present = present
        # Nothing to open means no bar at all — not an empty one. A row of
        # blank chrome above the composer would push the dock down for nothing.
        self.display = present
        self.set_class(present, "-ask-visible")
        self.set_class(self._expanded, "-ask-expanded")
        if changed:
            self.refresh(layout=True)

    def action_expand_or_collapse(self) -> None:
        self.post_message(self.Toggled())

    def on_click(self, event: events.Click) -> None:  # type: ignore[override]
        """A click anywhere on the bar toggles, chevron included.

        ``event.stop()`` for §5.0's overlay rule's sibling: a press that
        reaches the dock must not also reach whatever is behind it.
        """
        event.stop()
        self.post_message(self.Toggled())

    def render(self) -> Text:
        """The bar, laid out against the width it actually has.

        Explicit ``Style(color=...)`` rather than Rich markup names: a bare word
        like "accent" is not a Rich style, and Rich silently drops what it
        cannot parse — a colour that never paints and never errors.

        THE SACRIFICE ORDER IS DELIBERATE and is what design D5 asked for: the
        chevron is reserved first (it is the only thing saying which way this
        control goes), then the count, then the affordance words, and the head
        QUESTION is what gives way — it is the one part that is also available
        by expanding, while a bar with no chevron or no count is a bar that
        stopped being a control.
        """
        accent = Style(color=theme_mod.semantic_color("accent"))
        fg = Style(color=theme_mod.semantic_color("fg"))
        dim = Style(color=theme_mod.semantic_color("dim"))
        warning = Style(color=theme_mod.semantic_color("warning"))
        muted = Style(color=theme_mod.semantic_color("muted"))
        # A bar that owes nothing (only timed-out asks behind it) is painted in
        # the SETTLED register: the accent `?` is the "a question is waiting"
        # ink, and a lapsed ask is not one (#2481 design D4 / UX U2). The bar
        # stays — it is the door to `x` dismiss — it just stops calling.
        if self._urgent:
            glyph_style = Style(bold=True) + warning
        elif self._count:
            glyph_style = Style(bold=True) + accent
        else:
            glyph_style = muted
        copy_style = fg if self._count else muted

        # Read from `self.size` rather than assumed: a bar rendered before its
        # first layout pass has width 0.
        width = max(0, int(getattr(self.size, "width", 0) or 0))
        chevron = ASK_BAR_CHEVRON_EXPANDED if self._expanded else ASK_BAR_CHEVRON_COLLAPSED
        # One cell for the chevron and one of separation from the copy; a bar
        # too narrow for both keeps the glyph and the count and drops the rest
        # rather than butting the chevron against its own words (UX U10).
        avail = width - 2
        if avail <= 1:
            text = Text(no_wrap=True, overflow="ellipsis")
            text.append(ASK_MARKER, style=glyph_style)
            return text
        # SHED WHOLE CLAUSES (#2481 design D6): the full sentence, then without
        # its urgent clause (the amber glyph keeps saying it), then the compact
        # form. Measured against `avail`, which already reserves the chevron.
        headline = queue_headline(self._count, self._timed_out, self._urgent_count)
        for candidate in (
            headline,
            queue_headline(self._count, self._timed_out),
            compact_headline(self._count, self._timed_out),
        ):
            headline = candidate
            if cell_len(f"{ASK_MARKER} {candidate}") <= avail:
                break
        base = f"{ASK_MARKER} {headline}"
        # "view", not "answer", over a queue that owes nothing: a lapsed-only bar
        # opens a list of read-only rows (D10), and the verb must not promise an
        # answer box the list will not offer.
        verb = "collapse" if self._expanded else ("answer" if self._count else "view")
        tail = f" — {ASK_TOGGLE_KEY} or click to {verb}"
        head_part = f" · {self._head}" if self._head else ""
        used = cell_len(base)
        if head_part and used + cell_len(tail) + 8 <= avail:
            # The whole room that is left, with no second ceiling on top of it:
            # the tail is reserved first and the head gives way, so a further
            # cap can only truncate where the row had space (design round 2,
            # D14 — the old 48-cell ceiling still bit at 130 columns while the
            # frame showed eleven empty cells before the chevron).
            room = avail - used - cell_len(tail) - 3
            head_part = f" · {_clip_cells(self._head, room)}" if room >= 8 else ""
        else:
            head_part = ""
        if used + cell_len(head_part) + cell_len(tail) > avail:
            tail = ""
        text = Text(no_wrap=True, overflow="ellipsis")
        text.append(ASK_MARKER, style=glyph_style)
        # Clipped at the reserve, never past it: the chevron is the only thing
        # saying which way this control goes, so the copy gives way to it.
        text.append(f" {_clip_cells(headline, avail - 1)}", style=copy_style)
        if head_part:
            text.append(head_part, style=copy_style)
        if tail:
            text.append(tail, style=dim)
        padding = width - 1 - cell_len(text.plain)
        if padding > 0:
            text.append(" " * padding)
        # Muted with the rest of the bar when nothing is owed (#2481 UX U14):
        # one cell of accent on a bar that says nothing is waiting still calls.
        text.append(chevron, style=Style(bold=True) + accent if self._count else muted)
        return text


def _wrap_cells(text: str, room: int, *, max_lines: int = 2) -> list[str]:
    """``text`` split into at most ``max_lines`` lines of ``room`` cells, on spaces.

    The wrap side of :func:`_clip_cells`, for the two sub-lines that must stay
    readable on a narrow panel (UX round 1, U4). A space is the only break: a
    clause cut MID-WORD is what made the chooser's explanation unreadable at 60
    cells, and a second line costs nothing — the block already scrolls.

    The LAST line is clipped, never wrapped again: a bounded line count is what
    keeps the painted sequence (and so the pointer's row map) a function of the
    width rather than of the text's length.
    """
    if room <= 0:
        return [""]
    if cell_len(text) <= room:
        return [text]
    lines: list[str] = []
    rest = text
    while rest and len(lines) < max_lines - 1:
        cut = -1
        #: A space that OPENS a parenthetical wins over the last space: the
        #: context line's "(last 14:14)" is one fact, and splitting it printed
        #: `14:14)` alone on the second row (UX round 1, U4's own complaint in a
        #: new place).
        paren = -1
        used = ""
        for index, char in enumerate(rest):
            if cell_len(used + char) > room:
                break
            used += char
            if char == " ":
                cut = index
                if rest[index + 1 : index + 2] == "(":
                    paren = index
        if paren > 0:
            cut = paren
        if cut <= 0:
            break
        lines.append(rest[:cut].rstrip())
        rest = rest[cut:].lstrip()
    if rest:
        lines.append(_clip_cells(rest, room))
    return lines or [""]


def _clip_cells(text: str, room: int) -> str:
    """``text`` clipped to ``room`` cells with an ellipsis, cheaply.

    The local spelling of ``tool_card.truncate_cells`` rather than an import:
    this module is on the dock's paint path and the tool-card module is not
    otherwise needed here. Both agree on the ellipsis and the cell count.
    """
    if room <= 0:
        return ""
    if cell_len(text) <= room:
        return text
    if room == 1:
        return "…"
    clipped = ""
    for char in text:
        if cell_len(clipped + char) > room - 1:
            break
        clipped += char
    return clipped + "…"


#: The one key a PASSIVE list can honour from the composer (``AskQueueList(passive=True)``):
#: Tab hands it the caret. Spelled with the card's own glyph (``TAB_HINT_KEY``) so the
#: two ask surfaces name the same key the same way, and as a header hint so it sheds
#: under the same width rule as every other hint rather than being a second line.
PASSIVE_TAB_HINT = f"{TAB_HINT_KEY} answer here"


@dataclass(frozen=True)
class HeaderAtom:
    """One paintable piece of the list's ONE-LINE header.

    Atoms rather than a single string because two readers need different things
    from the same bytes: the painter needs each piece's INK, and the pointer
    needs the cell SPAN of each filter segment. Both read this one list, so the
    press target cannot drift from what was painted.

    ``ink`` is a role (``fg``/``muted``/``accent``/``active``) rather than a
    hex, resolved by the painter against the live ramp.
    """

    text: str
    ink: str
    filter_value: str = ""


def _atoms_width(atoms: Iterable[HeaderAtom]) -> int:
    return sum(cell_len(atom.text) for atom in atoms)


def _atoms_text(atoms: Iterable[HeaderAtom]) -> str:
    return "".join(atom.text for atom in atoms)


class AskQueueList(Widget):
    """The open-ask list: one row per ask, Enter or a click to answer one.

    A ``Widget`` rather than a ``Container`` of rows: the rows are painted from
    state and hit-tested by line, which is what keeps a queue of eight asks one
    widget instead of eight focusable children competing for the tab order.
    """

    can_focus = True

    class Picked(Message):
        """The user chose one ask to answer."""

        def __init__(self, ask_id: str) -> None:
            super().__init__()
            self.ask_id = ask_id

    class Collapse(Message):
        """The user closed the surface."""

    class Decline(Message):
        """The user declined one ask — terminal, so it is an explicit action."""

        def __init__(self, ask_id: str) -> None:
            super().__init__()
            self.ask_id = ask_id

    class Dismiss(Message):
        """The user dismissed one ask from the view.

        ``reason`` is the operator's word for WHY, and it is only ever set on an
        OPEN ask: a lapsed one was walked past by the agent already, so its
        dismissal carries no reason and the queue records its own default
        (``asks/store.DISMISS_REASON_DEFAULT``). The chooser is this widget's,
        because the list is where the gesture and the row both live.
        """

        def __init__(self, ask_id: str, *, reason: str | None = None) -> None:
            super().__init__()
            self.ask_id = ask_id
            self.reason = reason

    BINDINGS = [
        Binding("escape", "collapse", "Collapse", show=False),
        Binding("enter", "pick", "Answer", show=False),
        Binding("up", "move(-1)", "Up", show=False),
        Binding("down", "move(1)", "Down", show=False),
        Binding("k", "move(-1)", "Up", show=False),
        Binding("j", "move(1)", "Down", show=False),
        Binding("d", "decline", "Decline", show=False),
        Binding("x", "dismiss", "Dismiss", show=False),
        # The filter's own keys (design §4). None of `1/2/3`, `[`/`]` collide
        # with the bindings above (verified against this list), and the segments
        # are also press targets — the mouse path A9 requires.
        Binding("1", "filter_all", "All asks", show=False),
        Binding("2", "filter_outstanding", "Pending", show=False),
        Binding("3", "filter_settled", "Settled asks", show=False),
        Binding("left_square_bracket", "filter_prev", "Previous filter", show=False),
        Binding("right_square_bracket", "filter_next", "Next filter", show=False),
    ]

    #: The header occupies the first painted line; a click on it resolves to the
    #: FILTER SEGMENT under the pointer rather than to a row (`_row_at` returns
    #: ``None`` for it). ONE painted row by construction — see `header_atoms`,
    #: which drops whole hints rather than letting the line wrap.
    HEADER_ROWS = 1

    #: The header's hints, in the order they are SPENT — earlier entries are
    #: kept longest.
    #:
    #: `enter` LEADS since the open-by-default round (U3/D4): the auto-opened
    #: surface teaches `⇥` while passive, and the moment the user pressed it the
    #: ladder named only `d decline` — the primary action was the first casualty
    #: at the width the repo assumes. The earlier order spent `d` first by
    #: IRREVERSIBILITY (review round 2, D3) on the argument that `enter` already
    #: carries its own cue in the row's `❯` caret; the walk found the caret says
    #: WHICH row, not that Enter takes it — and on the passive→live handover the
    #: caret arrives in the same frame the hint is shed, so `enter answer` is
    #: the hint that cannot be spared. `d` still outlives `x`: a decline cannot
    #: be undone, and `x` is only ever offered over a timed-out row.
    HEADER_HINTS = (
        "enter answer",
        "d decline",
        # `x` before `enter view` (#2481 UX U8): on a lapsed row dismiss is the
        # only key that changes anything, so it is the hint a narrow header keeps
        # (the first hint outbids the drawer clause, see `header_atoms`).
        "x dismiss",
        "enter view",
        PASSIVE_TAB_HINT,
        "esc collapse",
        # THE CHOOSER'S OWN TWO (R5), last because they are live only while it is
        # up: `header_hints` filters by what the CURRENT view can honour, so the
        # list's hints and the chooser's never mix — and the order here is the
        # order they paint in when it is up. `esc cancel` and not `esc back`
        # (design N1): with no chooser up Esc collapses the whole surface, so one
        # word cannot name two exits — the chooser's footer is the only place
        # either key set is painted while it is up (`header_hints` returns none).
        "enter dismiss",
        "esc cancel",
    )

    #: The cells between the scope clause and the filter control, and between
    #: the control and the hints. Three, not two, because the control's own
    #: segments are two apart and the reader has to be able to see where the
    #: control ends — two paddings of the same size read as one row of words.
    SEGMENT_GAP = "   "

    def __init__(
        self,
        rows: Sequence[AskRow],
        widget_id: str = "ask-queue-list",
        now_ms: int = 0,
        *,
        scope: str = SCOPE_SESSION,
        open_count: int | None = None,
        truncated: bool = False,
        session_titles: Mapping[str, str] | None = None,
        passive: bool = False,
    ) -> None:
        super().__init__(id=widget_id, classes="prompt-slot")
        self._rows: list[AskRow] = list(rows)
        # THE CURSOR STARTS ON THE FIRST ANSWERABLE ROW (#2481 UX U3): rows are
        # ordered by status, not by what the user owes, and a list opened over a
        # timed-out row first put the cursor on the one row Enter cannot answer.
        self._index = next((i for i, row in enumerate(self._rows) if row.answerable), 0)
        self._now_ms = now_ms
        #: True for a list the open-by-default policy put up and the user has not yet
        #: engaged: the composer still holds the caret, so NONE of this list's keys
        #: (arrows, Enter, ``d``, ``x``, the filter digits) can reach it. It is painted
        #: as a read-only view of the queue — no ``❯`` marker, no key hints it cannot
        #: honour — because a list that looked live over a dead keyboard is exactly the
        #: dead end UX round 1 (U2) removed, and it says how to take the keyboard
        #: instead (Tab, or a click on a row). Ends on an explicit user gesture
        #: (:meth:`engage`), and NOT on merely receiving focus — Textual lends focus to
        #: whatever is next when a neighbouring prompt is removed, and a list that
        #: took that for engagement would paint a live ``❯`` and live key hints over a
        #: composer that holds the caret (see ``AskPickerScreen._passive``). A
        #: door-opened list (f4, the bar) takes focus at mount and is never passive.
        self._passive = passive
        # The three-way filter is CLIENT-LOCAL view state (§5.0's rule for the
        # EXPANDED state): which slice of one surface's list the reader is
        # looking at has no wire meaning, and persisting it would be a second
        # store to keep in step with a queue that changes under it.
        self._filter = FILTER_ALL
        #: The queue this list is reading, and its door's promise. The TUI has
        #: no session-less screen, so a FLEET list is only ever reached from a
        #: fleet surface — the door decides the scope and the list offers no
        #: toggle, which is what keeps this ONE affordance (design §4).
        self._scope = scope if scope in SCOPE_SUBJECTS else SCOPE_SESSION
        #: The backend's own outstanding tally, when the frame carries one. Only
        #: consulted on a truncated frame, where the rows are a prefix and the
        #: split between the halves is not knowable (see ``drawer_headline``).
        self._open_count = open_count
        self._truncated = truncated
        #: Ask ids whose answer is IN FLIGHT through the engage seam. A row in
        #: here paints ``IN_FLIGHT_MARK`` and is inert, which is the whole
        #: double-fire guard: the engage can take the phone's 30 s + 15 s
        #: window, and a second Enter while it runs would send the answer twice.
        self._in_flight: set[str] = set()
        #: ``{session_id: title}`` for FLEET rows, supplied by the app from the
        #: session catalogue. A fleet row names the conversation it belongs to,
        #: and the catalogue's own title is the name the user sees on every other
        #: surface — the cwd's last segment is only the fallback for a session the
        #: catalogue no longer carries (round 2: U8/U6).
        self._session_titles: dict[str, str] = dict(session_titles or {})
        #: The OPEN ask whose dismiss reason is being chosen, or ``None``. The
        #: chooser is a MODE of this list rather than a second surface: the
        #: gesture and the row it acts on are here, `x` already belongs to the
        #: list, and a separate widget would have to re-derive which row the
        #: user meant on both sides of the swap.
        self._dismiss_for: str | None = None
        #: The highlighted reason of that chooser (an index into
        #: ``DISMISS_CHOICES``). `moot` is the default for the same reason the
        #: queue records it when a caller omits the reason: it is the reading
        #: that commits the agent to nothing.
        self._dismiss_index = 0

    # -- the view-model ------------------------------------------------------

    @property
    def rows(self) -> list[AskRow]:
        return list(self._rows)

    @property
    def index(self) -> int:
        """The highlight's position among the VISIBLE rows (the painter's index)."""
        return self._index

    @property
    def visible_rows(self) -> list[AskRow]:
        """The rows the CURRENT filter shows, which is what the pointer hits.

        Every reader — the highlight, the hit test, the painter — works in
        these indexes and not in ``rows``, so a filtered view cannot open a row
        the user cannot see (the class of bug a filter introduces when only the
        painter knows about it).
        """
        if self._filter == FILTER_OUTSTANDING:
            return [row for row in self._rows if row.pending]
        if self._filter == FILTER_SETTLED:
            return [row for row in self._rows if row.settled]
        return list(self._rows)

    @property
    def filter_value(self) -> str:
        return self._filter

    @property
    def counts(self) -> dict[str, int]:
        """``{filter: count}`` for the three segments, over every row."""
        return filter_counts(self._rows)

    def set_filter(self, value: str) -> None:
        """Select one half. The highlight follows the ROW, not the index.

        THE KEYBOARD PATH IS CLOSED WHILE A CHOOSER IS UP (agent review round 1,
        m2): `1`/`2`/`3`/`[`/`]` and a click on a segment are inert, exactly like
        `d`, `x`, `enter` and the arrows, because the chooser owns the keyboard
        until a reason is chosen or Escaped out of. The earlier rule — a filter
        key silently CANCELS the chooser — was defensible on paper and advertised
        nowhere: the header named only `enter dismiss`/`esc`, so a reader who
        pressed `2` lost their unconfirmed reason without being told.

        This method stays the programmatic door (a caller re-pointing the slice
        from outside the keyboard loop), and it closes the chooser rather than
        letting it hover over a slice that may not draw its row.
        """
        if value not in FILTER_LABELS or value == self._filter:
            return
        self._dismiss_for = None
        current = self.current()
        self._filter = value
        self._reindex(current.ask_id if current else "")
        self.refresh(layout=True)

    def cycle_filter(self, delta: int) -> None:
        """``[``/``]``: walk all → pending → settled → all."""
        at = FILTER_ORDER.index(self._filter)
        self.set_filter(FILTER_ORDER[(at + delta) % len(FILTER_ORDER)])

    def set_in_flight(self, ask_id: str, active: bool) -> None:
        """Paint ``ask_id`` as being sent (or stop painting it).

        Called by the app around a fleet op: for an answer or a decline the window
        is the DIAL's (the phone's 30 s + 15 s), and for a dismissal it is the
        append that settles it against the log with nothing dialled (fork A §14) —
        the row must both SAY it is spoken for and REFUSE a second gesture for that
        window, whichever op set it. The mark and the guard are the same state,
        deliberately, so they cannot disagree.
        """
        if active:
            self._in_flight.add(ask_id)
        else:
            self._in_flight.discard(ask_id)
        self.refresh()

    def in_flight(self, ask_id: str) -> bool:
        return ask_id in self._in_flight

    def _reindex(self, ask_id: str) -> None:
        """Put the highlight on ``ask_id`` among the VISIBLE rows, clamped.

        By ask id and not by index: a queue that answered one ask and dropped
        another would otherwise slide the highlight onto a different question
        under the user's finger, which is the same defect the picker's held
        answer key exists to prevent.
        """
        visible = self.visible_rows
        if not visible:
            self._index = 0
            return
        self._index = next(
            (i for i, row in enumerate(visible) if row.ask_id == ask_id),
            0,
        )

    def set_rows(
        self,
        rows: Sequence[AskRow],
        *,
        now_ms: int = 0,
        open_count: int | None = None,
        truncated: bool = False,
        scope: str | None = None,
        session_titles: Mapping[str, str] | None = None,
    ) -> None:
        """Replace the list, keeping the highlight on the same ASK where it can.

        ``scope``/``session_titles`` are only passed when the app is RE-POINTING
        an already-mounted list at another scope (round 2: U1). Re-mounting
        would put a second widget with the same id in the prompt host while the
        first one's deferred removal is still pending — the `DuplicateIds`
        crash that ended the session — and retargeting is also the better answer
        to a second press on the door: the reader keeps their place and gets
        fresh rows.

        Called from the app's frontend-snapshot writer, which is what makes the
        panel follow the wire: without it an ask answered on another surface
        kept its row (and its answerability) until the user collapsed and
        re-expanded (QA Q3 / design D4).

        It CARRIES the settled rows too now (design §4) — the filter, not this
        call, decides which half is on screen — so a reconcile that settles a
        row moves it between halves rather than deleting it.

        ``open_count``/``truncated`` ride with the rows because the header's
        drawer register prints the backend's own number on a truncated frame
        (amendment A6). THEY ARE ASSIGNED UNCONDITIONALLY — a caller that omits
        them RESETS both, and the header then falls back to a split computed from
        whatever rows it happens to carry. Every caller must therefore pass the
        facts that belong to the rows it is handing over: the session scope
        passes the WIRE's tally, and the FLEET scope passes the index read's own
        (``_refresh_fleet_count_facts``). Round 2's F12 was exactly this bug in
        the fleet branch, whose rows are the whole index only when that index was
        not capped — which is not a property a caller may assume.
        """
        current = self.current()
        if scope is not None and scope in SCOPE_SUBJECTS:
            self._scope = scope
        if session_titles is not None:
            self._session_titles = dict(session_titles)
        self._rows = list(rows)
        self._now_ms = now_ms
        self._open_count = open_count
        self._truncated = truncated
        self._reindex(current.ask_id if current else "")
        # A CHOOSER FOR A ROW THAT IS GONE IS A DOOR TO NOTHING: the ask was answered
        # on the phone, expired, or was dismissed elsewhere while the reason was
        # on screen, and the next Enter would post a dismissal for an id this list
        # no longer draws — or, worse, for a row it no longer draws in THIS slice
        # (the Pending filter drops a row the moment it times out). The chooser
        # closes with the row it belongs to, and the row must be BOTH visible and
        # still answerable: a row that settled to `answered` or `timed_out` keeps
        # its id but has left the door the reason belongs to (agent review round
        # 1, m1 — Enter used to be live on a state the reader never saw).
        if self._dismiss_for is not None:
            keeper = next(
                (row for row in self.visible_rows if row.ask_id == self._dismiss_for),
                None,
            )
            if keeper is None or not keeper.answerable:
                self._dismiss_for = None
        self.refresh(layout=True)

    def set_now(self, now_ms: int) -> None:
        """Advance the countdown's clock without touching the rows.

        The app's countdown tick calls THIS rather than :meth:`set_rows`: no row
        changed, and re-deriving the highlight by ask id every 30 s is work that
        can only get the highlight wrong. A plain ``refresh`` and not
        ``refresh(layout=True)`` — the words ``expiry_text`` paints are clipped
        against the width read at paint time, so a countdown that gains a cell
        (``9m`` → ``10m``) shortens the question under it rather than reflowing
        the row, which is what keeps the one-painted-row-per-ask invariant the
        pointer hit test rests on.
        """
        if now_ms == self._now_ms:
            return
        self._now_ms = now_ms
        self.refresh()

    def select(self, index: int) -> None:
        """Put the cursor on a visible row by index, clamped — used to hand a
        card's user back to the row they came from (UX round 1, U6, where
        Escaping a card reset the highlight to row 1 and made them hunt for
        their place)."""
        visible = self.visible_rows
        self._index = max(0, min(len(visible) - 1, index)) if visible else 0
        self.refresh()

    def current(self) -> AskRow | None:
        visible = self.visible_rows
        if not visible:
            return None
        return visible[min(self._index, len(visible) - 1)]

    # -- keys ----------------------------------------------------------------

    @property
    def choosing(self) -> bool:
        """Whether the dismiss reason chooser is up (this list's second mode)."""
        return self._dismiss_for is not None

    def dismiss_choice(self) -> str | None:
        """The reason currently highlighted in the chooser, or ``None``. The QUEUE's
        word, not the painted label — that is what the dismissal will record."""
        if self._dismiss_for is None:
            return None
        return DISMISS_CHOICES[self._dismiss_index][0]

    def action_move(self, delta: int) -> None:
        if self.choosing:
            # Clamped like the row cursor: two choices, and a wrap under a held
            # arrow would look like the list re-ordering itself.
            self._dismiss_index = max(0, min(len(DISMISS_CHOICES) - 1, self._dismiss_index + delta))
            self.refresh()
            return
        visible = self.visible_rows
        if not visible:
            return
        # Clamped, not wrapped: the list is short, overlaid, and a wrap would
        # throw the user from the last row to the first under a held key.
        self._index = max(0, min(len(visible) - 1, self._index + delta))
        self.refresh()

    def _gesture(self, ask_id: str) -> bool:
        """Whether a row may be acted on at all, from the row's own state.

        A1: a SETTLED row is a READ-ONLY one-liner. Enter/d/x on one are INERT —
        no message, no toast, no refusal — because there is nothing to refuse:
        the ask is finished, and an expiry or a decline is not a failure. The
        same guard covers a row whose answer is already IN FLIGHT (A7): the
        second gesture is dropped rather than queued, which is the double-fire
        guard.

        `answerable` and NOT `pending` since round 2 (F3/U4): `pending` folds in
        the DELIVERING half (§10), and a delivering row has already been
        answered — declining or re-answering it is not a gesture this surface
        offers, and it is exactly the window in which a second answer would
        contradict the first. The row still COUNTS as pending (it sits in the
        middle segment; the session's mark cleared when it was answered, per
        §4.8's outstanding set), it just is not a row the picker opens.
        """
        row = self.current()
        if row is None or row.ask_id != ask_id:
            return False
        if not row.answerable or row.ask_id in self._in_flight:
            return False
        return True

    def action_pick(self) -> None:
        if self.choosing:
            self._choose_dismiss_reason()
            return
        row = self.current()
        if row is not None and _openable(row) and row.ask_id not in self._in_flight:
            self.post_message(self.Picked(row.ask_id))

    def action_collapse(self) -> None:
        if self.choosing:
            self._cancel_dismiss_reason()
            return
        self.post_message(self.Collapse())

    def action_decline(self) -> None:
        if self.choosing:
            # The chooser owns the keyboard until it is answered or escaped: a
            # `d` typed while a reason is on screen would decline a question the
            # user was in the middle of putting away, which is irreversible.
            return
        row = self.current()
        if row is not None and row.answerable and row.ask_id not in self._in_flight:
            self.post_message(self.Decline(row.ask_id))

    def action_dismiss(self) -> None:
        """`x`: put the row under the cursor away — OPEN asks ask WHY first.

        A TIMED-OUT row has already been walked past by the agent (its deadline
        fired, design §14), so its dismissal stays one key and records the
        queue's default reason. An OPEN one is still owed by the agent, and the
        reason is what it acts on: `moot` closes the question, `answered_in_chat`
        tells it the answer is already in the conversation (SA-D7/R5). So the
        key opens the two-choice reason card instead of dismissing outright.
        """
        if self.choosing:
            return
        row = self.current()
        if row is None or row.ask_id in self._in_flight:
            return
        if row.lapsed:
            self.post_message(self.Dismiss(row.ask_id))
            return
        if not row.answerable:
            return
        self._dismiss_for = row.ask_id
        self._dismiss_index = 0
        self.refresh(layout=True)

    def _choose_dismiss_reason(self) -> None:
        """Enter in the chooser: post the dismissal with the highlighted reason."""
        ask_id = self._dismiss_for
        if ask_id is None:
            return
        reason = DISMISS_CHOICES[self._dismiss_index][0]
        self._dismiss_for = None
        self.refresh(layout=True)
        self.post_message(self.Dismiss(ask_id, reason=reason))

    def _cancel_dismiss_reason(self) -> None:
        """Esc in the chooser: close it and leave the ask exactly where it was."""
        if self._dismiss_for is None:
            return
        self._dismiss_for = None
        self.refresh(layout=True)

    def action_filter_all(self) -> None:
        if self.choosing:
            return
        self.set_filter(FILTER_ALL)

    def action_filter_outstanding(self) -> None:
        if self.choosing:
            return
        self.set_filter(FILTER_OUTSTANDING)

    def action_filter_settled(self) -> None:
        if self.choosing:
            return
        self.set_filter(FILTER_SETTLED)

    def action_filter_next(self) -> None:
        if self.choosing:
            return
        self.cycle_filter(1)

    def action_filter_prev(self) -> None:
        if self.choosing:
            return
        self.cycle_filter(-1)

    # -- the pointer ---------------------------------------------------------

    def _top_inset(self) -> int:
        """Rows between this widget's OUTER edge and its first painted row.

        Taken from Textual's own regions rather than from a copied constant: the
        panel carries ``padding: 1 1``, and ``event.y`` arrives relative to the
        OUTER region, so a hit test that treated ``y`` as content-relative was a
        row ahead of the paint — which is exactly what review round 2's MAJOR
        found (a click opened the ask BELOW the one under the pointer, and the
        last ask was unreachable). ``content_region - region`` is Textual's own
        arithmetic for padding plus border, so it cannot drift from the
        stylesheet the way a constant would.
        """
        try:
            return max(0, int(self.content_region.y - self.region.y))
        except Exception:  # noqa: BLE001 — pre-layout, when there is nothing to hit
            return 0

    def _line_owners(self, width: int) -> list[tuple[str, int] | None]:
        """What each painted line BELOW THE HEADER belongs to.

        ``("row", i)`` — a visible ask's own line; ``("context", i)`` — a stale-ask
        context line under it (one entry per PAINTED line, so a wrapped context
        line owns both its lines); ``("reason", k)`` — a choice in the dismiss
        reason block (the label line and its note line both belong to that
        reason); ``None`` — a line that answers no click (the empty-half sentence
        and the chooser's heading and hint).

        ONE definition of the geometry, read by the painter's own sequence and by
        the pointer hit test. The two used to be the same arithmetic by luck: the
        row renderer emitted one line per ask and `_row_at` assumed it, which is
        how round 3's wrap defect opened the ask below the pointer. A line that is
        painted but unowned is exactly as visible here as one that is not painted,
        so the chooser and the context line cannot silently shift the rows below
        them.

        ``width`` is the painted width the caller is laying out for: the context
        line WRAPS to two lines on a narrow panel (UX round 1, U4 — clipping it
        mid-sentence lost the countdown, which is the half that says how stale the
        ask is), so its line count is a function of the width it was given. The
        painter and the hit test pass the same number, which is why this takes it
        rather than reading `content_size` twice.
        """
        visible = self.visible_rows
        owners: list[tuple[str, int] | None] = []
        if not visible and self._empty_slice():
            owners.append(None)
        for index, row in enumerate(visible):
            owners.append(("row", index))
            owners.extend([("context", index)] * self._context_line_count(row, width))
            if self._dismiss_for == row.ask_id:
                owners.append(None)  # the heading
                for k in range(len(DISMISS_CHOICES)):
                    owners.append(("reason", k))  # the label
                    owners.append(("reason", k))  # its note, one line down
                owners.append(None)  # the key hints
        return owners

    @staticmethod
    def _context_line_count(row: AskRow, width: int) -> int:
        """How many painted lines ``row``'s context line takes at ``width``."""
        context = context_text(row)
        if not context:
            return 0
        return len(_wrap_cells(CONTEXT_CUE + context, max(1, width - SUB_LINE_INDENT)))

    def _line_at(self, y: int) -> tuple[str, int] | None:
        """The owned thing under a widget-relative y, or ``None``."""
        index = y - self._top_inset() - self.HEADER_ROWS
        owners = self._line_owners(max(1, int(self.content_size.width)))
        if 0 <= index < len(owners):
            return owners[index]
        return None

    def _row_at(self, y: int) -> int | None:
        """The VISIBLE-row index under a widget-relative y, or ``None``.

        Both offsets are MEASURED: the widget's own top inset, and a header that
        is one painted line by construction (``header_atoms`` never wraps, and
        ``HEADER_HINTS`` is spent rather than broken mid-phrase). The index is
        into ``visible_rows`` — the rows the current filter draws — so a filtered
        view can never open a row the user cannot see.

        A row's CONTEXT line belongs to that row: clicking the sentence under an
        ask selects the ask, not the one below it (``_line_owners`` is the single
        statement of that).
        """
        owner = self._line_at(y)
        if owner is not None and owner[0] in {"row", "context"}:
            return owner[1]
        return None

    def _reason_at(self, y: int) -> int | None:
        """The chooser choice index under a widget-relative y, or ``None``."""
        owner = self._line_at(y)
        if owner is not None and owner[0] == "reason":
            return owner[1]
        return None

    def _segment_at(self, x: int, y: int) -> str | None:
        """The filter segment under a widget-relative point, or ``None``.

        The mouse path A9 requires beside the 1/2/3 and `[`/`]` keys. It reads
        the SAME atom list the painter drew (``_segment_spans`` walks
        ``header_atoms``), so the press target cannot drift from the painted
        bytes — the rule the sidebar's footer chip already follows.
        """
        if y - self._top_inset() != 0:  # HEADER_ROWS == 1; only the header row
            return None
        for value, start, end in self._segment_spans(max(1, int(self.content_size.width))):
            if start <= x < end:
                return value
        return None

    def _segment_spans(self, width: int) -> list[tuple[str, int, int]]:
        """``[(filter, start, end)]`` in cells, over the painted header."""
        spans: list[tuple[str, int, int]] = []
        cursor = 0
        for atom in self.header_atoms(width):
            end = cursor + cell_len(atom.text)
            if atom.filter_value:
                spans.append((atom.filter_value, cursor, end))
            cursor = end
        return spans

    def on_click(self, event: events.Click) -> None:  # type: ignore[override]
        """A click on a row selects it and opens it; the header picks a HALF.

        Without this the list was mouse-dead AND worse than dead: Textual's own
        click-to-focus walked up from the row and landed on the composer, so a
        click both failed to choose and disarmed the keyboard the list was
        using (UX round 1, U2 — `down`/`enter` stopped moving the highlight
        while `❯` was still painted, so the panel read as live).

        A settled row takes the highlight and NOTHING else (A1): it is inert, so
        the click must not mount a picker that could only refuse an answer. A
        DELIVERING row is the same case with a different reason (round 2:
        F3/U4) — it is already answered — so the gate is `answerable`, not
        `pending`.
        """
        event.stop()
        # The user chose this list: end its passivity for good (see :meth:`engage`),
        # not only while the click's own focus is on it.
        if self._passive:
            self._passive = False
            self.refresh(layout=True)
        y = int(event.y)
        if self.choosing:
            # THE CHOOSER TAKES THE CLICKS IT DRAWS: a tap on a choice sends THAT
            # reason — the pointer has already said which, so asking for a second
            # tap to confirm the row it is on would be a dead press. Every other
            # line in the block is inert, the same rule the keys follow — and that
            # now includes the header's filter segments (agent review round 1,
            # m2), which used to cancel the chooser from under the pointer.
            choice = self._reason_at(y)
            if choice is not None:
                self._dismiss_index = choice
                self._choose_dismiss_reason()
            return
        segment = self._segment_at(int(event.x), y)
        if segment is not None:
            self.set_filter(segment)
            return
        row = self._row_at(y)
        if row is None:
            return
        self._index = row
        self.refresh()
        target = self.visible_rows[row]
        if _openable(target) and target.ask_id not in self._in_flight:
            self.post_message(self.Picked(target.ask_id))

    def on_focus(self, event: events.Focus) -> None:  # type: ignore[override]
        """Repaint: a focused list paints live, an unfocused passive one does not.

        :attr:`passive` includes "not holding the caret", so the look changes with
        focus and has to be repainted on both edges (see :meth:`on_blur`). It does NOT
        end passivity — only :meth:`engage` does, because focus also arrives by
        accident.
        """
        if self._passive:
            self.refresh(layout=True)

    def on_blur(self, event: events.Blur) -> None:  # type: ignore[override]
        if self._passive:
            self.refresh(layout=True)

    @property
    def passive(self) -> bool:
        """Whether the list is up, un-engaged, and NOT holding the caret.

        The same two-part reading as ``AskPickerScreen.passive``, for the same reason:
        un-engaged alone would paint a list passive while the user is driving it,
        focus alone would paint it live for the one frame Textual lends it the caret.
        """
        return self._passive and not self.has_focus

    def engage(self) -> None:
        """The user chose this list: end its passivity and give it the caret.

        The Tab handover from the composer and a click both come through here.
        """
        self._passive = False
        self.refresh(layout=True)
        self.focus()

    def on_mouse_move(self, event: events.MouseMove) -> None:  # type: ignore[override]
        """Hover moves the highlight, so the row under the pointer is the one
        a click would take — the picker card's own rule.

        While the chooser is up the LIST's highlight does not move: the rows are
        not what a click acts on any more, and a highlight sliding under a choice
        would say the pointer had chosen an ask.
        """
        if self.choosing:
            choice = self._reason_at(int(event.y))
            if choice is None or choice == self._dismiss_index:
                return
            self._dismiss_index = choice
            self.refresh()
            return
        row = self._row_at(int(event.y))
        if row is None or row == self._index:
            return
        self._index = row
        self.refresh()

    def on_mouse_scroll_down(self, event: events.MouseScrollDown) -> None:  # type: ignore[override]
        event.stop()
        self.action_move(1)

    def on_mouse_scroll_up(self, event: events.MouseScrollUp) -> None:  # type: ignore[override]
        event.stop()
        self.action_move(-1)

    # -- paint ---------------------------------------------------------------

    def header_hints(self) -> tuple[str, ...]:
        """The hints the CURRENT view can actually honour (round 2: D2/F7/U9).

        The header used to spend its hint set purely by WIDTH, so an all-settled
        queue advertised `enter answer · d decline` over rows that are inert by
        construction (A1) — a dead-end affordance, and one that landed the hints
        exactly inverted against the rows: advertised where the keys do nothing,
        absent where they are live.

        So the set is derived from the row under the cursor and the keys' own
        objects: `enter answer`/`d` need an ANSWERABLE row, `enter view` a lapsed
        one, and `x dismiss` EITHER — an open ask can be put away too since R5
        (it is asked its reason first; a timed-out one goes at once, §14), so
        hiding the hint over open rows would name a key the row now honours.
        `esc` needs nothing — it is how a reader leaves any view, including an
        empty half. Order is still ``HEADER_HINTS``' own.
        """
        visible = self.visible_rows
        live = {"esc collapse"}
        if self.choosing:
            # THE CHOOSER OWNS THE KEYBOARD, AND ITS OWN BLOCK NAMES ITS KEYS.
            # Every key the list advertises is inert until a reason is chosen or
            # Escaped out of, and the block under the row paints `enter dismiss ·
            # esc cancel` one line below the choices — so a second copy in the
            # header was the same string twice within seven lines (design N2/n2).
            # A hint for a key this mode ignores is the dead-end affordance this
            # method exists to prevent; the header paints none at all.
            return ()
        if self.passive:
            # A PASSIVE list owns no key: the composer holds the caret, so `enter`,
            # `d` and `x` would type into the composer, not act on a row. It names
            # the one key that works (Tab hands it the caret) and `esc`, which the
            # app handles for the whole surface, so the header never advertises a
            # key the keyboard cannot deliver.
            if any(row.answerable for row in visible):
                live.add(PASSIVE_TAB_HINT)
            return tuple(hint for hint in self.HEADER_HINTS if hint in live)
        # THE ROW UNDER THE CURSOR decides, not "any visible row" (#2481 UX U3):
        # with an open and a timed-out row on screen the header used to promise
        # `enter answer · d decline` while the cursor sat on the timed-out one,
        # where Enter could not answer. A lapsed row takes `enter view` (the
        # read-only card, U1) and `x dismiss`; an answerable one takes the two
        # answer keys and `x dismiss` (R5). A row in flight takes nothing until
        # its reply lands.
        row = self.current()
        if row is not None and row.ask_id not in self._in_flight:
            if row.answerable:
                live |= {"enter answer", "d decline", "x dismiss"}
            elif row.lapsed:
                live |= {"enter view", "x dismiss"}
        return tuple(hint for hint in self.HEADER_HINTS if hint in live)

    def header_atoms(self, width: int) -> list[HeaderAtom]:
        """The ONE-LINE header, as paintable atoms, at this width.

        THE LINE CARRIES THREE THINGS, in this order: the queue's count in the
        DRAWER register (``drawer_headline`` — with the scope's own subject when
        the list is reading the fleet), the three-way FILTER control, and the
        key hints. They do not all fit at every width, so the line has a
        sacrifice order, and it is the one the operator's own ask dictates:

        * The CONTROL yields last. A filtered view that cannot be un-filtered is
          worse than an unadvertised key, so the three segments are never
          dropped — they are clipped only at the 47-cell floor, below which no
          arrangement of them fits anyway.
        * The DRAWER CLAUSE yields next, WHOLE: the segments already carry the
          three counts, so at a narrow width the sentence adds nothing the
          numbers do not say. Dropping it is why this is one line and not two.
        * The HINTS yield left to right, whole and never mid-phrase — design
          round 2's D12, which is what stops the header wrapping and every
          pointer hit below it shifting by a row — with ONE exception: the
          FIRST hint outbids the drawer clause (U3), which already yields to
          the numbers at a narrow width and is measured below with the first
          hint reserved, so the surface that teaches a key is never the
          surface that names no key at all.

        Atoms rather than a string because the segments are PRESSED controls:
        the painter inks them differently and the hit test (``_segment_spans``)
        reads these same atoms, so the press target cannot drift from the
        painted bytes.
        """
        counts = self.counts
        clause = drawer_headline(self._rows, open_count=self._open_count, truncated=self._truncated)
        # `chip-live` and not `accent` for the marker: the ask panel paints on
        # `overlay`, and the brand light accent lands at 3.49:1 there — under
        # the palette gate's own 4.0 state floor (design round 2, D1). The repo
        # already derives the remedy for exactly this ground
        # (`theme._fill_chip_live`, built because the light accent fails on
        # `overlay`), and on every ramp where the accent DOES clear, that token
        # IS the accent — so the dark ramp is unchanged.
        atoms = [HeaderAtom(f"{ASK_MARKER} ", "chip-live")]
        subject = SCOPE_SUBJECTS[self._scope] if self._scope == SCOPE_FLEET else ""
        segments: list[HeaderAtom] = []
        if self._rows:
            # Drawn only when there is something to filter (the desktop's rule):
            # three segments over an empty queue would be a control for nothing.
            for index, value in enumerate(FILTER_ORDER):
                if index:
                    segments.append(HeaderAtom("  ", "muted"))
                segments.append(
                    HeaderAtom(
                        filter_label(value, counts[value]),
                        "active" if value == self._filter else "muted",
                        value,
                    )
                )

        def _seg_block(with_head: bool) -> list[HeaderAtom]:
            """The control, with the gap it only needs when something precedes it."""
            if not segments:
                return []
            gap = [HeaderAtom(self.SEGMENT_GAP, "muted")] if with_head else []
            return [*gap, *segments]

        # THE HEAD'S OWN TWO-RUNG LADDER. The subject is kept only when the row
        # can also carry the CONTROL — the filter is a live destination and a
        # filtered view that cannot be un-filtered is worse than a subject the
        # rows themselves partly say (a fleet row names its conversation). So
        # the order is: subject + drawer count, then subject alone, then
        # nothing; the drawer count goes first because the three segments below
        # it already carry those numbers.
        head: list[HeaderAtom] = []
        # THE CLAUSE YIELDS TO THE WHOLE LIVE HINT SET (design D2 = UX U1). It
        # used to yield to the FIRST hint only, so at the default 100-column
        # terminal an OPEN row reserved room for `enter answer` and then shed
        # `d decline` and `x dismiss` — the one key R5 adds was never named at the
        # widths this repo is used at, while a NARROWER window still named two of
        # them. The clause repeats what the segments below already count, and the
        # dock prints the same sentence in full one line away, so the clause is
        # the thing that goes; the hints are what the ladder exists to keep.
        hints = self.header_hints()
        live_hints = [HeaderAtom("  ·  " + hint, "muted") for hint in hints]
        if clause:
            with_clause = [HeaderAtom(subject, "fg")] if subject else []
            with_clause.append(HeaderAtom(f"{' · ' if subject else ''}{clause}", "fg"))
            if _atoms_width([*atoms, *with_clause, *_seg_block(True), *live_hints]) <= width:
                head = with_clause
        if not head and subject:
            candidate = [HeaderAtom(subject, "fg")]
            if _atoms_width([*atoms, *candidate, *_seg_block(True), *live_hints]) <= width:
                head = candidate
        atoms.extend(head)
        atoms.extend(_seg_block(bool(head)))

        def shed(prefix: str) -> list[HeaderAtom]:
            """As many hints as fit, in order, from the left."""
            kept: list[HeaderAtom] = []
            for hint in hints:
                candidate = HeaderAtom(prefix + hint, "muted")
                if _atoms_width([*atoms, *kept, candidate]) > width:
                    break
                kept.append(candidate)
            return kept

        # THE SEPARATOR COMPACTS BEFORE A HINT IS DROPPED (design D2/UX U1). The
        # measured case is the repo's own 80-column default: the three live hints
        # need 80 cells with the roomy `  ·  ` separator and fit in 78 with the
        # dock's own ` · ` — and 78 is exactly the content width that terminal
        # leaves. A rung, not a second vocabulary: both separators are already in
        # this header, and the compact one is only used when it keeps MORE hints
        # than the roomy one does (never to prettify).
        kept = max((shed("  ·  "), shed(" · ")), key=len)
        atoms.extend(kept)
        if _atoms_width(atoms) > width:
            # The floor: not even the control fits. CUT rather than wrap — a
            # wrapped header is what shifted every pointer hit below it.
            atoms = [HeaderAtom(_clip_cells(_atoms_text(atoms), width), "fg")]
        return atoms

    def header_text(self, width: int) -> str:
        return _atoms_text(self.header_atoms(width))

    def _session_label(self, row: AskRow) -> str:
        """The conversation a FLEET row belongs to, as a short handle.

        The catalogue's own TITLE first, then the working directory's last
        segment, then the session id (round 2: U8/U6). The title is what the
        sidebar and the picker call that conversation, so a fleet row and the
        row the user knows cannot name one session two ways; the cwd is only a
        fallback for a session the catalogue no longer carries (an archived or
        deleted one is still a question someone asked). Clipped, because a
        title is a sentence and this is a tail.

        Empty in session scope, where the row belongs to whatever conversation
        is on screen and naming it would be noise.
        """
        if self._scope != SCOPE_FLEET or not row.session_id:
            return ""
        handle = self._session_titles.get(row.session_id) or (
            row.cwd.rstrip("/").split("/")[-1] or row.session_id
        )
        return _clip_cells(handle, SESSION_HANDLE_CELLS)

    def _empty_slice(self) -> str | None:
        return empty_sentence(self._rows, self._filter)

    def render(self) -> Text:
        # THE STATE INKS OF THIS PANEL ARE THE OVERLAY-SAFE ONES, and that is a
        # correction rather than a preference (design round 2, D1). Everywhere
        # else a state hue is solved against `bg`/`surface`; this panel's ground
        # is `overlay`, and on the brand light ramp the accent lands at 3.49:1
        # and `success` at 3.45:1 there — under the repo's own 4.0 state floor.
        # `theme._fill_chip_live` exists for exactly this ground (the quick-send
        # card is its other client), so the marker, the delivering glyph and the
        # settled chip words all take the derived `chip-*` inks: the hue itself
        # on every ramp that clears, the ramp's neutral ink where it does not.
        # The dark ramp is byte-identical either way (`chip-live` == accent).
        fg = Style(color=theme_mod.semantic_color("fg"))
        muted = Style(color=theme_mod.semantic_color("muted"))
        chip_live = Style(color=theme_mod.semantic_color("chip-live"))
        chip_success = Style(color=theme_mod.semantic_color("chip-success"))
        chip_warning = Style(color=theme_mod.semantic_color("chip-warning"))
        bold = Style(bold=True)
        accent = Style(color=theme_mod.semantic_color("accent"))
        inks = {
            "fg": fg,
            "muted": muted,
            "accent": accent,
            "active": bold + fg,
            "chip-live": chip_live,
        }
        # `content_size`, not `size`: the box the text is really painted in.
        width = max(1, int(self.content_size.width))
        text = Text()
        # `no_wrap` + ellipsis, so even a header too long for one row is CUT
        # rather than wrapped: a wrapped header is what shifted every hit below
        # it (round 2, MAJOR-1).
        header = Text(no_wrap=True, overflow="ellipsis")
        for atom in self.header_atoms(width):
            header.append(atom.text, style=inks.get(atom.ink, muted))
        text.append_text(header)
        visible = self.visible_rows
        sentence = self._empty_slice()
        # ONE PLACE THAT KNOWS WHAT A LINE IS. The list paints the header, then
        # for every ask its own row PLUS (on a stale OPEN ask) the context line
        # SA-D7 added, then the chooser's block while a reason is being picked.
        # `_line_owners` below is the same sequence as data, and the pointer hit
        # test reads IT rather than re-deriving the geometry: this surface already
        # paid for two copies of that arithmetic once (round 3: a click opened the
        # ask below the one under the pointer).
        #
        # `painted` starts TRUE because the header is already in the buffer: this
        # helper only decides the separators AFTER it, and starting false emitted
        # one blank row before the first ask.
        painted = True

        def start_line() -> None:
            nonlocal painted
            if painted:
                text.append("\n")
            painted = True

        if visible or sentence:
            start_line()
        if not visible and sentence:
            # AN EMPTY SLICE OWES A LINE (design round 1, D1 = UX U2): a blank
            # pane under a live filter reads as "there are no asks" over a queue
            # that plainly has some, so each half states its OWN emptiness and
            # names the half holding the rows. It is the one painted line this
            # panel spends on no ask, and it is why a filtered view can never be
            # mistaken for an empty queue.
            # The sentence takes `muted`, not `dim`: `dim` is this repo's
            # micro-label rung, and on the brand light ramp it reads 2.72:1 on
            # this panel's own ground — under even that rung's floor — while
            # this line is a SENTENCE telling the reader where their asks went
            # (design round 2, D1).
            text.append(sentence, style=muted)
        for index, row in enumerate(visible):
            if index:
                start_line()
            # A PASSIVE list marks no row: `❯` says "Enter acts on this one", and the
            # keyboard is not here. The column is kept (two blank cells) so a row's
            # question starts where it will once the list takes the caret, and the
            # frame does not shift when Tab hands it over.
            selected = index == self._index and not self.passive
            line = Text(no_wrap=True, overflow="ellipsis")
            line.append("❯ " if selected else "  ", style=bold if selected else muted)
            chip = settled_chip(row) if row.settled else delivering_chip(row)
            if row.ask_id in self._in_flight:
                # In flight through the engage seam (A7): the row says it is
                # being sent and refuses a second gesture. A WORD, not a
                # spinner — nothing here repaints on a clock, and a moving mark
                # would be the animated affordance §5.0 exists to forbid.
                line.append(f"{IN_FLIGHT_MARK} ", style=muted)
            elif row.delivering:
                # ANSWERED AND NOT YET DELIVERED (§10). It is pending (the user's
                # queue is not finished until the agent has been told) but it is
                # not something to answer, so it wears the delivering glyph
                # rather than an answerable one.
                line.append(f"{DELIVERING_MARK} ", style=chip_success)
            elif row.settled:
                # A SETTLED row has no state GLYPH: the chip word in the tail
                # IS the status (design §4: "a status CHIP in place of the
                # status glyph"). The two cells stay blank rather than being
                # reclaimed, so every row's question starts in the same column
                # and the one-painted-line-per-ask invariant is untouched.
                line.append("  ", style=muted)
            else:
                mark = STATUS_MARKS.get(row.status, "●")
                line.append(f"{mark} ", style=chip_warning if row.urgent else muted)
            # The SELECTED question is `fg` and not `accent`: accent on this
            # panel's ground reads 3.49:1 in the light ramp, under the 4.0 the
            # palette gate sets for a state hue (design D3). Selection is
            # carried by the `❯` marker and the weight instead, which is also
            # the one cue that survives NO_COLOR.
            # THE QUESTION GIVES WAY, not the row's tail, and the cut is made
            # HERE rather than left to the widget's overflow: a row that is
            # wider than the box is hard-cropped by the painter (no glyph), and
            # because the expiry is appended last it was the expiry — the
            # per-row WORD for urgency, which design D13 added so meaning did
            # not rest on hue alone — that fell off the end (agent review round
            # 4, MINOR-1). Clipping the question to the room the tail leaves
            # puts the cut where it belongs and paints the ellipsis that says
            # something was cut.
            #
            # A SETTLED row spends its tail on the STATUS CHIP instead of a
            # countdown (design §4): it has no deadline left to count, and the
            # chip is the row's whole meaning now. Neither the chip nor the tail
            # on a settled row is padded by `expiry_room` — nothing about a
            # settled row can move, which is the one place the tick's "a repaint
            # changes nothing else" promise is free to be exact.
            expiry = "" if chip is not None else expiry_text(row, self._now_ms)
            room = expiry_room(row, expiry) if expiry else 0
            # A FLEET row names its conversation (design §4: the index row
            # carries ``session_id``/``cwd`` precisely because a row for ANOTHER
            # session has to say whose it is). The cwd's own last segment is the
            # most recognisable handle the index holds — a row addressed to
            # another conversation, with nothing naming that conversation, is a
            # question the reader cannot place.
            label = self._session_label(row) if chip is None else ""
            tail = ""
            if expiry or label:
                tail = f"  {f'{label} · ' if label else ''}{expiry}"
                if expiry:
                    # Padded to the widest countdown this row can paint, so the
                    # cut in the question stays put as the clock moves.
                    tail += " " * max(0, room - 2 - cell_len(expiry))
            if chip is not None:
                # A SEPARATOR, not just a gap (design round 2, D6): the muted
                # statuses (`declined`/`dismissed`/`expired`) share the question's
                # own ink, so with two spaces between them the row read as one
                # sentence — `Should the retry budget double?  declined` — and the
                # word the reader has to classify was the one that did not look
                # like a field. `·` is the separator this row's tail already uses
                # for `handle · deadline`, so the status lands in the register the
                # row already has.
                tail = f"  · {chip[0]}"
            fixed = 4 + cell_len(tail)  # two marker cells, two state-glyph cells
            question = _clip_cells(
                row.head_question or f"(ask {row.ask_id})",
                max(1, width - fixed),
            )
            line.append(question, style=(bold + fg) if selected else muted)
            if tail:
                if chip is not None:
                    chip_inks = {"success": chip_success, "warning": chip_warning}
                    line.append(tail, style=chip_inks.get(chip[1], muted))
                else:
                    line.append(tail, style=chip_warning if row.urgent else muted)
            text.append(line)
            # THE STALE-ASK CONTEXT LINE (SA-D7/R5), on the row it belongs to and
            # nowhere else: indented under the question, in the muted register —
            # it is context for the question, never a second ask. It leads with a
            # `└` connector (design D6): the same indent is also where a LAPSED
            # ask's collapsed row sits, so without a cue of its own the slot meant
            # two things at once. `└` is this surface's existing "child of the row
            # above" glyph (the tree sections and the analytics panel use it), and
            # no ask row ever starts with it.
            context = context_text(row)
            if context:
                for number, piece in enumerate(
                    _wrap_cells(f"{CONTEXT_CUE}{context}", max(1, width - SUB_LINE_INDENT))
                ):
                    start_line()
                    sub = Text(no_wrap=True, overflow="ellipsis")
                    sub.append(" " * SUB_LINE_INDENT, style=muted)
                    if number:
                        # The continuation lines align under the words, not under
                        # the connector: the cue marks the block once.
                        sub.append(" " * cell_len(CONTEXT_CUE), style=muted)
                    sub.append(piece, style=muted)
                    text.append(sub)
            # THE CHOOSER SITS UNDER THE ROW IT IS ABOUT, never at the foot of the
            # list: with three asks on screen a reason card painted below the last
            # one reads as a question about THAT ask, and the operator has to hunt
            # for which row it belongs to. It is a sub-block of one row for the
            # same reason the context line is.
            if self._dismiss_for == row.ask_id:
                for line in self._chooser_lines(width, fg, muted, bold):
                    start_line()
                    text.append(line)
        return text

    def _chooser_lines(self, width: int, fg: Style, muted: Style, bold: Style) -> list[Text]:
        """The dismiss reason block: its heading, two choices, its key hints.

        No question text in the heading: the block sits DIRECTLY under the row it
        is about (`render`), so repeating the question one line below the row
        that asks it would be the same sentence twice — the distinction §5's
        preamble draws between stating a fact and padding a screen.

        THE NOTES ARE PAINTED RATHER THAN LEFT TO THE READER'S MEMORY: `moot` and
        `answered_in_chat` both end the agent's wait and differ only in what the
        agent is told, which is precisely the distinction a bare label cannot
        carry (design D2's own argument for `x dismiss` over the row's glyph).

        EACH CHOICE IS TWO PAINTED LINES — the label, then its note in the muted
        register one line down (design round 1, U4 = D1's shape). They used to
        share one row (`Answered in chat — you already replied in the conversation`)
        and clipped MID-WORD against a panel with the sidebar up, cutting the
        clause that carries the distinction. Two lines also give the geometry one
        fixed shape at every width, which is what lets `_line_owners` stay a
        function of the width alone.
        """
        head = _a(_asks.dismiss_title())
        out = [
            Text(
                " " * SUB_LINE_INDENT + _clip_cells(head, max(1, width - SUB_LINE_INDENT)),
                style=fg,
                no_wrap=True,
                overflow="ellipsis",
            )
        ]
        room = max(1, width - SUB_LINE_INDENT - 2)
        for index, (_, label, note) in enumerate(DISMISS_CHOICES):
            selected = index == self._dismiss_index
            line = Text(no_wrap=True, overflow="ellipsis")
            line.append(" " * SUB_LINE_INDENT, style=muted)
            line.append("❯ " if selected else "  ", style=bold if selected else muted)
            line.append(
                _clip_cells(_a(label()), room),
                style=(bold + fg) if selected else muted,
            )
            out.append(line)
            explanation = Text(no_wrap=True, overflow="ellipsis")
            explanation.append(" " * (SUB_LINE_INDENT + 2), style=muted)
            explanation.append(_clip_cells(_a(note()), room), style=muted)
            out.append(explanation)
        out.append(
            Text(
                " " * SUB_LINE_INDENT + _a(_asks.dismiss_hint()),
                style=muted,
                no_wrap=True,
                overflow="ellipsis",
            )
        )
        return out


def context_text(row: AskRow) -> str:
    """The stale-ask context line for ``row``, or ``""`` when it has none.

    SA-D7's line ("3 messages from you since this was asked (last 19:36)"),
    drawn under an OPEN row that has admitted operator messages since it was
    asked. It is the ONE thing on this surface that says a question may already
    have been answered in chat, so it is painted as its own line rather than
    competing with the question and its countdown for the same row.

    The clock is the CLIENT's, formatted here and passed as a finished string:
    ICU has no business reformatting a wall-clock time, and a locale that wants
    a different position for it reorders the sentence in its own catalogue (the
    time lives inside both plural arms for exactly that reason).
    """
    stamp = row.context_stamp
    if stamp is None:
        return ""
    count, last_ms = stamp
    when = datetime.fromtimestamp(last_ms / 1000).astimezone().strftime("%H:%M")
    return _a(_asks.context_messages(count=count, time=when))


def expiry_text(row: AskRow, now_ms: int) -> str:
    """``expires in 42 m`` / ``timed out`` / ``urgent`` — one honest word per state.

    Derived from ``expires_at`` against the client's own clock, which is what
    §5 says the countdown is: the server states a deadline, the client decides
    how long that is from here. A row with no deadline says nothing rather than
    ``in 0s``.

    URGENCY CARRIES WORDS, not only the amber glyph (design D8): the bar's `?`
    turning amber is invisible to a reader who cannot see the hue and to a
    transcript, so the row says ``urgent`` too.
    """
    if row.status == STATUS_TIMED_OUT:
        # Reached only by a caller painting a lapsed row's tail itself: the list
        # paints the settled chip instead (D11), and neither may promise an answer.
        return "timed out"
    if not row.expires_at or not now_ms:
        return "urgent" if row.urgent else ""
    remaining_s = int((row.expires_at - now_ms) // 1000)
    if remaining_s <= 0:
        return "urgent · expiring" if row.urgent else "expiring"
    if remaining_s < 60:
        left = f"{remaining_s} s"
    elif remaining_s < 3600:
        left = f"{remaining_s // 60} m"
    else:
        left = f"{remaining_s // 3600} h"
    return f"urgent · expires in {left}" if row.urgent else f"expires in {left}"


def expiry_room(row: AskRow, expiry: str) -> int:
    """Cells a row's ``  <expiry>`` TAIL may need, over its whole life.

    The tail CHANGES WIDTH as the clock runs — ``10 m`` is one cell wider than
    ``9 m``, the words change unit at the hour and minute boundaries, and
    ``expiring`` replaces them all — so a question that filled its line got
    RE-CUT by its own countdown: ``…which shard …`` became ``…which shard s…``
    without the user touching anything, on a surface whose whole promise is
    that a tick repaints and changes nothing else (design round 1, D2).
    Reserving the widest form the row can reach keeps the question's clip
    budget fixed, which is what makes the tick a pure repaint.

    Derived from the row's OWN span (``expires_at - created_at``, the deadline
    the ask was given) rather than a global constant: a constant wide enough
    for ``expires in 999 h`` would eat ten cells of question from every ask
    forever. ``expiry_text`` prints minutes and seconds below an hour — always
    at most two digits — so only the hours form can be wider, and the digits
    it can reach are the ones the row's own span implies.

    The returned width includes the two separator cells, and it never returns
    less than what ``expiry`` already needs: a client clock skewed far ahead
    of the wire's can paint a longer word than the span implies, and a tail is
    the one thing on this row that must not be cropped (the expiry and its
    urgency word are the meaning the hue alone cannot carry).
    """
    if row.status == STATUS_TIMED_OUT:
        widest = cell_len("timed out")
    else:
        prefix = "urgent · expires in " if row.urgent else "expires in "
        span_s = (row.expires_at - row.created_at) // 1000 if row.created_at else 0
        widest = cell_len(prefix) + max(2, len(str(max(0, span_s) // 3600))) + 2
    return 2 + max(widest, cell_len(expiry))


__all__ = [
    "ASK_BAR_CHEVRON_COLLAPSED",
    "ASK_BAR_CHEVRON_EXPANDED",
    "ASK_MARKER",
    "ASK_TOGGLE_KEY",
    "SETTLED_STATUSES",
    "STATUS_MARKS",
    "STATUS_OPEN",
    "STATUS_TIMED_OUT",
    "STATUS_WITHDRAWN",
    "AskBar",
    "AskQueueList",
    "AskRow",
    "ask_rows",
    "expiry_room",
    "expiry_text",
    "queue_headline",
]
