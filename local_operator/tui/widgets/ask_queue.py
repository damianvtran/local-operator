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
from typing import Any, Iterable, Mapping, Sequence

from rich.cells import cell_len
from rich.style import Style
from rich.text import Text
from textual import events
from textual.binding import Binding
from textual.message import Message
from textual.widget import Widget

from local_operator.asks import store
from local_operator.tui import theme as theme_mod

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
#: be refused. ``expired`` injects nothing at all (§2.2).
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
    }
)

#: `Moved on` is the drawer's own word for a row whose deadline passed but
#: which is still answerable (the fold's ``timed_out``) — §5's copy has said
#: "the agent moved on" since the first surface, and the filter's middle label
#: ("Waiting or moved on") only reads honestly if both halves are named the
#: same way here. The BAR still says "timed out": it is the chip register
#: (``queue_headline``), and the two registers are deliberately different
#: statements about one queue (see ``drawer_headline``).
STATUS_DELIVERING = frozenset({STATUS_ANSWERED, STATUS_LATE})

# ---------------------------------------------------------------------------
# The three-way filter (design §4, amendment A5)
# ---------------------------------------------------------------------------

FILTER_ALL = "all"
FILTER_OUTSTANDING = "outstanding"
FILTER_SETTLED = "settled"

#: Press order, left to right, and the cycle order for `[`/`]`.
FILTER_ORDER: tuple[str, ...] = (FILTER_ALL, FILTER_OUTSTANDING, FILTER_SETTLED)

#: EXACT labels, from the desktop's own filter control (``ask-panel.tsx``).
#: The middle one is deliberately NOT "Waiting": a moved-on ask is in that
#: half and the agent is not waiting for it, so the short word would be a
#: second name for a population the row itself spells "moved on" (design
#: round 1, M1 = UX U1 on the desktop — the same finding, same wording).
FILTER_LABELS: dict[str, str] = {
    FILTER_ALL: "All",
    FILTER_OUTSTANDING: "Waiting or moved on",
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
EMPTY_OUTSTANDING = "No asks are waiting or moved on. They have all settled — see Settled."
EMPTY_SETTLED = "No asks have settled yet. They are all still under Waiting or moved on."

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
}
#: ...which is to say: everything the reader must not be asked to answer. THE
#: OUTSTANDING SET, taken from the fold rather than re-listed here: a row the
#: user can still answer is ``open`` or timed out and unanswered, and that ONE
#: rule decides the wire's tally, this surface's rows and the sidebar mark, so
#: it lives in ``asks.store.OUTSTANDING_STATUSES`` and nowhere else.
_ANSWERABLE = store.OUTSTANDING_STATUSES

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
#: ``●`` open, ``◷`` timed out but still answerable (the wake glyph's meaning —
#: "a clock decided something"). A SETTLED row paints no glyph and no fallback
#: bit of punctuation either: its status is a WORD in the tail
#: (``settled_chip``), because once there is nothing left to act on the status
#: IS the row.
STATUS_MARKS: dict[str, str] = {
    STATUS_OPEN: "●",
    STATUS_TIMED_OUT: "◷",
}

#: The glyph an ANSWERED-but-undelivered row wears in the list's status column
#: (design §10's delivering half): the answer is recorded and the response has
#: not reached the transcript yet, so the row is still PENDING — it is not
#: something to answer, but it is the reason the pending count is honest.
DELIVERING_MARK = "▸"

#: The glyph a row wears while an answer for it is IN FLIGHT through the engage
#: seam (fleet scope only — the current session's ops are synchronous). Sized
#: for the phone's own window (engage 30 s + ack 15 s), it is not a spinner:
#: nothing repaints on a clock here, and a row that looked busy would be the
#: animated affordance §5.0 forbids. It exists so the row cannot be fired twice.
IN_FLIGHT_MARK = "…"


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

    @property
    def question_count(self) -> int:
        return len(self.questions)

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
        """Whether the user can still do something about this row."""
        return self.status in _ANSWERABLE

    @property
    def moved_on(self) -> bool:
        """Whether the agent walked past this row's deadline but can still hear it.

        The drawer's half of the outstanding set: a timed-out ask is not
        something the agent waits on any more (see :attr:`waiting`) and it is
        not settled either — a late answer is still attributable, which is the
        whole reason it stays outstanding.
        """
        return self.status == STATUS_TIMED_OUT

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

        Narrower than :attr:`answerable`, and the distinction is the copy's: a
        timed-out ask is answerable (the user may still reply, and the agent is
        told) but nobody is waiting on it — the agent moved on.
        """
        return self.status == STATUS_OPEN


def _value(row: Any, key: str) -> Any:
    """Read one field from a wire model OR a plain mapping."""
    if isinstance(row, Mapping):
        return row.get(key)
    return getattr(row, key, None)


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
    OUTSTANDING set (``open``/``timed_out``, §4.8) unioned with the current
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
        noun = "question" if open_count == 1 else "questions"
        parts.append(f"{open_count} {noun} waiting")
    if urgent_count:
        # URGENCY IN WORDS, because the bar's amber glyph is a hue and a hue is
        # not a channel a reader without colour has: D8's own principle, which
        # had reached the list row and not the bar (design round 2, D13). It
        # also removes the mismatch D13 found — the glyph is painted from
        # `any(row.urgent)`, so with no word for it the amber annotated the ONE
        # question the bar happened to name, which need not be the urgent one.
        parts.append(f"{urgent_count} urgent")
    if timed_out_count:
        noun = "ask" if timed_out_count == 1 else "asks"
        parts.append(f"{timed_out_count} {noun} timed out")
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

    So a MIXED queue spells both halves here, in the drawer's own words, and
    the single-half cases name that half in the DRAWER's vocabulary — "moved
    on", never the chip's "timed out" — because the segments above and the rows
    below this line already say "moved on" for that half, and one surface must
    not carry two words for one state (round 2: F3/D4). The bar keeps the chip
    clause; the two registers are named in §11.

    THE DELIVERING HALF IS PENDING AND MUST NEVER READ AS SETTLED (round 2:
    F3/Q1/U4). ``pending`` = answerable ∪ delivering (§10), so an
    answered-but-undelivered row sits in the "Waiting or moved on" segment while
    the old fallback printed `N settled` over it — the header contradicting its
    own segment. It says so in §5's words instead, and without the desktop's
    "you can still change it": the TUI has no revise wire at all, and promising
    a door that does not exist is worse than the missing sentence. The
    all-settled case takes the desktop drawer's own constant (`All asks
    settled`) rather than inventing a second form for it.

    THE TRUNCATED FRAME STATES THE BACKEND'S TALLY (``open_count``) AND
    WITHHOLDS THE SPLIT: the wire caps the list, so a waiting/moved-on split
    computed from the rows this frame carries would be a prefix passing for the
    whole queue. This is the desktop's `askSplitIsKnowable` rule ("the louder
    number wins"); `open_count is None` means the caller has no backend tally to
    state, and then the row-derived split is all there is — which is exactly the
    session-scope case, where the wire carries every ask the session has.
    """
    if truncated:
        return f"{open_count or 0} outstanding"
    waiting = sum(1 for row in rows if row.waiting)
    moved_on = sum(1 for row in rows if row.moved_on)
    delivering = sum(1 for row in rows if row.delivering)
    # OPEN rows only, the same set the bar counts (review round 3, MINOR-1): an
    # urgent ask is one whose deadline is imminent, and a timed-out ask has no
    # imminent deadline left — counting it here made the list say `1 urgent · 1
    # ask timed out` over a bar showing only the amber hue.
    urgent = sum(1 for row in rows if row.waiting and row.urgent)
    if waiting and moved_on:
        clause = f"{waiting} waiting, {moved_on} moved on"
        # THE URGENCY WORD IS NOT PART OF THE MIXED CLAUSE and must not be lost
        # with it: the drawer's own sentence replaced the chip's, and the chip's
        # was the only place urgency was ever stated in words (D13). It is
        # appended rather than interleaved so the half-counts stay a prefix.
        clause = f"{clause} · {urgent} urgent" if urgent else clause
    elif moved_on:
        noun = "question" if moved_on == 1 else "questions"
        clause = f"{moved_on} {noun} moved on"
    else:
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
        of nothing but TIMED-OUT asks owes nobody a wait (the agent moved on),
        but those asks are still answerable (§5's copy), so a bar hidden by
        ``count == 0`` would make them unreachable from the TUI.
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
        glyph_style = Style(bold=True) + (warning if self._urgent else accent)

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
        base = f"{ASK_MARKER} {queue_headline(self._count, self._timed_out, self._urgent_count)}"
        verb = "collapse" if self._expanded else "answer"
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
        text.append(
            f" {queue_headline(self._count, self._timed_out, self._urgent_count)}", style=fg
        )
        if head_part:
            text.append(head_part, style=fg)
        if tail:
            text.append(tail, style=dim)
        padding = width - 1 - cell_len(text.plain)
        if padding > 0:
            text.append(" " * padding)
        text.append(chevron, style=Style(bold=True) + accent)
        return text


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
        """The user dismissed one TIMED-OUT ask from the view."""

        def __init__(self, ask_id: str) -> None:
            super().__init__()
            self.ask_id = ask_id

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
        Binding("2", "filter_outstanding", "Waiting or moved on", show=False),
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
    #: kept longest, so the irreversible `d` outlives the reversible tips when
    #: the row runs out of room (UX round 1, U7 asked for `d` to be named
    #: precisely because it cannot be undone).
    #:
    #: `d` LEADS since review round 2 (D3): the spend order is by
    #: IRREVERSIBILITY, and `enter` already carries its own cue on the row (the
    #: `❯` caret marks the selected one), so at a width where only one hint fits
    #: it has to be the hint with no other teacher. The base named `d decline`
    #: at 100x30 and the first cut of this header had silently stopped doing so.
    HEADER_HINTS = (
        "d decline",
        "enter answer",
        "x dismiss",
        "esc collapse",
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
    ) -> None:
        super().__init__(id=widget_id, classes="prompt-slot")
        self._rows: list[AskRow] = list(rows)
        self._index = 0
        self._now_ms = now_ms
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
        """Select one half. The highlight follows the ROW, not the index."""
        if value not in FILTER_LABELS or value == self._filter:
            return
        current = self.current()
        self._filter = value
        self._reindex(current.ask_id if current else "")
        self.refresh(layout=True)

    def cycle_filter(self, delta: int) -> None:
        """``[``/``]``: walk all → waiting-or-moved-on → settled → all."""
        at = FILTER_ORDER.index(self._filter)
        self.set_filter(FILTER_ORDER[(at + delta) % len(FILTER_ORDER)])

    def set_in_flight(self, ask_id: str, active: bool) -> None:
        """Paint ``ask_id`` as being sent (or stop painting it).

        Called by the app around a fleet answer: the engage-and-dial window is
        the phone's 30 s + 15 s, and the row must both SAY it is in flight and
        REFUSE a second gesture for that window — the mark and the guard are the
        same state, deliberately, so they cannot disagree.
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

    def action_move(self, delta: int) -> None:
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
        row = self.current()
        if row is not None and row.answerable and row.ask_id not in self._in_flight:
            self.post_message(self.Picked(row.ask_id))

    def action_collapse(self) -> None:
        self.post_message(self.Collapse())

    def action_decline(self) -> None:
        row = self.current()
        if row is not None and row.answerable and row.ask_id not in self._in_flight:
            self.post_message(self.Decline(row.ask_id))

    def action_dismiss(self) -> None:
        row = self.current()
        if row is not None and row.answerable and row.ask_id not in self._in_flight:
            self.post_message(self.Dismiss(row.ask_id))

    def action_filter_all(self) -> None:
        self.set_filter(FILTER_ALL)

    def action_filter_outstanding(self) -> None:
        self.set_filter(FILTER_OUTSTANDING)

    def action_filter_settled(self) -> None:
        self.set_filter(FILTER_SETTLED)

    def action_filter_next(self) -> None:
        self.cycle_filter(1)

    def action_filter_prev(self) -> None:
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

    def _row_at(self, y: int) -> int | None:
        """The VISIBLE-row index under a widget-relative y, or ``None``.

        Both offsets are MEASURED: the widget's own top inset, and a header that
        is one painted line by construction (``header_atoms`` never wraps, and
        ``HEADER_HINTS`` is spent rather than broken mid-phrase). The index is
        into ``visible_rows`` — the rows the current filter draws — so a filtered
        view can never open a row the user cannot see.
        """
        index = y - self._top_inset() - self.HEADER_ROWS
        if 0 <= index < len(self.visible_rows):
            return index
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
        y = int(event.y)
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
        if target.answerable and target.ask_id not in self._in_flight:
            self.post_message(self.Picked(target.ask_id))

    def on_mouse_move(self, event: events.MouseMove) -> None:  # type: ignore[override]
        """Hover moves the highlight, so the row under the pointer is the one
        a click would take — the picker card's own rule."""
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

        So the set is derived from the VISIBLE rows (the filter half on screen)
        and the keys' own objects: `enter`/`d` need an ANSWERABLE row, `x` needs
        a moved-on one (dismiss is only ever offered on a timed-out ask, so
        advertising it over open rows names a key the row would refuse), and
        `esc` needs nothing — it is how a reader leaves any view, including an
        empty half. Order is still ``HEADER_HINTS``' own.
        """
        visible = self.visible_rows
        live = {"esc collapse"}
        if any(row.answerable for row in visible):
            live |= {"enter answer", "d decline"}
        if any(row.moved_on for row in visible):
            live.add("x dismiss")
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
        * The HINTS yield first, left to right, whole and never mid-phrase —
          design round 2's D12, which is what stops the header wrapping and
          every pointer hit below it shifting by a row.

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
        if clause:
            with_clause = [HeaderAtom(subject, "fg")] if subject else []
            with_clause.append(HeaderAtom(f"{' · ' if subject else ''}{clause}", "fg"))
            if _atoms_width([*atoms, *with_clause, *_seg_block(True)]) <= width:
                head = with_clause
        if not head and subject:
            candidate = [HeaderAtom(subject, "fg")]
            if _atoms_width([*atoms, *candidate, *_seg_block(True)]) <= width:
                head = candidate
        atoms.extend(head)
        atoms.extend(_seg_block(bool(head)))
        kept: list[HeaderAtom] = []
        for hint in self.header_hints():
            prefix = "  ·  "
            if _atoms_width([*atoms, *kept, HeaderAtom(prefix + hint, "muted")]) > width:
                break
            kept.append(HeaderAtom(prefix + hint, "muted"))
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
        if visible or sentence:
            text.append("\n")
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
            selected = index == self._index
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
            if index + 1 < len(visible):
                text.append("\n")
        return text


def expiry_text(row: AskRow, now_ms: int) -> str:
    """``expires in 42m`` / ``timed out`` / ``urgent`` — one honest word per state.

    Derived from ``expires_at`` against the client's own clock, which is what
    §5 says the countdown is: the server states a deadline, the client decides
    how long that is from here. A row with no deadline says nothing rather than
    ``in 0s``.

    URGENCY CARRIES WORDS, not only the amber glyph (design D8): the bar's `?`
    turning amber is invisible to a reader who cannot see the hue and to a
    transcript, so the row says ``urgent`` too.
    """
    if row.status == STATUS_TIMED_OUT:
        return "timed out — still answerable"
    if not row.expires_at or not now_ms:
        return "urgent" if row.urgent else ""
    remaining_s = int((row.expires_at - now_ms) // 1000)
    if remaining_s <= 0:
        return "urgent · expiring" if row.urgent else "expiring"
    if remaining_s < 60:
        left = f"{remaining_s}s"
    elif remaining_s < 3600:
        left = f"{remaining_s // 60}m"
    else:
        left = f"{remaining_s // 3600}h"
    return f"urgent · expires in {left}" if row.urgent else f"expires in {left}"


def expiry_room(row: AskRow, expiry: str) -> int:
    """Cells a row's ``  <expiry>`` TAIL may need, over its whole life.

    The tail CHANGES WIDTH as the clock runs — ``10m`` is one cell wider than
    ``9m``, the words change unit at the hour and minute boundaries, and
    ``expiring`` replaces them all — so a question that filled its line got
    RE-CUT by its own countdown: ``…which shard …`` became ``…which shard s…``
    without the user touching anything, on a surface whose whole promise is
    that a tick repaints and changes nothing else (design round 1, D2).
    Reserving the widest form the row can reach keeps the question's clip
    budget fixed, which is what makes the tick a pure repaint.

    Derived from the row's OWN span (``expires_at - created_at``, the deadline
    the ask was given) rather than a global constant: a constant wide enough
    for ``expires in 999h`` would eat ten cells of question from every ask
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
        widest = cell_len("timed out — still answerable")
    else:
        prefix = "urgent · expires in " if row.urgent else "expires in "
        span_s = (row.expires_at - row.created_at) // 1000 if row.created_at else 0
        widest = cell_len(prefix) + max(2, len(str(max(0, span_s) // 3600))) + 1
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
