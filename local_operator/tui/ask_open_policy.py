"""When the queued-ask surface opens BY ITSELF — the TUI half of the shared open policy.

Design: ``docs/design/ask-nonblocking.md`` §5.0, which this NARROWS. Its rule that the
EXPANDED state is "never automatic on ask arrival" stays true for the case it was
written about (an ask that ARRIVES while the conversation is on screen); this module
governs the other moment, a conversation being OPENED with asks already waiting.

WHY THIS EXISTS. The minimized bar above the composer is easy to miss, and a
first-time user who opens a conversation with questions waiting on them may never
discover that they are there. The operator asked for the primary ask interaction
of EVERY surface to be open by default when a conversation with pending asks is
opened. §5.0's original rule — the answer surface is "never automatic on ask
arrival" — is narrowed, not repealed: it still holds for an ask that ARRIVES while
the conversation is on screen. What changes is the moment a conversation is
OPENED.

THE SHARED OPEN-POLICY CONTRACT. The desktop UI's drawer, this TUI's ask list, the
mobile relay's ``AsksSheet`` and the native app's sheet implement the same six
clauses, so a conversation's asks behave alike wherever it is opened. They are
numbered here once and cited by number where each is kept:

1. **No asks on open → closed**, as before.
2. **Pending asks on open → auto-open ONCE** for that view of that conversation.
   A "view" begins when the conversation is opened or switched to.
3. **Everything already addressed on open → closed**, and a view that decided
   "closed" never auto-opens afterwards: a new ask arriving later is covered by the
   bar/chip/dock, which is the discoverability this feature is not meant to replace.
4. **A deliberate close while asks remain is respected.** It records the ASK IDS
   it waved off (the pending set at close time) against the conversation id, in
   memory, and survives re-renders, queue refreshes, asks arriving or changing,
   and switching away and back within the same app lifetime. It is FORGOTTEN
   once none of those ids is still outstanding (answered, declined, withdrawn,
   expired): from then on the conversation is a fresh one and its next batch of
   pending asks opens afresh. A fresh app start may open it again. Goal:
   discoverability, not insistence.
5. **Never steal the keyboard, never trap, never open on a guess.** Auto-open does
   nothing while the user is typing or another surface owns the screen, the user
   can always close it, and a frame that does not carry the rows (an unsupported
   runtime, a not-yet-loaded view, a tally-only frame) is NOT "pending asks".
6. **Auto-open is not a user pressing the door.** The door (the bar, ``f4``, the
   fleet note) keeps its own semantics; opening by policy is a separate path that
   shares the surface but not the gesture.

WHY THE DISMISSAL IS KEYED BY ASK ID, NOT BY CONVERSATION. A bare "this
conversation was dismissed" flag is sticky for the whole process, so a user who
waved off two questions, left, and came back after the agent had answered both
and asked a fresh pair would find the new pair hidden behind a refusal that was
about the OLD ones. The fix has to survive the one case that cannot be observed:
while the user is away the conversation is not on screen, so no frame for it is
ever seen and "I saw the queue empty" can never be witnessed. Ids make the rule a
pure function of the first frame on return — "is anything I waved off still
outstanding?" — so an emptied-and-refilled queue is told apart from an unchanged
one with no observation in between. A NEW ask arriving beside an old dismissed one
does not force it open (the bar covers it); it only stops being held once the old
ones are gone.

WHAT THIS MODULE IS. A pure decision function over a handful of facts plus a tiny
in-memory record of which view is current, whether it has decided, and which
conversations were dismissed. It imports nothing from Textual or from the app, so
the whole matrix is unit-testable without booting a UI, and so the app's wiring
stays a few lines of glue rather than a second copy of the rules.

"PENDING ON OPEN" MEANS THE ASK EXISTED BEFORE THE VIEW BEGAN. The first frame of a
view often cannot say what the queue holds (a conversation resumed in this process
only builds its queue once it is engaged), so the decision has to wait for the
first frame that can. Without a second test, the first question a brand-new
conversation asks — landing on that same first resolved frame — would read as
"pending on open" and pop the panel over a user who is watching the agent work,
which is the arrival §5.0 forbids. :func:`read_queue` therefore compares each
outstanding ask's own ``created_at`` with the instant the view began.

WHY A TIME WINDOW TOO. Waiting for a resolved frame needs a bound, or a view that
never gets one would open at minute ten over whatever the user is doing then. The
bound is the engage seam's own: a cold conversation is brought up within 30 s and
acknowledged within 15 s, so 45 s covers "the queue only exists once it is
engaged" and nothing past it can honestly be called "on open".
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from enum import Enum

#: How long after a view begins it may still choose to open (clauses 2 and 5). The
#: engage seam's own bound — 30 s to bring a cold runtime up plus 15 s to be
#: acknowledged — so a conversation whose queue only exists once it is engaged is
#: still "pending on open", while a frame that resolves well into the view is not.
OPEN_WINDOW_S = 45.0

#: THE TEST AND CAPTURE SEAM, in the idiom ``asks.policy.NONBLOCKING_ASK`` set: a
#: module constant read when a policy is built, flipped by a test or an evidence
#: capture and by nothing else. The product never changes it and there is no
#: environment variable behind it — the operator asked for the behaviour, not for
#: a switch. It exists because most of the ask-surface suite is about an ask
#: ARRIVING in a conversation that is already open (the minimized default, the
#: door-opened surfaces), and its fixtures are dated before the view they are fed
#: into, which this policy rightly reads as "pending on open".
AUTO_OPEN = True

#: How far past the instant a view began an ask's ``created_at`` may sit and still
#: count as having existed on open. ``created_at`` is stamped by the machine that
#: OWNS the queue, and a conversation viewed across the mesh is compared against
#: this machine's clock, so a skew of a few seconds would turn a genuinely pending
#: ask into an "arrival" and leave the panel shut. NTP keeps peers well inside
#: this; the cost of the other direction (a real arrival in the view's first
#: seconds read as pending) is a panel opening a moment after the conversation did,
#: which is the behaviour being asked for anyway.
ARRIVAL_SKEW_MS = 5000


class QueueReading(Enum):
    """What ONE frontend snapshot says about a conversation's queue, at view open."""

    #: No usable rows: the runtime does not publish the queue (absence is the
    #: capability proxy), the view has not loaded it yet, or the frame carried only
    #: the tally because the wire bound dropped the rows. NOT "no asks" and NOT
    #: "pending asks" — a decision made on it would be a guess (clause 5).
    UNRESOLVED = "unresolved"
    #: A live queue with nothing in it (``asks_open: 0``): clause 1.
    EMPTY = "empty"
    #: Rows are published and every one is already addressed: clause 3.
    SETTLED = "settled"
    #: Asks are outstanding, but every one was created AFTER the view began: they
    #: arrived, they were not pending on open (clause 3's "a new ask arriving
    #: later does not force it open").
    ARRIVED = "arrived"
    #: At least one outstanding ask existed before the view began: clause 2.
    PENDING = "pending"


class OpenDecision(Enum):
    """What the app should do about the surface for this frame."""

    #: Mount the surface now. Returned at most once per view.
    OPEN = "open"
    #: Nothing yet — the queue is not resolved. Ask again on the next frame.
    WAIT = "wait"
    #: Leave it closed. The view has decided; no later frame reopens the question.
    SKIP = "skip"


def read_queue(
    *,
    published: int,
    outstanding_created_ms: Sequence[int],
    tally: int | None,
    opened_at_ms: int,
) -> QueueReading:
    """Classify a snapshot's ask fields against the instant the view began.

    ``published`` is how many ROWS the frame carried; ``outstanding_created_ms``
    the ``created_at`` of each of them that can still be answered; ``tally`` the
    wire's own ``asks_open`` — ``None`` when the runtime does not publish it,
    ``0`` for a live queue with nothing to fold, ``N`` for N outstanding asks.

    The rows decide whenever they name an outstanding ask. With none, only an explicit
    ``0`` tally is an answer ("nothing is queued"); ``None`` (unsupported) and
    ``N > 0`` (the asks exist but their rows were dropped to fit the frame) both leave
    the question open, because neither one lets a surface draw anything — opening on
    the tally alone would mount an empty list over a queue that has asks. The same
    holds for rows that are ALL settled beside a positive tally: the outstanding asks
    are the ones the bound dropped.

    A ``created_at`` of ``0`` (the wire's default for a row that did not carry one)
    is "older than anything", so such a row is pending on open rather than an
    arrival: the safe reading of a fact that was not stated is the one that matches
    what the row looks like, an ask that has been there a while.
    """
    if published > 0:
        if not outstanding_created_ms:
            # Rows, and none of them outstanding. That is "everything addressed" only if
            # the tally agrees: ``asks_open`` counts outstanding asks BEFORE the wire's
            # text bound drops rows, and the drop order keeps newer settled rows over an
            # older timed-out one, so a frame can carry five settled rows beside
            # ``asks_open: 1`` — a pending ask it cannot show. Calling that SETTLED would
            # spend the view's one decision on a claim the frame contradicts (clause 5:
            # not on a guess), so it waits like any frame that cannot say. A runtime that
            # publishes no tally cannot contradict its own rows.
            if tally is not None and tally > 0:
                return QueueReading.UNRESOLVED
            return QueueReading.SETTLED
        cutoff = opened_at_ms + ARRIVAL_SKEW_MS
        if any(created <= cutoff for created in outstanding_created_ms):
            return QueueReading.PENDING
        return QueueReading.ARRIVED
    if tally == 0:
        return QueueReading.EMPTY
    return QueueReading.UNRESOLVED


def names_every_outstanding(*, tally: int | None, named_outstanding: int) -> bool:
    """Whether a frame's rows name EVERY outstanding ask — what a dismissal is judged on.

    A dismissal is forgotten only on evidence that every ask the user waved off has
    left the queue, and that evidence is "the frame names all the outstanding asks and
    none of them is one of mine". A frame that cannot name them all proves nothing: a
    dropped row may be exactly the one still outstanding.

    THE TEST IS THE TALLY, NOT ``asks_truncated``, and the flag is deliberately
    IGNORED. ``asks_truncated`` is set whenever the wire's text budget drops ANY row —
    including an already-ANSWERED one — and it stays set after every ask is answered,
    for as long as those dropped settled rows live (up to the 7-day horizon). A
    predicate of the shape ``not truncated and ...`` can therefore never become true
    again on that conversation, and the surface stays muted for the life of rows
    nobody can act on. Measured on the shipped wire (8 long asks, then all 8 answered):
    an attached viewer reads ``rows=5, asks_open=0, asks_truncated=True``.

    ``asks_open`` counts OUTSTANDING rows BEFORE the text bound drops any (it is
    summed over the same fold the rows are cut from), so a dropped outstanding row
    always shows as ``tally > named``, and it falls to what is really outstanding the
    moment the asks are answered, which the sticky flag does not.

    ``tally is None`` is the frame that cannot attest (a runtime that predates the
    field, or a last-resort yield that dropped it): not complete, so the caller holds.
    Failing closed here can only ever under-open, never override a refusal.

    ONE RESIDUAL LIMIT, and it is not closable on the client. The runtime projects at
    most ``PROJECTION_CAP`` (20) rows BEFORE it computes the tally, so past 20
    outstanding asks (reachable only by timed-out asks piling up: the open cap is 8)
    the surplus is on neither field and no client can name it. The residue is one
    extra auto-open — the ids the user waved off were all visible to them, so they
    leave the queue, the dismissal is forgotten and the surplus reads as a fresh
    batch — which is harmless: it is the surface the feature exists to show.
    """
    return tally is not None and tally <= named_outstanding


class AskOpenPolicy:
    """The in-memory record behind the open policy — one per app.

    Three pieces of state and nothing else: the CURRENT VIEW (which conversation is
    on screen and when it began), whether that view has taken its one decision, and
    the dismissals — for each conversation the user waved off, WHICH ASKS they
    waved off. All of it dies with the process, which is clause 4's "a fresh app
    start may auto-open again" — persisting a dismissal would turn a courtesy into
    a setting nobody asked for.
    """

    def __init__(self, *, window_s: float = OPEN_WINDOW_S, enabled: bool | None = None) -> None:
        self._window_s = window_s
        #: ``False`` makes the policy inert: no view ever arms. ``None`` follows
        #: :data:`AUTO_OPEN`, read NOW so a test that flips the constant before
        #: building its app is honoured.
        self._enabled = AUTO_OPEN if enabled is None else enabled
        #: conversation id -> the ask ids the user waved off by closing the surface
        #: while they were pending (clause 4). Evaluated and forgotten in
        #: :meth:`decide`, the only place the queue's current ids are known.
        self._dismissed: dict[str, frozenset[str]] = {}
        self._view = ""
        self._opened_at = 0.0
        #: True until a view arms, and again once it has decided. A view with no
        #: decision left to take is inert: every later frame is a no-op.
        self._decided = True

    # -- the view ------------------------------------------------------------

    @property
    def view_id(self) -> str:
        """The conversation the current view belongs to ("" before the first)."""
        return self._view

    @property
    def opened_at_ms(self) -> int:
        """When the current view began, in the unit an ask's ``created_at`` uses."""
        return int(self._opened_at * 1000)

    def begin_view(self, conversation_id: str, *, now: float) -> None:
        """A conversation was opened or switched to — arm its one decision.

        Called on the edge only (the conversation on screen CHANGED), never per
        frame, so a re-render or a queue refresh of the SAME view cannot re-arm it
        (clause 4). A takeover or reload that swaps the session object under the
        same conversation is not a new view either: the user never switched to
        anything.

        ``now`` is wall-clock epoch SECONDS (``time.time()``), not a monotonic
        reading: it is compared with an ask's ``created_at``, which is epoch
        milliseconds stamped by the machine that owns the queue.

        The SAME conversation is not a new view. A takeover, a reload or the
        sidebar's refresh commit swaps the session OBJECT under a conversation the
        user never left, and re-arming there would open a surface the view already
        decided about (or whose dismissal it had just recorded a moment ago).

        A view with no conversation id is never armed: its dismissal could not be
        keyed, so opening it would break clause 4 for exactly the sessions this
        record cannot remember. Production sessions always carry an id.
        """
        if conversation_id and conversation_id == self._view:
            return
        self._view = conversation_id
        self._opened_at = now
        self._decided = not conversation_id or not self._enabled

    def awaiting(self, conversation_id: str) -> bool:
        """Whether this view still owes a decision — the app's cheap per-frame gate.

        Frames land on every state change, so the caller asks this before doing
        any work: once a view has decided, every later frame costs one boolean.
        """
        return bool(conversation_id) and conversation_id == self._view and not self._decided

    def decide(
        self,
        conversation_id: str,
        reading: QueueReading,
        *,
        now: float,
        occupied: bool,
        surface_open: bool,
        settling: bool = False,
        outstanding_ids: Collection[str] | None = None,
    ) -> OpenDecision:
        """The view's one decision, taken from the first frame that can take it.

        ``occupied`` is everything that already has the user's hands or the
        screen: a non-empty composer (the user is typing — and opening would take
        the caret off their sentence), a live prompt, an aside, a full-page mode, a
        modal. Clause 5: auto-open yields to all of them, and it yields for GOOD —
        a user who was busy when the conversation opened is not interrupted a
        minute later by a panel that appears when they stop.

        ``surface_open`` is whether the surface is already up: the user got to the
        door first, so there is nothing left to do (clause 6).

        ``settling`` is a conversation switch that has not finished: the composer
        still holds the PREVIOUS conversation's draft and the new one's has not
        been loaded, so neither "the user is typing" nor "the composer is empty"
        is a fact yet. It is a wait, not a refusal — the same shape as an
        unresolved frame — because the switch ends in a moment and a refusal here
        would spend the one decision on a state the user never saw.

        ``outstanding_ids`` are ALL the asks that can still be answered, as this
        frame shows them. They are what a dismissal is judged against (clause 4):
        held while any waved-off ask is still among them, forgotten when none is.

        ``None`` — the DEFAULT — means the frame cannot name them all: its tally
        counts outstanding asks its rows do not carry (see
        :func:`names_every_outstanding`, and note it is NOT the ``asks_truncated``
        flag), or the caller did not say. A
        dismissal is then HELD, because an ask the bound dropped may be exactly
        the one still outstanding, and releasing on that guess would open a panel
        the user refused (clause 5: never on a guess). Failing closed by default
        means a caller that forgets the ids can only ever under-open, never
        override a refusal.

        A dismissal needs a RESOLVED frame to be judged, so it waits like any other
        decision rather than being spent on an unresolved one: until the frame can
        name the outstanding asks, "is anything I waved off still pending?" has no
        answer.
        """
        if not self.awaiting(conversation_id):
            return OpenDecision.SKIP
        if (
            surface_open  # clause 6: the door got there first
            or now - self._opened_at > self._window_s  # clause 3: too late to be "on open"
        ):
            self._decided = True
            return OpenDecision.SKIP
        if reading is QueueReading.UNRESOLVED or settling:
            return OpenDecision.WAIT  # clause 5: not a guess
        self._decided = True
        waved_off = self._dismissed.get(conversation_id)
        if waved_off is not None:
            if outstanding_ids is None or not waved_off.isdisjoint(outstanding_ids):
                return OpenDecision.SKIP  # clause 4: something they refused is still there
            # Every ask they waved off has left the queue. The refusal was about
            # THOSE asks, so the conversation is a fresh one from here on.
            del self._dismissed[conversation_id]
        if reading is QueueReading.PENDING and not occupied:
            return OpenDecision.OPEN  # clause 2
        return OpenDecision.SKIP  # clauses 1, 3 and 5

    # -- what the user did ---------------------------------------------------

    def note_user_closed(self, conversation_id: str, *, pending_ids: Collection[str]) -> None:
        """The user deliberately closed the surface — remember WHICH asks (clause 4).

        ``pending_ids`` is the pending set at the moment of the close: the asks the
        user looked at and waved off. Only while asks REMAIN — closing a list of
        nothing but settled rows was a glance at history, not a refusal of anything,
        and recording it would suppress the open for the next conversation view
        that does have asks.

        A second dismissal ADDS to the record rather than replacing it: with whole
        frames the two are equivalent (an id that left the queue cannot return), and
        with a frame that dropped rows the union keeps the ids the shorter list could
        not show.

        The record is STICKY for the process: it is not cleared if the user later
        opens the surface by hand. A hand-opened surface that is left behind
        returns to its door like any other, and "I closed it once" is the fact the
        contract asks to be respected. It is forgotten only by :meth:`decide`, when
        none of the waved-off asks is outstanding any more.
        """
        ids = frozenset(pending_ids)
        if not conversation_id or not ids:
            return
        self._dismissed[conversation_id] = self._dismissed.get(conversation_id, frozenset()) | ids
        if conversation_id == self._view:
            self._decided = True

    def note_user_opened(self, conversation_id: str) -> None:
        """The user opened the surface through a door — the policy has nothing to add.

        Settles the current view's decision, so a policy still waiting on an
        unresolved frame cannot open a surface the user has just opened themselves
        (clause 6). Does NOT touch the dismissal record (see
        :meth:`note_user_closed`).
        """
        if conversation_id == self._view:
            self._decided = True

    def is_dismissed(self, conversation_id: str) -> bool:
        """Whether a dismissal is ON RECORD for the conversation.

        "On record", not "still in force": a record is forgotten by :meth:`decide`,
        which is the only place the queue's current ids are known, so one whose asks
        have all left the queue reads ``True`` until the conversation is next viewed.
        """
        return conversation_id in self._dismissed
