"""The TUI's honest liveness term — what a viewer may say about its owner.

WHY THIS MODULE EXISTS. `is_cold` is the TUI's ONLY liveness term today
(``app.py:3059``, ``:6805``, ``:6903``, ``:6912``, ``:7330``, ``:7526``,
``:9048``; ``session_navigation.py:58``), and it is a purely LOCAL predicate:
``self._client is None or not self._client.connected or not
self._ready_for_events``. It has no round trip in it, so a viewer whose owner
stopped answering — a SIGSTOPped runtime, a machine mid-swap — keeps reading
live for as long as its socket stays open, because the kernel keeps the
connection and nothing observable locally changes. #1624 fixed exactly that on
the wire by publishing ``verified_at`` and refusing to claim live without it;
this module is the reader half, so the same honesty reaches the operator's own
screen instead of living only in a frame nobody reads (RR1-4).

THE TERM. ``AttachedSession.verified_at`` — when a round trip to the owner was
last ANSWERED — aged by the reader against ``LIVE_FRESHNESS_BUDGET_S``, plus
``attaching``, which is what lets a viewer tell **"coming"** from **"never"**.

NO ROUND TRIP LIVES HERE, and that is a constraint rather than an omission. The
read path answers in 170.7-350.6 ms and the warm receipt in 6.3-100.6 ms today;
a reader that dialled to look honest would trade the operator's latency target
for truthfulness he already has. Everything below is a clock comparison against
state the session already holds, and :func:`owner_liveness` is synchronous and
I/O-free by construction — ``test_the_reader_never_touches_the_owner`` is that
property, expressed as a test rather than as a promise.

THE COPY IS NOT MINE. :data:`LIVENESS_TEXT` is deliberately the single place the
words live, so the designer's strings are a substitution and not a refactor.
Nothing outside this module should spell a liveness word.
"""

from __future__ import annotations

import enum
import time
from collections.abc import Awaitable, Callable
from typing import Any

from local_operator.session.runtime.types import LIVE_FRESHNESS_BUDGET_S
from local_operator.tui.widgets.tool_card import format_duration

__all__ = [
    "LIVENESS_PROBE_BUDGET_S",
    "LIVENESS_PROBE_EVERY_S",
    "LIVENESS_TEXT",
    "LivenessProbe",
    "LIVE_FRESHNESS_BUDGET_S",
    "OwnerLiveness",
    "liveness_text",
    "owner_liveness",
]


class OwnerLiveness(enum.Enum):
    """What the viewer may honestly say about the session's owner.

    Ordered by precedence in :func:`owner_liveness`, and the order is the
    contract: a fresh answer outranks everything, and "coming" is claimed ONLY
    when a live claim cannot be made — which is the distinction the ``attaching``
    token exists for ("the runtime has accepted us and its state has not arrived
    yet" is not "the runtime is gone").
    """

    #: An answered round trip inside the budget: the owner is there.
    LIVE = "live"
    #: A dial is retained and its canonical state has not arrived yet. COMING,
    #: explicitly not NEVER — the whole reason the token survived review.
    COMING = "coming"
    #: The owner answered once, and has not answered for longer than the budget.
    #: Stale is not dead, and must never be drawn as live.
    STALE = "stale"
    #: No answered round trip, and nothing arriving: the honest "never".
    NEVER = "never"


#: THE ONE PLACE THE WORDS LIVE. These are placeholders chosen to be obviously
#: provisional rather than copy: the designer's three (or four) strings replace
#: the values here and nothing else moves. A caller must read them through
#: :func:`liveness_text` rather than reaching for a literal, so a reword cannot
#: miss a site — the same discipline the status band already uses for its
#: readings ("glyph must not imply a live reading, which is why the WORD beside
#: it is the part …", ``widgets/status_line.py``).
LIVENESS_TEXT: dict[OwnerLiveness, str] = {
    # LIVE renders NOTHING, and that is deliberate rather than an omission: the
    # stamp is a clock, so a live claim is up to LIVE_FRESHNESS_BUDGET_S stale by
    # construction, and painting one would re-introduce the fresh
    # confident-wrong statement this whole phase exists to remove. Absence is
    # already the app's live signal on the sidebar; the band agrees with it. The
    # checkable form: the connection row is present IFF the state is not LIVE.
    OwnerLiveness.LIVE: "",
    OwnerLiveness.COMING: "Connecting\u2026",
    # The AGE is composed by the caller (:func:`liveness_text`), because STALE is
    # the only state that has one -- NEVER has no stamp to measure from, which is
    # itself the fact separating them.
    OwnerLiveness.STALE: "Not answering",
    # "owner", not "runtime": the wire's own token for a pid holding the lease
    # that did not answer is `owner-silent`, so this asserts non-SERVICE, not
    # non-existence -- which is all the stamp can see.
    OwnerLiveness.NEVER: "No owner",
}

#: Cadence and bound for the foreground probe (see :class:`LivenessProbe`).
#:
#: THIRD OF THE BUDGET, so a healthy owner is re-verified well inside the window
#: the reader ages against: at 15 s a probe every 5 s leaves two missed rounds
#: before a verdict could go stale. The BOUND is the repo's own "one socket round
#: trip plus the leg's own work" envelope; a probe has to be cheap enough that a
#: frozen owner does not hold the prober's slot open, and a healthy owner answers
#: `ping` synchronously (the op is exempt from the runtime's op chain).
LIVENESS_PROBE_EVERY_S = LIVE_FRESHNESS_BUDGET_S / 3
LIVENESS_PROBE_BUDGET_S = 2.0


def liveness_text(
    state: OwnerLiveness,
    *,
    now: float | None = None,
    verified_at: float | None = None,
) -> str:
    """The row's text for ``state``, from the single copy table above.

    STALE carries an AGE and it is composed here rather than in the table, so the
    vocabulary stays one word per state and the one state with a measurement gets
    it: ``Not answering · 4m``. The HEAD COMES FIRST deliberately -- the row's
    existing right-side ellipsis then eats the age and never the word, which a
    ``4m · Not answering`` spelling would truncate into nonsense.

    An age is given only when both ``now`` and ``verified_at`` are known. Without
    them STALE degrades to its bare head rather than inventing a number: a
    made-up duration is the same class of confident-wrong statement as a made-up
    liveness, and the head still reads on its own.
    """
    head = LIVENESS_TEXT[state]
    if state is not OwnerLiveness.STALE or verified_at is None:
        return head
    clock = time.time() if now is None else now
    return f"{head} \u00b7 {format_duration(max(0.0, clock - verified_at))}"


def owner_liveness(
    session: Any,
    *,
    now: float | None = None,
    budget: float = LIVE_FRESHNESS_BUDGET_S,
) -> OwnerLiveness:
    """Classify what ``session`` may say about its owner RIGHT NOW.

    ``verified_at`` is read with ``getattr`` and a ``None`` default: an owner-side
    ``Session`` has no stamp (it is not a viewer and never dials), and neither
    does a facade that has never had an answered round trip. Both are honestly
    :attr:`OwnerLiveness.NEVER` rather than an error — the term must be usable on
    every session the status band can be asked to render.

    ``attaching`` is likewise optional, and is consulted only when a live claim
    cannot be made. A resync on a session whose owner answered a moment ago is
    ``LIVE``: the stamp is the evidence, and treating a mid-refresh viewer as
    absent is the conflation ``owner_reachable`` exists to prevent.

    ``now``/``budget`` are parameters so the classification is testable without a
    clock or a runtime. Neither is a hook for behaviour: this function reads
    nothing but its argument and the clock.
    """
    stamp = getattr(session, "verified_at", None)
    if stamp is not None:
        if (now if now is not None else time.time()) - stamp <= budget:
            return OwnerLiveness.LIVE
        return OwnerLiveness.COMING if getattr(session, "attaching", False) else OwnerLiveness.STALE
    if getattr(session, "attaching", False):
        return OwnerLiveness.COMING
    return OwnerLiveness.NEVER


class LivenessProbe:
    """Re-verify the FOREGROUND session's stamp on a cadence, off the paint path.

    THE PRECONDITION THAT MAKES `STALE` TRUE. `verified_at` is written only on a
    wire answer -- a canonical ``FrontendSync``, a display refresh, or
    :meth:`AttachedSession.verify_live` -- and every one of those is
    EVENT-DRIVEN. On a healthy but QUIET session nothing asks and nothing
    answers, so the stamp ages past the budget with nothing wrong, and the row
    would paint ``Not answering · 4m`` on a session that is fine: the age would be
    measuring the absence of QUESTIONS rather than of ANSWERS. That is the same
    confident-wrong class the phase exists to remove, arriving through the fix.

    So this asks. It is the instrument the design already anticipates ("what
    refreshes a viewer's stamp on an idle session is its OWN probe",
    ``session/runtime/types.py``) and the protocol already carries it: ``ping`` is
    op-chain-exempt so a health probe is answered while mutations queue.

    IT NEVER RUNS ON THE PAINT PATH. :meth:`tick` is awaited by a caller that owns
    a cadence (the app's own interval), the paint reads only the resulting clock,
    and the read path's 170.7-350.6 ms / warm receipt's 6.3-100.6 ms band is
    therefore untouched. Nothing here raises: a probe that fails leaves the stamp
    where it was, so a failure NARROWS the window rather than resetting it.
    """

    def __init__(
        self,
        *,
        every: float = LIVENESS_PROBE_EVERY_S,
        budget: float = LIVENESS_PROBE_BUDGET_S,
    ) -> None:
        self.every = every
        self.budget = budget
        self.probes = 0
        self.answers = 0

    def due(self, last: float | None, *, now: float | None = None) -> bool:
        """Whether a probe is owed, given the last one's clock.

        A pure clock comparison, so the caller's cadence can be coarse without
        probing more often than the budget needs.
        """
        if last is None:
            return True
        return (time.time() if now is None else now) - last >= self.every

    async def tick(self, session: Any) -> bool:
        """One bounded probe. Returns whether the owner answered.

        Guarded to an actual viewer: an owner-side ``Session`` has no
        :meth:`verify_live` (it never dials), and calling one would be asking a
        process about itself.
        """
        # Annotated rather than inferred: `getattr` types this `object`, and
        # `callable()` narrows it for the RUNTIME check without giving pyright an
        # `Awaitable` to accept, which is a real error and not a checker
        # preference -- this is the type the guard is asserting.
        verify: Callable[[float], Awaitable[bool]] | None = getattr(session, "verify_live", None)
        if not callable(verify):
            return False
        self.probes += 1
        try:
            answered = bool(await verify(self.budget))
        except Exception:  # noqa: BLE001 -- an unanswered probe is a False, never a crash
            return False
        if answered:
            self.answers += 1
        return answered
